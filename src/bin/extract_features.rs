use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids, OptionGrid, OptionTick};
use proarbitrage::calibration::calibrate_surface;
use proarbitrage::activation::{compute_activation_score, extract_candidate_features, ActivationConfig};
use chrono::NaiveDateTime;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::env;
use std::collections::HashMap;

fn parse_date(s: &str) -> Option<NaiveDateTime> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.3f") {
        return Some(dt);
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.6f") {
        return Some(dt);
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt);
    }
    if s.contains('+') {
        let parts: Vec<&str> = s.split('+').collect();
        if let Ok(dt) = NaiveDateTime::parse_from_str(parts[0], "%Y-%m-%d %H:%M:%S%.3f") {
            return Some(dt);
        }
        if let Ok(dt) = NaiveDateTime::parse_from_str(parts[0], "%Y-%m-%d %H:%M:%S") {
            return Some(dt);
        }
    }
    None
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct ContractKey {
    option_type: char,
    strike_micro: u64,
    expiry: String,
}

impl ContractKey {
    fn from_tick(tick: &OptionTick) -> Self {
        Self {
            option_type: tick.option_type,
            strike_micro: (tick.strike * 1_000_000.0).round() as u64,
            expiry: tick.expiry.clone(),
        }
    }
}

fn find_future_price_hist(
    price_history: &HashMap<ContractKey, Vec<(NaiveDateTime, f64)>>,
    key: &ContractKey,
    target_time: NaiveDateTime,
    tolerance_seconds: i64,
) -> Option<f64> {
    let history = price_history.get(key)?;
    if history.is_empty() {
        return None;
    }
    
    // Find index of first element >= target_time
    let idx = history.partition_point(|x| x.0 < target_time);
    
    let mut best_price = None;
    let mut min_diff = i64::MAX;
    
    if idx < history.len() {
        let diff = (history[idx].0 - target_time).num_seconds().abs();
        if diff <= tolerance_seconds && diff < min_diff {
            min_diff = diff;
            best_price = Some(history[idx].1);
        }
    }
    
    if idx > 0 {
        let diff = (history[idx - 1].0 - target_time).num_seconds().abs();
        if diff <= tolerance_seconds && diff < min_diff {
            min_diff = diff;
            best_price = Some(history[idx - 1].1);
        }
    }
    
    best_price
}

fn main() -> anyhow::Result<()> {
    println!("=== proarbitrage high-speed feature and target extraction ===");

    // Parse arguments manually to avoid clap rebuild overhead
    let args: Vec<String> = env::args().collect();
    let mut input_path = "data/510300_surface.parquet".to_string();
    let mut output_path = "data/510300_extracted.csv".to_string();
    let mut limit = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                if i + 1 < args.len() {
                    input_path = args[i + 1].clone();
                    i += 2;
                } else {
                    anyhow::bail!("Missing value for --input");
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 2;
                } else {
                    anyhow::bail!("Missing value for --output");
                }
            }
            "--limit" | "-l" => {
                if i + 1 < args.len() {
                    limit = Some(args[i + 1].parse::<usize>()?);
                    i += 2;
                } else {
                    anyhow::bail!("Missing value for --limit");
                }
            }
            _ => {
                println!("Unknown argument: {}, skipping", args[i]);
                i += 1;
            }
        }
    }

    println!("Input path:  {}", input_path);
    println!("Output path: {}", output_path);
    if let Some(l) = limit {
        println!("Limit:       {} ticks", l);
    } else {
        println!("Limit:       None (full file)");
    }

    // 1. Load ticks
    let start_load = Instant::now();
    let ticks = load_ticks_from_parquet(&input_path, limit)?;
    println!("Loaded {} ticks in {} ms", ticks.len(), start_load.elapsed().as_millis());

    // 2. Build global chronological price history map (using only liquid, non-NaN ticks)
    let start_hist = Instant::now();
    let mut price_history: HashMap<ContractKey, Vec<(NaiveDateTime, f64)>> = HashMap::new();
    for tick in &ticks {
        if tick.is_liquid && !tick.mid.is_nan() && tick.mid > 0.0 {
            if let Some(dt) = parse_date(&tick.date) {
                let key = ContractKey::from_tick(tick);
                price_history.entry(key).or_default().push((dt, tick.mid));
            }
        }
    }
    println!("Built history map for {} unique contracts in {} ms", price_history.len(), start_hist.elapsed().as_millis());

    // 3. Reconstruct microsecond-level chronological grids
    let grids = reconstruct_grids(&ticks);
    println!("Reconstructed {} chronological timestamp groups", grids.len());

    // 4. Open output CSV
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "date,option_type,strike,expiry,immediate_execution_gap,spot,moneyness,tau,is_put,spread,target_1m,target_3m,target_5m,target_10m"
    )?;

    // Setup calibration / activation
    let config = ActivationConfig::default();
    let mut current_surface = None;
    let r = 0.02;
    let lambda_reg = 0.0001;
    let lambda_gate = 0.0005; // 5 bps

    // Running cache representing dense state of option surface
    let mut running_cache: HashMap<ContractKey, OptionTick> = HashMap::new();

    let start_extract = Instant::now();
    let mut total_records = 0;

    for (k, grid) in grids.iter().enumerate() {
        let current_time = match parse_date(&grid.date) {
            Some(t) => t,
            None => continue,
        };

        // Update running cache with liquid ticks from this timestamp group
        for contract in &grid.contracts {
            if contract.is_liquid && !contract.mid.is_nan() && contract.mid > 0.0 {
                let key = ContractKey::from_tick(contract);
                running_cache.insert(key, contract.clone());
            }
        }

        // Build dense grid at current spot and time to maturity
        let mut dense_contracts = Vec::with_capacity(running_cache.len());
        for cached_tick in running_cache.values() {
            let mut tick = cached_tick.clone();
            tick.s_t = grid.s_t; // Update spot to current
            dense_contracts.push(tick);
        }
        
        let dense_grid = OptionGrid {
            date: grid.date.clone(),
            s_t: grid.s_t,
            contracts: dense_contracts,
        };

        // Calculate activation score on dense grid state
        let score = compute_activation_score(&dense_grid, &current_surface, &config);
        let should_calibrate = current_surface.is_none() || score > config.tau_enter;

        if should_calibrate {
            if let Ok(surf) = calibrate_surface(&dense_grid, r, lambda_reg) {
                current_surface = Some(surf);
            }
        }

        // Extract features only for contracts that actually updated (ticked) in this group
        if let Some(ref surface) = current_surface {
            for contract in &grid.contracts {
                if contract.is_liquid && !contract.mid.is_nan() && contract.mid > 0.0 {
                    if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                        let key = ContractKey::from_tick(contract);

                        // Look up chronological targets using binary search on history
                        let fut_1m = find_future_price_hist(&price_history, &key, current_time + chrono::Duration::seconds(60), 30);
                        let fut_3m = find_future_price_hist(&price_history, &key, current_time + chrono::Duration::seconds(180), 30);
                        let fut_5m = find_future_price_hist(&price_history, &key, current_time + chrono::Duration::seconds(300), 30);
                        let fut_10m = find_future_price_hist(&price_history, &key, current_time + chrono::Duration::seconds(600), 30);

                        // Only output if we have at least one valid future target
                        if fut_1m.is_some() || fut_3m.is_some() || fut_5m.is_some() || fut_10m.is_some() {
                            let t_1m = fut_1m.map(|f| f - contract.mid).unwrap_or(0.0);
                            let t_3m = fut_3m.map(|f| f - contract.mid).unwrap_or(0.0);
                            let t_5m = fut_5m.map(|f| f - contract.mid).unwrap_or(0.0);
                            let t_10m = fut_10m.map(|f| f - contract.mid).unwrap_or(0.0);

                            writeln!(
                                writer,
                                "{},{},{:.4},{},{:.5},{:.4},{:.4},{:.5},{:.0},{:.5},{:.5},{:.5},{:.5},{:.5}",
                                grid.date,
                                contract.option_type,
                                contract.strike,
                                contract.expiry,
                                feat.immediate_execution_gap,
                                feat.spot,
                                feat.moneyness,
                                feat.tau,
                                feat.is_put,
                                feat.spread,
                                t_1m,
                                t_3m,
                                t_5m,
                                t_10m
                            )?;
                            total_records += 1;
                        }
                    }
                }
            }
        }

        if k > 0 && k % 50000 == 0 {
            println!("Processed {} timestamp groups, extracted {} records...", k, total_records);
        }
    }

    writer.flush()?;
    println!("Extraction complete! Total records saved: {}", total_records);
    println!("Extraction duration: {} ms", start_extract.elapsed().as_millis());
    Ok(())
}
