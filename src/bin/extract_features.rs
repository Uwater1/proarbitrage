use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids, OptionGrid};
use proarbitrage::calibration::calibrate_surface;
use proarbitrage::activation::{compute_activation_score, extract_candidate_features, ActivationConfig};
use chrono::NaiveDateTime;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::env;

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

struct ValidGrid<'a> {
    index: usize,
    time: NaiveDateTime,
    grid: &'a OptionGrid,
}

struct TargetPointer {
    ptr: usize,
    target_seconds: i64,
    tolerance_seconds: i64,
}

impl TargetPointer {
    fn new(target_seconds: i64, tolerance_seconds: i64) -> Self {
        Self {
            ptr: 0,
            target_seconds,
            tolerance_seconds,
        }
    }

    fn advance(&mut self, current_time: NaiveDateTime, valid_grids: &[ValidGrid]) -> Option<usize> {
        let target_time = current_time + chrono::Duration::seconds(self.target_seconds);
        let start_time = target_time - chrono::Duration::seconds(self.tolerance_seconds);
        
        // Move sliding pointer forward to target_time - tolerance
        while self.ptr < valid_grids.len() && valid_grids[self.ptr].time < start_time {
            self.ptr += 1;
        }
        
        let mut best_idx = None;
        let mut min_diff = i64::MAX;
        let mut scan_ptr = self.ptr;
        
        // Look within the tolerance window
        while scan_ptr < valid_grids.len() {
            let t = valid_grids[scan_ptr].time;
            let diff = (t - target_time).num_seconds().abs();
            if diff <= self.tolerance_seconds {
                if diff < min_diff {
                    min_diff = diff;
                    best_idx = Some(valid_grids[scan_ptr].index);
                }
                scan_ptr += 1;
            } else {
                break;
            }
        }
        
        best_idx
    }
}

fn find_future_mid(grid: &OptionGrid, option_type: char, strike: f64, expiry: &str) -> Option<f64> {
    for contract in &grid.contracts {
        if contract.option_type == option_type 
            && (contract.strike - strike).abs() < 1e-6 
            && contract.expiry == expiry 
        {
            return Some(contract.mid);
        }
    }
    None
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

    // 2. Reconstruct grids
    let grids = reconstruct_grids(&ticks);
    println!("Reconstructed {} chronological grids", grids.len());

    // 3. Pre-parse and filter grids that have valid times
    let mut valid_grids = Vec::with_capacity(grids.len());
    for (idx, grid) in grids.iter().enumerate() {
        if let Some(time) = parse_date(&grid.date) {
            valid_grids.push(ValidGrid {
                index: idx,
                time,
                grid,
            });
        }
    }
    println!("Found {} valid grids with parsed timestamps.", valid_grids.len());

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

    // Initialize monotonically sliding window pointers
    let mut pointer_1m = TargetPointer::new(60, 30);
    let mut pointer_3m = TargetPointer::new(180, 30);
    let mut pointer_5m = TargetPointer::new(300, 30);
    let mut pointer_10m = TargetPointer::new(600, 30);

    let start_extract = Instant::now();
    let mut total_records = 0;

    for (k, vg) in valid_grids.iter().enumerate() {
        let grid = vg.grid;
        
        // Calculate activation score
        let score = compute_activation_score(grid, &current_surface, &config);
        let should_calibrate = current_surface.is_none() || score > config.tau_enter;

        if should_calibrate {
            if let Ok(surf) = calibrate_surface(grid, r, lambda_reg) {
                current_surface = Some(surf);
            }
        }

        if let Some(ref surface) = current_surface {
            // Find future grid indices using high-speed sliding window
            let idx_1m = pointer_1m.advance(vg.time, &valid_grids);
            let idx_3m = pointer_3m.advance(vg.time, &valid_grids);
            let idx_5m = pointer_5m.advance(vg.time, &valid_grids);
            let idx_10m = pointer_10m.advance(vg.time, &valid_grids);

            for contract in &grid.contracts {
                if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                    // Extract targets
                    let fut_1m = idx_1m.and_then(|idx| find_future_mid(&grids[idx], contract.option_type, contract.strike, &contract.expiry));
                    let fut_3m = idx_3m.and_then(|idx| find_future_mid(&grids[idx], contract.option_type, contract.strike, &contract.expiry));
                    let fut_5m = idx_5m.and_then(|idx| find_future_mid(&grids[idx], contract.option_type, contract.strike, &contract.expiry));
                    let fut_10m = idx_10m.and_then(|idx| find_future_mid(&grids[idx], contract.option_type, contract.strike, &contract.expiry));

                    // Only output if we have at least one valid future target to train on
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

        if k > 0 && k % 50000 == 0 {
            println!("Processed {} valid grids, extracted {} records...", k, total_records);
        }
    }

    writer.flush()?;
    println!("Extraction complete! Total records saved: {}", total_records);
    println!("Extraction duration: {} ms", start_extract.elapsed().as_millis());
    Ok(())
}
