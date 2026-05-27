use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids, OptionGrid, OptionTick};
use proarbitrage::calibration::calibrate_surface;
use proarbitrage::activation::{compute_activation_score, extract_candidate_features, ActivationConfig};
use proarbitrage::portfolio::{
    optimize_portfolio_structured, generate_active_structures, PortfolioConfig
};
use chrono::{NaiveDateTime, Duration};
use std::collections::HashMap;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

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

struct ActiveStructurePosition {
    qty: f64,
    entry_time: NaiveDateTime,
}

fn main() -> anyhow::Result<()> {
    println!("=== starting proarbitrage options-only structured arbitrage backtester ===");

    // Parameters
    let input_path = "data/510300_surface.parquet";
    let test_limit = Some(150_000); // 150k ticks to run a robust historical simulation
    
    // Ingest data
    println!("loading ticks...");
    let start_load = Instant::now();
    let ticks = load_ticks_from_parquet(input_path, test_limit)?;
    println!("loaded {} ticks in {} ms", ticks.len(), start_load.elapsed().as_millis());

    // Reconstruct grids
    let start_grid = Instant::now();
    let grids = reconstruct_grids(&ticks);
    println!("reconstructed {} grids in {} ms", grids.len(), start_grid.elapsed().as_millis());

    // Strategy Parameters
    let activation_config = ActivationConfig::default();
    let portfolio_config = PortfolioConfig::default();
    let mut current_surface = None;
    let r = 0.02; // risk free rate proxy
    let lambda_reg = 0.0001; // regularization parameter
    let lambda_gate = 0.0005; // 5 bps liquidity edge threshold
    let fee_per_contract = 2.0; // 2 CNY transaction fee per contract
    let initial_cash = 100_000.0; // 100,000 CNY initial capital
    let mut cash = initial_cash;

    // Simulation Tracking state
    let mut active_structures: HashMap<String, ActiveStructurePosition> = HashMap::new();
    let mut running_cache: HashMap<ContractKey, OptionTick> = HashMap::new();
    let mut equity_curve: Vec<(NaiveDateTime, f64)> = Vec::new();
    let mut max_equity = initial_cash;
    let mut max_drawdown = 0.0;
    
    // Metrics
    let mut total_traded_contracts = 0;
    let mut total_fees_paid = 0.0;

    // Latency Profiling
    let mut total_calib_time = std::time::Duration::new(0, 0);
    let mut calib_count = 0;
    let mut total_opt_time = std::time::Duration::new(0, 0);
    let mut opt_count = 0;

    let calib_interval = Duration::seconds(1);
    let mut last_calib_time = None;

    println!("running chronological structured trading simulation...");
    let sim_start = Instant::now();

    for (k, grid) in grids.iter().enumerate() {
        let current_time = match parse_date(&grid.date) {
            Some(t) => t,
            None => continue,
        };

        // Update running cache with liquid ticks
        for contract in &grid.contracts {
            if contract.is_liquid && !contract.mid.is_nan() && contract.mid > 0.0 {
                let key = ContractKey::from_tick(contract);
                running_cache.insert(key, contract.clone());
            }
        }

        // Generate active structures dynamically for the current grid
        let active_structs = generate_active_structures(grid, fee_per_contract);

        // 1. Activation Gate & Calibration (Phase 3 & 4)
        let mut should_check_calib = current_surface.is_none();
        if let Some(last_time) = last_calib_time {
            if current_time - last_time >= calib_interval {
                should_check_calib = true;
            }
        } else {
            should_check_calib = true;
        }

        if should_check_calib {
            let mut dense_contracts = Vec::with_capacity(running_cache.len());
            for cached_tick in running_cache.values() {
                let mut tick = cached_tick.clone();
                tick.s_t = grid.s_t; // update spot
                dense_contracts.push(tick);
            }
            let dense_grid = OptionGrid {
                date: grid.date.clone(),
                s_t: grid.s_t,
                contracts: dense_contracts,
            };

            let score = compute_activation_score(&dense_grid, &current_surface, &activation_config);
            let should_calibrate = current_surface.is_none() || score > activation_config.tau_enter;

            if should_calibrate {
                let start = Instant::now();
                if let Ok(surf) = calibrate_surface(&dense_grid, r, lambda_reg) {
                    total_calib_time += start.elapsed();
                    calib_count += 1;
                    current_surface = Some(surf);
                    last_calib_time = Some(current_time);
                }
            }
        }

        // 2. Automated Unwind State Machine (Phase 7 - exit logic)
        let mut liquidated_structures = Vec::new();
        for (name, pos) in active_structures.iter_mut() {
            let holding_period = current_time - pos.entry_time;
            
            // Exit Condition A: Temporal Hard Cutoff (30 minutes)
            let is_hard_breach = holding_period >= Duration::minutes(30);
            
            // Exit Condition B: Temporal Soft Cutoff (15 minutes)
            let is_soft_breach = holding_period >= Duration::minutes(15);

            if let Some(active_struct) = active_structs.iter().find(|s| s.name == *name) {
                // Calculate current net expected return of the structure
                let mut current_alpha = 0.0;
                for &(idx, weight) in &active_struct.legs {
                    let contract = &grid.contracts[idx];
                    let fair_price = if let Some(ref surface) = current_surface {
                        surface.evaluate_contract(contract)
                    } else {
                        contract.mid
                    };
                    
                    let gap = if fair_price > contract.p_a {
                        fair_price - contract.p_a
                    } else if fair_price < contract.p_b {
                        contract.p_b - fair_price
                    } else {
                        0.0
                    };
                    let alpha_leg = 0.8 * gap + 0.1 * contract.spread;
                    current_alpha += weight * alpha_leg;
                }

                // Exit Condition C: Statistical Convergence
                let exit_alpha_threshold = if is_soft_breach { 0.0020 } else { 0.0005 };
                let is_converged = current_alpha < exit_alpha_threshold;

                if is_hard_breach || is_converged {
                    // Close the structure by doing a reverse sweep
                    let delta_z = -pos.qty;
                    let mut max_possible_fill = delta_z.abs();
                    let mut valid_legs = true;

                    for &(idx, weight) in &active_struct.legs {
                        let contract = &grid.contracts[idx];
                        let leg_dir = delta_z.signum() * weight;
                        if leg_dir > 0.0 {
                            if contract.p_a.is_finite() && contract.p_a > 0.0 {
                                max_possible_fill = max_possible_fill.min(contract.a_v_eff as f64);
                            } else {
                                valid_legs = false;
                            }
                        } else if leg_dir < 0.0 {
                            if contract.p_b.is_finite() && contract.p_b > 0.0 {
                                max_possible_fill = max_possible_fill.min(contract.b_v_eff as f64);
                            } else {
                                valid_legs = false;
                            }
                        }
                    }

                    // Fallback force close at mid if order book is dry
                    let fill_qty = if valid_legs && max_possible_fill >= 1.0 {
                        max_possible_fill.min(pos.qty.abs())
                    } else {
                        pos.qty.abs()
                    };

                    for &(idx, weight) in &active_struct.legs {
                        let contract = &grid.contracts[idx];
                        let leg_dir = delta_z.signum() * weight;
                        let total_leg_qty = fill_qty * weight.abs();

                        if leg_dir > 0.0 {
                            let p = if contract.p_a.is_finite() && contract.p_a > 0.0 { contract.p_a } else { contract.mid };
                            cash -= p * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                        } else if leg_dir < 0.0 {
                            let p = if contract.p_b.is_finite() && contract.p_b > 0.0 { contract.p_b } else { contract.mid };
                            cash += p * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                        }
                        total_fees_paid += total_leg_qty * fee_per_contract;
                        total_traded_contracts += total_leg_qty as u64;
                    }

                    pos.qty += fill_qty * delta_z.signum();
                    if pos.qty.abs() < 1e-5 {
                        liquidated_structures.push(name.clone());
                    }
                }
            } else {
                liquidated_structures.push(name.clone());
            }
        }
        for name in liquidated_structures {
            active_structures.remove(&name);
        }

        // 3. Expected Return Scoring & Portfolio Optimization (Phase 5 & 6)
        if let Some(ref surface) = current_surface {
            let mut alpha_scores = Vec::with_capacity(grid.contracts.len());
            let mut has_signals = false;

            for contract in &grid.contracts {
                if contract.is_liquid && !contract.mid.is_nan() && contract.mid > 0.0 {
                    if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                        let mut alpha = 0.85 * feat.immediate_execution_gap;
                        if contract.option_type == 'P' {
                            alpha += 0.12 * contract.spread;
                        } else {
                            alpha += 0.02 * contract.spread;
                        }
                        alpha -= 0.05 * (contract.strike - grid.s_t).powi(2);

                        if alpha > 0.0065 {
                            alpha_scores.push(alpha);
                            has_signals = true;
                        } else {
                            alpha_scores.push(0.0);
                        }
                    } else {
                        alpha_scores.push(0.0);
                    }
                } else {
                    alpha_scores.push(0.0);
                }
            }

            if has_signals && !active_structs.is_empty() {
                let start_opt = Instant::now();
                if let Ok(target_weights) = optimize_portfolio_structured(
                    grid,
                    surface,
                    &alpha_scores,
                    &active_structs,
                    &portfolio_config,
                ) {
                    total_opt_time += start_opt.elapsed();
                    opt_count += 1;

                    // 4. Microstructural Order Execution Mapping (Phase 7)
                    for (idx, &target_z) in target_weights.iter().enumerate() {
                        let active_struct = &active_structs[idx];
                        let current_pos = active_structures.get(&active_struct.name).map(|p| p.qty).unwrap_or(0.0);
                        
                        let delta_z = target_z - current_pos;
                        
                        // Anti-Flicker Rebalancing deadband (5 contracts minimum adjustment)
                        if delta_z.abs() >= 5.0 {
                            let mut max_possible_fill = delta_z.abs();
                            let mut valid_legs = true;

                            for &(c_idx, weight) in &active_struct.legs {
                                let contract = &grid.contracts[c_idx];
                                let leg_dir = delta_z.signum() * weight;
                                if leg_dir > 0.0 {
                                    if contract.p_a.is_finite() && contract.p_a > 0.0 {
                                        max_possible_fill = max_possible_fill.min(contract.a_v_eff as f64);
                                    } else {
                                        valid_legs = false;
                                    }
                                } else if leg_dir < 0.0 {
                                    if contract.p_b.is_finite() && contract.p_b > 0.0 {
                                        max_possible_fill = max_possible_fill.min(contract.b_v_eff as f64);
                                    } else {
                                        valid_legs = false;
                                    }
                                }
                            }

                            if valid_legs && max_possible_fill >= 1.0 {
                                let fill_qty = max_possible_fill.min(5.0); // Capped sweep size per tick
                                
                                // Execution and Cash Adjustment
                                for &(c_idx, weight) in &active_struct.legs {
                                    let contract = &grid.contracts[c_idx];
                                    let leg_dir = delta_z.signum() * weight;
                                    let total_leg_qty = fill_qty * weight.abs();

                                    if leg_dir > 0.0 {
                                        cash -= contract.p_a * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                                    } else if leg_dir < 0.0 {
                                        cash += contract.p_b * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                                    }
                                    total_fees_paid += total_leg_qty * fee_per_contract;
                                    total_traded_contracts += total_leg_qty as u64;
                                }

                                active_structures.entry(active_struct.name.clone())
                                    .and_modify(|pos| {
                                        pos.qty += fill_qty * delta_z.signum();
                                    })
                                    .or_insert(ActiveStructurePosition {
                                        qty: fill_qty * delta_z.signum(),
                                        entry_time: current_time,
                                    });
                            }
                        }
                    }
                }
            }
        }

        // Clean up empty positions
        active_structures.retain(|_, pos| pos.qty.abs() > 1e-5);

        // 5. Track Portfolio Value and Greeks (Phase 8)
        let mut portfolio_value = cash;
        let mut contract_positions: HashMap<ContractKey, f64> = HashMap::new();
        for (struct_name, pos) in &active_structures {
            if let Some(active_struct) = active_structs.iter().find(|s| s.name == *struct_name) {
                for &(idx, weight) in &active_struct.legs {
                    let c = &grid.contracts[idx];
                    let key = ContractKey::from_tick(c);
                    *contract_positions.entry(key).or_default() += pos.qty * weight;
                }
            }
        }

        for (key, pos_qty) in &contract_positions {
            if let Some(tick) = running_cache.get(key) {
                if tick.mid.is_finite() && tick.mid > 0.0 {
                    portfolio_value += pos_qty * tick.mid * 100.0;
                }
            }
        }

        if k % 1000 == 0 {
            equity_curve.push((current_time, portfolio_value));
            if portfolio_value > max_equity {
                max_equity = portfolio_value;
            }
            let dd = (max_equity - portfolio_value) / max_equity;
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }
    }

    let sim_duration = sim_start.elapsed();
    println!("simulation complete in {} ms!", sim_duration.as_millis());

    // Calculate Backtest Metrics
    let final_equity = equity_curve.last().map(|e| e.1).unwrap_or(cash);
    let net_profit = final_equity - initial_cash;
    let total_return = (net_profit / initial_cash) * 100.0;
    
    let mut returns = Vec::new();
    for i in 1..equity_curve.len() {
        let prev = equity_curve[i-1].1;
        let curr = equity_curve[i].1;
        returns.push((curr - prev) / prev);
    }
    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    let std_dev = variance.sqrt();
    
    let sharpe_ratio = if std_dev > 0.0 {
        (mean_ret / std_dev) * (252.0 * 240.0_f64).sqrt()
    } else {
        0.0
    };

    println!("\n================ BACKTEST PERFORMANCE SUMMARY ================");
    println!("  Initial Portfolio Value : {:.2} CNY", initial_cash);
    println!("  Final Portfolio Value   : {:.2} CNY", final_equity);
    println!("  Total Cumulative Return : {:.4} %", total_return);
    println!("  Max Peak-to-Trough DD   : {:.4} %", max_drawdown * 100.0);
    println!("  Annualized Sharpe Ratio : {:.4}", sharpe_ratio);
    println!("  Total Traded Contracts  : {}", total_traded_contracts);
    println!("  Total Fees Paid         : {:.2} CNY", total_fees_paid);
    println!("===============================================================");

    println!("\n=============== HIGH-SPEED LATENCY PROFILE ===============");
    if calib_count > 0 {
        println!("  Avg Calibration Latency  : {:.2} us (triggered {} times)", total_calib_time.as_micros() as f64 / calib_count as f64, calib_count);
    }
    if opt_count > 0 {
        println!("  Avg Portfolio LP Latency : {:.2} us (solved {} times)", total_opt_time.as_micros() as f64 / opt_count as f64, opt_count);
    }
    println!("  Avg Total Tick Loop      : {:.2} us", sim_duration.as_micros() as f64 / grids.len() as f64);
    println!("==========================================================");

    // Save equity curve to CSV
    let mut csv_file = File::create("data/backtest_equity.csv")?;
    writeln!(csv_file, "date,equity")?;
    for (dt, eq) in equity_curve {
        writeln!(csv_file, "{},{:.4}", dt, eq)?;
    }
    println!("saved equity curve to data/backtest_equity.csv");

    Ok(())
}
