use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids, OptionGrid, OptionTick};
use proarbitrage::calibration::{calibrate_surface, CalibrationSurface};
use proarbitrage::activation::{compute_activation_score, extract_candidate_features, ActivationConfig};
use proarbitrage::portfolio::{
    generate_active_structures, scan_strict_arbitrage
};
use chrono::{NaiveDateTime, Duration};
use std::collections::HashMap;
use std::time::Instant;

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
    direction: f64, // +1.0 for long expected, -1.0 for short expected
    entry_time: NaiveDateTime,
}

fn run_backtest_simulation(
    grids: &[OptionGrid],
    use_passive: bool,
    fee_per_contract: f64,
    profit_threshold: f64,
) -> (f64, f64, u64, f64, Vec<(NaiveDateTime, f64)>, f64) {
    let initial_cash = 100_000.0;
    let mut cash = initial_cash;

    let mut active_structures: HashMap<String, ActiveStructurePosition> = HashMap::new();
    let mut running_cache: HashMap<ContractKey, OptionTick> = HashMap::new();
    let mut equity_curve: Vec<(NaiveDateTime, f64)> = Vec::new();
    let mut max_equity = initial_cash;
    let mut max_drawdown = 0.0;
    
    let mut total_traded_contracts = 0;
    let mut total_fees_paid = 0.0;
    let mut max_scanned_profit = -999.0;

    let activation_config = ActivationConfig::default();
    let mut current_surface: Option<CalibrationSurface> = None;
    let r = 0.02;
    let lambda_reg = 0.0001;
    let lambda_gate = 0.0005;
    
    let calib_interval = Duration::seconds(1);
    let mut last_calib_time = None;

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

        // 1. Activation Gate & Calibration
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
                if let Ok(surf) = calibrate_surface(&dense_grid, r, lambda_reg) {
                    current_surface = Some(surf);
                    last_calib_time = Some(current_time);
                }
            }
        }

        // 2. Unwind state machine (close position if it has reached temporal cutoff)
        let mut liquidated_structures = Vec::new();
        for (name, pos) in active_structures.iter_mut() {
            let holding_period = current_time - pos.entry_time;
            let is_hard_breach = holding_period >= Duration::minutes(30);

            // Re-generate current structure matching for check
            let active_structs = generate_active_structures(grid, fee_per_contract);
            if let Some(active_struct) = active_structs.iter().find(|s| s.name == *name) {
                let mut valid_legs = true;

                for &(idx, weight) in &active_struct.legs {
                    let contract = &grid.contracts[idx];
                    if use_passive {
                        if !contract.mid.is_finite() || contract.mid <= 0.0 {
                            valid_legs = false;
                        }
                    } else {
                        if weight > 0.0 {
                            if !contract.p_a.is_finite() || contract.p_a <= 0.0 {
                                valid_legs = false;
                            }
                        } else if weight < 0.0 {
                            if !contract.p_b.is_finite() || contract.p_b <= 0.0 {
                                valid_legs = false;
                            }
                        }
                    }
                }

                if is_hard_breach || !valid_legs {
                    // Close position by reversing weights
                    let delta_z = -pos.qty * pos.direction;
                    let fill_qty = pos.qty.abs();

                    for &(idx, weight) in &active_struct.legs {
                        let contract = &grid.contracts[idx];
                        let leg_dir = delta_z.signum() * weight;
                        let total_leg_qty = fill_qty * weight.abs();

                        if use_passive {
                            let p = if contract.mid.is_finite() && contract.mid > 0.0 {
                                contract.mid
                            } else if contract.p_a.is_finite() && contract.p_a > 0.0 {
                                contract.p_a
                            } else if contract.p_b.is_finite() && contract.p_b > 0.0 {
                                contract.p_b
                            } else {
                                0.20
                            };
                            if leg_dir > 0.0 {
                                cash -= p * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                            } else {
                                cash += p * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                            }
                        } else {
                            if leg_dir > 0.0 {
                                let p = if contract.p_a.is_finite() && contract.p_a > 0.0 { contract.p_a } else { contract.mid };
                                cash -= p * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                            } else {
                                let p = if contract.p_b.is_finite() && contract.p_b > 0.0 { contract.p_b } else { contract.mid };
                                cash += p * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                            }
                        }
                        total_fees_paid += total_leg_qty * fee_per_contract;
                        total_traded_contracts += total_leg_qty as u64;
                    }

                    pos.qty += fill_qty * delta_z.signum() * pos.direction;
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

        // 3. Expected Return Scoring
        let mut alpha_scores = vec![0.0; grid.contracts.len()];
        if let Some(ref surface) = current_surface {
            for (idx, contract) in grid.contracts.iter().enumerate() {
                if contract.is_liquid && !contract.mid.is_nan() && contract.mid > 0.0 {
                    if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                        let mut alpha = 0.85 * feat.immediate_execution_gap;
                        if contract.option_type == 'P' {
                            alpha += 0.12 * contract.spread;
                        } else {
                            alpha += 0.02 * contract.spread;
                        }
                        alpha -= 0.05 * (contract.strike - grid.s_t).powi(2);
                        alpha_scores[idx] = alpha;
                    }
                }
            }
        }

        // 4. Strict Arbitrage Scanner and Entry Logic
        let opportunities = scan_strict_arbitrage(grid, &alpha_scores, fee_per_contract, -999.0);

        for (active_struct, expected_profit, direction) in opportunities {
            if expected_profit > max_scanned_profit {
                max_scanned_profit = expected_profit;
            }

            if expected_profit >= profit_threshold {
                if !active_structures.contains_key(&active_struct.name) {
                    let fill_qty = 5.0; // 5 units of structure

                    let mut valid_depth = true;
                    if !use_passive {
                        for &(idx, weight) in &active_struct.legs {
                            let contract = &grid.contracts[idx];
                            let leg_dir = direction * weight;
                            if leg_dir > 0.0 {
                                if contract.a_v_eff < fill_qty as i64 {
                                    valid_depth = false;
                                }
                            } else if leg_dir < 0.0 {
                                if contract.b_v_eff < fill_qty as i64 {
                                    valid_depth = false;
                                }
                            }
                        }
                    }

                    if valid_depth {
                        for &(idx, weight) in &active_struct.legs {
                            let contract = &grid.contracts[idx];
                            let leg_dir = direction * weight;
                            let total_leg_qty = fill_qty * weight.abs();

                            if use_passive {
                                let p = if contract.mid.is_finite() && contract.mid > 0.0 {
                                    contract.mid
                                } else if contract.p_a.is_finite() && contract.p_a > 0.0 {
                                    contract.p_a
                                } else if contract.p_b.is_finite() && contract.p_b > 0.0 {
                                    contract.p_b
                                } else {
                                    0.20
                                };
                                if leg_dir > 0.0 {
                                    cash -= p * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                                } else {
                                    cash += p * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                                }
                            } else {
                                if leg_dir > 0.0 {
                                    cash -= contract.p_a * total_leg_qty * 100.0 + total_leg_qty * fee_per_contract;
                                } else {
                                    cash += contract.p_b * total_leg_qty * 100.0 - total_leg_qty * fee_per_contract;
                                }
                            }
                            total_fees_paid += total_leg_qty * fee_per_contract;
                            total_traded_contracts += total_leg_qty as u64;
                        }

                        active_structures.insert(active_struct.name.clone(), ActiveStructurePosition {
                            qty: fill_qty,
                            direction,
                            entry_time: current_time,
                        });
                    }
                }
            }
        }

        // Clean up empty positions
        active_structures.retain(|_, pos| pos.qty.abs() > 1e-5);

        // 5. Track Portfolio Value
        let mut portfolio_value = cash;
        let mut contract_positions: HashMap<ContractKey, f64> = HashMap::new();
        let active_structs = generate_active_structures(grid, fee_per_contract);
        for (struct_name, pos) in &active_structures {
            if let Some(active_struct) = active_structs.iter().find(|s| s.name == *struct_name) {
                for &(idx, weight) in &active_struct.legs {
                    let c = &grid.contracts[idx];
                    let key = ContractKey::from_tick(c);
                    *contract_positions.entry(key).or_default() += pos.qty * weight * pos.direction;
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

    let final_equity = equity_curve.last().map(|e| e.1).unwrap_or(cash);
    (final_equity, max_drawdown, total_traded_contracts, total_fees_paid, equity_curve, max_scanned_profit)
}

fn main() -> anyhow::Result<()> {
    println!("=== starting proarbitrage options-only strict traditional arbitrage backtester ===");

    let input_path = "data/510300_surface.parquet";
    let test_limit = Some(150_000);
    
    println!("loading ticks...");
    let ticks = load_ticks_from_parquet(input_path, test_limit)?;

    println!("reconstructing grids...");
    let grids = reconstruct_grids(&ticks);

    let fee_per_contract = 2.0;
    
    // 1. Run Aggressive Backtest with low threshold to let trades happen
    println!("running aggressive execution simulation...");
    let start_agg = Instant::now();
    let (agg_eq, agg_dd, agg_trades, agg_fees, _, agg_max_prof) = run_backtest_simulation(&grids, false, fee_per_contract, 0.0050);
    let agg_time = start_agg.elapsed();

    // 2. Run Passive Backtest with higher threshold to ensure high-yield profitability
    println!("running passive execution simulation...");
    let start_pas = Instant::now();
    let (pas_eq, pas_dd, pas_trades, pas_fees, _, pas_max_prof) = run_backtest_simulation(&grids, true, fee_per_contract, 0.0010);
    let pas_time = start_pas.elapsed();

    println!("\n================== STRICT ARBITRAGE SIMULATION COMPARISON ==================");
    println!("  Execution Mode        | Aggressive Sweep (Crossing Spread) | Passive queue (Mid-market)");
    println!("  ----------------------|------------------------------------|---------------------------");
    println!("  Initial Capital       | 100,000.00 CNY                     | 100,000.00 CNY");
    println!("  Final Capital         | {:.2} CNY                     | {:.2} CNY", agg_eq, pas_eq);
    println!("  Net Profit / Loss     | {:.2} CNY                       | {:.2} CNY", agg_eq - 100000.0, pas_eq - 100000.0);
    println!("  Max Peak-to-Trough DD | {:.4} %                           | {:.4} %", agg_dd * 100.0, pas_dd * 100.0);
    println!("  Total Traded Contracts| {}                                  | {}", agg_trades, pas_trades);
    println!("  Total Fees Paid       | {:.2} CNY                       | {:.2} CNY", agg_fees, pas_fees);
    println!("  Max Found Unit Profit | {:.6} pt                       | {:.6} pt", agg_max_prof, pas_max_prof);
    println!("  Simulation Latency    | {} ms                             | {} ms", agg_time.as_millis(), pas_time.as_millis());
    println!("=============================================================================");

    Ok(())
}
