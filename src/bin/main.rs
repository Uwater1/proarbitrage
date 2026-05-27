use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids, OptionTick};
use proarbitrage::calibration::{calibrate_surface, CalibrationSurface};
use proarbitrage::activation::{
    compute_activation_score, extract_candidate_features, ActivationConfig,
};
use proarbitrage::portfolio::{optimize_portfolio, PortfolioConfig, calculate_greeks};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== proarbitrage integration test: phases 1-6 ===");

    // 1. Ingest sample ticks from parquet file
    let path = "data/510300_surface.parquet";
    println!("loading ticks from: {}", path);
    let start_load = Instant::now();
    // Load first 20,000 rows for verification
    let ticks = load_ticks_from_parquet(path, Some(20_000))?;
    let load_time = start_load.elapsed();
    println!(
        "loaded {} ticks in {} ms",
        ticks.len(),
        load_time.as_millis()
    );

    // 2. Reconstruct option grids and measure matrix construction latency
    let start_matrix = Instant::now();
    let grids = reconstruct_grids(&ticks);
    let matrix_time = start_matrix.elapsed();
    println!("reconstructed {} chronological grids in {:.2} ms", grids.len(), matrix_time.as_secs_f64() * 1000.0);
    
    // Average matrix construction time per tick
    let avg_matrix_per_tick_ns = if !ticks.is_empty() {
        matrix_time.as_nanos() as f64 / ticks.len() as f64
    } else {
        0.0
    };
    println!("average matrix reconstruction latency per tick: {:.2} ns", avg_matrix_per_tick_ns);

    if grids.is_empty() {
        println!("no grids reconstructed! exiting.");
        return Ok(());
    }

    // 3. Process grids sequentially to measure calibration, greeks, and portfolio LP latencies
    let activation_config = ActivationConfig::default();
    let portfolio_config = PortfolioConfig::default();
    let mut current_surface = None;
    
    let r = 0.02; // 2% risk-free rate proxy
    let lambda_reg = 0.0001; // regularization parameter
    let lambda_gate = 0.0005; // liquidity edge threshold (5 bps)

    let mut total_calibrations = 0;
    let mut total_calibration_duration = std::time::Duration::new(0, 0);
    
    let mut total_greeks_calculated = 0;
    let mut total_greeks_duration = std::time::Duration::new(0, 0);

    let mut total_portfolios_optimized = 0;
    let mut total_portfolio_duration = std::time::Duration::new(0, 0);

    let mut total_ticks_processed = 0;
    let mut total_tick_processing_duration = std::time::Duration::new(0, 0);

    println!("processing grids chronologically...");
    for (i, grid) in grids.iter().enumerate() {
        let tick_start = Instant::now();

        // Step A: Calculate activation score
        let score = compute_activation_score(grid, &current_surface, &activation_config);
        
        let should_calibrate = current_surface.is_none() || score > activation_config.tau_enter;

        // Step B: LP Surface calibration (if activated)
        if should_calibrate {
            let start_calib = Instant::now();
            match calibrate_surface(grid, r, lambda_reg) {
                Ok(surf) => {
                    let elapsed = start_calib.elapsed();
                    total_calibration_duration += elapsed;
                    total_calibrations += 1;
                    
                    if total_calibrations <= 3 {
                        println!(
                            "  [calibration #{}] date: {}, spot: {:.4}, score: {:.4}, duration: {} us",
                            total_calibrations,
                            grid.date,
                            grid.s_t,
                            score,
                            elapsed.as_micros()
                        );
                    }
                    current_surface = Some(surf);
                }
                Err(e) => {
                    println!("  [error] calibration failed on grid {}: {:?}", i, e);
                }
            }
        }

        // Step C: If surface is available, extract candidate features, calculate Greeks and run Portfolio LP
        if let Some(ref surface) = current_surface {
            let mut alpha_scores = Vec::with_capacity(grid.contracts.len());
            let mut has_alpha = false;

            // Compute Greeks and expected return scores for each contract
            for contract in &grid.contracts {
                if contract.is_liquid {
                    // Extract candidate features
                    if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                        // Measure Greeks extraction latency (Implied Vol Bisection + BS formulas)
                        let start_greeks = Instant::now();
                        let fair_price = surface.evaluate_contract(contract);
                        let gk = calculate_greeks(grid.s_t, contract.strike, contract.tau, r, contract.option_type, fair_price);
                        total_greeks_duration += start_greeks.elapsed();
                        total_greeks_calculated += 1;

                        // Mock tree expected return output based on the immediate execution gap
                        let mock_alpha = 0.8 * feat.immediate_execution_gap + 0.2 * gk.delta.abs() * 0.01;
                        alpha_scores.push(mock_alpha);
                        has_alpha = true;
                    } else {
                        alpha_scores.push(0.0);
                    }
                } else {
                    alpha_scores.push(0.0);
                }
            }

            // Step D: Solve the Multi-Greek Constrained Portfolio LP (Phase 6)
            if has_alpha {
                let start_port = Instant::now();
                match optimize_portfolio(grid, surface, &alpha_scores, &portfolio_config) {
                    Ok(portfolio) => {
                        let elapsed = start_port.elapsed();
                        total_portfolio_duration += elapsed;
                        total_portfolios_optimized += 1;

                        if total_portfolios_optimized == 1 {
                            println!("\n  [first portfolio optimization example]");
                            println!("    liquid contracts in grid: {}", grid.contracts.iter().filter(|c| c.is_liquid).count());
                            println!("    optimization duration: {} us", elapsed.as_micros());
                            
                            // Print first few non-zero allocations
                            let mut non_zeros = 0;
                            for (idx, alloc) in portfolio.iter().enumerate() {
                                if alloc.abs() > 1e-3 {
                                    let c = &grid.contracts[idx];
                                    println!("      option: {} {} strike {} -> weight: {:.4} (delta: {:.4}, mid: {:.4})", 
                                        c.option_type, c.expiry, c.strike, alloc, deltas_helper(c, surface), c.mid);
                                    non_zeros += 1;
                                    if non_zeros >= 5 {
                                        break;
                                    }
                                }
                            }
                            println!();
                        }
                    }
                    Err(e) => {
                        println!("  [error] portfolio optimization failed: {:?}", e);
                    }
                }
            }
        }

        total_tick_processing_duration += tick_start.elapsed();
        total_ticks_processed += 1;
    }

    println!("=== performance latency benchmarks ===");
    println!("total ticks processed in chronological grids: {}", ticks.len());
    println!("total grids processed: {}", grids.len());
    
    if total_calibrations > 0 {
        let avg_calib = total_calibration_duration.as_micros() as f64 / total_calibrations as f64;
        println!("average surface calibration latency: {:.2} us", avg_calib);
    }
    
    if total_greeks_calculated > 0 {
        let avg_greeks = total_greeks_duration.as_nanos() as f64 / total_greeks_calculated as f64;
        println!("average Greeks calculation latency per contract: {:.2} ns", avg_greeks);
    }

    if total_portfolios_optimized > 0 {
        let avg_portfolio = total_portfolio_duration.as_micros() as f64 / total_portfolios_optimized as f64;
        println!("average Portfolio LP optimization latency: {:.2} us", avg_portfolio);
    }

    let avg_tick_total = total_tick_processing_duration.as_micros() as f64 / total_ticks_processed as f64;
    println!("average total loop processing latency per grid group: {:.2} us", avg_tick_total);
    
    // Check if total tick processing time satisfies the sub-5ms requirement
    let is_fast_enough = avg_tick_total < 5000.0;
    println!("verifying latency is under 5ms: {}", if is_fast_enough { "PASS" } else { "FAIL" });

    Ok(())
}

fn deltas_helper(contract: &OptionTick, surface: &CalibrationSurface) -> f64 {
    let fair = surface.evaluate_contract(contract);
    let gk = calculate_greeks(contract.s_t, contract.strike, contract.tau, surface.r, contract.option_type, fair);
    gk.delta
}
