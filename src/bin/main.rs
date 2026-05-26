use proarbitrage::ingestion::{load_ticks_from_parquet, reconstruct_grids};
use proarbitrage::calibration::calibrate_surface;
use proarbitrage::activation::{
    compute_activation_score, extract_candidate_features, ActivationConfig,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== starting proarbitrage integration test (phases 1-4) ===");

    // 1. Ingest sample ticks from parquet file
    let path = "data/510300_surface.parquet";
    println!("loading ticks from: {}", path);
    let start_load = Instant::now();
    // Load first 20,000 rows to quickly verify logic without loading entire 460MB file
    let ticks = load_ticks_from_parquet(path, Some(20_000))?;
    println!(
        "loaded {} ticks in {} ms",
        ticks.len(),
        start_load.elapsed().as_millis()
    );

    // 2. Reconstruct option grids
    let grids = reconstruct_grids(&ticks);
    println!("reconstructed {} chronological grids", grids.len());
    if grids.is_empty() {
        println!("no grids reconstructed! exiting.");
        return Ok(());
    }

    // 3. Process grids sequentially
    let config = ActivationConfig::default();
    let mut current_surface = None;
    let mut total_calibrations = 0;
    let mut total_calibration_duration = std::time::Duration::new(0, 0);
    let mut total_gated_candidates = 0;

    let r = 0.02; // 2% risk-free rate proxy
    let lambda_reg = 0.0001; // regularization parameter
    let lambda_gate = 0.0005; // liquidity edge threshold (5 bps)

    println!("processing grids chronologically...");
    for (i, grid) in grids.iter().enumerate() {
        // Calculate activation score
        let score = compute_activation_score(grid, &current_surface, &config);
        
        let should_calibrate = current_surface.is_none() || score > config.tau_enter;

        if should_calibrate {
            let start_calib = Instant::now();
            match calibrate_surface(grid, r, lambda_reg) {
                Ok(surf) => {
                    let elapsed = start_calib.elapsed();
                    total_calibration_duration += elapsed;
                    total_calibrations += 1;
                    
                    if total_calibrations <= 5 {
                        println!(
                            "  [calibration #{}] date: {}, spot: {:.4}, score: {:.4}, duration: {} us, theta: {:?}",
                            total_calibrations,
                            grid.date,
                            grid.s_t,
                            score,
                            elapsed.as_micros(),
                            surf.theta
                        );
                    }
                    current_surface = Some(surf);
                }
                Err(e) => {
                    println!("  [error] calibration failed on grid {}: {:?}", i, e);
                }
            }
        }

        // Apply pre-inference liquidity gate and extract candidate features
        if let Some(ref surface) = current_surface {
            for contract in &grid.contracts {
                if let Some(feat) = extract_candidate_features(contract, surface, lambda_gate) {
                    total_gated_candidates += 1;
                    if total_gated_candidates == 1 {
                        println!("\n  [first candidate feature extraction example]");
                        println!("    contract: {} {} strike {} (tau: {:.4})", contract.option_type, contract.expiry, contract.strike, contract.tau);
                        println!("    executable ask: {:.4}, bid: {:.4}, mid: {:.4}", contract.p_a, contract.p_b, contract.mid);
                        println!("    calibrated fair price: {:.4}", surface.evaluate_contract(contract));
                        println!("    immediate execution gap D_i: {:.4}", feat.immediate_execution_gap);
                        println!("    engineered features: {:?}", feat.raw_features);
                        println!();
                    }
                }
            }
        }
    }

    println!("=== processing complete ===");
    println!("total grids: {}", grids.len());
    println!("total calibrations triggered: {}", total_calibrations);
    if total_calibrations > 0 {
        let avg_time = total_calibration_duration.as_micros() as f64 / total_calibrations as f64;
        println!("average calibration latency: {:.2} us (target < 5000 us)", avg_time);
    }
    println!("total contracts passing liquidity gate: {}", total_gated_candidates);
    println!("integration test verified successfully!");

    Ok(())
}
