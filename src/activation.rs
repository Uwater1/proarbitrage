use crate::ingestion::{OptionGrid, OptionTick};
use crate::calibration::CalibrationSurface;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ActivationConfig {
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub alpha: f64,
    pub beta: f64,
    pub tau_enter: f64,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            w1: 1.0,
            w2: 2.0,
            w3: 0.5,
            w4: 1.5,
            alpha: 10.0,
            beta: 0.0,
            tau_enter: 0.5,
        }
    }
}

/// Calculate shape violation score L_t on option mid-prices
pub fn compute_shape_violations(grid: &OptionGrid) -> f64 {
    let mut total_violation = 0.0;

    // Group contracts by expiry to check strike monotonicity and convexity
    let mut by_expiry: HashMap<String, Vec<&OptionTick>> = HashMap::new();
    for contract in &grid.contracts {
        if contract.is_liquid {
            by_expiry.entry(contract.expiry.clone()).or_default().push(contract);
        }
    }

    for (_, mut contracts) in by_expiry {
        // Sort by strike price
        contracts.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap());
        let m = contracts.len();
        if m < 2 {
            continue;
        }

        // Strike Monotonicity: Call mid prices should be decreasing in strike K
        // Put mid prices should be increasing in strike K
        for i in 0..(m - 1) {
            let c1 = contracts[i];
            let c2 = contracts[i + 1];
            
            if c1.option_type == 'C' && c2.option_type == 'C' {
                if c2.mid > c1.mid {
                    total_violation += c2.mid - c1.mid;
                }
            } else if c1.option_type == 'P' && c2.option_type == 'P' {
                if c1.mid > c2.mid {
                    total_violation += c1.mid - c2.mid;
                }
            }
        }

        // Strike Convexity (Butterfly No-Arbitrage):
        // Call / Put prices should be convex in strike
        if m >= 3 {
            for i in 1..(m - 1) {
                let c_left = contracts[i - 1];
                let c_mid = contracts[i];
                let c_right = contracts[i + 1];

                if c_left.option_type == c_mid.option_type && c_mid.option_type == c_right.option_type {
                    let k_l = c_left.strike;
                    let k_m = c_mid.strike;
                    let k_r = c_right.strike;
                    
                    let y_l = c_left.mid;
                    let y_m = c_mid.mid;
                    let y_r = c_right.mid;

                    if k_m > k_l && k_r > k_m {
                        // Discrete convexity: slope_right >= slope_left
                        let slope_l = (y_m - y_l) / (k_m - k_l);
                        let slope_r = (y_r - y_m) / (k_r - k_m);
                        
                        if slope_l > slope_r {
                            total_violation += slope_l - slope_r;
                        }
                    }
                }
            }
        }
    }

    total_violation
}

/// Calculate activation score Gamma_t
pub fn compute_activation_score(
    grid: &OptionGrid,
    prev_surface: &Option<CalibrationSurface>,
    config: &ActivationConfig,
) -> f64 {
    // 1. Residual Intensity (R_t)
    let r_t = if let Some(surface) = prev_surface {
        let mut sum_res = 0.0;
        let mut count = 0;
        for contract in &grid.contracts {
            if contract.is_liquid {
                let fair = surface.evaluate_contract(contract);
                sum_res += (contract.mid - fair).abs();
                count += 1;
            }
        }
        if count > 0 { sum_res / count as f64 } else { 0.0 }
    } else {
        1.0 // Trigger calibration immediately on first step
    };

    // 2. Shape Violations (L_t)
    let l_t = compute_shape_violations(grid);

    // 3. Execution Quality (Q_t)
    let mut sum_spread = 0.0;
    let mut count_spread = 0;
    for contract in &grid.contracts {
        if contract.is_liquid {
            sum_spread += contract.spread;
            count_spread += 1;
        }
    }
    let avg_spread = if count_spread > 0 { sum_spread / count_spread as f64 } else { 0.0 };
    let q_t = 1.0 / (1.0 + config.alpha * avg_spread);

    // 4. Expected Edge Estimate (E_t)
    let e_t = if let Some(surface) = prev_surface {
        let mut sum_edge = 0.0;
        let mut count = 0;
        for contract in &grid.contracts {
            if contract.is_liquid {
                let fair = surface.evaluate_contract(contract);
                let raw_diff = (contract.mid - fair).abs();
                let cost = contract.spread * 0.5 + 0.0002; // transaction fee proxy
                let edge = ((raw_diff - cost) / cost).max(0.0);
                sum_edge += edge;
                count += 1;
            }
        }
        if count > 0 { sum_edge / count as f64 } else { 0.0 }
    } else {
        1.0
    };

    // Compute Gamma_t
    config.w1 * r_t + config.w2 * l_t + config.w3 * q_t + config.w4 * e_t
}

/// Feature Vector representable as an array of floats
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub immediate_execution_gap: f64,
    pub spot: f64,
    pub moneyness: f64,
    pub tau: f64,
    pub is_put: f64,
    pub spread: f64,
    pub raw_features: Vec<f64>,
}

/// Extract engineered feature vector and apply pre-inference liquidity gate
pub fn extract_candidate_features(
    tick: &OptionTick,
    surface: &CalibrationSurface,
    lambda: f64,
) -> Option<FeatureVector> {
    if !tick.is_liquid {
        return None;
    }

    let fair_price = surface.evaluate_contract(tick);
    
    // Immediate Execution Gap D_i
    let immediate_execution_gap = if fair_price > tick.p_a {
        fair_price - tick.p_a // Underpriced edge
    } else if fair_price < tick.p_b {
        tick.p_b - fair_price // Overpriced edge
    } else {
        0.0
    };

    // Pre-inference Liquidity Gate G_i
    // Contract is routed to the inference matrix only if the edge exceeds the threshold lambda
    if immediate_execution_gap <= lambda {
        return None;
    }

    let is_put = if tick.option_type == 'P' { 1.0 } else { 0.0 };
    let moneyness = tick.strike - tick.s_t;

    // Feature order: [D_i, S_t, moneyness, tau, option_type, spread]
    let raw_features = vec![
        immediate_execution_gap,
        tick.s_t,
        moneyness,
        tick.tau,
        is_put,
        tick.spread,
    ];

    Some(FeatureVector {
        immediate_execution_gap,
        spot: tick.s_t,
        moneyness,
        tau: tick.tau,
        is_put,
        spread: tick.spread,
        raw_features,
    })
}
