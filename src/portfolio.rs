use crate::ingestion::OptionGrid;
use crate::calibration::CalibrationSurface;
use anyhow::{Result, bail};
use good_lp::{variables, variable, default_solver, Solver, Solution, Expression, constraint, SolverModel};
use std::collections::{HashMap, BTreeSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LegConfig {
    pub expiry: String,
    pub strike: f64,
    pub option_type: char, // 'C' or 'P'
    pub weight: f64,       // +1.0 for buy, -1.0 for sell, etc.
}

#[derive(Debug, Clone)]
pub struct ArbitrageStructure {
    pub name: String,
    pub legs: Vec<LegConfig>,
    pub expected_fee: f64,
}

#[derive(Debug, Clone)]
pub struct ActiveStructure {
    pub name: String,
    pub legs: Vec<(usize, f64)>, // (contract_index, weight)
    pub expected_fee: f64,
}

struct StrikeInfo {
    strike_micro: u64,
    call_idx: Option<usize>,
    put_idx: Option<usize>,
}

/// Dynamically generate and map all active options-only arbitrage structures
pub fn generate_active_structures(
    grid: &OptionGrid,
    fee_per_contract: f64,
) -> Vec<ActiveStructure> {
    // Group contracts by expiry in a flat Vec
    let mut expiries: Vec<(Arc<str>, Vec<(usize, &crate::ingestion::OptionTick)>)> = Vec::with_capacity(4);

    for (idx, c) in grid.contracts.iter().enumerate() {
        if let Some(pos) = expiries.iter().position(|e| e.0 == c.expiry) {
            expiries[pos].1.push((idx, c));
        } else {
            expiries.push((c.expiry.clone(), vec![(idx, c)]));
        }
    }

    let mut active_structs = Vec::new();

    for (expiry, items) in expiries {
        let mut strikes: Vec<StrikeInfo> = Vec::with_capacity(32);

        for &(idx, c) in &items {
            let strike_micro = (c.strike * 1_000_000.0).round() as u64;
            if let Some(pos) = strikes.iter().position(|s| s.strike_micro == strike_micro) {
                if c.option_type == 'C' {
                    strikes[pos].call_idx = Some(idx);
                } else {
                    strikes[pos].put_idx = Some(idx);
                }
            } else {
                strikes.push(StrikeInfo {
                    strike_micro,
                    call_idx: if c.option_type == 'C' { Some(idx) } else { None },
                    put_idx: if c.option_type == 'P' { Some(idx) } else { None },
                });
            }
        }

        // Sort strikes chronologically
        strikes.sort_by_key(|s| s.strike_micro);

        if strikes.len() < 2 {
            continue;
        }

        // A. Box Arbitrage: strikes K_i < K_j (allow adjacent and 1-strike gap, index difference <= 2)
        for i in 0..strikes.len() - 1 {
            let limit_j = (i + 3).min(strikes.len());
            for j in (i + 1)..limit_j {
                let s1 = &strikes[i];
                let s2 = &strikes[j];
                
                let k1_f = s1.strike_micro as f64 / 1_000_000.0;
                let k2_f = s2.strike_micro as f64 / 1_000_000.0;

                if let (Some(c1_c), Some(c2_c), Some(c1_p), Some(c2_p)) = (
                    s1.call_idx,
                    s2.call_idx,
                    s1.put_idx,
                    s2.put_idx,
                ) {
                    active_structs.push(ActiveStructure {
                        name: format!("Box_{}_{:.3}_{:.3}", expiry, k1_f, k2_f),
                        legs: vec![
                            (c1_c, 1.0),
                            (c2_c, -1.0),
                            (c2_p, 1.0),
                            (c1_p, -1.0),
                        ],
                        expected_fee: 4.0 * fee_per_contract,
                    });
                }
            }
        }

        // B. Butterfly Spread: equal-interval strikes K_i < K_j < K_k (index limit <= 4 steps)
        if strikes.len() >= 3 {
            for i in 0..strikes.len() - 2 {
                let limit_k = (i + 5).min(strikes.len());
                for j in (i + 1)..limit_k {
                    for k in (j + 1)..limit_k {
                        let s1 = &strikes[i];
                        let s2 = &strikes[j];
                        let s3 = &strikes[k];

                        let k1_f = s1.strike_micro as f64 / 1_000_000.0;
                        let k2_f = s2.strike_micro as f64 / 1_000_000.0;
                        let k3_f = s3.strike_micro as f64 / 1_000_000.0;

                        if (s2.strike_micro - s1.strike_micro) == (s3.strike_micro - s2.strike_micro) {
                            if let (Some(c1), Some(c2), Some(c3)) = (s1.call_idx, s2.call_idx, s3.call_idx) {
                                active_structs.push(ActiveStructure {
                                    name: format!("C_Fly_{}_{:.3}_{:.3}_{:.3}", expiry, k1_f, k2_f, k3_f),
                                    legs: vec![
                                        (c1, 1.0),
                                        (c2, -2.0),
                                        (c3, 1.0),
                                    ],
                                    expected_fee: 4.0 * fee_per_contract,
                                });
                            }

                            if let (Some(p1), Some(p2), Some(p3)) = (s1.put_idx, s2.put_idx, s3.put_idx) {
                                active_structs.push(ActiveStructure {
                                    name: format!("P_Fly_{}_{:.3}_{:.3}_{:.3}", expiry, k1_f, k2_f, k3_f),
                                    legs: vec![
                                        (p1, 1.0),
                                        (p2, -2.0),
                                        (p3, 1.0),
                                    ],
                                    expected_fee: 4.0 * fee_per_contract,
                                });
                            }
                        }
                    }
                }
            }
        }

        // C. Iron Condor: strikes K_i < K_j < K_k < K_l (index limit <= 5 steps)
        if strikes.len() >= 4 {
            for i in 0..strikes.len() - 3 {
                let limit_l = (i + 6).min(strikes.len());
                for j in (i + 1)..limit_l {
                    for k in (j + 1)..limit_l {
                        for l in (k + 1)..limit_l {
                            let s1 = &strikes[i];
                            let s2 = &strikes[j];
                            let s3 = &strikes[k];
                            let s4 = &strikes[l];

                            let k1_f = s1.strike_micro as f64 / 1_000_000.0;
                            let k2_f = s2.strike_micro as f64 / 1_000_000.0;
                            let k3_f = s3.strike_micro as f64 / 1_000_000.0;
                            let k4_f = s4.strike_micro as f64 / 1_000_000.0;

                            if let (Some(p1), Some(p2), Some(c3), Some(c4)) = (
                                s1.put_idx,
                                s2.put_idx,
                                s3.call_idx,
                                s4.call_idx,
                            ) {
                                active_structs.push(ActiveStructure {
                                    name: format!("Condor_{}_{:.3}_{:.3}_{:.3}_{:.3}", expiry, k1_f, k2_f, k3_f, k4_f),
                                    legs: vec![
                                        (p1, 1.0),
                                        (p2, -1.0),
                                        (c3, -1.0),
                                        (c4, 1.0),
                                    ],
                                    expected_fee: 4.0 * fee_per_contract,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    active_structs
}

/// Structured Multi-Leg Options-Only Portfolio LP Optimizer
pub fn optimize_portfolio_structured(
    grid: &OptionGrid,
    surface: &CalibrationSurface,
    alpha_scores: &[f64],
    active_structs: &[ActiveStructure],
    config: &PortfolioConfig,
) -> Result<Vec<f64>> {
    let m = active_structs.len();
    let n = grid.contracts.len();
    if m == 0 || n == 0 {
        return Ok(vec![0.0; m]);
    }
    if n != alpha_scores.len() {
        bail!("Contracts list and alpha scores must have identical size");
    }

    // Pre-calculate Greeks and Capital Requirements
    let mut deltas = Vec::with_capacity(n);
    let mut vegas = Vec::with_capacity(n);
    let mut thetas = Vec::with_capacity(n);
    let mut c_long = Vec::with_capacity(n);
    let mut c_short = Vec::with_capacity(n);

    for i in 0..n {
        let contract = &grid.contracts[i];
        let fair_price = surface.evaluate_contract(contract);
        
        let gk = calculate_greeks(grid.s_t, contract.strike, contract.tau, surface.r, contract.option_type, fair_price);
        deltas.push(gk.delta);
        vegas.push(gk.vega);
        thetas.push(gk.theta);

        c_long.push(contract.p_a.max(0.0001));
        c_short.push(contract.p_b + config.margin_ratio * grid.s_t);
    }

    // Formulate LP variables using good_lp
    let mut vars = variables!();

    // z^+_p >= 0, z^-_p >= 0
    let z_plus: Vec<_> = (0..m).map(|_| vars.add(variable().min(0.0).max(config.theta_max))).collect();
    let z_minus: Vec<_> = (0..m).map(|_| vars.add(variable().min(0.0).max(config.theta_max))).collect();

    // Net expected return for each structure
    let mut alpha_structs = Vec::with_capacity(m);
    for p in 0..m {
        let mut alpha = 0.0;
        for &(idx, weight) in &active_structs[p].legs {
            alpha += weight * alpha_scores[idx];
        }
        alpha_structs.push(alpha);
    }

    // Objective function: sum_p (z^+_p - z^-_p) * alpha_struct_p - sum_p (z^+_p + z^-_p) * expected_fee_p
    // Scaling down expected_fee or standard objective formulation:
    let mut objective = Expression::from(0.0);
    for p in 0..m {
        // expected fee is in CNY, and option premiums are also in CNY. 
        // 1 contract = 100 units. So we must scale expected fee appropriately to option price terms:
        // Expected fee in premium points = fee / 100
        let fee_premium = active_structs[p].expected_fee / 100.0;
        objective += z_plus[p] * (alpha_structs[p] - fee_premium)
                   - z_minus[p] * (alpha_structs[p] + fee_premium);
    }

    let mut problem = vars.maximise(objective).using(default_solver);

    // Compute contract level long/short expressions
    let mut t_plus_exprs: Vec<Expression> = (0..n).map(|_| Expression::from(0.0)).collect();
    let mut t_minus_exprs: Vec<Expression> = (0..n).map(|_| Expression::from(0.0)).collect();

    for p in 0..m {
        for &(idx, w) in &active_structs[p].legs {
            if w > 0.0 {
                t_plus_exprs[idx] += z_plus[p] * w;
                t_minus_exprs[idx] += z_minus[p] * w;
            } else if w < 0.0 {
                t_plus_exprs[idx] += z_minus[p] * (-w);
                t_minus_exprs[idx] += z_plus[p] * (-w);
            }
        }
    }

    // Delta Constraints: -epsilon_delta <= sum_i (Theta^+_i - Theta^-_i)*Delta_i <= epsilon_delta
    let mut portfolio_delta = Expression::from(0.0);
    for i in 0..n {
        portfolio_delta += (t_plus_exprs[i].clone() - t_minus_exprs[i].clone()) * deltas[i];
    }
    problem = problem.with(constraint!(portfolio_delta.clone() >= -config.epsilon_delta));
    problem = problem.with(constraint!(portfolio_delta.clone() <= config.epsilon_delta));

    // Vega Constraints: -epsilon_vega <= sum_i (Theta^+_i - Theta^-_i)*Vega_i <= epsilon_vega
    let mut portfolio_vega = Expression::from(0.0);
    for i in 0..n {
        portfolio_vega += (t_plus_exprs[i].clone() - t_minus_exprs[i].clone()) * vegas[i];
    }
    problem = problem.with(constraint!(portfolio_vega.clone() >= -config.epsilon_vega));
    problem = problem.with(constraint!(portfolio_vega.clone() <= config.epsilon_vega));

    // Theta Constraints: sum_i (Theta^+_i - Theta^-_i)*Theta_i >= -epsilon_theta
    let mut portfolio_theta = Expression::from(0.0);
    for i in 0..n {
        portfolio_theta += (t_plus_exprs[i].clone() - t_minus_exprs[i].clone()) * thetas[i];
    }
    problem = problem.with(constraint!(portfolio_theta.clone() >= -config.epsilon_theta));

    // Capital constraints: sum_i (Theta^+_i * c_long_i + Theta^-_i * c_short_i) <= C_max
    let mut capital_charge = Expression::from(0.0);
    for i in 0..n {
        capital_charge += t_plus_exprs[i].clone() * c_long[i] + t_minus_exprs[i].clone() * c_short[i];
    }
    problem = problem.with(constraint!(capital_charge <= config.c_max));

    // Solve LP
    let solution = problem.solve().map_err(|e| anyhow::anyhow!("Structured Portfolio LP failed: {:?}", e))?;

    // Extract target structure weights
    let mut target_z = Vec::with_capacity(m);
    for p in 0..m {
        let val = solution.value(z_plus[p]) - solution.value(z_minus[p]);
        target_z.push(val);
    }

    Ok(target_z)
}

/// Scans grid for Box, Butterfly, and Iron Condor combinations with high XGBoost expected alpha
pub fn scan_strict_arbitrage(
    active_structs: &[ActiveStructure],
    alpha_scores: &[f64],
    profit_threshold: f64,
) -> Vec<(ActiveStructure, f64, f64)> {
    let mut profitable = Vec::new();

    for active_struct in active_structs {
        let mut expected_alpha = 0.0;
        for &(idx, weight) in &active_struct.legs {
            expected_alpha += weight * alpha_scores[idx];
        }

        let fee_premium = active_struct.expected_fee / 100.0;

        let long_profit = expected_alpha - fee_premium;
        let short_profit = -expected_alpha - fee_premium;

        if long_profit >= profit_threshold {
            profitable.push((active_struct.clone(), long_profit, 1.0));
        } else if short_profit >= profit_threshold {
            profitable.push((active_struct.clone(), short_profit, -1.0));
        }
    }

    profitable
}


#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    pub epsilon_delta: f64,
    pub epsilon_vega: f64,
    pub epsilon_theta: f64,
    pub c_max: f64,
    pub theta_max: f64,
    pub margin_ratio: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            epsilon_delta: 0.5,     // Delta exposure limit (very neutral)
            epsilon_vega: 2.0,      // Vega limit
            epsilon_theta: 10.0,    // Max theta decay allowed (negative decay budget)
            c_max: 50_000.0,        // Global capital budget
            theta_max: 50.0,        // Max absolute position size per contract
            margin_ratio: 0.15,     // 15% margin on short positions
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptionGreeks {
    pub delta: f64,
    pub vega: f64,
    pub theta: f64,
}

/// Standard normal PDF
pub fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF approximation
pub fn cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327;
    let probs = d * (-0.5 * x * x).exp() * t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    );
    if x >= 0.0 {
        1.0 - probs
    } else {
        probs
    }
}

/// Black-Scholes Option Pricing
pub fn bs_price(s_t: f64, strike: f64, tau: f64, r: f64, vol: f64, option_type: char) -> f64 {
    if tau <= 0.0 {
        if option_type == 'C' {
            return (s_t - strike).max(0.0);
        } else {
            return (strike - s_t).max(0.0);
        }
    }
    if vol <= 0.0 {
        return if option_type == 'C' { (s_t - strike).max(0.0) } else { (strike - s_t).max(0.0) };
    }
    let d1 = ((s_t / strike).ln() + (r + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();
    
    if option_type == 'C' {
        s_t * cdf(d1) - strike * (-r * tau).exp() * cdf(d2)
    } else {
        strike * (-r * tau).exp() * cdf(-d2) - s_t * cdf(-d1)
    }
}

/// Numerical bisection method to compute implied volatility from a calibrated fair price
pub fn compute_implied_vol(
    s_t: f64,
    strike: f64,
    tau: f64,
    r: f64,
    option_type: char,
    target_price: f64,
) -> f64 {
    if tau <= 0.0 {
        return 0.20;
    }
    let intrinsic = if option_type == 'C' { (s_t - strike).max(0.0) } else { (strike - s_t).max(0.0) };
    if target_price <= intrinsic {
        return 0.05; // floor
    }

    let mut low = 1e-4;
    let mut high = 5.0;
    let mut vol = 0.20;
    
    for _ in 0..30 {
        let mid_vol = 0.5 * (low + high);
        let p = bs_price(s_t, strike, tau, r, mid_vol, option_type);
        if p < target_price {
            low = mid_vol;
        } else {
            high = mid_vol;
        }
        vol = mid_vol;
        if (high - low).abs() < 1e-4 {
            break;
        }
    }
    vol
}

/// Analytical Black-Scholes Greeks calculation based on calibrated surface implied vol
pub fn calculate_greeks(
    s_t: f64,
    strike: f64,
    tau: f64,
    r: f64,
    option_type: char,
    calibrated_price: f64,
) -> OptionGreeks {
    let vol = compute_implied_vol(s_t, strike, tau, r, option_type, calibrated_price);
    
    if tau <= 0.0 || vol <= 0.0 {
        let d = if option_type == 'C' { if s_t >= strike { 1.0 } else { 0.0 } } else { if s_t < strike { -1.0 } else { 0.0 } };
        return OptionGreeks {
            delta: d,
            vega: 0.0,
            theta: 0.0,
        };
    }
    
    let d1 = ((s_t / strike).ln() + (r + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();
    
    let delta = if option_type == 'C' {
        cdf(d1)
    } else {
        cdf(d1) - 1.0
    };
    
    let vega = s_t * tau.sqrt() * pdf(d1);
    
    let term1 = -(s_t * pdf(d1) * vol) / (2.0 * tau.sqrt());
    let theta = if option_type == 'C' {
        term1 - r * strike * (-r * tau).exp() * cdf(d2)
    } else {
        term1 + r * strike * (-r * tau).exp() * cdf(-d2)
    };
    
    OptionGreeks { delta, vega, theta }
}

/// Decompose portfolio optimization into a standard primal-dual interior LP and solve it
pub fn optimize_portfolio(
    grid: &OptionGrid,
    surface: &CalibrationSurface,
    alpha_scores: &[f64], // Expected return W_i for each contract in grid
    config: &PortfolioConfig,
) -> Result<Vec<f64>> {
    let n = grid.contracts.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if n != alpha_scores.len() {
        bail!("Contracts list and alpha scores must have identical size");
    }

    // Pre-calculate Greeks and Capital Requirements
    let mut deltas = Vec::with_capacity(n);
    let mut vegas = Vec::with_capacity(n);
    let mut thetas = Vec::with_capacity(n);
    let mut c_long = Vec::with_capacity(n);
    let mut c_short = Vec::with_capacity(n);

    for i in 0..n {
        let contract = &grid.contracts[i];
        let fair_price = surface.evaluate_contract(contract);
        
        let gk = calculate_greeks(grid.s_t, contract.strike, contract.tau, surface.r, contract.option_type, fair_price);
        deltas.push(gk.delta);
        vegas.push(gk.vega);
        thetas.push(gk.theta);

        // Capital cost coefficients:
        // Long option: price to purchase
        c_long.push(contract.p_a.max(0.0001));
        // Short option margin: premium + 15% margin on spot
        c_short.push(contract.p_b + config.margin_ratio * grid.s_t);
    }

    // Formulate LP variables using good_lp
    let mut vars = variables!();

    // Theta^+_i >= 0, Theta^-_i >= 0
    let t_plus: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0).max(config.theta_max))).collect();
    let t_minus: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0).max(config.theta_max))).collect();

    // Objective function: sum_i (Theta^+_i - Theta^-_i) * W_i
    let mut objective = Expression::from(0.0);
    for i in 0..n {
        objective += (t_plus[i] - t_minus[i]) * alpha_scores[i];
    }

    let mut problem = vars.maximise(objective).using(default_solver);

    // Delta Constraints: -epsilon_delta <= sum_i (Theta^+_i - Theta^-_i)*Delta_i <= epsilon_delta
    let mut portfolio_delta = Expression::from(0.0);
    for i in 0..n {
        portfolio_delta += (t_plus[i] - t_minus[i]) * deltas[i];
    }
    problem = problem.with(constraint!(portfolio_delta.clone() >= -config.epsilon_delta));
    problem = problem.with(constraint!(portfolio_delta.clone() <= config.epsilon_delta));

    // Vega Constraints: -epsilon_vega <= sum_i (Theta^+_i - Theta^-_i)*Vega_i <= epsilon_vega
    let mut portfolio_vega = Expression::from(0.0);
    for i in 0..n {
        portfolio_vega += (t_plus[i] - t_minus[i]) * vegas[i];
    }
    problem = problem.with(constraint!(portfolio_vega.clone() >= -config.epsilon_vega));
    problem = problem.with(constraint!(portfolio_vega.clone() <= config.epsilon_vega));

    // Theta Constraints: sum_i (Theta^+_i - Theta^-_i)*Theta_i >= -epsilon_theta
    let mut portfolio_theta = Expression::from(0.0);
    for i in 0..n {
        portfolio_theta += (t_plus[i] - t_minus[i]) * thetas[i];
    }
    problem = problem.with(constraint!(portfolio_theta.clone() >= -config.epsilon_theta));

    // Capital constraints: sum_i (Theta^+_i * c_long_i + Theta^-_i * c_short_i) <= C_max
    let mut capital_charge = Expression::from(0.0);
    for i in 0..n {
        capital_charge += t_plus[i] * c_long[i] + t_minus[i] * c_short[i];
    }
    problem = problem.with(constraint!(capital_charge <= config.c_max));

    // Solve LP
    let solution = problem.solve().map_err(|e| anyhow::anyhow!("Portfolio LP failed: {:?}", e))?;

    // Extract net positions: Theta_i = Theta^+_i - Theta^-_i
    let mut portfolio = Vec::with_capacity(n);
    for i in 0..n {
        let net = solution.value(t_plus[i]) - solution.value(t_minus[i]);
        portfolio.push(net);
    }

    Ok(portfolio)
}
