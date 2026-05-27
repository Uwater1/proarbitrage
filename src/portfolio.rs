use crate::ingestion::OptionGrid;
use crate::calibration::CalibrationSurface;
use anyhow::{Result, bail};
use good_lp::{variables, variable, default_solver, Solver, Solution, Expression, constraint, SolverModel};

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
