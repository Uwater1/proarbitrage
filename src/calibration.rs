use anyhow::{Result, bail};
use good_lp::{variables, variable, default_solver, Solver, Solution, Expression, constraint, SolverModel};
use crate::ingestion::{OptionGrid, OptionTick};

#[derive(Debug, Clone)]
pub struct CalibrationSurface {
    pub s_t: f64,
    pub r: f64,
    pub theta: [f64; 6],
}

impl CalibrationSurface {
    /// Evaluate the calibrated Call price at strike K and maturity T (tau)
    pub fn evaluate(&self, strike: f64, tau: f64) -> f64 {
        let k = strike;
        let t = tau;
        self.s_t 
            + self.theta[0] * k 
            + self.theta[1] * k * t 
            + self.theta[2] * k * t * t 
            + self.theta[3] * k * k 
            + self.theta[4] * k * k * t 
            + self.theta[5] * k * k * t * t
    }
    
    /// First derivative of Call price with respect to Strike (K)
    pub fn d_k(&self, strike: f64, tau: f64) -> f64 {
        let k = strike;
        let t = tau;
        self.theta[0] 
            + self.theta[1] * t 
            + self.theta[2] * t * t 
            + 2.0 * self.theta[3] * k 
            + 2.0 * self.theta[4] * k * t 
            + 2.0 * self.theta[5] * k * t * t
    }
    
    /// Second derivative of Call price with respect to Strike (K) - must be >= 0 for no butterfly arbitrage
    pub fn d_kk(&self, _strike: f64, tau: f64) -> f64 {
        let t = tau;
        2.0 * self.theta[3] 
            + 2.0 * self.theta[4] * t 
            + 2.0 * self.theta[5] * t * t
    }
    
    /// First derivative of Call price with respect to Time to Maturity (T) - must be >= 0 for no calendar arbitrage
    pub fn d_t(&self, strike: f64, tau: f64) -> f64 {
        let k = strike;
        let t = tau;
        self.theta[1] * k 
            + 2.0 * self.theta[2] * k * t 
            + self.theta[4] * k * k 
            + 2.0 * self.theta[5] * k * k * t
    }

    /// Evaluate the equivalent fair price for a contract (applies Call-Put parity if it is a Put)
    pub fn evaluate_contract(&self, tick: &OptionTick) -> f64 {
        let call_fair = self.evaluate(tick.strike, tick.tau);
        if tick.option_type == 'P' {
            // P = C - S_t + K * e^(-r * tau)
            let disc = (-self.r * tick.tau).exp();
            call_fair - self.s_t + tick.strike * disc
        } else {
            call_fair
        }
    }
}

/// Calibrate the arbitrage-free surface from an OptionGrid using L1-norm Linear Programming
pub fn calibrate_surface(grid: &OptionGrid, r: f64, lambda: f64) -> Result<CalibrationSurface> {
    let s_t = grid.s_t;
    if s_t <= 0.0 {
        bail!("Invalid spot price: {}", s_t);
    }

    // Filter liquid contracts
    let liquid_contracts: Vec<&OptionTick> = grid.contracts.iter()
        .filter(|c| c.is_liquid && c.tau > 0.0 && c.strike > 0.0)
        .collect();

    if liquid_contracts.is_empty() {
        bail!("No liquid contracts available for calibration");
    }

    let n = liquid_contracts.len();

    // Define LP variables
    let mut vars = variables!();
    
    // 6 free variables for theta
    let theta0 = vars.add(variable().min(f64::NEG_INFINITY));
    let theta1 = vars.add(variable().min(f64::NEG_INFINITY));
    let theta2 = vars.add(variable().min(f64::NEG_INFINITY));
    let theta3 = vars.add(variable().min(f64::NEG_INFINITY));
    let theta4 = vars.add(variable().min(f64::NEG_INFINITY));
    let theta5 = vars.add(variable().min(f64::NEG_INFINITY));
    
    // Absolute deviation slack variables u_i >= 0
    let u_vars: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0))).collect();
    
    // L1 regularization slack variables v_m >= 0
    let v_vars: Vec<_> = (0..6).map(|_| vars.add(variable().min(0.0))).collect();

    // Objective function: sum_{i} u_i + lambda * sum_{m} v_m
    let mut objective = Expression::from(0.0);
    for i in 0..n {
        objective += u_vars[i];
    }
    for m in 0..6 {
        objective += v_vars[m] * lambda;
    }

    let mut problem = vars.minimise(objective).using(default_solver);

    // L1 regularization constraints: -v_m <= theta_m <= v_m
    let thetas = [theta0, theta1, theta2, theta3, theta4, theta5];
    for m in 0..6 {
        problem = problem.with(constraint!(thetas[m] <= v_vars[m]));
        problem = problem.with(constraint!(-thetas[m] <= v_vars[m]));
    }

    // Constraints for each option contract in the grid
    for i in 0..n {
        let contract = liquid_contracts[i];
        let k = contract.strike;
        let t = contract.tau;
        
        // C_hat(K, T) = s_t + theta0*K + theta1*K*T + theta2*K*T^2 + theta3*K^2 + theta4*K^2*T + theta5*K^2*T^2
        let c_hat = Expression::from(s_t) 
            + theta0 * k 
            + theta1 * (k * t) 
            + theta2 * (k * t * t) 
            + theta3 * (k * k) 
            + theta4 * (k * k * t) 
            + theta5 * (k * k * t * t);

        // Convert contract mid-price to equivalent Call mid-price using Call-Put parity if it is a Put
        let m_call = if contract.option_type == 'P' {
            let disc = (-r * t).exp();
            contract.mid + s_t - k * disc
        } else {
            contract.mid
        };

        // Absolute deviation constraints: -u_i <= C_hat - M_call <= u_i
        problem = problem.with(constraint!(c_hat.clone() - m_call <= u_vars[i]));
        problem = problem.with(constraint!(m_call - c_hat.clone() <= u_vars[i]));

        // Value bounds: 0 <= C_hat <= S_t
        problem = problem.with(constraint!(c_hat.clone() >= 0.0));
        problem = problem.with(constraint!(c_hat.clone() <= s_t));

        // Strike Monotonicity: dC/dK <= 0
        let d_k = theta0 
            + theta1 * t 
            + theta2 * (t * t) 
            + theta3 * (2.0 * k) 
            + theta4 * (2.0 * k * t) 
            + theta5 * (2.0 * k * t * t);
        problem = problem.with(constraint!(d_k <= 0.0));

        // Strike Convexity (Butterfly No-Arbitrage): d^2C/dK^2 >= 0
        let d_kk = theta3 * 2.0 
            + theta4 * (2.0 * t) 
            + theta5 * (2.0 * t * t);
        problem = problem.with(constraint!(d_kk >= 0.0));

        // Calendar Monotonicity: dC/dT >= 0 (since k > 0, we can use the derivative divided by k)
        let d_t_scaled = theta1 
            + theta2 * (2.0 * t) 
            + theta4 * k 
            + theta5 * (2.0 * k * t);
        problem = problem.with(constraint!(d_t_scaled >= 0.0));
    }

    // Solve LP
    let solution = problem.solve().map_err(|e| anyhow::anyhow!("LP solver failed: {:?}", e))?;
    
    // Extract coefficients
    let theta = [
        solution.value(theta0),
        solution.value(theta1),
        solution.value(theta2),
        solution.value(theta3),
        solution.value(theta4),
        solution.value(theta5),
    ];

    Ok(CalibrationSurface { s_t, r, theta })
}
