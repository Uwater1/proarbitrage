use anyhow::{Result, Context};
use polars::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OptionTick {
    pub date: Arc<str>,
    pub option_type: char, // 'C' or 'P'
    pub strike: f64,
    pub expiry: Arc<str>,
    pub days_to_maturity: f64,
    pub tau: f64,
    pub s_t: f64,
    pub moneyness: f64,
    pub p_a: f64,
    pub p_b: f64,
    pub mid: f64,
    pub spread: f64,
    pub a_v_eff: i64,
    pub b_v_eff: i64,
    pub is_liquid: bool,
}

#[derive(Debug, Clone)]
pub struct OptionGrid {
    pub date: Arc<str>,
    pub s_t: f64,
    pub contracts: Vec<OptionTick>,
}

/// Load option ticks from a parquet file using Polars
pub fn load_ticks_from_parquet(path: &str, limit: Option<usize>) -> Result<Vec<OptionTick>> {
    let mut df = if let Some(n) = limit {
        LazyFrame::scan_parquet(path, Default::default())?
            .limit(n as u32)
            .collect()?
    } else {
        LazyFrame::scan_parquet(path, Default::default())?
            .collect()?
    };

    let n_rows = df.height();
    let mut ticks = Vec::with_capacity(n_rows);

    // Get series and cast them to correct types to handle schema variance safely
    let date_series = df.column("date")?.cast(&DataType::String)?;
    let date_chunked = date_series.str()?;
    
    let type_series = df.column("type")?.cast(&DataType::String)?;
    let type_chunked = type_series.str()?;
    
    let strike_series = df.column("strike")?.cast(&DataType::Float64)?;
    let strike_chunked = strike_series.f64()?;
    
    let expiry_series = df.column("expiry")?.cast(&DataType::String)?;
    let expiry_chunked = expiry_series.str()?;
    
    let days_series = df.column("days_to_maturity")?.cast(&DataType::Float64)?;
    let days_chunked = days_series.f64()?;
    
    let tau_series = df.column("tau")?.cast(&DataType::Float64)?;
    let tau_chunked = tau_series.f64()?;
    
    let st_series = df.column("S_t")?.cast(&DataType::Float64)?;
    let st_chunked = st_series.f64()?;
    
    let moneyness_series = df.column("moneyness")?.cast(&DataType::Float64)?;
    let moneyness_chunked = moneyness_series.f64()?;
    
    let pa_series = df.column("P_A")?.cast(&DataType::Float64)?;
    let pa_chunked = pa_series.f64()?;
    
    let pb_series = df.column("P_B")?.cast(&DataType::Float64)?;
    let pb_chunked = pb_series.f64()?;
    
    let mid_series = df.column("mid")?.cast(&DataType::Float64)?;
    let mid_chunked = mid_series.f64()?;
    
    let spread_series = df.column("spread")?.cast(&DataType::Float64)?;
    let spread_chunked = spread_series.f64()?;
    
    let av_series = df.column("a_v_eff")?.cast(&DataType::Int64)?;
    let av_chunked = av_series.i64()?;
    
    let bv_series = df.column("b_v_eff")?.cast(&DataType::Int64)?;
    let bv_chunked = bv_series.i64()?;
    
    let liquid_series = df.column("is_liquid")?.cast(&DataType::Boolean)?;
    let liquid_chunked = liquid_series.bool()?;

    let mut expiry_cache: HashMap<String, Arc<str>> = HashMap::with_capacity(8);
    let mut last_date_str = String::new();
    let mut last_date_arc: Arc<str> = Arc::from("");

    for i in 0..n_rows {
        let date_str = date_chunked.get(i).unwrap_or("");
        let date = if date_str == last_date_str {
            last_date_arc.clone()
        } else {
            last_date_str = date_str.to_string();
            last_date_arc = Arc::from(date_str);
            last_date_arc.clone()
        };

        let opt_type_str = type_chunked.get(i).unwrap_or("C");
        let option_type = opt_type_str.chars().next().unwrap_or('C');
        let strike = strike_chunked.get(i).unwrap_or(0.0);

        let expiry_str = expiry_chunked.get(i).unwrap_or("");
        let expiry = expiry_cache.entry(expiry_str.to_string())
            .or_insert_with(|| Arc::from(expiry_str))
            .clone();

        let days_to_maturity = days_chunked.get(i).unwrap_or(0.0);
        let tau = tau_chunked.get(i).unwrap_or(0.0);
        let s_t = st_chunked.get(i).unwrap_or(0.0);
        let moneyness = moneyness_chunked.get(i).unwrap_or(0.0);
        let p_a = pa_chunked.get(i).unwrap_or(0.0);
        let p_b = pb_chunked.get(i).unwrap_or(0.0);
        let mid = mid_chunked.get(i).unwrap_or(f64::NAN);
        let spread = spread_chunked.get(i).unwrap_or(0.0);
        let a_v_eff = av_chunked.get(i).unwrap_or(0);
        let b_v_eff = bv_chunked.get(i).unwrap_or(0);
        let is_liquid = liquid_chunked.get(i).unwrap_or(false);

        ticks.push(OptionTick {
            date,
            option_type,
            strike,
            expiry,
            days_to_maturity,
            tau,
            s_t,
            moneyness,
            p_a,
            p_b,
            mid,
            spread,
            a_v_eff,
            b_v_eff,
            is_liquid,
        });
    }

    Ok(ticks)
}

/// Reconstruct chronologically ordered option grid groups from a list of option ticks
pub fn reconstruct_grids(ticks: &[OptionTick]) -> Vec<OptionGrid> {
    let mut grids = Vec::new();
    if ticks.is_empty() {
        return grids;
    }
    
    let mut current_date = ticks[0].date.clone();
    let mut current_st = ticks[0].s_t;
    let mut current_contracts = Vec::new();
    
    for tick in ticks {
        if tick.date != current_date {
            grids.push(OptionGrid {
                date: current_date,
                s_t: current_st,
                contracts: current_contracts,
            });
            current_date = tick.date.clone();
            current_st = tick.s_t;
            current_contracts = Vec::new();
        }
        current_contracts.push(tick.clone());
    }
    
    if !current_contracts.is_empty() {
        grids.push(OptionGrid {
            date: current_date,
            s_t: current_st,
            contracts: current_contracts,
        });
    }
    
    grids
}
