use live_boxx_rs::*;
use polars::prelude::*;
use std::collections::HashMap;

/// Format a Unix-seconds timestamp as a human-readable local datetime string.
/// The parquet data is stored in timezone-naive CST (UTC+8), so we add 8h to get
/// the wall-clock time that matches the printed tick times.
fn fmt_ts(ts_sec: i64) -> String {
    let secs = ts_sec % 86400;
    let days = ts_sec / 86400;
    // Rough date from 1970-01-01 epoch (good enough for display purposes)
    let base_days = days;
    let y400 = base_days / 146097;
    let rem = base_days % 146097;
    let y100 = rem / 36524;
    let rem = rem % 36524;
    let y4 = rem / 1461;
    let rem = rem % 1461;
    let y1 = rem / 365;
    let rem = rem % 365;
    let year = 1970 + y400 * 400 + y100 * 100 + y4 * 4 + y1;
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days: [u32; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 0u32;
    let mut day = rem as u32 + 1;
    for (m, &d) in month_days.iter().enumerate() {
        if day <= d {
            month = m as u32 + 1;
            break;
        }
        day -= d;
    }
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, h, m, s
    )
}

/// Load strategy params from `best_params.json` if it exists; fall back to backtest-optimized defaults.
fn load_params() -> StrategyParams {
    if let Ok(content) = std::fs::read_to_string("best_params.json") {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            let p = StrategyParams {
                min_return: val["min_return"].as_f64().unwrap_or(0.1126) as f32,
                min_return_butterfly: val["min_return_butterfly"]
                    .as_f64()
                    .unwrap_or(0.0546) as f32,
                min_exit_return: val["min_exit_return"].as_f64().unwrap_or(0.10) as f32,
            };
            println!(
                "Loaded params from best_params.json: mr={:.4} mr_fly={:.4} mer={:.4}",
                p.min_return, p.min_return_butterfly, p.min_exit_return
            );
            return p;
        }
    }
    println!("best_params.json not found or failed to parse. Using backtest default StrategyParams.");
    StrategyParams {
        min_return: 0.1126,
        min_return_butterfly: 0.0546,
        min_exit_return: 0.10,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let symbol = if args.len() > 1 {
        args[1].clone()
    } else {
        "510500XSHG".to_string()
    };

    let parquet_file = format!("{}.parquet", symbol);
    println!("Loading parquet file: {}...", parquet_file);
    let df = LazyFrame::scan_parquet(&parquet_file, ScanArgsParquet::default())?
        .with_columns([
            col("a1_v").cast(DataType::Float64),
            col("b1_v").cast(DataType::Float64),
        ])
        .collect()?;
    println!("DataFrame loaded. Shape: {:?}", df.shape());

    let s_date = df.column("date")?.datetime()?;
    let s_type = df.column("type")?.str()?;
    let s_strike = df.column("strike")?.f64()?;
    let s_a1 = df.column("a1")?.f64()?;
    let s_b1 = df.column("b1")?.f64()?;
    let s_a1_v = df.column("a1_v")?.f64()?;
    let s_b1_v = df.column("b1_v")?.f64()?;
    let s_dte = df.column("days_to_maturity")?.i64()?;

    let n_rows = df.height();

    // ── Build BoxContext ──────────────────────────────────────────────────────
    let mut strikes_raw: Vec<i64> = (0..n_rows)
        .filter_map(|i| s_strike.get(i).map(|s| (s * 1000.0).round() as i64))
        .collect();
    strikes_raw.sort_unstable();
    strikes_raw.dedup();

    let float_strikes: Vec<f32> = strikes_raw.iter().map(|&s| s as f32 / 1000.0).collect();
    let dte = s_dte.get(0).unwrap_or(30) as u8;

    let ctx = BoxContext {
        prefix: symbol.clone(),
        dte,
        strikes: float_strikes.clone(),
        call_oids: float_strikes
            .iter()
            .map(|s| format!("C_{:.3}", s))
            .collect(),
        put_oids: float_strikes
            .iter()
            .map(|s| format!("P_{:.3}", s))
            .collect(),
    };

    let mut oid_to_ctx_idx: HashMap<String, Vec<usize>> = HashMap::new();
    for oid in ctx.call_oids.iter().chain(ctx.put_oids.iter()) {
        oid_to_ctx_idx.entry(oid.clone()).or_default().push(0);
    }
    let mut eval_contexts = vec![ctx];

    // ── Timestamp divisor ─────────────────────────────────────────────────────
    let divisor = match s_date.time_unit() {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
    };

    // If the datetime has a timezone, the underlying data is stored as UTC.
    // For China/CST (UTC+8), we add 8 hours (28,800 seconds) to get the wall-clock time.
    let tz_offset_sec = if s_date.time_zone().is_some() {
        28800
    } else {
        0
    };

    // ── Run simulation via Portfolio ──────────────────────────────────────────
    // Load the optimizer-derived best parameters from best_params.json,
    // falling back to the backtest defaults if needed.
    let params = load_params();
    let mut portfolio = Portfolio::new(INITIAL_CASH);
    let mut state: HashMap<String, TickData> = HashMap::new();
    let mut last_ts_sec: i64 = 0;

    // Track closed trades for reporting / CSV export
    let mut trades: Vec<TradeRecord> = Vec::new();

    // Track simulation date range (seconds)
    let mut first_ts: i64 = 0;
    let mut last_ts: i64 = 0;

    println!(
        "Starting simulation with {:.0} RMB starting capital...",
        INITIAL_CASH
    );

    for i in 0..n_rows {
        let timestamp_sec = s_date.get(i).unwrap_or(0) / divisor + tz_offset_sec;
        let current_dte = s_dte.get(i).unwrap_or(30) as u8;

        let opt_type = s_type.get(i).unwrap_or("");
        let strike = s_strike.get(i).unwrap_or(0.0) as f32;
        let oid = format!("{}_{:.3}", opt_type, strike);

        let tick = TickUpdate {
            oid,
            a1: s_a1.get(i).unwrap_or(0.0) as f32,
            b1: s_b1.get(i).unwrap_or(0.0) as f32,
            a1_v: s_a1_v.get(i).unwrap_or(0.0) as f32,
            b1_v: s_b1_v.get(i).unwrap_or(0.0) as f32,
        };

        // ── Update state unconditionally so quotes are ready at open ──────
        // Portfolio::update will re-insert, so we do it here just for the
        // is_trading_hours gate that follows.
        state.insert(
            tick.oid.clone(),
            TickData {
                a1: tick.a1,
                b1: tick.b1,
                a1_v: tick.a1_v,
                b1_v: tick.b1_v,
                timestamp: timestamp_sec,
            },
        );

        // ── Update the BoxContext DTE so ann-return scales with time ──────
        eval_contexts[0].dte = current_dte;

        // ── Skip evaluation outside continuous trading hours ──────────────
        if !live_boxx_rs::is_trading_hours(timestamp_sec) {
            continue;
        }

        if first_ts == 0 {
            first_ts = timestamp_sec;
        }
        last_ts = timestamp_sec;

        // Only trigger evaluation when the second advances (save compute)
        if timestamp_sec <= last_ts_sec {
            continue;
        }
        last_ts_sec = timestamp_sec;

        let time_str = fmt_ts(timestamp_sec);
        let events = portfolio.update(
            &tick,
            &mut state,
            &oid_to_ctx_idx,
            &eval_contexts,
            &params,
            timestamp_sec,
            &time_str,
        );

        for event in events {
            match event {
                PortfolioEvent::Exit { trade_record, .. } => {
                    trades.push(trade_record);
                }
                _ => {}
            }
        }
    }

    // ── Final report ──────────────────────────────────────────────────────────
    // Value open positions at their maturity payout (same logic as optimize.rs).
    // Long box: receive payout at maturity, net of entry cost already deducted.
    // Short box: must pay payout at maturity; entry premium already received.
    let mut open_pos_value: f32 = 0.0;
    for pos in &portfolio.positions {
        match pos.pos_type {
            live_boxx_rs::PosType::LongBox => open_pos_value += pos.payout * CONTRACT_SIZE,
            live_boxx_rs::PosType::ShortBox => open_pos_value -= pos.payout * CONTRACT_SIZE,
            live_boxx_rs::PosType::ButterflyCall | live_boxx_rs::PosType::ButterflyPut => {}
        }
    }

    let final_capital = portfolio.available_cash + portfolio.locked_margin + open_pos_value;
    let total_pnl = final_capital - INITIAL_CASH;
    let pnl_pct = total_pnl / INITIAL_CASH * 100.0;

    // ── Simulation duration & annualised return ───────────────────────────────
    let sim_days = if first_ts > 0 && last_ts > first_ts {
        (last_ts - first_ts) as f32 / 86400.0
    } else {
        1.0
    };
    let ann_return_pct = pnl_pct * 365.0 / sim_days;

    // ── Per-trade return distribution ─────────────────────────────────────────
    // Return = pnl_rmb / capital_deployed, where:
    //   LongBox:  capital = entry_cost * CONTRACT_SIZE  (cash paid)
    //   ShortBox: capital = 2 * payout * CONTRACT_SIZE  (margin locked)
    // This avoids distortion from near-zero cost entries.
    let trade_returns: Vec<f32> = trades
        .iter()
        .filter_map(|r| {
            if r.pos_type == "LongBox" && r.entry_cost > 1e-6 {
                let capital = r.entry_cost * CONTRACT_SIZE;
                Some(r.pnl_rmb / capital)
            } else if r.pos_type == "ShortBox" && r.payout > 1e-6 {
                let capital = 2.0 * r.payout * CONTRACT_SIZE;
                Some(r.pnl_rmb / capital)
            } else {
                None
            }
        })
        .collect();
    let avg_trade_ret = if !trade_returns.is_empty() {
        trade_returns.iter().sum::<f32>() / trade_returns.len() as f32
    } else {
        0.0
    };
    // Median is more robust than mean for skewed distributions
    let median_trade_ret = if !trade_returns.is_empty() {
        let mut sorted = trade_returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    } else {
        0.0
    };
    let positive_ret_pct = if !trade_returns.is_empty() {
        trade_returns.iter().filter(|&&r| r > 0.0).count() as f32 / trade_returns.len() as f32
            * 100.0
    } else {
        0.0
    };
    // How many trades beat the configured min_return?
    let above_min = trade_returns
        .iter()
        .filter(|&&r| r >= params.min_return)
        .count();
    let above_min_pct = if !trade_returns.is_empty() {
        above_min as f32 / trade_returns.len() as f32 * 100.0
    } else {
        0.0
    };

    // ── Per-trade stats ───────────────────────────────────────────────────────
    let n_trades = trades.len();
    let wins: Vec<f32> = trades
        .iter()
        .filter(|r| r.pnl_rmb > 0.0)
        .map(|r| r.pnl_rmb)
        .collect();
    let losses: Vec<f32> = trades
        .iter()
        .filter(|r| r.pnl_rmb < 0.0)
        .map(|r| r.pnl_rmb)
        .collect();
    let gross_win: f32 = wins.iter().sum();
    let gross_loss: f32 = losses.iter().sum();
    let avg_pnl = if n_trades > 0 {
        total_pnl / n_trades as f32
    } else {
        0.0
    };
    let avg_win = if !wins.is_empty() {
        gross_win / wins.len() as f32
    } else {
        0.0
    };
    let avg_loss = if !losses.is_empty() {
        gross_loss / losses.len() as f32
    } else {
        0.0
    };
    let max_win: f32 = wins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_loss: f32 = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    let profit_factor = if gross_loss.abs() > 0.0 {
        gross_win / gross_loss.abs()
    } else {
        f32::INFINITY
    };
    let win_rate = if n_trades > 0 {
        wins.len() as f32 / n_trades as f32 * 100.0
    } else {
        0.0
    };

    // ── Holding Time Stats ───────────────────────────────────────────────────
    let holding_times: Vec<f32> = trades.iter().map(|r| r.holding_time_sec as f32).collect();
    let avg_hold_time = if !holding_times.is_empty() {
        holding_times.iter().sum::<f32>() / holding_times.len() as f32
    } else {
        0.0
    };
    let median_hold_time = if !holding_times.is_empty() {
        let mut sorted = holding_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    } else {
        0.0
    };
    let fmt_dur = |secs: f32| -> String {
        if secs < 60.0 {
            format!("{:.1}s", secs)
        } else if secs < 3600.0 {
            format!("{:.1}m", secs / 60.0)
        } else {
            format!("{:.1}h", secs / 3600.0)
        }
    };

    // ── Per-type breakdown ────────────────────────────────────────────────────
    let long_pnl: f32 = trades
        .iter()
        .filter(|r| r.pos_type == "LongBox")
        .map(|r| r.pnl_rmb)
        .sum();
    let short_pnl: f32 = trades
        .iter()
        .filter(|r| r.pos_type == "ShortBox")
        .map(|r| r.pnl_rmb)
        .sum();
    let n_long = trades.iter().filter(|r| r.pos_type == "LongBox").count();
    let n_short = trades.iter().filter(|r| r.pos_type == "ShortBox").count();
    let n_early = trades.iter().filter(|r| r.exit_type == "early").count();
    let n_maturity = trades.iter().filter(|r| r.exit_type == "maturity").count();

    println!("\n\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!(" Backtest Summary");
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!(" Data Range       : {:.1} days", sim_days);
    println!(" Starting Capital : {:.0} RMB", INITIAL_CASH);
    println!(" Final Capital    : {:.2} RMB", final_capital);
    println!(
        " Total P&L        : {:+.2} RMB ({:+.2}%)",
        total_pnl, pnl_pct
    );
    println!(" Annualised Return: {:+.2}%", ann_return_pct);
    println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    println!(" Closed Trades    : {}", n_trades);
    println!("   Avg Hold Time  : {}", fmt_dur(avg_hold_time));
    println!("   Median HoldTime: {}", fmt_dur(median_hold_time));
    println!(
        "   LongBox        : {}  ({:+.2} RMB total)",
        n_long, long_pnl
    );
    println!(
        "   ShortBox       : {}  ({:+.2} RMB total)",
        n_short, short_pnl
    );
    println!("   Early exits    : {}", n_early);
    println!("   Maturity exits : {}", n_maturity);
    println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    if n_trades > 0 {
        println!(
            " Win / Loss       : {} / {} ({:.1}% win rate)",
            wins.len(),
            losses.len(),
            win_rate
        );
        println!(" Gross Profit     : {:+.2} RMB", gross_win);
        println!(" Gross Loss       : {:+.2} RMB", gross_loss);
        println!(" Profit Factor    : {:.3}", profit_factor);
        println!(" Avg PnL/trade    : {:+.2} RMB", avg_pnl);
        println!(" Avg Win          : {:+.2} RMB", avg_win);
        println!(" Avg Loss         : {:+.2} RMB", avg_loss);
        if max_win > f32::NEG_INFINITY {
            println!(" Best Trade       : {:+.2} RMB", max_win);
        }
        if max_loss < f32::INFINITY {
            println!(" Worst Trade      : {:+.2} RMB", max_loss);
        }
        println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    }
    println!(" Open Positions   : {}", portfolio.positions.len());
    println!(
        " Open Pos Value   : {:.2} RMB (at maturity)",
        open_pos_value
    );
    println!(" Available Cash   : {:.2} RMB", portfolio.available_cash);
    println!(" Locked Margin    : {:.2} RMB", portfolio.locked_margin);
    println!(
        " Strategy Params  : min_ret={:.4}  mer={:.4}",
        params.min_return, params.min_exit_return
    );
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");

    // ── Return Realism Check ───────────────────────────────────────────────
    // Answers: "Is the configured min_return actually being achieved in practice?"
    // Return metric: pnl_rmb / capital_deployed (Long=cash paid, Short=margin locked)
    println!("\n{}", "═".repeat(50));
    println!(" Return Realism Check (pnl / capital deployed)");
    println!("{}", "═".repeat(50));
    if !trade_returns.is_empty() {
        println!(
            " Configured min_return    : {:.2}%",
            params.min_return * 100.0
        );
        println!(" Avg  return/trade        : {:+.2}%", avg_trade_ret * 100.0);
        println!(
            " Median return/trade      : {:+.2}%",
            median_trade_ret * 100.0
        );
        println!(" % trades profitable      : {:.1}%", positive_ret_pct);
        println!(
            " % trades >= min_return   : {:.1}%  ({}/{})",
            above_min_pct,
            above_min,
            trade_returns.len()
        );
        if median_trade_ret <= 0.0 {
            println!(" \u{26a0}  WARNING: Median return is negative — strategy is losing on typical trades.");
            println!("    The average may be distorted by a few large winners.");
        } else if median_trade_ret >= params.min_return {
            println!(" \u{2713}  Return level is ACHIEVABLE: median trade return ({:.2}%) meets min_return.",
                median_trade_ret * 100.0);
        } else {
            println!(
                " ~  Return level PARTIALLY achieved: median ({:.2}%) < min_return ({:.2}%).",
                median_trade_ret * 100.0,
                params.min_return * 100.0
            );
            println!("    Early exits are reducing the realised return below the entry threshold.");
        }
    } else {
        println!(" No closed trades to evaluate.");
    }
    println!("{}", "═".repeat(50));

    // ── Write trades_{symbol}.csv ─────────────────────────────────────────────
    let csv_path_str = format!("trades_{}.csv", symbol);
    let csv_path = csv_path_str.as_str();
    let mut wtr = csv::Writer::from_path(csv_path)?;
    wtr.write_record(&[
        "pos_type",
        "prefix",
        "strike_lo",
        "strike_hi",
        "dte_at_entry",
        "entry_time",
        "exit_time",
        "entry_cost",
        "payout",
        "exit_value",
        "pnl_rmb",
        "exit_type",
        "holding_time_sec",
        "realised_return_pct",    // (exit_value - entry_cost) / cost_base * 100
        "hold_to_mat_return_pct", // (payout - entry_cost) / entry_cost * 100  [LongBox]
    ])?;
    for r in &trades {
        // Realised return (what we actually captured)
        let realised_ret = if r.pos_type == "LongBox" && r.entry_cost > 0.0 {
            (r.exit_value - r.entry_cost) / r.entry_cost * 100.0
        } else if r.pos_type == "ShortBox" && r.payout > 0.0 {
            let margin = 2.0 * r.payout;
            (r.entry_cost - r.exit_value) / margin * 100.0
        } else {
            0.0
        };
        // Hold-to-maturity return (what we would have earned without early exit)
        let h2m_ret = if r.pos_type == "LongBox" && r.entry_cost > 0.0 {
            (r.payout - r.entry_cost) / r.entry_cost * 100.0
        } else if r.pos_type == "ShortBox" && r.payout > 0.0 {
            let margin = 2.0 * r.payout;
            (r.entry_cost - r.payout) / margin * 100.0
        } else {
            0.0
        };
        wtr.write_record(&[
            &r.pos_type,
            &r.prefix,
            &format!("{:.3}", r.strike_lo),
            &format!("{:.3}", r.strike_hi),
            &r.dte_at_entry.to_string(),
            &r.entry_time,
            &r.exit_time,
            &format!("{:.6}", r.entry_cost),
            &format!("{:.6}", r.payout),
            &format!("{:.6}", r.exit_value),
            &format!("{:.2}", r.pnl_rmb),
            &r.exit_type,
            &r.holding_time_sec.to_string(),
            &format!("{:.4}", realised_ret),
            &format!("{:.4}", h2m_ret),
        ])?;
    }
    wtr.flush()?;
    println!(
        "\n {} written -> {} rows  ({})", csv_path,
        trades.len(),
        csv_path
    );

    Ok(())
}
