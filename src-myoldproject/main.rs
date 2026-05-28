use chrono::{Datelike, FixedOffset, NaiveDate, Utc};
use crossbeam_channel::bounded;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use live_boxx_rs::*;

#[derive(Serialize, Deserialize, Debug)]
struct WsAuthMsg {
    action: String,
    license: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct WsSubMsg {
    action: String,
    channels: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct WsTick {
    action: String,
    order_book_id: String,
    ask: Vec<f32>,
    bid: Vec<f32>,
    #[serde(alias = "ask_vols")]
    ask_vol: Vec<f32>,
    #[serde(alias = "bid_vols")]
    bid_vol: Vec<f32>,
}

fn set_high_priority() {
    #[cfg(unix)]
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, -20);
    }
}

/// Load strategy params from `best_params.json` if it exists; fall back to defaults.
fn load_params() -> StrategyParams {
    if let Ok(content) = std::fs::read_to_string("best_params.json") {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            let p = StrategyParams {
                min_return: val["min_return"].as_f64().unwrap_or(0.015) as f32,
                min_return_butterfly: val["min_return_butterfly"]
                    .as_f64()
                    .unwrap_or(0.065) as f32,
                min_exit_return: val["min_exit_return"].as_f64().unwrap_or(0.05) as f32,
            };
            println!(
                "Loaded params from best_params.json: mr={:.3} mr_fly={:.3} mer={:.4}",
                p.min_return, p.min_return_butterfly, p.min_exit_return
            );
            return p;
        }
    }
    println!("Using default StrategyParams (no best_params.json found).");
    StrategyParams::default()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let license_key = std::fs::read_to_string(".license")
        .expect("Error: .license file not found. Please create a .license file.")
        .trim()
        .to_string();

    println!("Fetching token and option metadata...");
    let token = fetch_token(&license_key).await.expect("Failed to fetch token");
    let init_options = fetch_options(&token).await.expect("Failed to fetch options");

    if init_options.is_empty() {
        eprintln!("Error: No active options found.");
        return Ok(());
    }

    let tz_beijing = FixedOffset::east_opt(8 * 3600).unwrap();
    let today = Utc::now().with_timezone(&tz_beijing).date_naive();
    let mut internal_options = Vec::new();

    for opt in init_options {
        let m_date = if let Ok(d) = NaiveDate::parse_from_str(&opt.maturity, "%Y-%m-%d") {
            d
        } else {
            continue;
        };
        let expiry = get_4th_wednesday(m_date.year(), m_date.month());
        let dte = (expiry - today).num_days();
        if dte > 0 && dte <= 255 {
            let prefix = opt
                .underlying
                .split('.')
                .next()
                .unwrap_or(&opt.underlying)
                .to_string();
            internal_options.push((opt.oid, prefix, expiry.format("%Y-%m-%d").to_string(), opt.opt_type, opt.strike, dte as u8));
        }
    }

    if internal_options.is_empty() {
        eprintln!("Error: No valid near-term options found.");
        return Ok(());
    }

    let mut all_dtes: Vec<u8> = internal_options.iter().map(|o| o.5).collect();
    all_dtes.sort_unstable();
    all_dtes.dedup();
    let min_dte = all_dtes[0];

    let mut active_oids = Vec::new();
    let mut strike_map: HashMap<(String, String, u8), HashMap<String, HashMap<String, String>>> =
        HashMap::new();

    for (oid, prefix, expiry, opt_type, strike, dte) in internal_options {
        if dte == min_dte {
            active_oids.push(oid.clone());
            let key = (prefix, expiry, dte);
            let strike_k = format!("{:.1}", strike);
            strike_map.entry(key).or_default().entry(strike_k).or_default().insert(opt_type, oid);
        }
    }

    let mut box_contexts = Vec::new();
    let mut oid_to_ctx_idx: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ctx_idx = 0;

    for ((prefix, _expiry, dte), strikes) in strike_map {
        let mut sorted_keys: Vec<String> = strikes.keys().cloned().collect();
        sorted_keys.sort_by(|a, b| {
            a.parse::<f32>().unwrap().partial_cmp(&b.parse::<f32>().unwrap()).unwrap()
        });

        let mut strikes_vec = Vec::new();
        let mut call_oids = Vec::new();
        let mut put_oids = Vec::new();

        for k_str in sorted_keys {
            let types = strikes.get(&k_str).unwrap();
            if let (Some(c), Some(p)) = (types.get("C"), types.get("P")) {
                strikes_vec.push(k_str.parse::<f32>().unwrap());
                call_oids.push(c.clone());
                put_oids.push(p.clone());
                oid_to_ctx_idx.entry(c.clone()).or_default().push(ctx_idx);
                oid_to_ctx_idx.entry(p.clone()).or_default().push(ctx_idx);
            }
        }

        if strikes_vec.len() >= 2 {
            box_contexts.push(BoxContext { prefix, dte, strikes: strikes_vec, call_oids, put_oids });
            ctx_idx += 1;
        }
    }

    println!(
        "Loaded {} active options for dte={}. Initializing WebSocket...",
        active_oids.len(),
        min_dte
    );

    let (tick_tx, tick_rx) = bounded::<TickUpdate>(100000);
    let (log_tx, log_rx) = bounded::<String>(10000);

    std::thread::spawn(move || {
        while let Ok(msg) = log_rx.recv() {
            eprintln!("{}", msg);
        }
    });

    let eval_contexts = box_contexts;
    let mut state: HashMap<String, TickData> = HashMap::with_capacity(3000);

    // --- CSV persistence background thread ---
    let (db_tx, db_rx) = bounded::<TradeRecord>(4096);
    std::thread::spawn(move || {
        let file_exists = std::path::Path::new("trades_live.csv").exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("trades_live.csv")
            .expect("Cannot open trades_live.csv");

        if !file_exists {
            let _ = writeln!(
                file,
                "pos_type,prefix,strike_lo,strike_hi,dte_at_entry,entry_time,exit_time,entry_cost,payout,exit_value,pnl_rmb,exit_type"
            );
        }

        while let Ok(rec) = db_rx.recv() {
            let _ = writeln!(
                file,
                "{},{},{:.2},{:.2},{},{},{},{:.4},{:.4},{:.4},{:.2},{}",
                rec.pos_type, rec.prefix, rec.strike_lo, rec.strike_hi,
                rec.dte_at_entry, rec.entry_time, rec.exit_time,
                rec.entry_cost, rec.payout, rec.exit_value, rec.pnl_rmb, rec.exit_type
            );
        }
    });

    // --- Evaluator thread (now delegates to Portfolio::update) ---
    std::thread::spawn(move || {
        set_high_priority();
        let params = load_params();
        let mut portfolio = Portfolio::new(INITIAL_CASH);
        let mut last_print = std::time::Instant::now();
        let tz_beijing = FixedOffset::east_opt(8 * 3600).unwrap();

        while let Ok(tick) = tick_rx.recv() {
            let now = Utc::now().with_timezone(&tz_beijing);
            let now_ts = now.timestamp();
            let time_str = now.format("%H:%M:%S").to_string();

            let events = portfolio.update(
                &tick,
                &mut state,
                &oid_to_ctx_idx,
                &eval_contexts,
                &params,
                now_ts,
                &time_str,
            );

            for event in events {
                match event {
                    PortfolioEvent::Exit { pos_type, ctx_idx, strike_i, strike_j,
                                          close_net_or_buy_back, payout, pnl_rmb, trade_record } =>
                    {
                        let ctx = &eval_contexts[ctx_idx];
                        let label = match pos_type {
                            PosType::LongBox  => format!(
                                "EXIT LongBox [{pfx}] K{lo:.2}/{hi:.2} close_net={cn:.4} payout={py:.4} PnL={pnl:+.2} RMB\n    └─ Cash: {cash:.0} RMB | Open Positions: {cnt}",
                                pfx=ctx.prefix, lo=ctx.strikes[strike_i], hi=ctx.strikes[strike_j],
                                cn=close_net_or_buy_back, py=payout, pnl=pnl_rmb,
                                cash=portfolio.available_cash, cnt=portfolio.positions.len(),
                            ),
                            PosType::ShortBox => format!(
                                "EXIT ShortBox [{pfx}] K{lo:.2}/{hi:.2} buy_back={bb:.4} payout={py:.4} PnL={pnl:+.2} RMB\n    └─ Cash: {cash:.0} RMB | Open Positions: {cnt}",
                                pfx=ctx.prefix, lo=ctx.strikes[strike_i], hi=ctx.strikes[strike_j],
                                bb=close_net_or_buy_back, py=payout, pnl=pnl_rmb,
                                cash=portfolio.available_cash, cnt=portfolio.positions.len(),
                            ),
                            PosType::ButterflyCall | PosType::ButterflyPut => unreachable!(),
                        };
                        let _ = log_tx.try_send(format!("[{time_str}] *** {label}"));
                        let _ = db_tx.try_send(trade_record);
                    }

                    PortfolioEvent::EntryLong { ctx_idx, strike_i, strike_j, cost, payout, ann, msg: _, is_new } => {
                        let ctx = &eval_contexts[ctx_idx];
                        if is_new {
                            let _ = log_tx.try_send(format!(
                                "[{ts}] >>> ENTER LongBox [{pfx}] K{k1:.2}/{k2:.2} cost={cost:.4} payout={py:.4} ret={ret:.2}%\n    └─ Cash: {cash:.0} RMB | Margin: {margin:.0} RMB | Open Positions: {cnt}",
                                ts=time_str, pfx=ctx.prefix,
                                k1=ctx.strikes[strike_i], k2=ctx.strikes[strike_j],
                                py=payout, ret=ann*100.0,
                                cash=portfolio.available_cash, margin=portfolio.locked_margin,
                                cnt=portfolio.positions.len(),
                            ));
                        }
                    }

                    PortfolioEvent::EntryShort { ctx_idx, strike_i, strike_j, gain, payout_val, margin_rmb: _, ann, msg: _, is_new } => {
                        let ctx = &eval_contexts[ctx_idx];
                        if is_new {
                            let _ = log_tx.try_send(format!(
                                "[{ts}] >>> ENTER ShortBox [{pfx}] K{k1:.2}/{k2:.2} gain={gain:.4} payout={py:.4} ret={ret:.2}%\n    └─ Cash: {cash:.0} RMB | Margin: {margin:.0} RMB | Open Positions: {cnt}",
                                ts=time_str, pfx=ctx.prefix,
                                k1=ctx.strikes[strike_i], k2=ctx.strikes[strike_j],
                                py=payout_val, ret=ann*100.0,
                                cash=portfolio.available_cash, margin=portfolio.locked_margin,
                                cnt=portfolio.positions.len(),
                            ));
                        }
                    }

                    PortfolioEvent::EntryButterfly { ctx_idx, flavor, strike_i, strike_j, strike_k, profit, margin_rmb, ann, msg: _, is_new } => {
                        let ctx = &eval_contexts[ctx_idx];
                        if is_new {
                            let _ = log_tx.try_send(format!(
                                "[{ts}] >>> ENTER {flavor}Butterfly [{pfx}] K{k1:.2}/{k2:.2}/{k3:.2} profit={profit:.4} margin={mg:.0} RMB ret={ret:.2}%\n    └─ Cash: {cash:.0} RMB | Margin: {margin:.0} RMB | Open Positions: {cnt}",
                                ts=time_str, pfx=ctx.prefix,
                                k1=ctx.strikes[strike_i], k2=ctx.strikes[strike_j], k3=ctx.strikes[strike_k],
                                mg=margin_rmb, ret=ann*100.0,
                                cash=portfolio.available_cash, margin=portfolio.locked_margin,
                                cnt=portfolio.positions.len(),
                            ));
                        }
                    }
                }
            }

            if last_print.elapsed() >= std::time::Duration::from_secs(5) {
                // Mark-to-market net worth: use live bid/ask prices from state.
                // LongBox: available_cash already had entry_cost*CS deducted → add current bid value.
                // ShortBox: available_cash already has entry_gain*CS added, margin is in locked_margin
                //           → subtract current ask (buyback cost) as the open liability.
                // Butterfly: margin is in locked_margin, received credit is in available_cash → +0.
                let mut open_pos_value: f32 = 0.0;
                for pos in &portfolio.positions {
                    let ctx = &eval_contexts[pos.ctx_idx];
                    let si = pos.strike_i;
                    let sj = pos.strike_j;
                    match pos.pos_type {
                        live_boxx_rs::PosType::LongBox => {
                            // Current value = what we'd receive closing the box right now (bid side).
                            let mtm = match (
                                state.get(&ctx.call_oids[si]),
                                state.get(&ctx.call_oids[sj]),
                                state.get(&ctx.put_oids[sj]),
                                state.get(&ctx.put_oids[si]),
                            ) {
                                (Some(c_lo), Some(c_hi), Some(p_hi), Some(p_lo))
                                    if c_lo.b1 > 0.0 && c_hi.a1 > 0.0
                                        && p_hi.b1 > 0.0 && p_lo.a1 > 0.0 =>
                                {
                                    ((c_lo.b1 - c_hi.a1) + (p_hi.b1 - p_lo.a1))
                                        .max(0.0)
                                }
                                // Fall back to entry cost when prices are stale/missing.
                                _ => pos.entry_cost,
                            };
                            open_pos_value += mtm * CONTRACT_SIZE;
                        }
                        live_boxx_rs::PosType::ShortBox => {
                            // Current liability = what it costs to buy back right now (ask side).
                            let buyback = match (
                                state.get(&ctx.call_oids[si]),
                                state.get(&ctx.call_oids[sj]),
                                state.get(&ctx.put_oids[sj]),
                                state.get(&ctx.put_oids[si]),
                            ) {
                                (Some(c_lo), Some(c_hi), Some(p_hi), Some(p_lo))
                                    if c_lo.a1 > 0.0 && c_hi.b1 > 0.0
                                        && p_hi.a1 > 0.0 && p_lo.b1 > 0.0 =>
                                {
                                    ((c_lo.a1 - c_hi.b1) + (p_hi.a1 - p_lo.b1))
                                        .max(0.0)
                                }
                                // Fall back to payout (worst case) when stale.
                                _ => pos.payout,
                            };
                            open_pos_value -= buyback * CONTRACT_SIZE;
                        }
                        // Butterfly: margin already in locked_margin, net credit already in cash.
                        live_boxx_rs::PosType::ButterflyCall
                        | live_boxx_rs::PosType::ButterflyPut => {}
                    }
                }
                let net_worth = portfolio.available_cash + portfolio.locked_margin + open_pos_value;

                let _ = log_tx.try_send(format!(
                    "[{}] Evaluator: {} ticks | NetWorth={:.0} RMB | Cash={:.0} RMB | Margin={:.0} RMB | Positions={}",
                    Utc::now().with_timezone(&tz_beijing).format("%H:%M:%S"),
                    portfolio.eval_count,
                    net_worth,
                    portfolio.available_cash,
                    portfolio.locked_margin,
                    portfolio.positions.len()
                ));
                last_print = std::time::Instant::now();
            }
        }
    });

    let url = "wss://rqdata.ricequant.com/live_md";
    let (ws_stream, _) = connect_async(url).await.expect("Failed to connect to Ricequant WebSocket");
    let (mut write, mut read) = ws_stream.split();

    let auth_msg = WsAuthMsg { action: "auth".to_string(), license: license_key.clone() };
    write.send(Message::Text(serde_json::to_string(&auth_msg)?)).await?;

    for chunk in active_oids.chunks(100) {
        let sub_msg = WsSubMsg {
            action: "subscribe".to_string(),
            channels: chunk.iter().map(|oid| format!("tick_{}", oid)).collect(),
        };
        write.send(Message::Text(serde_json::to_string(&sub_msg)?)).await?;
    }

    println!("Listening for live ticks... (Press Ctrl+C to stop)");
    let close_time = chrono::NaiveTime::from_hms_opt(15, 5, 0).unwrap();
    while let Some(msg) = read.next().await {
        if Utc::now().with_timezone(&tz_beijing).time() > close_time {
            println!("Market closed (past 15:05:00). Exiting live engine.");
            break;
        }
        let msg = msg?;
        if let Message::Text(text) = msg {
            if let Ok(tick) = serde_json::from_str::<WsTick>(&text) {
                if tick.action == "feed" && !tick.ask.is_empty() && !tick.bid.is_empty() {
                    let _ = tick_tx.try_send(TickUpdate {
                        oid: tick.order_book_id,
                        a1: tick.ask[0],
                        b1: tick.bid[0],
                        a1_v: tick.ask_vol[0],
                        b1_v: tick.bid_vol[0],
                    });
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_logic() {
        let ctx = BoxContext {
            prefix: "TEST".to_string(),
            dte: 10,
            strikes: vec![3.0, 3.1],
            call_oids: vec!["c3.0".into(), "c3.1".into()],
            put_oids: vec!["p3.0".into(), "p3.1".into()],
        };

        let positions = vec![
            OpenPosition {
                pos_type: PosType::LongBox,
                ctx_idx: 0,
                strike_i: 0,
                strike_j: 1,
                strike_k: 0,
                entry_cost: 0.09,
                payout: 0.10,
                margin_rmb: 0.0,
                dte_at_entry: 20,
                _entry_ts: 0,
                entry_time_str: "10:00:00".into(),
            },
            OpenPosition {
                pos_type: PosType::ShortBox,
                ctx_idx: 0,
                strike_i: 0,
                strike_j: 1,
                strike_k: 0,
                entry_cost: 0.11,
                payout: 0.10,
                margin_rmb: 0.20 * CONTRACT_SIZE,
                dte_at_entry: 20,
                _entry_ts: 0,
                entry_time_str: "10:00:00".into(),
            },
        ];

        let mut state = HashMap::new();
        let now_ts = Utc::now().timestamp();

        state.insert("c3.0".to_string(), TickData { a1: 0.14, b1: 0.1508, a1_v: 10.0, b1_v: 10.0, timestamp: now_ts });
        state.insert("c3.1".to_string(), TickData { a1: 0.05, b1: 0.05,   a1_v: 10.0, b1_v: 10.0, timestamp: now_ts });
        state.insert("p3.1".to_string(), TickData { a1: 0.10, b1: 0.10,   a1_v: 10.0, b1_v: 10.0, timestamp: now_ts });
        state.insert("p3.0".to_string(), TickData { a1: 0.10, b1: 0.10,   a1_v: 10.0, b1_v: 10.0, timestamp: now_ts });

        let mut closed = Vec::new();
        for (pi, pos) in positions.iter().enumerate() {
            let si = pos.strike_i;
            let sj = pos.strike_j;
            let c_lo = state.get(&ctx.call_oids[si]).unwrap();
            let c_hi = state.get(&ctx.call_oids[sj]).unwrap();
            let p_hi = state.get(&ctx.put_oids[sj]).unwrap();
            let p_lo = state.get(&ctx.put_oids[si]).unwrap();

            match pos.pos_type {
                PosType::LongBox => {
                    let close_net = (c_lo.b1 - c_hi.a1) + (p_hi.b1 - p_lo.a1) - BOX_COMMISSION;
                    let fv_close = close_net * (1.0 + 0.02 * ctx.dte as f32 / 365.0);
                    if fv_close > pos.payout { closed.push(pi); }
                }
                PosType::ShortBox => {
                    let buy_back = (c_lo.a1 - c_hi.b1) + (p_hi.a1 - p_lo.b1) + BOX_COMMISSION;
                    let fv_buy_back = buy_back * (1.0 + 0.02 * ctx.dte as f32 / 365.0);
                    if buy_back > 0.0 && fv_buy_back < pos.payout { closed.push(pi); }
                }
                PosType::ButterflyCall | PosType::ButterflyPut => {}
            }
        }

        assert!(closed.contains(&0), "LongBox should have exited early!");
        assert!(closed.contains(&1), "ShortBox should have exited early!");
    }
}
