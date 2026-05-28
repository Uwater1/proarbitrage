use live_boxx_rs::*;
use polars::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::io::Write;

// ─── Integer-indexed tick (no String allocation per row) ────────────────────

/// A tick where the OID has been resolved to a `usize` index.
#[derive(Clone, Copy)]
struct IndexedTick {
    oid_idx: u32, // index into the flat state Vec
    a1: f32,
    b1: f32,
    a1_v: f32,
    b1_v: f32,
    timestamp: i64,
    dte: u8,
}

/// BoxContext equivalent with integer indices into the state Vec.
struct IndexedCtx {
    strikes: Vec<f32>,
    /// state-Vec index for call at strike[i]
    call_idx: Vec<u32>,
    /// state-Vec index for put at strike[i]
    put_idx: Vec<u32>,
}

struct Dataset {
    ticks: Vec<IndexedTick>,
    ctx: IndexedCtx,
    num_oids: usize,
}

// ─── Data loading ────────────────────────────────────────────────────────────

fn load_data(symbol: &str) -> Dataset {
    let parquet_file = format!("{}.parquet", symbol);
    println!("Loading parquet file: {}...", parquet_file);
    let df = LazyFrame::scan_parquet(&parquet_file, ScanArgsParquet::default())
        .unwrap()
        .with_columns([
            col("a1_v").cast(DataType::Float64),
            col("b1_v").cast(DataType::Float64),
        ])
        .collect()
        .unwrap();

    let s_date = df.column("date").unwrap().datetime().unwrap();
    let s_type = df.column("type").unwrap().str().unwrap();
    let s_strike = df.column("strike").unwrap().f64().unwrap();
    let s_a1 = df.column("a1").unwrap().f64().unwrap();
    let s_b1 = df.column("b1").unwrap().f64().unwrap();
    let s_a1_v = df.column("a1_v").unwrap().f64().unwrap();
    let s_b1_v = df.column("b1_v").unwrap().f64().unwrap();
    let s_dte = df.column("days_to_maturity").unwrap().i64().unwrap();

    let n_rows = df.height();

    // Collect unique strikes, sorted
    let mut strike_set: Vec<i64> = (0..n_rows)
        .filter_map(|i| s_strike.get(i).map(|s| (s * 1000.0).round() as i64))
        .collect();
    strike_set.sort_unstable();
    strike_set.dedup();

    let n_strikes = strike_set.len();
    let float_strikes: Vec<f32> = strike_set.iter().map(|&s| s as f32 / 1000.0).collect();

    // OID layout: call for strike i → index i, put for strike i → index n_strikes + i
    let num_oids = 2 * n_strikes;
    let call_idx: Vec<u32> = (0..n_strikes as u32).collect();
    let put_idx: Vec<u32> = (n_strikes as u32..(2 * n_strikes) as u32).collect();

    let ctx = IndexedCtx {
        strikes: float_strikes,
        call_idx,
        put_idx,
    };

    // Timestamp divisor
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

    // Build indexed ticks — no String anywhere
    let mut ticks = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let raw_strike = (s_strike.get(i).unwrap_or(0.0) * 1000.0).round() as i64;
        // Binary search into sorted strike_set
        let Ok(pos) = strike_set.binary_search(&raw_strike) else {
            continue;
        };

        let is_call = s_type.get(i).unwrap_or("") == "C";
        let oid_idx = if is_call {
            pos as u32
        } else {
            (n_strikes + pos) as u32
        };

        ticks.push(IndexedTick {
            oid_idx,
            a1: s_a1.get(i).unwrap_or(0.0) as f32,
            b1: s_b1.get(i).unwrap_or(0.0) as f32,
            a1_v: s_a1_v.get(i).unwrap_or(0.0) as f32,
            b1_v: s_b1_v.get(i).unwrap_or(0.0) as f32,
            timestamp: s_date.get(i).unwrap_or(0) / divisor + tz_offset_sec,
            dte: s_dte.get(i).unwrap_or(30) as u8,
        });
    }

    println!(
        "Loaded {} ticks, {} unique strikes, {} OIDs.",
        ticks.len(),
        n_strikes,
        num_oids
    );
    Dataset {
        ticks,
        ctx,
        num_oids,
    }
}

// ─── Fast evaluation helpers (Vec-indexed, no HashMap) ───────────────────────

struct SimplePos {
    strike_i: usize,
    strike_j: usize,
    strike_k: usize,
    pos_type: u8,    // 0=LongBox, 1=ShortBox, 2=Butterfly
    entry_cost: f32, // cost paid (long) or premium received (short)
    payout: f32,
    margin_rmb: f32,
    entry_ts: i64,
}

/// Evaluate long-box across all valid strike pairs using Vec indexing.
/// Returns (strike_i, strike_j, cost, payout, ann_return).
fn eval_long_box(
    ctx: &IndexedCtx,
    state: &[TickData],
    now_ts: i64,
    current_dte: u8,
    params: &StrategyParams,
) -> Option<(usize, usize, f32, f32, f32)> {
    let n = ctx.strikes.len();
    let mut best: Option<(usize, usize, f32, f32, f32)> = None;
    let mut best_ret = params.min_return;

    // Collect valid strikes
    let mut valid: Vec<(usize, u32, u32)> = Vec::with_capacity(n); // (strike_pos, call_idx, put_idx)
    for i in 0..n {
        let ci = ctx.call_idx[i] as usize;
        let pi = ctx.put_idx[i] as usize;
        let c = &state[ci];
        let p = &state[pi];
        if now_ts - c.timestamp <= MAX_STALE_SECONDS
            && now_ts - p.timestamp <= MAX_STALE_SECONDS
            && c.a1 > 0.0
            && c.b1 > 0.0
            && p.a1 > 0.0
            && p.b1 > 0.0
            && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
            && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
        {
            valid.push((i, ctx.call_idx[i], ctx.put_idx[i]));
        }
    }

    for vi in 0..valid.len() {
        let (i, ci, pi) = valid[vi];
        for vj in (vi + 1)..valid.len() {
            let (j, cj, pj) = valid[vj];
            let payout = ctx.strikes[j] - ctx.strikes[i];
            let cost = (state[ci as usize].a1 - state[cj as usize].b1)
                + (state[pj as usize].a1 - state[pi as usize].b1)
                + BOX_COMMISSION;
            if cost > 0.0 {
                let ret = (payout - cost) / cost;
                if ret > best_ret {
                    best_ret = ret;
                    best = Some((i, j, cost, payout, ret));
                }
            }
        }
    }
    best
}

/// Evaluate short-box. Returns (strike_i, strike_j, gain, payout_val, margin_rmb).
fn eval_short_box(
    ctx: &IndexedCtx,
    state: &[TickData],
    now_ts: i64,
    current_dte: u8,
    params: &StrategyParams,
) -> Option<(usize, usize, f32, f32, f32)> {
    let n = ctx.strikes.len();
    let mut best: Option<(usize, usize, f32, f32, f32)> = None;
    let mut best_ret = params.min_return;

    let mut valid: Vec<(usize, u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let ci = ctx.call_idx[i] as usize;
        let pi = ctx.put_idx[i] as usize;
        let c = &state[ci];
        let p = &state[pi];
        if now_ts - c.timestamp <= MAX_STALE_SECONDS
            && now_ts - p.timestamp <= MAX_STALE_SECONDS
            && c.a1 > 0.0
            && c.b1 > 0.0
            && p.a1 > 0.0
            && p.b1 > 0.0
            && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
            && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
        {
            valid.push((i, ctx.call_idx[i], ctx.put_idx[i]));
        }
    }

    for vi in 0..valid.len() {
        let (i, ci, pi) = valid[vi];
        for vj in (vi + 1)..valid.len() {
            let (j, cj, pj) = valid[vj];
            let width = ctx.strikes[j] - ctx.strikes[i];
            let gain = (state[ci as usize].b1 - state[cj as usize].a1)
                + (state[pj as usize].b1 - state[pi as usize].a1)
                - BOX_COMMISSION;
            let profit = gain - width;
            if profit > 0.0 {
                let margin_price = 2.0 * width;
                let ret = profit / margin_price;
                if ret > best_ret {
                    best_ret = ret;
                    let margin_rmb = margin_price * CONTRACT_SIZE;
                    best = Some((i, j, gain, width, margin_rmb));
                }
            }
        }
    }
    best
}

/// Evaluate butterfly
fn eval_butterfly(
    ctx: &IndexedCtx,
    state: &[TickData],
    now_ts: i64,
    current_dte: u8,
    params: &StrategyParams,
) -> Option<(usize, usize, usize, f32, f32)> {
    // (i, j, k, profit, margin_rmb)
    let n = ctx.strikes.len();
    if n < 3 {
        return None;
    }
    let mut best: Option<(usize, usize, usize, f32, f32)> = None;
    let mut best_ret = params.min_return_butterfly;

    let mut valid: Vec<(usize, u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let ci = ctx.call_idx[i] as usize;
        let pi = ctx.put_idx[i] as usize;
        let c = &state[ci];
        let p = &state[pi];
        if now_ts - c.timestamp <= MAX_STALE_SECONDS
            && now_ts - p.timestamp <= MAX_STALE_SECONDS
            && c.a1 > 0.0
            && c.b1 > 0.0
            && p.a1 > 0.0
            && p.b1 > 0.0
            && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
            && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
        {
            valid.push((i, ctx.call_idx[i], ctx.put_idx[i]));
        }
    }

    let m_len = valid.len();
    if m_len < 3 {
        return None;
    }

    for idx_1 in 0..(m_len - 2) {
        let (i, ci_1, pi_1) = valid[idx_1];
        for idx_2 in (idx_1 + 1)..(m_len - 1) {
            let (j, ci_2, pi_2) = valid[idx_2];
            let width_1 = ctx.strikes[j] - ctx.strikes[i];

            for idx_3 in (idx_2 + 1)..m_len {
                let (k, ci_3, pi_3) = valid[idx_3];
                let width_2 = ctx.strikes[k] - ctx.strikes[j];

                if (width_1 - width_2).abs() > 1e-4 {
                    continue;
                }

                let margin = width_1;

                // Call Butterfly
                let call_cost = state[ci_1 as usize].a1 - 2.0 * state[ci_2 as usize].b1
                    + state[ci_3 as usize].a1
                    + BUTTERFLY_COMMISSION;
                if call_cost < 0.0 {
                    let profit = -call_cost;
                    let ret = profit / margin;
                    if ret > best_ret {
                        best_ret = ret;
                        best = Some((i, j, k, profit, margin * CONTRACT_SIZE));
                    }
                }

                // Put Butterfly
                let put_cost = state[pi_1 as usize].a1 - 2.0 * state[pi_2 as usize].b1
                    + state[pi_3 as usize].a1
                    + BUTTERFLY_COMMISSION;
                if put_cost < 0.0 {
                    let profit = -put_cost;
                    let ret = profit / margin;
                    if ret > best_ret {
                        best_ret = ret;
                        best = Some((i, j, k, profit, margin * CONTRACT_SIZE));
                    }
                }
            }
        }
    }
    best
}

// ─── Simulation (zero String/HashMap in hot loop) ────────────────────────────

fn simulate(params: &StrategyParams, data: &Dataset) -> f32 {
    let ctx = &data.ctx;
    // Flat state Vec — indexed by oid_idx
    let mut state = vec![TickData::default(); data.num_oids];
    let mut positions: Vec<SimplePos> = Vec::with_capacity(8);
    let mut available_cash: f32 = INITIAL_CASH;
    let mut locked_margin: f32 = 0.0;
    let mut last_ts: i64 = 0;
    let mut current_dte: u8 = 30;

    for t in &data.ticks {
        current_dte = t.dte;
        // Update state — direct array write, no hashing
        let idx = t.oid_idx as usize;
        state[idx] = TickData {
            a1: t.a1,
            b1: t.b1,
            a1_v: t.a1_v,
            b1_v: t.b1_v,
            timestamp: t.timestamp,
        };

        // ── Skip ticks outside continuous trading hours (09:30-11:30, 13:00-15:00) ──
        // Prevents the call auction (09:15-09:25) from triggering unrealistic fills.
        if !live_boxx_rs::is_trading_hours(t.timestamp) {
            continue;
        }

        if t.timestamp <= last_ts {
            continue; // only evaluate once per timestamp
        }
        let now_ts = t.timestamp;
        last_ts = now_ts;

        // ── Phase 0: Settle positions at maturity (DTE = 0) ──────────────────────
        // A position that has reached expiry is settled at its known payout value.
        // This frees capital and removes the position so it is not counted twice
        // in the final open-position valuation at the end of the run.
        if current_dte == 0 {
            // Settle all expired positions at their known payout value.
            // Drain from the back to avoid index-shifting issues.
            while let Some(pos) = positions.pop() {
                if pos.pos_type == 0 {
                    available_cash += pos.payout * CONTRACT_SIZE;
                } else {
                    available_cash += pos.margin_rmb - pos.payout * CONTRACT_SIZE;
                    locked_margin -= pos.margin_rmb;
                }
            }
        }

        // ── Phase 1: Check exits ──────────────────────────────────────────────
        let mut i = 0;
        while i < positions.len() {
            let si = positions[i].strike_i;
            let sj = positions[i].strike_j;
            let is_long = positions[i].pos_type == 0;
            let payout = positions[i].payout;
            let margin_rmb = positions[i].margin_rmb;

            let ci = ctx.call_idx[si] as usize;
            let cj = ctx.call_idx[sj] as usize;
            let pj = ctx.put_idx[sj] as usize;
            let pi = ctx.put_idx[si] as usize;

            let fresh = now_ts - state[ci].timestamp <= MAX_STALE_SECONDS
                && now_ts - state[cj].timestamp <= MAX_STALE_SECONDS
                && now_ts - state[pj].timestamp <= MAX_STALE_SECONDS
                && now_ts - state[pi].timestamp <= MAX_STALE_SECONDS;

            let hold_met = now_ts - positions[i].entry_ts >= 60;

            let mut exit_threshold_long = params.min_exit_return;
            let mut exit_threshold_short = params.min_exit_return;
            if current_dte <= 4 {
                exit_threshold_long += EARLY_EXIT_ADJUSTMENT;
                exit_threshold_short = (exit_threshold_short - EARLY_EXIT_ADJUSTMENT).max(0.0);
            }

            let should_close = if fresh && hold_met && positions[i].pos_type == 0 {
                let close_net =
                    (state[ci].b1 - state[cj].a1) + (state[pj].b1 - state[pi].a1) - BOX_COMMISSION;
                close_net * (1.0 + exit_threshold_long) > payout
            } else if fresh && hold_met && positions[i].pos_type == 1 {
                let buy_back =
                    (state[ci].a1 - state[cj].b1) + (state[pj].a1 - state[pi].b1) + BOX_COMMISSION;
                buy_back > 0.0 && buy_back * (1.0 + exit_threshold_short) < payout
            } else {
                false
            };

            if should_close {
                if positions[i].pos_type == 0 {
                    let close_net = (state[ci].b1 - state[cj].a1) + (state[pj].b1 - state[pi].a1)
                        - BOX_COMMISSION;
                    available_cash += close_net * CONTRACT_SIZE;
                } else {
                    let buy_back = (state[ci].a1 - state[cj].b1)
                        + (state[pj].a1 - state[pi].b1)
                        + BOX_COMMISSION;
                    available_cash += margin_rmb - buy_back * CONTRACT_SIZE;
                    locked_margin -= margin_rmb;
                }
                positions.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // ── Phase 2: Scan for entries ─────────────────────────────────────────
        if current_dte > 5 && !positions.iter().any(|p| p.pos_type == 0) {
            if let Some((si, sj, cost, payout, _ann)) =
                eval_long_box(ctx, &state, now_ts, current_dte, params)
            {
                let cost_rmb = cost * CONTRACT_SIZE;
                if available_cash - cost_rmb >= MIN_CASH_THRESHOLD {
                    available_cash -= cost_rmb;
                    positions.push(SimplePos {
                        strike_i: si,
                        strike_j: sj,
                        strike_k: 0,
                        pos_type: 0,
                        entry_cost: cost,
                        payout,
                        margin_rmb: 0.0,
                        entry_ts: now_ts,
                    });
                }
            }
        }

        if current_dte > 5 && !positions.iter().any(|p| p.pos_type == 1) {
            if let Some((si, sj, gain, payout_val, margin_rmb)) =
                eval_short_box(ctx, &state, now_ts, current_dte, params)
            {
                if available_cash - margin_rmb >= MIN_CASH_THRESHOLD {
                    available_cash += gain * CONTRACT_SIZE - margin_rmb;
                    locked_margin += margin_rmb;
                    positions.push(SimplePos {
                        strike_i: si,
                        strike_j: sj,
                        strike_k: 0,
                        pos_type: 1,
                        entry_cost: gain,
                        payout: payout_val,
                        margin_rmb,
                        entry_ts: now_ts,
                    });
                }
            }
        }

        if current_dte > 5 && !positions.iter().any(|p| p.pos_type == 2) {
            if let Some((si, sj, sk, profit, margin_rmb)) =
                eval_butterfly(ctx, &state, now_ts, current_dte, params)
            {
                if available_cash - margin_rmb >= MIN_CASH_THRESHOLD {
                    available_cash += profit * CONTRACT_SIZE - margin_rmb;
                    locked_margin += margin_rmb;
                    positions.push(SimplePos {
                        strike_i: si,
                        strike_j: sj,
                        strike_k: sk,
                        pos_type: 2,
                        entry_cost: profit,
                        payout: 0.0,
                        margin_rmb,
                        entry_ts: now_ts,
                    });
                }
            }
        }
    }

    let mut final_val = available_cash + locked_margin;
    for pos in positions {
        if pos.pos_type == 0 {
            final_val += pos.payout * CONTRACT_SIZE;
        } else {
            final_val -= pos.payout * CONTRACT_SIZE;
        }
    }
    final_val
}

// ─── Particle Swarm Optimisation ─────────────────────────────────────────────

#[derive(Clone)]
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    best_position: [f32; 3],
    best_score: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let symbol = if args.len() > 1 {
        args[1].clone()
    } else {
        "510500XSHG".to_string()
    };

    let data = load_data(&symbol);

    // Reduced PSO for a fast first run; bump up once performance is confirmed.
    let num_particles = 20;
    let num_iterations = 25;

    // Bounds: [min_return, min_return_butterfly, risk_free_rate]
    let bounds_min = [0.005_f32, 0.02, 0.01];
    let bounds_max = [0.15_f32, 0.20, 0.10];

    let mut rng = rand::thread_rng();
    let mut particles: Vec<Particle> = (0..num_particles)
        .map(|_| {
            let pos = [
                rng.gen_range(bounds_min[0]..bounds_max[0]),
                rng.gen_range(bounds_min[1]..bounds_max[1]),
                rng.gen_range(bounds_min[2]..bounds_max[2]),
            ];
            Particle {
                position: pos,
                velocity: [0.0; 3],
                best_position: pos,
                best_score: -1.0,
            }
        })
        .collect();

    let mut global_best_position = particles[0].position;
    let mut global_best_score = f32::NEG_INFINITY;

    let (w, c1, c2) = (0.5_f32, 1.5_f32, 1.5_f32);

    for iter in 0..num_iterations {
        println!("--- Iteration {}/{} ---", iter + 1, num_iterations);

        let scores: Vec<f32> = particles
            .par_iter()
            .map(|p| {
                let params = StrategyParams {
                    min_return: p.position[0],
                    min_return_butterfly: 0.1, // Hardcoded
                    min_exit_return: p.position[2],
                };
                simulate(&params, &data)
            })
            .collect();

        let mut local_rng = rand::thread_rng();
        for (i, p) in particles.iter_mut().enumerate() {
            let score = scores[i];
            if score > p.best_score {
                p.best_score = score;
                p.best_position = p.position;
            }
            if score > global_best_score {
                global_best_score = score;
                global_best_position = p.position;
                println!(
                    "  ✓ New Global Best: {:.2} RMB  (mar={:.3}, mar_fly={:.3}, rfr={:.4})",
                    global_best_score, global_best_position[0], 0.1, global_best_position[2]
                );
            }

            // Update velocity and position
            for j in 0..3 {
                let r1: f32 = local_rng.gen();
                let r2: f32 = local_rng.gen();
                p.velocity[j] = w * p.velocity[j]
                    + c1 * r1 * (p.best_position[j] - p.position[j])
                    + c2 * r2 * (global_best_position[j] - p.position[j]);
                p.position[j] = (p.position[j] + p.velocity[j]).clamp(bounds_min[j], bounds_max[j]);
            }
        }
    }

    println!("\n=== Optimisation Complete ===");
    println!(
        "Max Final Capital : {:.2} RMB  (start: {:.0} RMB)",
        global_best_score, INITIAL_CASH
    );
    println!("Best Parameters:");
    println!(
        "  min_return                  = {:.4}",
        global_best_position[0]
    );
    println!("  min_return_butterfly        = {:.4} (Hardcoded)", 0.1);
    println!(
        "  min_exit_return             = {:.4}",
        global_best_position[2]
    );

    // ── Persist best params to JSON ──────────────────────────────────────────
    let json = serde_json::json!({
        "min_return": global_best_position[0],
        "min_return_butterfly": 0.1,
        "min_exit_return": global_best_position[2],
        "final_capital_rmb": global_best_score,
    });
    match std::fs::File::create("best_params.json") {
        Ok(mut f) => {
            let _ = writeln!(f, "{}", serde_json::to_string_pretty(&json).unwrap());
            println!("\nBest params saved to best_params.json");
        }
        Err(e) => eprintln!("Warning: could not save best_params.json: {}", e),
    }
}
