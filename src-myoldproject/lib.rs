use chrono::{Datelike, Duration, NaiveDate};
use std::collections::HashMap;

// --- Constants ---
pub const COMMISSION_PER_LEG: f32 = 0.0002; // Assume 0.0001 Slippage, added to 1 RMB commission per trade
pub const BOX_COMMISSION: f32 = 4.0 * COMMISSION_PER_LEG;
pub const BUTTERFLY_COMMISSION: f32 = 4.0 * COMMISSION_PER_LEG;
pub const MAX_STALE_SECONDS: i64 = 2; // Entry guard: reject stale quotes for NEW entries
pub const MAX_STALE_SECONDS_EXIT: i64 = 60; // Exit guard: last known price valid for up to 60s
pub const MIN_VOL: f32 = 3.0;
pub const MIN_HOLDING_TIME: i64 = 10; // Minimum holding time in seconds before allowing early exit
pub const EARLY_EXIT_ADJUSTMENT: f32 = 0.04; // Adjustment to return threshold near expiry to encourage exit

/// Returns true if the given timestamp (seconds) falls within continuous
/// trading hours: 09:30–11:30 or 13:00–15:00.
/// Ticks during the call auction (09:15–09:25) are excluded.
///
/// NOTE: The parquet file stores timestamps as **timezone-naive local CST**
/// (confirmed: raw values read directly as 09:15, 09:30, etc.).
/// No UTC→CST conversion is applied here.
pub fn is_trading_hours(ts_sec: i64) -> bool {
    let seconds_since_midnight = ts_sec.rem_euclid(86400) as u32;

    // Helper to convert HHMM format to seconds since midnight
    let hm = |time: u32| -> u32 { (time / 100) * 3600 + (time % 100) * 60 };

    let is_in_range = |start: u32, end: u32| -> bool {
        seconds_since_midnight >= hm(start) && seconds_since_midnight < hm(end)
    };

    // To prevent trading in certain times, simply split or remove intervals.
    // e.g. To pause trading from 09:40 to 09:50, you could do:
    // is_in_range(930, 940) || is_in_range(950, 1130)

    is_in_range(930, 1129) || is_in_range(1300, 1459)
}

#[derive(Debug, Clone)]
pub struct StrategyParams {
    pub min_return: f32,
    pub min_return_butterfly: f32,
    pub min_exit_return: f32,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            min_return: 0.2,
            min_return_butterfly: 0.9,
            min_exit_return: 0.2,
        }
    }
}

// --- Position & Cash Management ---
pub const CONTRACT_SIZE: f32 = 10_000.0; // units per ETF option contract
pub const INITIAL_CASH: f32 = 100_000.0; // RMB available at start
pub const MIN_CASH_THRESHOLD: f32 = 10_000.0; // stop entering new positions below this

// Ricequant License Key (Stored in .license file for security)
// Please create a .license file in the project root and paste your Ricequant license key there.
// DO NOT commit your .license file to version control.

// --- Data Structures ---

#[derive(Debug, Clone)]
pub struct OptionMeta {
    pub oid: String,
    pub underlying: String,
    pub maturity: String,
    pub opt_type: String,
    pub strike: f32,
}

#[derive(Debug, Clone, Default)]
pub struct TickData {
    pub a1: f32,
    pub b1: f32,
    pub a1_v: f32,
    pub b1_v: f32,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct TickUpdate {
    pub oid: String,
    pub a1: f32,
    pub b1: f32,
    pub a1_v: f32,
    pub b1_v: f32,
}

#[derive(Debug, Clone)]
pub struct BoxContext {
    pub prefix: String,
    pub dte: u8,
    pub strikes: Vec<f32>,
    pub call_oids: Vec<String>,
    pub put_oids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PosType {
    LongBox,
    ShortBox,
    ButterflyCall,
    ButterflyPut,
}

impl std::fmt::Display for PosType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PosType::LongBox => write!(f, "LongBox"),
            PosType::ShortBox => write!(f, "ShortBox"),
            PosType::ButterflyCall => write!(f, "ButterflyCall"),
            PosType::ButterflyPut => write!(f, "ButterflyPut"),
        }
    }
}

/// An open virtual position recorded when a signal fires.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    pub pos_type: PosType,
    pub ctx_idx: usize,
    pub strike_i: usize, // lower-strike index
    pub strike_j: usize, // middle/upper-strike index
    pub strike_k: usize, // upper-strike index (for butterfly)
    pub entry_cost: f32, // price/unit (long: cost paid; short: premium received)
    pub payout: f32,     // K_hi - K_lo (price/unit)
    pub margin_rmb: f32, // RMB locked as margin (short box only)
    pub dte_at_entry: u8,
    pub _entry_ts: i64,
    pub entry_time_str: String,
}

/// Sent to the background thread for CSV persistence.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub pos_type: String,
    pub prefix: String,
    pub strike_lo: f32,
    pub strike_hi: f32,
    pub dte_at_entry: u8,
    pub entry_time: String,
    pub exit_time: String,
    pub entry_cost: f32,
    pub payout: f32,
    pub exit_value: f32,
    pub pnl_rmb: f32,
    pub exit_type: String, // "early" | "maturity"
    pub holding_time_sec: i64,
}

// --- Logic ---

pub fn get_4th_wednesday(year: i32, month: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let weekday = first.weekday().num_days_from_monday(); // Mon=0 .. Wed=2
    let offset = (2 + 7 - (weekday as i32)) % 7;
    first + Duration::days((offset + 21) as i64)
}

pub async fn fetch_token(license_key: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://rqdata.ricequant.com/auth")
        .json(&serde_json::json!({
            "user_name": "license",
            "password": license_key
        }))
        .send()
        .await?;

    let token = resp.text().await?;
    Ok(token)
}

pub async fn fetch_options(token: &str) -> Result<Vec<OptionMeta>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://rqdata.ricequant.com/api")
        .header("token", token)
        .json(&serde_json::json!({
            "method": "all_instruments",
            "type": "Option"
        }))
        .send()
        .await?;

    let csv_data = resp.text().await?;
    let mut reader = csv::Reader::from_reader(csv_data.as_bytes());

    let mut options = Vec::new();
    let target_underlyings = ["510300.XSHG", "510500.XSHG", "588000.XSHG"];

    for result in reader.deserialize() {
        let record: HashMap<String, String> = result?;
        let underlying = record.get("underlying_symbol").cloned().unwrap_or_default();

        if target_underlyings.contains(&underlying.as_str()) {
            options.push(OptionMeta {
                oid: record.get("order_book_id").cloned().unwrap_or_default(),
                underlying,
                maturity: record.get("maturity_date").cloned().unwrap_or_default(),
                opt_type: record.get("option_type").cloned().unwrap_or_default(),
                strike: record
                    .get("strike_price")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
            });
        }
    }

    Ok(options)
}

pub fn evaluate_long_box(
    ctx: &BoxContext,
    state: &HashMap<String, TickData>,
    now_ts: i64,
    params: &StrategyParams,
) -> Option<(usize, usize, f32, f32, f32, f32, String)> {
    let n = ctx.strikes.len();
    let mut best = None;
    let mut best_ret = params.min_return;

    let mut valid: Vec<(usize, &TickData, &TickData)> = Vec::with_capacity(n);
    for i in 0..n {
        if let (Some(c), Some(p)) = (state.get(&ctx.call_oids[i]), state.get(&ctx.put_oids[i])) {
            if now_ts - c.timestamp <= MAX_STALE_SECONDS
                && now_ts - p.timestamp <= MAX_STALE_SECONDS
                && c.a1 > 0.0
                && c.b1 > 0.0
                && p.a1 > 0.0
                && p.b1 > 0.0
                && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
                && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
            {
                valid.push((i, c, p));
            }
        }
    }

    for (vi, &(i, c_k1, p_k1)) in valid.iter().enumerate() {
        for &(j, c_k2, p_k2) in valid.iter().skip(vi + 1) {
            let payout = ctx.strikes[j] - ctx.strikes[i];
            let cost = (c_k1.a1 - c_k2.b1) + (p_k2.a1 - p_k1.b1) + BOX_COMMISSION;
            if cost > 0.0 {
                let ret = (payout - cost) / cost;
                if ret > best_ret {
                    best_ret = ret;
                    let legs = [
                        (&ctx.call_oids[i], c_k1.a1_v),
                        (&ctx.call_oids[j], c_k2.b1_v),
                        (&ctx.put_oids[j], p_k2.b1_v),
                        (&ctx.put_oids[i], p_k1.a1_v),
                    ];
                    let mut min_idx = 0usize;
                    for k in 1..4 {
                        if legs[k].1 < legs[min_idx].1 {
                            min_idx = k;
                        }
                    }
                    let (least_liquid_oid, min_leg_vol) = legs[min_idx];
                    let legging_msg = if min_leg_vol >= MIN_VOL {
                        "All 4 legs are limit orders".to_string()
                    } else {
                        format!("Place passive limit order on leg {} (most illiquid), limit orders on others.", least_liquid_oid)
                    };
                    best = Some((i, j, cost, payout, ret, ret, legging_msg)); // Passing ret twice to avoid signature change
                }
            }
        }
    }
    best
}

pub fn evaluate_short_box(
    ctx: &BoxContext,
    state: &HashMap<String, TickData>,
    now_ts: i64,
    params: &StrategyParams,
) -> Option<(usize, usize, f32, f32, f32, f32, String)> {
    let n = ctx.strikes.len();
    let mut best = None;
    let mut best_ret = params.min_return;

    let mut valid: Vec<(usize, &TickData, &TickData)> = Vec::with_capacity(n);
    for i in 0..n {
        if let (Some(c), Some(p)) = (state.get(&ctx.call_oids[i]), state.get(&ctx.put_oids[i])) {
            if now_ts - c.timestamp <= MAX_STALE_SECONDS
                && now_ts - p.timestamp <= MAX_STALE_SECONDS
                && c.a1 > 0.0
                && c.b1 > 0.0
                && p.a1 > 0.0
                && p.b1 > 0.0
                && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
                && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
            {
                valid.push((i, c, p));
            }
        }
    }

    for (vi, &(i, c_k1, p_k1)) in valid.iter().enumerate() {
        for &(j, c_k2, p_k2) in valid.iter().skip(vi + 1) {
            let width = ctx.strikes[j] - ctx.strikes[i];
            let margin = width * 2.0;
            let gain = (c_k1.b1 - c_k2.a1) + (p_k2.b1 - p_k1.a1) - BOX_COMMISSION;
            let profit = gain - width;

            if profit > 0.0 {
                let ret = profit / margin;
                if ret > best_ret {
                    best_ret = ret;
                    let legs = [
                        (&ctx.call_oids[i], c_k1.b1_v),
                        (&ctx.call_oids[j], c_k2.a1_v),
                        (&ctx.put_oids[j], p_k2.b1_v),
                        (&ctx.put_oids[i], p_k1.a1_v),
                    ];
                    let mut min_idx = 0usize;
                    for k in 1..4 {
                        if legs[k].1 < legs[min_idx].1 {
                            min_idx = k;
                        }
                    }
                    let (least_liquid_oid, min_leg_vol) = legs[min_idx];
                    let legging_msg = if min_leg_vol >= MIN_VOL {
                        "All 4 legs are limit orders".to_string()
                    } else {
                        format!("Place passive limit order on leg {} (most illiquid), limit orders on others.", least_liquid_oid)
                    };
                    best = Some((i, j, margin, gain, ret, ret, legging_msg)); // Pass ret twice
                }
            }
        }
    }
    best
}

pub fn evaluate_butterfly(
    ctx: &BoxContext,
    state: &HashMap<String, TickData>,
    now_ts: i64,
    params: &StrategyParams,
) -> Option<(
    usize,
    usize,
    usize,
    &'static str,
    f32,
    f32,
    f32,
    f32,
    String,
)> {
    let n = ctx.strikes.len();
    if n < 3 {
        return None;
    }
    let mut best = None;
    let mut best_ret = params.min_return_butterfly;

    let mut valid: Vec<(usize, &TickData, &TickData)> = Vec::with_capacity(n);
    for i in 0..n {
        if let (Some(c), Some(p)) = (state.get(&ctx.call_oids[i]), state.get(&ctx.put_oids[i])) {
            if now_ts - c.timestamp <= MAX_STALE_SECONDS
                && now_ts - p.timestamp <= MAX_STALE_SECONDS
                && c.a1 > 0.0
                && c.b1 > 0.0
                && p.a1 > 0.0
                && p.b1 > 0.0
                && (c.a1_v >= MIN_VOL || c.b1_v >= MIN_VOL)
                && (p.a1_v >= MIN_VOL || p.b1_v >= MIN_VOL)
            {
                valid.push((i, c, p));
            }
        }
    }

    let m_len = valid.len();
    if m_len < 3 {
        return None;
    }

    for idx_1 in 0..(m_len - 2) {
        let &(i, c_k1, p_k1) = &valid[idx_1];
        for idx_2 in (idx_1 + 1)..(m_len - 1) {
            let &(j, c_k2, p_k2) = &valid[idx_2];
            let width_1 = ctx.strikes[j] - ctx.strikes[i];

            for idx_3 in (idx_2 + 1)..m_len {
                let &(k, c_k3, p_k3) = &valid[idx_3];
                let width_2 = ctx.strikes[k] - ctx.strikes[j];

                if (width_1 - width_2).abs() > 1e-4 {
                    continue;
                }

                let margin = width_1;

                // Call Butterfly
                let call_cost = c_k1.a1 - 2.0 * c_k2.b1 + c_k3.a1 + BUTTERFLY_COMMISSION;
                if call_cost < 0.0 {
                    let profit = -call_cost;
                    let ret = profit / margin;
                    if ret > best_ret {
                        best_ret = ret;
                        let legs = [
                            (&ctx.call_oids[i], c_k1.a1_v),
                            (&ctx.call_oids[j], c_k2.b1_v),
                            (&ctx.call_oids[k], c_k3.a1_v),
                        ];
                        let mut min_idx = 0usize;
                        for z in 1..3 {
                            if legs[z].1 < legs[min_idx].1 {
                                min_idx = z;
                            }
                        }
                        let legging_msg = if legs[min_idx].1 >= MIN_VOL {
                            "All Call legs are limit orders".to_string()
                        } else {
                            format!(
                                "Place passive limit order on leg {} (most illiquid).",
                                legs[min_idx].0
                            )
                        };
                        best = Some((i, j, k, "Call", margin, profit, ret, ret, legging_msg));
                        // Pass ret twice
                    }
                }

                // Put Butterfly
                let put_cost = p_k1.a1 - 2.0 * p_k2.b1 + p_k3.a1 + BUTTERFLY_COMMISSION;
                if put_cost < 0.0 {
                    let profit = -put_cost;
                    let ret = profit / margin;
                    if ret > best_ret {
                        best_ret = ret;
                        let legs = [
                            (&ctx.put_oids[i], p_k1.a1_v),
                            (&ctx.put_oids[j], p_k2.b1_v),
                            (&ctx.put_oids[k], p_k3.a1_v),
                        ];
                        let mut min_idx = 0usize;
                        for z in 1..3 {
                            if legs[z].1 < legs[min_idx].1 {
                                min_idx = z;
                            }
                        }
                        let legging_msg = if legs[min_idx].1 >= MIN_VOL {
                            "All Put legs are limit orders".to_string()
                        } else {
                            format!(
                                "Place passive limit order on leg {} (most illiquid).",
                                legs[min_idx].0
                            )
                        };
                        best = Some((i, j, k, "Put", margin, profit, ret, ret, legging_msg));
                        // Pass ret twice
                    }
                }
            }
        }
    }
    best
}

#[derive(Debug, Clone)]
pub enum PortfolioEvent {
    Exit {
        pos_type: PosType,
        ctx_idx: usize,
        strike_i: usize,
        strike_j: usize,
        close_net_or_buy_back: f32,
        payout: f32,
        pnl_rmb: f32,
        trade_record: TradeRecord,
    },
    EntryLong {
        ctx_idx: usize,
        strike_i: usize,
        strike_j: usize,
        cost: f32,
        payout: f32,
        ann: f32,
        msg: String,
        is_new: bool,
    },
    EntryShort {
        ctx_idx: usize,
        strike_i: usize,
        strike_j: usize,
        gain: f32,
        payout_val: f32,
        margin_rmb: f32,
        ann: f32,
        msg: String,
        is_new: bool,
    },
    EntryButterfly {
        ctx_idx: usize,
        flavor: &'static str,
        strike_i: usize,
        strike_j: usize,
        strike_k: usize,
        profit: f32,
        margin_rmb: f32,
        ann: f32,
        msg: String,
        is_new: bool,
    },
}

pub struct Portfolio {
    pub available_cash: f32,
    pub locked_margin: f32,
    pub positions: Vec<OpenPosition>,
    pub eval_count: u64,
    pub last_lb: HashMap<usize, (usize, usize)>,
    pub last_sb: HashMap<usize, (usize, usize)>,
    pub last_fly: HashMap<usize, (usize, usize, usize, &'static str)>,
}

impl Portfolio {
    pub fn new(initial_cash: f32) -> Self {
        Self {
            available_cash: initial_cash,
            locked_margin: 0.0,
            positions: Vec::with_capacity(32),
            eval_count: 0,
            last_lb: HashMap::new(),
            last_sb: HashMap::new(),
            last_fly: HashMap::new(),
        }
    }

    pub fn update(
        &mut self,
        tick: &TickUpdate,
        state: &mut HashMap<String, TickData>,
        oid_to_ctx_idx: &HashMap<String, Vec<usize>>,
        eval_contexts: &[BoxContext],
        params: &StrategyParams,
        now_ts: i64,
        time_str: &str,
    ) -> Vec<PortfolioEvent> {
        let mut events = Vec::new();
        state.insert(
            tick.oid.clone(),
            TickData {
                a1: tick.a1,
                b1: tick.b1,
                a1_v: tick.a1_v,
                b1_v: tick.b1_v,
                timestamp: now_ts,
            },
        );

        if let Some(ctx_indices) = oid_to_ctx_idx.get(&tick.oid) {
            for &idx in ctx_indices {
                let ctx = &eval_contexts[idx];

                // ── PHASE 1: Check exits ──
                let mut closed: Vec<usize> = Vec::new();
                for (pi, pos) in self.positions.iter().enumerate() {
                    if pos.ctx_idx != idx {
                        continue;
                    }
                    // Butterfly positions are always held to maturity — skip exit check.
                    if matches!(pos.pos_type, PosType::ButterflyCall | PosType::ButterflyPut) {
                        continue;
                    }

                    if now_ts - pos._entry_ts < MIN_HOLDING_TIME {
                        continue;
                    }

                    let si = pos.strike_i;
                    let sj = pos.strike_j;

                    let c_lo = state.get(&ctx.call_oids[si]);
                    let c_hi = state.get(&ctx.call_oids[sj]);
                    let p_hi = state.get(&ctx.put_oids[sj]);
                    let p_lo = state.get(&ctx.put_oids[si]);
                    // For exits, use a much more relaxed staleness window: the last
                    // known bid/ask of an illiquid leg is still a valid reference for
                    // a long time.
                    let fresh = [c_lo, c_hi, p_hi, p_lo].iter().all(|opt| {
                        opt.map_or(false, |t| now_ts - t.timestamp <= 3600) // 1 hour for exits
                    });
                    if !fresh {
                        continue;
                    }

                    let (c_lo, c_hi, p_hi, p_lo) =
                        (c_lo.unwrap(), c_hi.unwrap(), p_hi.unwrap(), p_lo.unwrap());

                    // Maturity exit: if DTE is 0, we must exit at payout.
                    if ctx.dte == 0 {
                        let payout_val = pos.payout;
                        let pnl_rmb = match pos.pos_type {
                            PosType::LongBox => (payout_val - pos.entry_cost) * CONTRACT_SIZE,
                            PosType::ShortBox => (pos.entry_cost - payout_val) * CONTRACT_SIZE,
                            _ => 0.0,
                        };
                        if pos.pos_type == PosType::ShortBox {
                            self.available_cash += pos.margin_rmb;
                            self.locked_margin -= pos.margin_rmb;
                            self.available_cash -= payout_val * CONTRACT_SIZE;
                        } else if pos.pos_type == PosType::LongBox {
                            self.available_cash += payout_val * CONTRACT_SIZE;
                        }

                        events.push(PortfolioEvent::Exit {
                            pos_type: pos.pos_type,
                            ctx_idx: idx,
                            strike_i: si,
                            strike_j: sj,
                            close_net_or_buy_back: payout_val,
                            payout: pos.payout,
                            pnl_rmb,
                            trade_record: TradeRecord {
                                pos_type: format!("{}", pos.pos_type),
                                prefix: ctx.prefix.clone(),
                                strike_lo: ctx.strikes[si],
                                strike_hi: ctx.strikes[sj],
                                dte_at_entry: pos.dte_at_entry,
                                entry_time: pos.entry_time_str.clone(),
                                exit_time: time_str.to_string(),
                                entry_cost: pos.entry_cost,
                                payout: pos.payout,
                                exit_value: payout_val,
                                pnl_rmb,
                                exit_type: "maturity".into(),
                                holding_time_sec: now_ts - pos._entry_ts,
                            },
                        });
                        closed.push(pi);
                        continue;
                    }

                    match pos.pos_type {
                        PosType::LongBox => {
                            if c_lo.b1 > 0.0 && c_hi.a1 > 0.0 && p_hi.b1 > 0.0 && p_lo.a1 > 0.0 {
                                let close_value = (c_lo.b1 - c_hi.a1) + (p_hi.b1 - p_lo.a1);
                                let close_net = close_value - BOX_COMMISSION;
                                let mut exit_threshold = params.min_exit_return;
                                if ctx.dte <= 4 {
                                    exit_threshold += EARLY_EXIT_ADJUSTMENT; // Encourage early exit
                                }

                                if close_net * (1.0 + exit_threshold) > pos.payout {
                                    let immediate_profit = close_net - pos.entry_cost;
                                    let pnl_rmb = immediate_profit * CONTRACT_SIZE;
                                    self.available_cash += close_net * CONTRACT_SIZE;
                                    events.push(PortfolioEvent::Exit {
                                        pos_type: PosType::LongBox,
                                        ctx_idx: idx,
                                        strike_i: si,
                                        strike_j: sj,
                                        close_net_or_buy_back: close_net,
                                        payout: pos.payout,
                                        pnl_rmb,
                                        trade_record: TradeRecord {
                                            pos_type: "LongBox".into(),
                                            prefix: ctx.prefix.clone(),
                                            strike_lo: ctx.strikes[si],
                                            strike_hi: ctx.strikes[sj],
                                            dte_at_entry: pos.dte_at_entry,
                                            entry_time: pos.entry_time_str.clone(),
                                            exit_time: time_str.to_string(),
                                            entry_cost: pos.entry_cost,
                                            payout: pos.payout,
                                            exit_value: close_net,
                                            pnl_rmb,
                                            exit_type: "early".into(),
                                            holding_time_sec: now_ts - pos._entry_ts,
                                        },
                                    });
                                    closed.push(pi);
                                }
                            }
                        }
                        PosType::ShortBox => {
                            if c_lo.a1 > 0.0 && c_hi.b1 > 0.0 && p_hi.a1 > 0.0 && p_lo.b1 > 0.0 {
                                let buy_back =
                                    (c_lo.a1 - c_hi.b1) + (p_hi.a1 - p_lo.b1) + BOX_COMMISSION;
                                let mut exit_threshold = params.min_exit_return;
                                if ctx.dte <= 4 {
                                    exit_threshold =
                                        (exit_threshold - EARLY_EXIT_ADJUSTMENT).max(0.0);
                                    // Encourage early exit
                                }

                                if buy_back > 0.0 && buy_back * (1.0 + exit_threshold) < pos.payout
                                {
                                    let immediate_profit = pos.entry_cost - buy_back;
                                    let pnl_rmb = immediate_profit * CONTRACT_SIZE;
                                    self.available_cash += pos.margin_rmb;
                                    self.locked_margin -= pos.margin_rmb;
                                    self.available_cash -= buy_back * CONTRACT_SIZE;
                                    events.push(PortfolioEvent::Exit {
                                        pos_type: PosType::ShortBox,
                                        ctx_idx: idx,
                                        strike_i: si,
                                        strike_j: sj,
                                        close_net_or_buy_back: buy_back,
                                        payout: pos.payout,
                                        pnl_rmb,
                                        trade_record: TradeRecord {
                                            pos_type: "ShortBox".into(),
                                            prefix: ctx.prefix.clone(),
                                            strike_lo: ctx.strikes[si],
                                            strike_hi: ctx.strikes[sj],
                                            dte_at_entry: pos.dte_at_entry,
                                            entry_time: pos.entry_time_str.clone(),
                                            exit_time: time_str.to_string(),
                                            entry_cost: pos.entry_cost,
                                            payout: pos.payout,
                                            exit_value: buy_back,
                                            pnl_rmb,
                                            exit_type: "early".into(),
                                            holding_time_sec: now_ts - pos._entry_ts,
                                        },
                                    });
                                    closed.push(pi);
                                }
                            }
                        }
                        // Butterfly arms are unreachable here because we skip them
                        // at the top of the loop with an early `continue`.
                        PosType::ButterflyCall | PosType::ButterflyPut => unreachable!(),
                    }
                }
                for pi in closed.into_iter().rev() {
                    self.positions.swap_remove(pi);
                }

                // ── PHASE 2: Scan for entries ──
                if ctx.dte <= 5 {
                    self.last_lb.remove(&idx);
                    self.last_sb.remove(&idx);
                    self.last_fly.remove(&idx);
                    continue;
                }

                if let Some((i, j, cost, payout, _ret, ann, msg)) =
                    evaluate_long_box(ctx, state, now_ts, params)
                {
                    let already_in = self.positions.iter().any(|p| {
                        p.ctx_idx == idx
                            && p.pos_type == PosType::LongBox
                            && p.strike_i == i
                            && p.strike_j == j
                    });
                    let cost_rmb = cost * CONTRACT_SIZE;
                    let mut is_new = false;
                    if !already_in && self.available_cash - cost_rmb >= MIN_CASH_THRESHOLD {
                        self.available_cash -= cost_rmb;
                        self.positions.push(OpenPosition {
                            pos_type: PosType::LongBox,
                            ctx_idx: idx,
                            strike_i: i,
                            strike_j: j,
                            strike_k: 0,
                            entry_cost: cost,
                            payout,
                            margin_rmb: 0.0,
                            dte_at_entry: ctx.dte,
                            _entry_ts: now_ts,
                            entry_time_str: time_str.to_string(),
                        });
                        is_new = true;
                    }
                    if self.last_lb.get(&idx) != Some(&(i, j)) || is_new {
                        self.last_lb.insert(idx, (i, j));
                        events.push(PortfolioEvent::EntryLong {
                            ctx_idx: idx,
                            strike_i: i,
                            strike_j: j,
                            cost,
                            payout,
                            ann,
                            msg,
                            is_new,
                        });
                    }
                } else {
                    self.last_lb.remove(&idx);
                }

                if let Some((i, j, _margin_price, gain, _ret, ann, msg)) =
                    evaluate_short_box(ctx, state, now_ts, params)
                {
                    let payout_val = ctx.strikes[j] - ctx.strikes[i];
                    let margin_rmb = 2.0 * payout_val * CONTRACT_SIZE;
                    let already_in = self.positions.iter().any(|p| {
                        p.ctx_idx == idx
                            && p.pos_type == PosType::ShortBox
                            && p.strike_i == i
                            && p.strike_j == j
                    });
                    let mut is_new = false;
                    if !already_in && self.available_cash - margin_rmb >= MIN_CASH_THRESHOLD {
                        self.available_cash += gain * CONTRACT_SIZE - margin_rmb;
                        self.locked_margin += margin_rmb;
                        self.positions.push(OpenPosition {
                            pos_type: PosType::ShortBox,
                            ctx_idx: idx,
                            strike_i: i,
                            strike_j: j,
                            strike_k: 0,
                            entry_cost: gain,
                            payout: payout_val,
                            margin_rmb,
                            dte_at_entry: ctx.dte,
                            _entry_ts: now_ts,
                            entry_time_str: time_str.to_string(),
                        });
                        is_new = true;
                    }
                    if self.last_sb.get(&idx) != Some(&(i, j)) || is_new {
                        self.last_sb.insert(idx, (i, j));
                        events.push(PortfolioEvent::EntryShort {
                            ctx_idx: idx,
                            strike_i: i,
                            strike_j: j,
                            gain,
                            payout_val,
                            margin_rmb,
                            ann,
                            msg,
                            is_new,
                        });
                    }
                } else {
                    self.last_sb.remove(&idx);
                }

                if let Some((i, j, k, flavor, margin, profit, ret, ann, msg)) =
                    evaluate_butterfly(ctx, state, now_ts, params)
                {
                    let flavor_type = if flavor == "Call" {
                        PosType::ButterflyCall
                    } else {
                        PosType::ButterflyPut
                    };
                    let already_in = self.positions.iter().any(|p| {
                        p.ctx_idx == idx
                            && p.pos_type == flavor_type
                            && p.strike_i == i
                            && p.strike_j == j
                            && p.strike_k == k
                    });

                    let margin_rmb = margin * CONTRACT_SIZE;
                    let mut is_new = false;

                    if !already_in && self.available_cash - margin_rmb >= MIN_CASH_THRESHOLD {
                        self.available_cash += profit * CONTRACT_SIZE - margin_rmb;
                        self.locked_margin += margin_rmb;
                        self.positions.push(OpenPosition {
                            pos_type: flavor_type,
                            ctx_idx: idx,
                            strike_i: i,
                            strike_j: j,
                            strike_k: k,
                            entry_cost: profit,
                            payout: 0.0, // Minimum payout is 0
                            margin_rmb,
                            dte_at_entry: ctx.dte,
                            _entry_ts: now_ts,
                            entry_time_str: time_str.to_string(),
                        });
                        is_new = true;
                    }

                    if self.last_fly.get(&idx) != Some(&(i, j, k, flavor)) || is_new {
                        self.last_fly.insert(idx, (i, j, k, flavor));
                        events.push(PortfolioEvent::EntryButterfly {
                            ctx_idx: idx,
                            flavor,
                            strike_i: i,
                            strike_j: j,
                            strike_k: k,
                            profit,
                            margin_rmb,
                            ann: ret,
                            msg,
                            is_new,
                        });
                    }
                } else {
                    self.last_fly.remove(&idx);
                }
            }
        }
        self.eval_count += 1;
        events
    }
}
