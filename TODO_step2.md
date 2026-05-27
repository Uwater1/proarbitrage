# proarbitrage Phase 2: Option-Only Structured Arbitrage & ML Hybrid Roadmap

This document outlines the step-by-step technical implementation path to pivot the relative-value trading engine from raw single-option execution into a **hybrid options-only structured arbitrage** model. By blending **traditional arbitrage (Box, Butterfly, Iron Condor, Vertical Spreads)** with the **XGBoost expected alpha model**, we bypass ETF short-selling constraints, maximize margin efficiency, and eliminate HFT over-trading.

---

## 1. Mathematical Architecture: Blending Structures with XGBoost

Instead of allocating capital to single-option contracts $\boldsymbol{\Theta} \in \mathbb{R}^N$ (which creates spot/delta hedging friction and high short-margin requirements), the trading engine will select allocations over a set of $M$ pre-defined multi-leg **Arbitrage Structures** $\mathbf{z} \in \mathbb{R}^M$.

Let $S_p$ for $p \in \{1, \dots, M\}$ denote a structured basket (e.g. Box, Iron Condor, Butterfly, Vertical Spread). Each structure is defined by a sparse mapping vector $\mathbf{a}_p \in \mathbb{R}^N$, where:
* $a_{p,i} = +1$ (long 1 contract of option $i$),
* $a_{p,i} = -1$ (short 1 contract of option $i$),
* $a_{p,i} = 0$ (no position).

The unified option allocation vector is:
$$ \boldsymbol{\Theta} = \mathbf{A} \mathbf{z} $$
where $\mathbf{A} \in \mathbb{R}^{N \times M}$ is the structural matrix whose columns are $\{\mathbf{a}_p\}$.

### The Hybrid Objective Function
At each tick group update, we evaluate individual expected returns $W_i$ via the XGBoost model. The net expected return of structure $p$ is:
$$ \alpha^{\text{struct}}_p = \mathbf{a}_p^\top \mathbf{W} $$

The optimizer solves for optimal structural allocations $\mathbf{z} = \mathbf{z}^+ - \mathbf{z}^-$:
$$ \max_{\mathbf{z}^+, \mathbf{z}^-} \quad \sum_{p=1}^M \left( z^+_p - z^-_p \right) \alpha^{\text{struct}}_p - \sum_{p=1}^M \mathrm{TC}_p \cdot \left( z^+_p + z^-_p \right) $$
subject to:
* **Delta Neutrality**: $-\epsilon_{\Delta} \le (\mathbf{A}\mathbf{z})^\top \boldsymbol{\Delta} \le \epsilon_{\Delta}$
* **Volatility Bounding**: $-\epsilon_{\nu} \le (\mathbf{A}\mathbf{z})^\top \boldsymbol{\nu} \le \epsilon_{\nu}$
* **Margin Capacity**: $(\mathbf{A}\mathbf{z}^+ + \mathbf{A}\mathbf{z}^-)^\top \mathbf{C}_{\text{margin}} \le C_{\max}$
* **Anti-Flicker Rebalancing Band**: Only submit order changes if $|z_p - z_{\text{current},p}| \ge \delta_p$ (where $\delta_p \ge 5$ contracts).

Notice that because Arbitrage Structures are risk low, the restriction above do not need to be strictly followed, we can relax them to allow for more trading opportunities.

---

## 2. Core Arbitrage Structures

We prioritize options-only structures to eliminate the need to borrow and short-sell A-share ETF underlyings.

### A. Box Arbitrage (Options-Only Synthetic Spot)
A Box Arbitrage consists of a bull call spread paired with a bear put spread using strikes $K_1 < K_2$:
* **Leg 1**: Long $C(K_1)$
* **Leg 2**: Short $C(K_2)$
* **Leg 3**: Long $P(K_2)$
* **Leg 4**: Short $P(K_1)$

**Mathematical Edge**: The terminal payoff of this structure is strictly guaranteed to be $(K_2 - K_1) \times 100$ CNY at expiry, completely immune to spot price $S_T$.
* **Fair Price**: $V^{\text{fair}}_{\text{box}} = (K_2 - K_1) e^{-r \tau}$
* **Arbitrage Condition**: Trade if the market price $V^{\text{mkt}}_{\text{box}} = [P^A(K_1) - P^B(K_2)] + [P^A(K_2) - P^B(K_1)]$ satisfies:
  $$ V^{\text{mkt}}_{\text{box}} + \mathrm{Fees}_{\text{box}} < V^{\text{fair}}_{\text{box}} - \lambda_{\text{arbitrage}} $$
* **XGBoost Integration**: Only trigger Box execution if the aggregate XGBoost expected return $\alpha^{\text{box}} = \mathbf{a}_{\text{box}}^\top \mathbf{W}$ is strongly positive, signaling rapid microstructural convergence of the mispriced legs.

### B. Butterfly Spread (Volatility & Pinning Arbitrage)
For three consecutive strikes $K_1 < K_2 < K_3$ with equal intervals ($K_2 - K_1 = K_3 - K_2$):
* **Leg 1**: Long $C(K_1)$ (or Put)
* **Leg 2**: Short $2 \times C(K_2)$ (or Put)
* **Leg 3**: Long $C(K_3)$ (or Put)

**Mathematical Edge**: Maximize premium capture around strike $K_2$ while strictly capping maximum loss to the premium paid. Requires **zero short-selling of the ETF spot**.
* **XGBoost Integration**: XGBoost model flags when the center strike $K_2$ is highly overpriced relative to the tail wings $K_1, K_3$ (due to local L1 convexity/butterfly constraint violations).

### C. Iron Condor (Range Volatility Arbitrage)
For four strikes $K_1 < K_2 < K_3 < K_4$:
* **Leg 1**: Long $P(K_1)$
* **Leg 2**: Short $P(K_2)$
* **Leg 3**: Short $C(K_3)$
* **Leg 4**: Long $C(K_4)$

**Mathematical Edge**: Pairs a put spread and call spread to collect premium. Under SSE/SZSE portfolio margin rules, paired spreads **share risk**, reducing required margin by up to 80% compared to raw single-option writing.

---

## 3. Roadmap: Step-by-Step Implementation

### [X] Phase 1: Structural Grid Definition & Mapping in Rust
* [x] Define `ArbitrageStructure` struct in Rust representing multi-leg portfolios:
  ```rust
  pub struct ArbitrageStructure {
      pub name: String,
      pub legs: Vec<LegConfig>,
      pub expected_fee: f64,
  }
  pub struct LegConfig {
      pub expiry: String,
      pub strike: f64,
      pub option_type: char, // 'C' or 'P'
      pub weight: f64,       // +1.0 for buy, -1.0 for sell, etc.
  }
  ```
* [x] Implement an automated scanner in `src/portfolio.rs` that dynamically generates all valid Box Arbitrage, Butterfly, and Iron Condor combinations from the active strike-expiry matrix (typically ~50 combinations).

### [X] Phase 2: Decomposed Structural LP Solver Integration
* [x] Integrate structure matrix $\mathbf{A}$ into the `optimize_portfolio` LP.
* [x] Decompose structural variables $z_p = z^+_p - z^-_p$ where $z^+_p, z^-_p \ge 0$.
* [x] Incorporate transaction cost coefficient $\mathrm{TC}_p$ directly into the objective function to penalize entry of multi-leg spreads with high fee drag.
* [x] Solve LP in `good_lp`. Verify sub-millisecond solving latency (<30 us) for $M \le 100$ candidate structures.

### [X] Phase 3: Anti-Flicker Rebalancing & Order Routing Gates
* [x] Implement the structural rebalancing deadband in `src/bin/backtest.rs`:
  ```rust
  let delta_z = target_z - current_z;
  if delta_z.abs() < 5.0 {
      continue; // Skip tiny, fee-negative adjustments
  }
  ```
* [x] Implement multi-leg execution mapping: submit the legs of the chosen structure ($S_p$) simultaneously as aggressive/passive limit orders.

### [X] Phase 4: Validation via Tick Backtester
* [x] Run simulation loop on `data/510300_surface.parquet`.
* [x] Verify that:
  1. **Total traded contracts** drops by **>90%** (e.g. from 123,948 contracts to <10,000 contracts).
  2. **Total transaction fees** paid drops below the net trading profit.
  3. **Delta and Vega exposure** stays relative neutral throughout the run.
  4. **Drawdowns** are structurally minimized due to options-only risk capping.
