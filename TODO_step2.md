# proarbitrage Phase 2: Profitable Optimization Roadmap

This document outlines the step-by-step technical implementation path to convert the low-latency options relative-value trading engine from an aggressive, fee-negative model into a highly profitable, friction-hedged system.

---

## [ ] Phase 1: Passive Execution Routing & Order Book Queue Tracking
Aggressively crossing the bid-ask spread ($P^A_i, P^B_i$) is the primary source of alpha erosion. The execution engine must pivot to passive limit order placement.
- [ ] **Queue Position Tracker**: Implement a microsecond-level queue position estimator using Level 2 depth changes and Order Flow Imbalance ($\mathrm{OFI}$).
- [ ] **Passive Price Placement**: Route buy orders exactly at the best bid $b_{1,i}$ (or $b_{1,i} + 0.0001$ if tick intervals allow) and sell orders at best ask $a_{1,i}$.
- [ ] **Fill Probability Engine**: Add a statistical model to backtest estimating passive fill probability conditional on localized microstructure volume cues.
- [ ] **Adverse Selection Filter**: Cancel passive orders immediately if the underlying spot price $S_t$ moves against the position, avoiding getting filled only when the distortion expands.

---

## [ ] Phase 2: Anti-Flicker Deadbands & Rebalancing Bands
Continuous target weight adjustments ($\boldsymbol{\Delta\Theta}$) on micro-ticks trigger massive over-trading and fee accumulation.
- [ ] **Hysteresis Thresholds (Deadbands)**: Impose a minimum trade size threshold. Only issue executable orders if the absolute difference exceeds a rebalancing barrier:
  $$ |\Theta^\star_i - \Theta_{\text{current},i}| \ge \delta_i, \quad \text{where } \delta_i = \max\left(5.0, \; 0.15 \cdot |\Theta_{\text{current},i}|\right) $$
- [ ] **LP Transaction Cost Penalty**: Reformulate the Portfolio LP objective function to include a flat-rate transaction cost penalty $\mathbf{TC}$ per unit of trade:
  $$ \max_{\boldsymbol{\Theta}^+, \boldsymbol{\Theta}^-} \quad (\boldsymbol{\Theta}^+ - \boldsymbol{\Theta}^-)^\top \mathbf{W} - \sum_i \mathrm{Fee}_i \cdot \left| \Theta_i - \Theta_{\text{current},i} \right| $$
  *(Note: Enforced linearly by splitting target adjustments into positive/negative changes $\mathbf{y}^+, \mathbf{y}^- \ge 0$)*.

---

## [ ] Phase 3: High-Alpha Hurdle & Regime Classification
Relative-value anomalies must be large enough to clear the bid-ask spread and fee hurdle.
- [ ] **Alpha Entry Hurdle**: Restrict execution to extreme tail events. Raise the pre-inference candidate threshold ($\lambda_{\text{gate}}$) from 5 bps to **65–85 bps**.
- [ ] **Market Regime Detector**: Implement a high-speed volatility and trend classifier. Automatically deactivate the trading engine during high-momentum spot regimes where mean-reversion fails and relative-value anomalies continue to expand.
- [ ] **Parity Skew Filter**: Account for A-share short-selling constraints. Since writing puts or shorting calls faces severe directional margin friction, implement asymmetric alpha hurdles for long vs. short option structures.

---

## [ ] Phase 4: Quadratic Transaction Cost Formulation (QP Solver Integration)
Standard Linear Programs (LP) encourage corner solutions, leading to full position flips and large transaction costs.
- [ ] **Quadratic Program (QP) Solver**: Replace the `minilp` Linear Program with a high-speed quadratic solver (e.g. osqp-rust).
- [ ] **L2 Cost Penalty**: Add a quadratic penalty on position changes to damp down large allocations and favor smooth, incremental portfolio adjustments:
  $$ \max_{\boldsymbol{\Theta}} \quad \boldsymbol{\Theta}^\top \mathbf{W} - \gamma \left( \boldsymbol{\Theta} - \boldsymbol{\Theta}_{\text{current}} \right)^\top \mathbf{\Sigma}_{\text{friction}} \left( \boldsymbol{\Theta} - \boldsymbol{\Theta}_{\text{current}} \right) $$

---

## [ ] Phase 5: Institutional Exchange Fee Optimization
The retail 2.0 CNY/contract fee makes high-frequency trading mathematically impossible.
- [ ] **Market-Maker Fee Tiering**: Structure the trading entities under a Tier-1 Exchange Membership or Liquidity Provider (LP) status to reduce transaction fees from **2.0 CNY** to **<0.15 CNY** per contract.
- [ ] **Rebate Capture**: Implement execution logic to capture maker rebates (getting paid by the exchange to provide liquidity via passive limit orders).
