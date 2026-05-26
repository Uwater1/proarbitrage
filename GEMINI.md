# proarbitrage System State & Architecture Memory

## Project Goal
Statistical relative-value options trading on SSE A-share ETF options. Limit latency under 10ms. Reconstruct arbitrage-free surface. Map mispricings to Greek-hedged portfolio. Execute over 1-10 minute mean-reversion horizons.

## Current State
* **Completed:** Phases 1 to 4 (Environment setup, Data ingestion, LP surface calibration, Activation/Feature extraction).
* **Ingestion performance:** Loaded 20,000 tick records and reconstructed 8,811 chronological grids in **30 ms** (~0.66M rows/sec).
* **Calibration performance:** Completed L1-norm surface calibration in **72.05 us** average latency (70x faster than 5ms target).
* **Workspace:** Unified dependency versions. Compiles with Rust zero-dependency `minilp` solver.

## Data Assessment
Parquet datasets in `data/` folder verified. Fully compatible. Directly supports math in `strategy_framework.tex`.

### Dataset Properties
* **Asset 510300 (Huatai-PineBridge CSI 300 ETF Options):**
  * File: `data/510300_surface.parquet`
  * Rows: 33,757,074 ticks
  * Date range: `2026-04-20 09:30:00+08:00` to `2026-05-20 15:00:00+08:00`
  * Unique expiries: 3
  * Unique strikes: 27
  * Liquidity: 33,225,659 rows (98.4%) marked `is_liquid`
* **Asset 510500 (China Southern CSI 500 ETF Options):**
  * File: `data/510500_surface.parquet`
  * Rows: 22,520,273 ticks
  * Date range: `2026-04-27 09:30:00+08:00` to `2026-05-21 15:00:00+08:00`
  * Unique expiries: 2
  * Unique strikes: 30
  * Liquidity: 22,037,825 rows (97.9%) marked `is_liquid`

### Mathematical Alignment to strategy_framework.tex
* **L2 Order Book Parameters:** `P_A`, `P_B` map to effective ask $P^A_i$ and bid $P^B_i$. `a_v_eff`, `b_v_eff` provide effective volume thresholds for swept order-book liquidity filtering (Section 1.1).
* **Mid-price Matrix:** `mid` maps to $M_i = (P^A_i + P^B_i)/2$ (Equation 2).
* **Calibration Inputs:** `strike` ($K_i$), `tau` ($\tau_j$), `S_t` (underlying price), `mid` ($M_{ij}$) match inputs for surface calibration (Equation 6) and non-uniform convexity constraints (Equation 8).
* **Tree Features:** `spread`, `moneyness`, `tau`, `type` ('C' or 'P'), `is_liquid` map to elements of state vector $\boldsymbol{\xi}_{i,t}$ for XGBoost/LightGBM scoring (Equation 10).
* **Backtesting Horizon:** Millisecond-level tick timestamps (`date`) allow simulated returns over 1-10 minute mean-reversion targets.

## System Architecture

```mermaid
graph TD
    A[data/*.parquet] --> B[Data Ingestion / Streaming]
    B --> C[Liquidity Filter & Matrix Transformation]
    C --> D{Selective Activation Gate}
    D -- Pass --> E[Arbitrage-Free Volatility Surface Calibration LP]
    D -- Skip --> F[Reuse Previous Surface]
    E --> G[Unified Tree Model Inference W]
    F --> G
    G --> H[Multi-Greek Constrained Portfolio LP]
    H --> I[Microstructural Order Execution Routing]
    I --> J[Automated Unwind State Machine]
```

1. **Ingestion & Streaming:** Read high-frequency parquet ticks using `polars`. Stream by timestamp sequence.
2. **Matrix Reconstruct:** Group contracts sequentially into strike-expiry grid (e.g. 3x27 or 2x30 active subset).
3. **Activation Gate:** Compute lightweight score $\Gamma_t$ (Equation 7). Trigger calibration if $\Gamma_t > \tau_{\text{enter}}$.
4. **Surface Calibration:** Solve constrained Linear Program (LP) under L1-norm with non-uniform finite-difference convexity boundaries (Equation 8). Incorporates Call-Put parity.
5. **Return Scoring:** Compute Immediate Execution Gap $D_i$. Route candidates to compiled tree predictor.
6. **Portfolio LP:** Split weights $\boldsymbol{\Theta} = \boldsymbol{\Theta}^+ - \boldsymbol{\Theta}^-$. Solve LP under Delta, Vega, Theta, and Capital capacity constraints (Equations 11-15).
7. **Execution Routing:** Map target change $\boldsymbol{\Delta\Theta}$ to aggressive bids/asks matching current depth.
8. **Unwind Manager:** Monitor statistical convergence, temporal limits (15m/30m), and Greek violations.

## Environment & Tech Stack
* **Core:** Rust for low-latency pipeline (Parquet parser, LP surface solver, LP portfolio optimizer). Python for model training and research.
* **Resolved Dependency Layout:**
  * `polars = "0.38.0"` - Fast parquet ingestion.
  * `ndarray = "0.15.6"` - Grid matrices.
  * `good_lp = { version = "1.8.1", default-features = false, features = ["minilp"] }` - Pure Rust simplex solver. Prevents linking failures against system CBC libraries.
  * `half = "=2.4.1"` - Pins `half` to unify `rand` version.
  * `candle-core = "0.6.0"`, `candle-nn = "0.6.0"`, `candle-transformers = "0.6.0"` - Deep learning models framework.
