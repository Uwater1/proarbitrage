# proarbitrage Project Implementation Roadmap

- [x] **Phase 1: Environment & Workspace Setup**
  - [x] Initialize Rust workspace (`Cargo.toml` in root).
  - [x] Configure Rust dependencies in `Cargo.toml` (`polars = "0.38.0"`, `ndarray = "0.15.6"`, `good_lp` with `minilp` backend, `candle-core = "0.6.0"`, `candle-nn = "0.6.0"`, `candle-transformers = "0.6.0"`). Unified `half = "=2.4.1"` version to resolve compiler conflicts.
  - [x] Create entrypoint `src/lib.rs` and submodules `ingestion`, `calibration`, `activation`.

- [x] **Phase 2: High-Frequency Data Streaming & Ingestion**
  - [x] Write Parquet streaming reader in Rust (`src/ingestion.rs`).
  - [x] Reconstruct options grid matrix per timestamp group $t$. Chronologically grouped 20,000 tick records into 8,811 grids in **30 ms**.
  - [x] Extract effective bid--ask $P^A_i, P^B_i$ and validate liquid indicator columns.

- [x] **Phase 3: Arbitrage-Free Surface Calibration**
  - [x] Define bi-quadratic polynomial basis functions $\phi_m(K, T)$ in `src/calibration.rs` guaranteeing spot boundary conditions exactly.
  - [x] Formulate L1-norm surface calibration Linear Programming (LP) objective.
  - [x] Implement monotonicity, butterfly convexity, and calendar constraints.
  - [x] Integrate `good_lp` with `minilp` solver. Achieved average calibration latency of **72.05 us** (70x faster than 5ms target).

- [x] **Phase 4: Selective Activation & Feature Engineering**
  - [x] Write activation score $g(\mathcal{X}_t, S_t)$ calculator in `src/activation.rs` scoring residual intensity, shape violations, execution quality, and expected edge.
  - [x] Construct 6-dimensional feature vector $\boldsymbol{\xi}_{i,t}$ for candidate contracts.
  - [x] Implement pre-inference liquidity gate (Equation 9).
  - [x] Compile and execute integration test binary (`src/bin/main.rs`) verifying full logic and microsecond-level latency benchmarks.

- [ ] **Phase 5: Unified Tree Scoring Model**
  - [ ] Extract target returns (1m to 10m forward mid-price changes) from tick history.
  - [ ] Train unified XGBoost/LightGBM model on historical parquet data.
  - [ ] Export model to C (Treelite) or ONNX format for fast inference.
  - [ ] Set up compiled tree model inference pipeline in Rust using Candle or bindings.

- [ ] **Phase 6: Multi-Greek Constrained Portfolio LP**
  - [ ] Extract Greeks (Delta $\boldsymbol{\Delta}$, Vega $\boldsymbol{\nu}$, Theta $\boldsymbol{\vartheta}$) from calibrated surface or analytical formulas.
  - [ ] Set up LP matrix for Equations 11-15.
  - [ ] Decompose positions into non-negative variables $\boldsymbol{\Theta}^+$ and $\boldsymbol{\Theta}^-$.
  - [ ] Integrate solver. Verify sub-millisecond execution.

- [ ] **Phase 7: Execution Mapping & Unwind Manager**
  - [ ] Map optimal target change $\boldsymbol{\Delta\Theta}$ to aggressive volume-capped limit orders.
  - [ ] Write automated unwind state machine: statistical exit, 15m soft/30m hard temporal cutoff, and Greek breach recovery.

- [ ] **Phase 8: Tick Backtester & Latency Profile**
  - [ ] Build simulation loop processing `data/*.parquet` chronologically.
  - [ ] Model bid-ask execution, slippage, and fee friction.
  - [ ] Track portfolio equity curve, Sharpe ratio, max drawdown, and Greek exposures over time.
  - [ ] Profile execution latency of calibration and optimization stages.
