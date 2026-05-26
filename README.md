# proarbitrage: Quantitative Options Relative-Value Trading

A statistical arbitrage options trading engine on A-share ETF options (`510300` and `510500`), achieving sub-10ms latency. The engine reconstructs an arbitrage-free pricing surface under strict L1-norm monotonicity and butterfly convexity constraints, applies a selective liquidity gate, and maps candidate contracts to an expected return vector $\mathbf{W}$ via a unified gradient-boosted decision tree.

---

## Phase 5: Unified Tree Scoring Model & GPU Training Pipeline

To run expected-return predictions without sacrificing sub-10ms latency, the system uses a **Unified Tree-Based Prediction Engine**. This page documents the workflow to extract options microstructure features and multi-horizon target returns in Rust, and train the unified predictive model on an NVIDIA GPU-enabled machine.

### System Architecture Workflow

```
[Parquet Datasets] ---> (Rust extract_features) ---> [Extracted CSV]
                                                            | (Transfer)
                                                            v
[NVIDIA GPU Machine] <-- (train_xgboost.py) <---------------'
       |
       +---> [xgboost_target.json] (Native tree model)
       +---> [xgboost_target.onnx] (Low-latency deployment)
```

---

### Step 1: Rust Feature & Target Extraction

The high-speed extraction binary (`src/bin/extract_features.rs`) loads ticks, builds chronological strike-expiry option grids, calibrates the arbitrage-free surface in Rust, scores execution edge against the pre-inference liquidity gate ($D_i > \lambda$), searches forward in time to match future mid-prices, and writes the dataset to a structured CSV file.

> [!WARNING]
> **CRITICAL PERFORMANCE REQUIREMENT:** You **MUST** compile and run this binary in **Release Mode** (`--release`).
> Rust debug mode compiles without optimizations, causing math-heavy libraries (like the `minilp` simplex solver) to run **50x to 100x slower**, leading to >1 hour runtimes.

To compile the extraction binary in release mode:
```bash
cargo build --release --bin extract_features
```

To run feature extraction on the Huatai CSI 300 ETF Options dataset (`510300`):
```bash
# Run on the full dataset (has 33757074 would always take too long)
./target/release/extract_features --input data/510300_surface.parquet --output data/510300_extracted.csv

# Run with a limit of rows for quick testing (about 1 minute)
./target/release/extract_features --input data/510300_surface.parquet --output data/510300_extracted_subset.csv --limit 1000000
```

To run on the China Southern CSI 500 ETF Options dataset (`510500`):
```bash
./target/release/extract_features --input data/510500_surface.parquet --output data/510500_extracted.csv
```

The resulting CSV contains the following columns:
* **Option Metadata:** `date`, `option_type`, `strike`, `expiry`
* **6D Engineered Features:**
  1. `immediate_execution_gap` ($D_i$): Directional distance from surface to executable layer.
  2. `spot` ($S_t$): Underlying ETF price.
  3. `moneyness` ($K_i - S_t$): Distance of strike to spot.
  4. `tau` ($\tau_i$): Time-to-maturity (years).
  5. `is_put`: Boolean indicator (1.0 for Put, 0.0 for Call).
  6. `spread`: Bid-ask spread ($P^A_i - P^B_i$).
* **Forward Target Returns:** `target_1m`, `target_3m`, `target_5m`, `target_10m` (absolute forward mid-price changes).

---

### Step 2: GPU Training Machine Setup

Transfer the generated CSV dataset (`data/510300_extracted.csv`) and the training script (`train_xgboost.py`) to your GPU-enabled machine.

#### System Requirements
* NVIDIA GPU (Ampere, Ada Lovelace, Hopper, or newer recommended).
* CUDA Toolkit (11.8 or 12.x) and compatible NVIDIA drivers.
* Python 3.8+ installed.

#### Python Environment Setup
Create a virtual environment and install the required dependencies (using `uv` for ultra-fast setup, or standard `pip`):

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install pandas numpy scikit-learn packaging

# Install GPU-enabled XGBoost
pip install xgboost

# Install ONNX conversion libraries (optional, for low-latency integration)
pip install onnx onnxmltools
```

---

### Step 3: Run GPU-Accelerated Training

The training script `train_xgboost.py` performs a chronological split to prevent future time-series data leakage, trains an XGBoost regressor using CUDA, evaluates performance ($R^2$ and RMSE), and saves the resulting model.

To run the training script:
```bash
python train_xgboost.py --input data/510300_extracted.csv --target target_5m --output-dir models --gpu True
```

#### Key Command-line Arguments:
* `--input`: Path to the extracted CSV dataset (default: `data/510300_extracted.csv`).
* `--target`: Target return horizon to train on (choices: `target_1m`, `target_3m`, `target_5m`, `target_10m`; default: `target_5m`).
* `--output-dir`: Directory to save the trained models (default: `models`).
* `--gpu`: Set to `True` to enable CUDA acceleration (default: `True`).
* `--train-split`: Fraction of data to use for chronological training vs testing (default: `0.8`).

---

### Step 4: Model Exports & Deployment

Once training is complete, the script exports the model into multiple formats in the `--output-dir` folder:

1. **`xgboost_target_5m.ubj`**: XGBoost Universal Binary JSON model. Highly optimized native representation.
2. **`xgboost_target_5m.json`**: Standard JSON format model.
3. **`xgboost_target_5m.onnx`**: ONNX format model. Perfect for integration with ONNX Runtime or `candle-onnx` in the low-latency Rust/C++ production trading loops (executing in <100 microseconds).

---

### Verification and Evaluation Indicators
During training, the script outputs key regression metrics:
* **Root Mean Squared Error (RMSE)**: Direct pricing error in currency units.
* **$R^2$ (Coefficient of Determination)**: Proportion of variance explained.
* **Feature Importances**: Ranked listing of structural features driving predictive power (typically led by `immediate_execution_gap` and `spread`).
