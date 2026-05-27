import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="proarbitrage GPU-Accelerated XGBoost Training Script")
    parser.add_argument(
        "--input",
        type=str,
        default="data/510300_extracted.csv",
        help="Path to the extracted CSV dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_5m",
        choices=["target_1m", "target_3m", "target_5m", "target_10m"],
        help="Target horizon return to predict"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained models"
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="Use GPU for training"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for chronological training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== proarbitrage XGBoost Training Pipeline ===")
    print(f"Loading dataset: {args.input}")
    
    if not os.path.exists(args.input):
        print(f"Error: Dataset {args.input} not found!")
        return

    # Load data
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records.")
    
    if len(df) == 0:
        print("Error: Empty dataset!")
        return

    # Define feature set
    features = [
        "immediate_execution_gap",
        "spot",
        "moneyness",
        "tau",
        "is_put",
        "spread"
    ]
    
    print(f"Features: {features}")
    print(f"Target:   {args.target}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Chronological Split (No random split to prevent future leakage in time-series)
    # Sort by date just in case
    df = df.sort_values("date").reset_index(drop=True)
    
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTrain set: {len(train_df)} records (from {train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test set:  {len(test_df)} records (from {test_df['date'].min()} to {test_df['date'].max()})")

    X_train, y_train = train_df[features], train_df[args.target]
    X_test, y_test = test_df[features], test_df[args.target]

    # Import XGBoost
    try:
        import xgboost as xgb
    except ImportError:
        print("\nError: xgboost package not installed! Please run 'pip install xgboost'")
        return

    # Configure XGBoost parameters for GPU training
    xgb_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }

    if args.gpu:
        print("\nConfiguring GPU-accelerated training...")
        try:
            from packaging import version
            xgb_ver = version.parse(xgb.__version__)
            if xgb_ver >= version.parse("2.0.0"):
                xgb_params["tree_method"] = "hist"
                xgb_params["device"] = "cuda"
                print("Using tree_method='hist' and device='cuda' (XGBoost 2.0+ pattern)")
            else:
                xgb_params["tree_method"] = "gpu_hist"
                print("Using tree_method='gpu_hist' (XGBoost <2.0 pattern)")
        except Exception as e:
            # Fallback to standard GPU parameter config
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda"
            print(f"Defaulting to device='cuda' (encountered check error: {e})")
    else:
        print("\nTraining on CPU...")

    # Train model
    model = xgb.XGBRegressor(**xgb_params)
    print("\nStarting training...")
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    print("Training complete.")

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\n=== Model Performance ===")
    print(f"Train RMSE: {train_rmse:.6f} | Train R2: {train_r2:.4f}")
    print(f"Test RMSE:  {test_rmse:.6f} | Test R2:  {test_r2:.4f}")

    # Feature Importance
    print("\n=== Feature Importance ===")
    importance = model.feature_importances_
    for name, score in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:<25}: {score:.4f}")

    # Save Native Model (UBJ & JSON)
    native_ubj_path = os.path.join(args.output_dir, f"xgboost_{args.target}.ubj")
    native_json_path = os.path.join(args.output_dir, f"xgboost_{args.target}.json")
    
    print(f"\nSaving native models...")
    model.save_model(native_ubj_path)
    model.save_model(native_json_path)
    print(f"Saved UBJ model to: {native_ubj_path}")
    print(f"Saved JSON model to: {native_json_path}")

    # Save to ONNX for ultra-low latency inference
    try:
        print("\nAttempting ONNX export...")
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        initial_types = [("input", FloatTensorType([None, len(features)]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=15)
        
        onnx_path = os.path.join(args.output_dir, f"xgboost_{args.target}.onnx")
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        print(f"ONNX Model saved successfully to: {onnx_path}")
    except ImportError:
        print("Note: 'onnxmltools' or 'onnx' not installed. Skipping ONNX export.")
        print("To enable ONNX export, run: pip install onnxmltools onnx")
    except Exception as e:
        print(f"ONNX conversion failed: {e}")

if __name__ == "__main__":
    main()
