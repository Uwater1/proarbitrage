import pandas as pd
from pathlib import Path
import sys

def fix_parquet_file(file_path: Path):
    print(f"Loading {file_path.name}...")
    df = pd.read_parquet(file_path, engine="pyarrow")
    print(f"  Shape: {df.shape}")
    
    # 1. Convert categorical columns to string
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    if cat_cols:
        print(f"  Converting categorical columns to string: {cat_cols}")
        for col in cat_cols:
            df[col] = df[col].astype(str)
            
    # 2. Convert float16 columns to float32
    f16_cols = df.select_dtypes(include=['float16']).columns.tolist()
    if f16_cols:
        print(f"  Converting float16 columns to float32: {f16_cols}")
        for col in f16_cols:
            df[col] = df[col].astype('float32')
            
    # Save a backup just in case
    backup_path = file_path.with_suffix('.parquet.bak')
    if not backup_path.exists():
        print(f"  Creating backup at {backup_path.name}...")
        file_path.rename(backup_path)
    else:
        print(f"  Backup already exists. Overwriting original file...")
        file_path.unlink(missing_ok=True)
        
    print(f"  Writing fixed parquet back to {file_path.name}...")
    df.to_parquet(file_path, compression='snappy', index=False)
    print(f"  Done fixing {file_path.name}!\n")

def main():
    data_dir = Path("data")
    parquet_files = [
        data_dir / "510300_surface.parquet",
        data_dir / "510500_surface.parquet",
        data_dir / "588000_surface.parquet",
    ]
    
    for fp in parquet_files:
        if fp.exists():
            try:
                fix_parquet_file(fp)
            except Exception as e:
                print(f"Error fixing {fp.name}: {e}")
        else:
            print(f"File {fp.name} not found, skipping.")

if __name__ == "__main__":
    main()
