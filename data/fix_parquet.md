# Fix Parquet Files for Rust Polars Compatibility

Downloaded `*_surface.parquet` files contain `float16` and `category` types, which crash Rust's Polars v0.38.0 reader.

## Quick Fix

Run the helper script:

```bash
python data/fix_parquet.py
```

## Script Details

Convert the columns in your python download pipeline before writing Parquet:

```python
import pandas as pd

df = pd.read_parquet("path/to/surface.parquet")

# 1. Convert categorical columns to string
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].astype(str)

# 2. Convert float16 to float32
for col in df.select_dtypes(include=['float16']).columns:
    df[col] = df[col].astype('float32')

df.to_parquet("path/to/surface.parquet", compression='snappy', index=False)
```
