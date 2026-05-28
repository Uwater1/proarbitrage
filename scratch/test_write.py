import pandas as pd
import os

os.makedirs("scratch", exist_ok=True)

# Read 100 rows
df = pd.read_parquet("data/510300_surface.parquet", engine="pyarrow").head(100)

# 1. Convert categories to standard strings
df_str = df.copy()
for col in df_str.select_dtypes(include=['category']).columns:
    df_str[col] = df_str[col].astype(str)

# Write df_str as-is (with float16, but str categories)
df_str.to_parquet("scratch/test_str_float16.parquet", index=False)

# 2. Also convert float16 to float32
df_str_f32 = df_str.copy()
for col in df_str_f32.select_dtypes(include=['float16']).columns:
    df_str_f32[col] = df_str_f32[col].astype('float32')

df_str_f32.to_parquet("scratch/test_str_float32.parquet", index=False)
print("Saved both test_str_float16.parquet and test_str_float32.parquet successfully!")
