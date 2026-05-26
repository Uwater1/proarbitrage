#!/usr/bin/env python3
"""Simple script to convert CSV files to Parquet format with snappy compression."""

import sys
from pathlib import Path
import pandas as pd


def convert_csv_to_parquet(csv_path: str) -> str:
    """
    Convert a CSV file to Parquet with snappy compression.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Path to created Parquet file
    """
    csv_path = Path(csv_path)
    parquet_path = csv_path.with_suffix('.parquet')
    
    print(f"Converting {csv_path.name} to {parquet_path.name}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"  Input shape: {df.shape}")
    
    # Write Parquet with snappy compression (auto-detects engine)
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    # Show compression stats
    csv_size = csv_path.stat().st_size / 1024**2
    parquet_size = parquet_path.stat().st_size / 1024**2
    print(f"  CSV size: {csv_size:.2f} MB")
    print(f"  Parquet size: {parquet_size:.2f} MB")
    print(f"  Compression ratio: {csv_size/parquet_size:.2f}x")
    print(f"  Space saved: {(1 - parquet_size/csv_size)*100:.1f}%")
    
    return str(parquet_path)


def main():
    """Convert all CSV files in current directory to Parquet."""
    current_dir = Path.cwd()
    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return 1
    
    print(f"Found {len(csv_files)} CSV file(s)")
    print("=" * 50)
    
    for csv_file in csv_files:
        try:
            parquet_file = convert_csv_to_parquet(csv_file)
            print(f"✓ Created: {Path(parquet_file).name}")
            print("-" * 50)
        except Exception as e:
            print(f"✗ Error converting {csv_file.name}: {e}")
            print("-" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())