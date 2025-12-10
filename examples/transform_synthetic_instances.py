"""
Transform synthetic demand files to match H&M instances structure.

For each case file:
1. Create folder: initial_synthetic_demand_files/{case_name}/
2. Create train.csv: 5 rows from Normal(100,25), only exact_dates and demand columns
3. Create test.csv: All rows from original file with modifications:
   - First column renamed to exact_dates_chips(Regular) with values Period_1, Period_2, ...
   - profit=2, holding_cost=1
   - Remove news column
4. Delete original file
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
SYNTHETIC_DIR = Path(__file__).resolve().parent / "initial_synthetic_demand_files"

# Files to transform and their folder names
FILES_TO_TRANSFORM = {
    "demand_case1_iid_normal.csv": "case1_iid_normal",
    "demand_case2_sudden_shift_cp15.csv": "case2_sudden_shift_cp15",
    "demand_case3_increasing.csv": "case3_increasing",
    "demand_case4_normal_to_uniform_cp15.csv": "case4_normal_to_uniform_cp15",
}

# Item ID used in synthetic data
ITEM_ID = "chips(Regular)"

def generate_train_data() -> pd.DataFrame:
    """Generate unified train data from Normal(100, 25) distribution."""
    # Generate 5 samples from Normal(100, 25)
    demands = np.random.normal(100, 25, size=5)
    # Round to integers and ensure non-negative
    demands = np.maximum(0, np.round(demands)).astype(int)
    
    # Create DataFrame with exact_dates format (Period_1, Period_2, ...)
    train_df = pd.DataFrame({
        f'exact_dates_{ITEM_ID}': [f'Period_{i}' for i in range(1, 6)],
        f'demand_{ITEM_ID}': demands
    })
    
    return train_df

def transform_to_test_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Transform original DataFrame to test.csv format."""
    # Get the number of rows
    num_rows = len(df)
    
    # Create new DataFrame with required columns
    test_df = pd.DataFrame()
    
    # Add exact_dates column with Period_X format
    test_df[f'exact_dates_{ITEM_ID}'] = [f'Period_{i}' for i in range(1, num_rows + 1)]
    
    # Copy demand column
    demand_col = f'demand_{ITEM_ID}'
    test_df[demand_col] = df[demand_col].values
    
    # Copy description column
    desc_col = f'description_{ITEM_ID}'
    test_df[desc_col] = df[desc_col].values
    
    # Copy lead_time column
    lead_time_col = f'lead_time_{ITEM_ID}'
    test_df[lead_time_col] = df[lead_time_col].values
    
    # Set profit to 2 and holding_cost to 1
    profit_col = f'profit_{ITEM_ID}'
    holding_col = f'holding_cost_{ITEM_ID}'
    test_df[profit_col] = 2
    test_df[holding_col] = 1
    
    # Note: news column is intentionally omitted
    
    return test_df

def transform_file(csv_path: Path, folder_name: str, train_df: pd.DataFrame) -> None:
    """Transform a single synthetic CSV file."""
    print(f"\nProcessing {csv_path.name} -> {folder_name}/...")
    
    # Read the original CSV file
    df = pd.read_csv(csv_path)
    
    # Create folder
    folder_path = SYNTHETIC_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    print(f"  Created folder: {folder_path}")
    
    # Save train.csv (shared across all instances)
    train_path = folder_path / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  Created train.csv: {len(train_df)} rows")
    
    # Transform and save test.csv
    test_df = transform_to_test_csv(df)
    test_path = folder_path / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  Created test.csv: {len(test_df)} rows")
    
    # Delete original file
    csv_path.unlink()
    print(f"  Deleted original file: {csv_path.name}")
    
    print(f"  ✓ Completed transformation for {folder_name}")

def main():
    """Transform all specified synthetic files."""
    print("=" * 70)
    print("Transforming synthetic demand files")
    print("=" * 70)
    
    if not SYNTHETIC_DIR.exists():
        raise FileNotFoundError(f"Synthetic directory not found: {SYNTHETIC_DIR}")
    
    # Generate unified train data (shared by all instances)
    print("\nGenerating unified train data from Normal(100, 25)...")
    train_df = generate_train_data()
    print(f"  Generated samples: {train_df[f'demand_{ITEM_ID}'].tolist()}")
    print(f"  Mean: {train_df[f'demand_{ITEM_ID}'].mean():.1f}")
    print(f"  Std: {train_df[f'demand_{ITEM_ID}'].std():.1f}")
    
    # Check which files exist
    files_found = []
    for filename, folder_name in FILES_TO_TRANSFORM.items():
        csv_path = SYNTHETIC_DIR / filename
        if csv_path.exists():
            files_found.append((csv_path, folder_name))
        else:
            print(f"Warning: File not found: {filename}")
    
    if not files_found:
        print("No files found to transform!")
        return
    
    print(f"\nFound {len(files_found)} file(s) to transform:")
    for csv_path, folder_name in files_found:
        print(f"  - {csv_path.name} -> {folder_name}/")
    
    # Transform each file
    for csv_path, folder_name in files_found:
        try:
            transform_file(csv_path, folder_name, train_df)
        except Exception as e:
            print(f"\n❌ Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Transformation complete!")
    print("=" * 70)
    
    # Summary
    print("\nSummary:")
    print(f"  - Created 4 instance folders in {SYNTHETIC_DIR}")
    print(f"  - Each folder contains train.csv (5 rows) and test.csv (50 rows)")
    print(f"  - train.csv: exact_dates and demand columns only")
    print(f"  - test.csv: all columns with profit=2, holding_cost=1")
    print(f"  - Deleted original CSV files")

if __name__ == "__main__":
    main()

