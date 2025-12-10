"""
Transform weekly_H&M_instances CSV files to match biweely_H&M_instances structure.

For each hm_weekly_instance_article_{article_id}.csv:
1. Create folder: weekly_H&M_instances/{article_id}/
2. Create train.csv: First 5 rows (excluding header), only exact_dates and demand columns
3. Create test.csv: Remaining rows (including header), all columns
4. Delete original file
"""

import os
import re
import pandas as pd
from pathlib import Path

# Paths
WEEKLY_DIR = Path(__file__).resolve().parent / "H&M_instances" / "weekly_H&M_instances"

def extract_article_id(filename: str) -> str:
    """Extract article_id from filename like 'hm_weekly_instance_article_372860001.csv'"""
    match = re.search(r'hm_weekly_instance_article_(\d+)\.csv', filename)
    if not match:
        raise ValueError(f"Cannot extract article_id from filename: {filename}")
    return match.group(1)

def transform_weekly_file(csv_path: Path) -> None:
    """Transform a single weekly CSV file to train/test structure."""
    filename = csv_path.name
    article_id = extract_article_id(filename)
    
    print(f"\nProcessing {filename} (article_id: {article_id})...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get column names
    exact_dates_col = f'exact_dates_{article_id}'
    demand_col = f'demand_{article_id}'
    
    # Validate columns exist
    if exact_dates_col not in df.columns or demand_col not in df.columns:
        raise ValueError(f"Missing required columns in {filename}: {exact_dates_col}, {demand_col}")
    
    # Create folder for this article_id
    article_dir = WEEKLY_DIR / article_id
    article_dir.mkdir(exist_ok=True)
    print(f"  Created folder: {article_dir}")
    
    # Create train.csv: First 5 rows, only exact_dates and demand
    train_df = df.iloc[:5][[exact_dates_col, demand_col]].copy()
    train_path = article_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  Created train.csv: {len(train_df)} rows (columns: {exact_dates_col}, {demand_col})")
    
    # Create test.csv: Remaining rows (from row 5 onwards), all columns
    test_df = df.iloc[5:].copy()
    test_path = article_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  Created test.csv: {len(test_df)} rows (all columns)")
    
    # Delete original file
    csv_path.unlink()
    print(f"  Deleted original file: {filename}")
    
    print(f"  ✓ Completed transformation for {article_id}")

def main():
    """Transform all weekly instance CSV files."""
    print("=" * 70)
    print("Transforming weekly_H&M_instances to match biweely_H&M_instances structure")
    print("=" * 70)
    
    if not WEEKLY_DIR.exists():
        raise FileNotFoundError(f"Weekly directory not found: {WEEKLY_DIR}")
    
    # Find all CSV files matching the pattern
    csv_files = list(WEEKLY_DIR.glob("hm_weekly_instance_article_*.csv"))
    
    if not csv_files:
        print("No files found matching pattern 'hm_weekly_instance_article_*.csv'")
        return
    
    print(f"\nFound {len(csv_files)} file(s) to transform:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Transform each file
    for csv_path in sorted(csv_files):
        try:
            transform_weekly_file(csv_path)
        except Exception as e:
            print(f"\n❌ Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Transformation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()



