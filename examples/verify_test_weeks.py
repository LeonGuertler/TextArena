"""
Verify that all test.csv files have the correct week range and data.

Expected:
- Week range: 1-50
- Number of rows: 50
- Columns should include: week, demand_{item_id}, description_{item_id}, 
  lead_time_{item_id}, profit_{item_id}, holding_cost_{item_id}, news
"""

import pandas as pd
from pathlib import Path


def verify_test_file(folder_path: Path, item_id: str):
    """Verify a single test.csv file."""
    test_file = folder_path / "test.csv"
    
    if not test_file.exists():
        return False, f"File not found: {test_file}"
    
    try:
        df = pd.read_csv(test_file)
        
        # Check week range
        min_week = df['week'].min()
        max_week = df['week'].max()
        num_rows = len(df)
        
        # Expected columns
        expected_cols = [
            'week',
            f'demand_{item_id}',
            f'description_{item_id}',
            f'lead_time_{item_id}',
            f'profit_{item_id}',
            f'holding_cost_{item_id}',
            'news'
        ]
        
        # Check columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check week range
        if min_week != 1 or max_week != 50 or num_rows != 50:
            return False, f"weeks {min_week}-{max_week} ({num_rows} rows), expected 1-50 (50 rows)"
        
        return True, f"weeks {min_week}-{max_week} ({num_rows} rows)"
    
    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    base_path = Path(__file__).parent / "real_instances_50_weeks"
    
    print("\n" + "="*70)
    print("Verification: All test.csv files")
    print("="*70)
    
    # Get all instance folders
    instance_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    all_valid = True
    for folder in instance_folders:
        item_id = folder.name
        is_valid, message = verify_test_file(folder, item_id)
        
        status = "✓" if is_valid else "✗"
        print(f"{status} {item_id}: {message}")
        
        if not is_valid:
            all_valid = False
    
    print("="*70)
    if all_valid:
        print("✓ All test.csv files verified successfully!")
    else:
        print("✗ Some test.csv files have issues!")
    print("="*70)


if __name__ == "__main__":
    main()
