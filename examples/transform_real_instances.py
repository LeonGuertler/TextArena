"""
Transform real_instances test.csv files to match the expected format.

Transformations:
1. Rename week_number (10-59) to week (1-50)
2. Merge holiday + weeks_to_christmas into news column with descriptive text
3. Add required columns: lead_time, profit, holding_cost with default values
4. Rename demand to demand_{item_id} and description to description_{item_id}
5. Use folder name as item_id
"""

import os
import pandas as pd
from pathlib import Path


def merge_news_columns(row):
    """
    Merge holiday and weeks_to_christmas into a descriptive news string.
    
    Examples:
    - holiday="Holiday, National", weeks_to_christmas=8 
      -> "Holiday, National (8 weeks to Christmas)"
    - holiday="", weeks_to_christmas=25
      -> "25 weeks to Christmas"
    - holiday="Additional, National; Event, National", weeks_to_christmas=33
      -> "Additional, National; Event, National (33 weeks to Christmas)"
    """
    holiday = str(row['holiday']).strip() if pd.notna(row['holiday']) else ""
    weeks = int(row['weeks_to_christmas'])
    
    # Build news string
    if holiday and holiday != 'nan':
        # Has holiday info - append weeks_to_christmas in parentheses
        news = f"{holiday} ({weeks} weeks to Christmas)"
    else:
        # No holiday - just mention weeks to Christmas
        if weeks == 0:
            news = "Christmas week!"
        elif weeks == 1:
            news = "1 week to Christmas"
        else:
            news = f"{weeks} weeks to Christmas"
    
    return news


def transform_test_csv(folder_path: Path, item_id: str):
    """
    Transform a single test.csv file.
    
    Args:
        folder_path: Path to the instance folder (e.g., real_instances_50_weeks/1047675/)
        item_id: Item identifier to use (folder name)
    """
    test_file = folder_path / "test.csv"
    
    if not test_file.exists():
        print(f"Warning: {test_file} not found, skipping...")
        return
    
    print(f"Processing {folder_path.name}...")
    
    # Read the original test.csv
    df = pd.read_csv(test_file)
    
    # 1. Adjust week_number: 10-59 -> 1-50
    df['week'] = df['week_number'] - 9  # 10->1, 11->2, ..., 59->50
    
    # 2. Merge holiday + weeks_to_christmas into news
    df['news'] = df.apply(merge_news_columns, axis=1)
    
    # 3. Add required columns with default values
    # Using specified defaults for real instances
    df[f'lead_time_{item_id}'] = 4    # Default lead time = 4 days
    df[f'profit_{item_id}'] = 2.0     # Default profit = $2 per unit
    df[f'holding_cost_{item_id}'] = 1.0  # Default holding cost = $1 per unit per day
    
    # 4. Rename demand and description to include item_id
    df[f'demand_{item_id}'] = df['demand']
    df[f'description_{item_id}'] = df['description']
    
    # 5. Select final columns in expected order
    final_columns = [
        'week',
        f'demand_{item_id}',
        f'description_{item_id}',
        f'lead_time_{item_id}',
        f'profit_{item_id}',
        f'holding_cost_{item_id}',
        'news'
    ]
    
    df_final = df[final_columns]
    
    # 6. Save the transformed file
    output_file = folder_path / "test.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"  ✓ Transformed {folder_path.name}/test.csv")
    print(f"    - Weeks: {df_final['week'].min()} to {df_final['week'].max()}")
    print(f"    - Rows: {len(df_final)}")
    print(f"    - Item ID: {item_id}")
    print()


def main():
    # Path to real_instances_50_weeks folder
    base_path = Path(__file__).parent / "real_instances_50_weeks"
    
    if not base_path.exists():
        print(f"Error: {base_path} not found!")
        return
    
    print("="*70)
    print("Transforming real_instances_50_weeks test.csv files")
    print("="*70)
    print()
    
    # Get all instance folders
    instance_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    print(f"Found {len(instance_folders)} instance folders:")
    for folder in instance_folders:
        print(f"  - {folder.name}")
    print()
    
    # Transform each instance
    for folder in instance_folders:
        # Use folder name as item_id
        item_id = folder.name
        transform_test_csv(folder, item_id)
    
    print("="*70)
    print(f"✓ Successfully transformed {len(instance_folders)} test.csv files!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Review the transformed files to ensure they look correct")
    print("2. Test with existing demo scripts (or_csv_demo.py, llm_csv_demo.py, etc.)")
    print("3. You may need to adjust lead_time, profit, holding_cost defaults if needed")


if __name__ == "__main__":
    main()
