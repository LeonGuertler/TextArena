"""
Re-extract real instances from original CSV files.

PROBLEM: Original extraction used weeks 0-9 for training, but week 0 is always invalid.
SOLUTION: Extract weeks 1-10 for training and weeks 11-60 for testing (50 weeks).

Original CSV structure (from favorita data):
- Columns: week_number, demand, description, holiday, weeks_to_christmas
- Week numbers: 0, 1, 2, ..., 59 (60 weeks total)

Target structure:
- train.csv: weeks 1-10 from original (10 weeks)
  Columns: week_number, demand, description, holiday, weeks_to_christmas
  
- test.csv: weeks 11-60 from original, renumbered to 1-50
  Columns: week, demand_{item_id}, description_{item_id}, lead_time_{item_id}, 
           profit_{item_id}, holding_cost_{item_id}, news
"""

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


def extract_instance(original_csv_path: Path, output_folder: Path, item_id: str):
    """
    Extract training and test data from original CSV.
    
    Args:
        original_csv_path: Path to original CSV file (e.g., output_csv/1047675.csv)
        output_folder: Path to output folder (e.g., real_instances_50_weeks/1047675/)
        item_id: Item identifier (folder name)
    """
    print(f"\nProcessing {item_id}...")
    
    # Read original CSV
    df = pd.read_csv(original_csv_path)
    print(f"  Original CSV: {len(df)} rows, weeks {df['week_number'].min()}-{df['week_number'].max()}")
    
    # ============================================
    # 1. Extract TRAINING data: weeks 1-10
    # ============================================
    train_df = df[df['week_number'].between(1, 10)].copy()
    
    # Keep original columns and format
    train_df = train_df[['week_number', 'demand', 'description', 'holiday', 'weeks_to_christmas']]
    
    # Save train.csv
    train_file = output_folder / "train.csv"
    train_df.to_csv(train_file, index=False)
    print(f"  ✓ train.csv: {len(train_df)} rows, weeks {train_df['week_number'].min()}-{train_df['week_number'].max()}")
    
    # ============================================
    # 2. Extract TEST data: weeks 11-60 -> 1-50
    # ============================================
    test_df = df[df['week_number'].between(11, 60)].copy()
    
    # Renumber weeks: 11->1, 12->2, ..., 60->50
    test_df['week'] = test_df['week_number'] - 10
    
    # Merge holiday + weeks_to_christmas into news
    test_df['news'] = test_df.apply(merge_news_columns, axis=1)
    
    # Add required columns with default values
    test_df[f'lead_time_{item_id}'] = 4    # Default lead time = 4 weeks
    test_df[f'profit_{item_id}'] = 2.0     # Default profit = $2 per unit
    test_df[f'holding_cost_{item_id}'] = 1.0  # Default holding cost = $1 per unit per week
    
    # Rename demand and description to include item_id
    test_df[f'demand_{item_id}'] = test_df['demand']
    test_df[f'description_{item_id}'] = test_df['description']
    
    # Select final columns in expected order
    final_columns = [
        'week',
        f'demand_{item_id}',
        f'description_{item_id}',
        f'lead_time_{item_id}',
        f'profit_{item_id}',
        f'holding_cost_{item_id}',
        'news'
    ]
    
    test_df = test_df[final_columns]
    
    # Save test.csv
    test_file = output_folder / "test.csv"
    test_df.to_csv(test_file, index=False)
    print(f"  ✓ test.csv: {len(test_df)} rows, weeks {test_df['week'].min()}-{test_df['week'].max()}")


def main():
    """
    Main extraction function.
    
    IMPORTANT: You need to manually specify the path to the original CSV folder.
    The original CSV files should be in a folder like: d:\OR agent docs\favorita data\output_csv\
    """
    
    # ===========================================================================
    # CONFIGURATION: Update this path to point to your original CSV folder
    # ===========================================================================
    ORIGINAL_CSV_FOLDER = r"d:\OR agent docs\favorita data\output_csv"
    
    # Path to output folder
    base_output_path = Path(__file__).parent / "real_instances_50_weeks"
    
    # List of instance IDs (same as CSV filenames without .csv)
    instance_ids = [
        "1047675", "168927", "168989", "172343", "279137",
        "521818", "527757", "827911", "864511", "938576"
    ]
    
    print("="*70)
    print("Re-extracting Real Instances (weeks 1-10 for training, 11-60 for test)")
    print("="*70)
    print(f"\nOriginal CSV folder: {ORIGINAL_CSV_FOLDER}")
    print(f"Output folder: {base_output_path}")
    print(f"\nInstances to process: {len(instance_ids)}")
    
    # Verify original CSV folder exists
    original_folder = Path(ORIGINAL_CSV_FOLDER)
    if not original_folder.exists():
        print(f"\n❌ ERROR: Original CSV folder not found: {ORIGINAL_CSV_FOLDER}")
        print("\nPlease update the ORIGINAL_CSV_FOLDER variable in this script")
        print("to point to the folder containing the original CSV files:")
        print("  - 1047675.csv")
        print("  - 168927.csv")
        print("  - etc.")
        return
    
    # Process each instance
    success_count = 0
    for item_id in instance_ids:
        # Paths
        original_csv = original_folder / f"{item_id}.csv"
        output_folder = base_output_path / item_id
        
        # Check if original CSV exists
        if not original_csv.exists():
            print(f"\n⚠️  WARNING: {original_csv} not found, skipping...")
            continue
        
        # Create output folder if needed
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        try:
            extract_instance(original_csv, output_folder, item_id)
            success_count += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {item_id}: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"✓ Successfully processed {success_count}/{len(instance_ids)} instances!")
    print("="*70)
    
    print("\n📋 Summary of changes:")
    print("  - Training data: NOW uses weeks 1-10 (previously 0-9)")
    print("  - Test data: NOW uses weeks 11-60 mapped to 1-50 (previously 10-59)")
    print("  - Week 0 is excluded (was causing issues)")
    
    print("\n✅ Next steps:")
    print("  1. Verify the extracted data looks correct")
    print("  2. Run verify_train_weeks.py to check training data")
    print("  3. Test with demo scripts (or_csv_demo.py, llm_csv_demo.py, etc.)")


if __name__ == "__main__":
    main()
