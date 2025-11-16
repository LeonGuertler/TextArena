"""
Generate 2-week aggregated train and test data for instance 172343.

This script:
1. Reads the original 172343.csv (week 1-170)
2. Creates train_2weeks.csv: weeks 1-20 aggregated into 10 2-week units
3. Creates test_2weeks.csv: weeks 21-120 aggregated into 50 2-week units

Time unit conversion: 2 original weeks = 1 new week unit
"""

import pandas as pd
from pathlib import Path


def generate_2week_data():
    """Generate 2-week aggregated train and test datasets."""
    
    # Read original CSV
    original_csv = Path(r"d:\OR agent docs\output_csv\172343.csv")
    df_original = pd.read_csv(original_csv)
    
    print("="*70)
    print("Generating 2-week aggregated data for instance 172343")
    print("="*70)
    print(f"\nOriginal data: {len(df_original)} rows (weeks 0-{len(df_original)-1})")
    print(f"Valid data: weeks 1-{len(df_original)-1} ({len(df_original)-1} weeks)")
    
    # =========================================================================
    # 1. Generate train_2weeks.csv
    # Source: weeks 1-20 (20 weeks) → 10 2-week units
    # =========================================================================
    
    print("\n" + "-"*70)
    print("Creating train_2weeks.csv")
    print("-"*70)
    
    train_data = []
    for i in range(10):  # 10 2-week units
        # Original weeks: (1,2), (3,4), (5,6), ..., (19,20)
        week1_idx = i * 2 + 1  # 1, 3, 5, ..., 19
        week2_idx = i * 2 + 2  # 2, 4, 6, ..., 20
        
        week1 = df_original[df_original['week_number'] == week1_idx].iloc[0]
        week2 = df_original[df_original['week_number'] == week2_idx].iloc[0]
        
        # Aggregate demand (sum of two weeks)
        demand_sum = week1['demand'] + week2['demand']
        
        train_data.append({
            'week_number': i + 1,  # 1, 2, 3, ..., 10
            'demand': demand_sum,
            'description': 'GROCERY I',
            'holiday': '',  # Leave empty as requested
            'weeks_to_christmas': ''  # Leave empty as requested
        })
        
        print(f"  Week {i+1}: Original weeks {week1_idx}+{week2_idx} → demand = {week1['demand']}+{week2['demand']} = {demand_sum}")
    
    # Create DataFrame and save
    df_train = pd.DataFrame(train_data)
    train_output = Path(__file__).parent / "train_2weeks.csv"
    df_train.to_csv(train_output, index=False)
    
    print(f"\n✓ Created: {train_output}")
    print(f"  Rows: {len(df_train)}")
    print(f"  Columns: {list(df_train.columns)}")
    
    # =========================================================================
    # 2. Generate test_2weeks.csv
    # Source: weeks 21-120 (100 weeks) → 50 2-week units
    # =========================================================================
    
    print("\n" + "-"*70)
    print("Creating test_2weeks.csv")
    print("-"*70)
    
    test_data = []
    for i in range(50):  # 50 2-week units
        # Original weeks: (21,22), (23,24), (25,26), ..., (119,120)
        week1_idx = i * 2 + 21  # 21, 23, 25, ..., 119
        week2_idx = i * 2 + 22  # 22, 24, 26, ..., 120
        
        week1 = df_original[df_original['week_number'] == week1_idx].iloc[0]
        week2 = df_original[df_original['week_number'] == week2_idx].iloc[0]
        
        # Aggregate demand (sum of two weeks)
        demand_sum = week1['demand'] + week2['demand']
        
        test_data.append({
            'week': i + 1,  # 1, 2, 3, ..., 50
            'demand_172343': demand_sum,
            'description_172343': 'GROCERY I',
            'lead_time_172343': 4,
            'profit_172343': 4.0,
            'holding_cost_172343': 1.0,
            'news': ''  # Leave empty as requested
        })
        
        if i < 5 or i >= 48:  # Print first 5 and last 2 for verification
            print(f"  Week {i+1}: Original weeks {week1_idx}+{week2_idx} → demand = {week1['demand']}+{week2['demand']} = {demand_sum}")
        elif i == 5:
            print("  ...")
    
    # Create DataFrame and save
    df_test = pd.DataFrame(test_data)
    test_output = Path(__file__).parent / "test_2weeks.csv"
    df_test.to_csv(test_output, index=False)
    
    print(f"\n✓ Created: {test_output}")
    print(f"  Rows: {len(df_test)}")
    print(f"  Columns: {list(df_test.columns)}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*70)
    print("✓ Generation Complete!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Original time unit: 1 week")
    print(f"  New time unit: 2 weeks (aggregated)")
    print(f"")
    print(f"  train_2weeks.csv:")
    print(f"    - Source: original weeks 1-20")
    print(f"    - Output: 10 2-week units")
    print(f"    - Total demand: {df_train['demand'].sum()}")
    print(f"")
    print(f"  test_2weeks.csv:")
    print(f"    - Source: original weeks 21-120")
    print(f"    - Output: 50 2-week units")
    print(f"    - Total demand: {df_test['demand_172343'].sum()}")
    print(f"")
    print(f"  Files saved in: {Path(__file__).parent}")


if __name__ == "__main__":
    generate_2week_data()
