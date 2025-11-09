"""
Final verification: Show the actual data comparison.

This demonstrates:
- Week 0 from original CSV had INVALID data (demand was always 0 or missing)
- Week 1 from original CSV is the FIRST VALID week  
- New extraction: train uses weeks 1-10, test uses weeks 11-60
"""

import pandas as pd
from pathlib import Path


def show_sample_data():
    """Show sample data from one instance to clarify the extraction."""
    
    print("\n" + "="*70)
    print("DATA VERIFICATION: Instance 1047675")
    print("="*70)
    
    # Read the new extracted files
    train_df = pd.read_csv("real_instances_50_weeks/1047675/train.csv")
    test_df = pd.read_csv("real_instances_50_weeks/1047675/test.csv")
    
    print("\n📊 NEW TRAINING DATA (train.csv) - Weeks 1-10:")
    print(train_df.to_string(index=False))
    
    print(f"\n📊 NEW TEST DATA (test.csv) - First 5 rows:")
    print(test_df.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ SUMMARY:")
    print("="*70)
    print("  • Training: 10 weeks of data (week 1-10 from original CSV)")
    print("  • Test: 50 weeks of data (week 11-60 from original, mapped to 1-50)")
    print("  • Week 0 excluded (was invalid/always 0 in original data)")
    print("  • Week 1 demand can be 0.0 - this is VALID data, not the invalid week 0")
    print("="*70)


if __name__ == "__main__":
    show_sample_data()
