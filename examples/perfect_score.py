"""
Perfect Score Calculator for Vending Machine Environment.

This script calculates the theoretical maximum reward achievable with perfect
foreknowledge of demand. It assumes:
- Orders are placed perfectly to meet exactly the demand for each period
- 100% demand fulfillment (after initial lead time period)
- 0 holding cost (no excess inventory)

The initial lead time period (Period 1 to Period L) cannot be served because
orders placed on Period 1 arrive on Period L+1.

Usage:
  python perfect_score.py --demand-file path/to/test.csv
"""

import os
import sys
import argparse
import pandas as pd

# Fix stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_item_ids(df: pd.DataFrame) -> list:
    """Extract item IDs from CSV columns (columns starting with 'demand_')."""
    item_ids = []
    for col in df.columns:
        if col.startswith('demand_'):
            item_id = col.replace('demand_', '')
            item_ids.append(item_id)
    return item_ids


def calculate_perfect_score(csv_path: str) -> dict:
    """
    Calculate the perfect score for a given test CSV file.
    
    Args:
        csv_path: Path to the test.csv file
        
    Returns:
        Dictionary containing perfect score details for each item
    
    Logic:
        1. Find the first period X where lead_time != inf
        2. Get lead_time L from period X
        3. Calculate perfect score from Period X+L onwards (inclusive)
    """
    df = pd.read_csv(csv_path)
    item_ids = extract_item_ids(df)
    
    if not item_ids:
        raise ValueError("No item columns found in CSV. Expected columns like 'demand_<item_id>'")
    
    results = {}
    total_perfect_score = 0
    
    for item_id in item_ids:
        # Extract columns for this item
        demand_col = f'demand_{item_id}'
        lead_time_col = f'lead_time_{item_id}'
        profit_col = f'profit_{item_id}'
        
        # Validate required columns exist
        required_cols = [demand_col, lead_time_col, profit_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for item {item_id}: {missing_cols}")
        
        # Get data
        demands = df[demand_col].tolist()
        lead_times = df[lead_time_col].tolist()
        profit = float(df[profit_col].iloc[0])
        
        total_periods = len(demands)
        
        # Find the first period X where lead_time != inf (0-indexed)
        first_valid_idx = None
        first_valid_lead_time = None
        for idx, lt in enumerate(lead_times):
            # Check if lead_time is not inf (could be string 'inf' or float inf)
            lt_str = str(lt).lower().strip()
            if lt_str != 'inf' and lt_str != 'infinity':
                try:
                    first_valid_lead_time = int(float(lt))
                    first_valid_idx = idx
                    break
                except (ValueError, TypeError):
                    continue
        
        if first_valid_idx is None:
            # All lead times are inf - cannot calculate perfect score
            results[item_id] = {
                'total_periods': total_periods,
                'first_valid_period': None,
                'lead_time': None,
                'profit': profit,
                'effective_start_period': None,
                'effective_periods': 0,
                'sum_demand': 0,
                'sum_missed_demand': sum(demands),
                'perfect_score': 0,
                'error': 'All lead times are inf'
            }
            continue
        
        # X is 1-indexed period, first_valid_idx is 0-indexed
        period_x = first_valid_idx + 1  # Convert to 1-indexed
        lead_time = first_valid_lead_time
        
        # Start from Period X+L (1-indexed), which is index X+L-1 (0-indexed)
        # In 0-indexed: start_idx = first_valid_idx + lead_time
        start_idx = first_valid_idx + lead_time
        
        # Calculate perfect score from start_idx onwards
        if start_idx >= total_periods:
            # Not enough periods after first valid period + lead time
            effective_demands = []
        else:
            effective_demands = demands[start_idx:]
        
        sum_demand = sum(effective_demands)
        perfect_score = profit * sum_demand
        
        # Calculate what was missed (Period 1 to Period X+L-1)
        missed_demands = demands[:start_idx]
        sum_missed = sum(missed_demands)
        
        results[item_id] = {
            'total_periods': total_periods,
            'first_valid_period': period_x,  # Period X (1-indexed)
            'lead_time': lead_time,
            'profit': profit,
            'effective_start_period': start_idx + 1 if start_idx < total_periods else None,  # Period X+L (1-indexed)
            'effective_periods': len(effective_demands),
            'sum_demand': sum_demand,
            'sum_missed_demand': sum_missed,
            'perfect_score': perfect_score
        }
        
        total_perfect_score += perfect_score
    
    results['_total'] = {
        'total_perfect_score': total_perfect_score
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate perfect score for vending machine test case')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data (test.csv)')
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.demand_file):
        print(f"Error: File not found: {args.demand_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("=== Perfect Score Calculator ===")
    print("=" * 70)
    print(f"\nInput file: {args.demand_file}")
    
    # Calculate perfect score
    results = calculate_perfect_score(args.demand_file)
    
    # Display results for each item
    for item_id, data in results.items():
        if item_id == '_total':
            continue
            
        print(f"\n{'-' * 70}")
        print(f"Item: {item_id}")
        print(f"{'-' * 70}")
        print(f"  Total Periods: {data['total_periods']}")
        print(f"  Profit per unit: ${data['profit']}")
        print(f"")
        
        if data.get('error'):
            print(f"  ERROR: {data['error']}")
            print(f"  Perfect Score: $0.00 (cannot calculate)")
            continue
        
        print(f"  First Valid Period (X): Period {data['first_valid_period']}")
        print(f"  Lead Time (L): {data['lead_time']} (from Period {data['first_valid_period']})")
        print(f"")
        print(f"  Skipped Periods: Period 1 to Period {data['effective_start_period'] - 1}")
        print(f"    - Demand during skipped periods: {data['sum_missed_demand']} (cannot be served)")
        print(f"")
        print(f"  Effective Periods: Period {data['effective_start_period']} to Period {data['total_periods']}")
        print(f"    - Number of effective periods: {data['effective_periods']}")
        print(f"    - Total demand: {data['sum_demand']}")
        print(f"")
        print(f"  Perfect Score for {item_id}: ${data['perfect_score']:.2f}")
    
    # Display total
    print("\n" + "=" * 70)
    print("=== TOTAL PERFECT SCORE ===")
    print("=" * 70)
    print(f"\n>>> Perfect Score: ${results['_total']['total_perfect_score']:.2f} <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()

