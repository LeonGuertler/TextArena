"""
Generate table showing average standard deviation across instances for each method
Shows the variance ordering: OR < LLM to OR < OR to LLM < LLM
"""

import json
from pathlib import Path
import pandas as pd

# Method names mapping
METHOD_KEYS = ['or', 'llm', 'simple_llm_to_or', 'or_to_llm']
METHOD_LABELS = ['OR', 'LLM', 'LLM to OR', 'OR to LLM']

def load_json_files(folder_path):
    """Load all JSON files from a folder and return a dict keyed by instance_name."""
    data = {}
    for file in Path(folder_path).glob('*.json'):
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            instance_name = content['instance_name']
            data[instance_name] = content
    return data

def calculate_avg_std(data):
    """Calculate average std across all instances for each method."""
    method_stds = {key: [] for key in METHOD_KEYS}
    
    for instance_name, instance_data in data.items():
        for method_key in METHOD_KEYS:
            if method_key in instance_data['results']:
                std_val = instance_data['results'][method_key]['std']
                method_stds[method_key].append(std_val)
    
    # Calculate statistics for each method
    results = {}
    for method_key in METHOD_KEYS:
        stds = method_stds[method_key]
        if stds:
            results[method_key] = {
                'mean': sum(stds) / len(stds),
                'min': min(stds),
                'max': max(stds),
                'count': len(stds)
            }
        else:
            results[method_key] = {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
    
    return results

def calculate_avg_cv(data):
    """Calculate average Coefficient of Variation (CV = std/mean) across all instances for each method."""
    method_cvs = {key: [] for key in METHOD_KEYS}
    
    for instance_name, instance_data in data.items():
        for method_key in METHOD_KEYS:
            if method_key in instance_data['results']:
                std_val = instance_data['results'][method_key]['std']
                mean_val = instance_data['results'][method_key]['mean']
                
                # Calculate CV = std / mean
                # Handle division by zero (OR method has std=0, mean>0, so CV=0)
                if mean_val != 0:
                    cv_val = std_val / mean_val
                    method_cvs[method_key].append(cv_val)
                else:
                    # If mean is 0, CV is undefined, but we can set it to 0
                    method_cvs[method_key].append(0.0)
    
    # Calculate average CV for each method
    results = {}
    for method_key in METHOD_KEYS:
        cvs = method_cvs[method_key]
        if cvs:
            results[method_key] = {
                'mean': sum(cvs) / len(cvs),
                'min': min(cvs),
                'max': max(cvs),
                'count': len(cvs)
            }
        else:
            results[method_key] = {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
    
    return results

def generate_table():
    """Generate table showing average std for each method across lead time settings."""
    script_dir = Path(__file__).parent
    
    # Define folder paths
    folders = {
        0: (script_dir / "Lead Time = 0", "Lead Time = 0"),
        4: (script_dir / "Lead Time = 4", "Lead Time = 4"),
        "Mixed": (script_dir / "Lead Time = 1, 2, 3, inf", "Lead Time = Mixed"),
    }
    
    # Collect data for all lead times (for std table)
    table_data = []
    # Collect data for CV table
    cv_table_data = []
    
    for lead_time_key, (folder_path, display_name) in folders.items():
        if folder_path.exists():
            print(f"\nProcessing {display_name}...")
            data = load_json_files(folder_path)
            print(f"  Found {len(data)} instances")
            
            # Calculate std statistics (use all instances)
            results = calculate_avg_std(data)
            
            # For CV calculation, exclude 599580017 for Lead Time = 4
            cv_data = data.copy()
            if lead_time_key == 4:
                if '599580017' in cv_data:
                    del cv_data['599580017']
                    print(f"  Excluded 599580017 for CV calculation")
            
            # Calculate CV statistics
            cv_results = calculate_avg_cv(cv_data)
            
            for method_key, method_label in zip(METHOD_KEYS, METHOD_LABELS):
                stats = results[method_key]
                table_data.append({
                    'Lead Time': display_name,
                    'Method': method_label,
                    'Avg Std': stats['mean'],
                    'Min Std': stats['min'],
                    'Max Std': stats['max'],
                    'N Instances': stats['count']
                })
                
                cv_stats = cv_results[method_key]
                cv_table_data.append({
                    'Lead Time': display_name,
                    'Method': method_label,
                    'Avg CV': cv_stats['mean'],
                    'Min CV': cv_stats['min'],
                    'Max CV': cv_stats['max'],
                    'N Instances': cv_stats['count']
                })
                
                print(f"  {method_label}: avg_std={stats['mean']:.2f}, avg_cv={cv_stats['mean']:.4f}")
        else:
            print(f"Warning: Folder not found: {folder_path}")
    
    # Create DataFrame for std
    df = pd.DataFrame(table_data)
    
    # Create pivot table for std
    pivot_mean = df.pivot(index='Method', columns='Lead Time', values='Avg Std')
    pivot_mean = pivot_mean.reindex(METHOD_LABELS)  # Ensure correct order
    
    # Create DataFrame for CV
    df_cv = pd.DataFrame(cv_table_data)
    
    # Create pivot table for CV
    pivot_cv = df_cv.pivot(index='Method', columns='Lead Time', values='Avg CV')
    pivot_cv = pivot_cv.reindex(METHOD_LABELS)  # Ensure correct order
    
    # Save to CSV
    csv_path = script_dir / "variance_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed std table to: {csv_path}")
    
    # Save pivot table (std mean only)
    pivot_csv_path = script_dir / "variance_table_pivot.csv"
    try:
        pivot_mean.to_csv(pivot_csv_path)
        print(f"Saved pivot table (std mean) to: {pivot_csv_path}")
    except PermissionError:
        print(f"Warning: Could not save {pivot_csv_path} (file may be open in another program)")
    
    # Save CV pivot table
    cv_pivot_csv_path = script_dir / "variance_table_cv_pivot.csv"
    try:
        pivot_cv.to_csv(cv_pivot_csv_path)
        print(f"Saved CV pivot table to: {cv_pivot_csv_path}")
    except PermissionError:
        print(f"Warning: Could not save {cv_pivot_csv_path} (file may be open in another program)")
    
    # Print formatted tables
    print("\n" + "="*80)
    print("Average Standard Deviation Across Instances")
    print("="*80)
    print(pivot_mean.to_string())
    print("="*80)
    
    print("\n" + "="*80)
    print("Average Coefficient of Variation (CV = std/mean) Across Instances")
    print("="*80)
    print(pivot_cv.to_string())
    print("="*80)
    
    return df, pivot_mean, df_cv, pivot_cv

if __name__ == "__main__":
    generate_table()

