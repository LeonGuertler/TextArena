"""
Combined Box Plot: All 10 instances combined, showing 4 methods
Each method has data points from all instances (OR: 10 points, others: 50 points)
Generates 2 charts: one for Lead Time = 0, one for Lead Time = 4
Normalization: by best average reward of the 4 methods for each instance
Display: Mean + 95% CI
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Method names mapping
METHOD_KEYS = ['or', 'llm', 'simple_llm_to_or', 'or_to_llm']
METHOD_LABELS = ['OR', 'LLM', 'LLM to OR', 'OR to LLM']
METHOD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# Instances to exclude for specific lead times (due to anomalous data)
EXCLUDE_INSTANCES = {
    4: ['599580017'],  # Exclude for Lead Time = 4 (negative values)
}

def load_json_files(folder_path):
    """Load all JSON files from a folder and return a dict keyed by instance_name."""
    data = {}
    for file in Path(folder_path).glob('*.json'):
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            instance_name = content['instance_name']
            data[instance_name] = content
    return data

def get_best_avg_for_instance(instance_data):
    """Get the best average reward among the 4 methods for an instance."""
    avg_rewards = []
    for method_key in METHOD_KEYS:
        if method_key in instance_data['results']:
            avg_rewards.append(instance_data['results'][method_key]['mean'])
    return max(avg_rewards) if avg_rewards else 1.0

def normalize_rewards(rewards, best_avg):
    """Normalize rewards as % of best average."""
    return [(r / best_avg) * 100 for r in rewards]

def calculate_95ci(data):
    """Calculate mean and 95% CI for a list of values."""
    n = len(data)
    mean = np.mean(data)
    if n <= 1:
        return mean, 0.0  # No CI for single data point
    std = np.std(data, ddof=1)
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci

def create_combined_plot(data, lead_time, output_path, display_name=None):
    """
    Create combined Mean + 95% CI plot: All instances combined.
    X-axis: 4 methods
    Each method shows mean + 95% CI of all normalized data points from all instances
    """
    if display_name is None:
        display_name = f"Lead Time = {lead_time}"
    
    # Filter out excluded instances for this lead time
    exclude_list = EXCLUDE_INSTANCES.get(lead_time, [])
    filtered_data = {k: v for k, v in data.items() if k not in exclude_list}
    
    if exclude_list:
        excluded = [k for k in data.keys() if k in exclude_list]
        if excluded:
            print(f"  Excluded instances: {excluded}")
    
    # Collect all normalized data points for each method
    method_data = {key: [] for key in METHOD_KEYS}
    
    for instance_name, instance_data in filtered_data.items():
        best_avg = get_best_avg_for_instance(instance_data)
        
        for method_key in METHOD_KEYS:
            if method_key in instance_data['results']:
                rewards = instance_data['results'][method_key]['rewards']
                normalized = normalize_rewards(rewards, best_avg)
                method_data[method_key].extend(normalized)
    
    # Calculate mean and 95% CI for each method
    means = []
    cis = []
    for method_key in METHOD_KEYS:
        if method_data[method_key]:
            mean, ci = calculate_95ci(method_data[method_key])
            means.append(mean)
            cis.append(ci)
            print(f"  {method_key}: {len(method_data[method_key])} data points, mean={mean:.2f}%, 95% CI=±{ci:.2f}%")
        else:
            means.append(0)
            cis.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # X positions for the 4 methods
    x_positions = np.arange(len(METHOD_KEYS))
    
    # Plot error bars with markers
    for i, (pos, mean, ci, color) in enumerate(zip(x_positions, means, cis, METHOD_COLORS)):
        ax.errorbar(pos, mean, yerr=ci, 
                    fmt='o', markersize=10, capsize=6, capthick=2,
                    color=color, elinewidth=2, 
                    markeredgecolor='black', markeredgewidth=1)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(METHOD_LABELS, fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('% of Best Average', fontsize=12)
    ax.set_title(f'Combined Performance Across All Instances ({display_name})', fontsize=14)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis to start from a reasonable value
    y_min = min(means) - max(cis) - 5
    y_max = max(means) + max(cis) + 5
    ax.set_ylim(max(0, y_min), min(110, y_max))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Define folder paths (key is used for output filename and display)
    # Use tuples: (folder_name, display_name)
    folders = {
        0: (script_dir / "Lead Time = 0", "Lead Time = 0"),
        4: (script_dir / "Lead Time = 4", "Lead Time = 4"),
        "Mixed": (script_dir / "Lead Time = 1, 2, 3, inf", "Lead Time = Mixed"),
    }
    
    # Generate plots for each lead time
    for lead_time_key, (folder_path, display_name) in folders.items():
        if folder_path.exists():
            print(f"\nProcessing {display_name}...")
            data = load_json_files(folder_path)
            print(f"  Found {len(data)} instances: {sorted(data.keys())}")
            
            output_path = script_dir / f"combined_lead_time_{lead_time_key}.png"
            create_combined_plot(data, lead_time_key, output_path, display_name)
        else:
            print(f"Warning: Folder not found: {folder_path}")

if __name__ == "__main__":
    main()

