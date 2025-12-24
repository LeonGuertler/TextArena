"""
SET 1: Mean + 95% CI plots - For each instance, showing 4 methods side-by-side
Generates 2 charts: one for Lead Time = 0, one for Lead Time = 4
Normalization: by best average reward of the 4 methods for each instance
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

def create_plot_set1(data, lead_time, output_path, display_name=None):
    """
    Create Mean + 95% CI plot for SET 1: For each instance, 4 methods side-by-side.
    X-axis: instances
    Each instance has 4 points with error bars (one per method)
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
    
    # Sort instances for consistent ordering
    instance_names = sorted(filtered_data.keys())
    n_instances = len(instance_names)
    n_methods = len(METHOD_KEYS)
    
    # Width and spacing
    bar_width = 0.18
    group_width = n_methods * bar_width + 0.2
    
    # Create figure with extra space on the right for legend
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Plot each method
    for j, method_key in enumerate(METHOD_KEYS):
        means = []
        cis = []
        positions = []
        
        for i, instance_name in enumerate(instance_names):
            instance_data = filtered_data[instance_name]
            best_avg = get_best_avg_for_instance(instance_data)
            
            if method_key in instance_data['results']:
                rewards = instance_data['results'][method_key]['rewards']
                normalized = normalize_rewards(rewards, best_avg)
                mean, ci = calculate_95ci(normalized)
                means.append(mean)
                cis.append(ci)
                positions.append(i * group_width + j * bar_width)
        
        # Plot error bars with markers
        ax.errorbar(positions, means, yerr=cis, 
                    fmt='o', markersize=6, capsize=4, capthick=1.5,
                    color=METHOD_COLORS[j], label=METHOD_LABELS[j],
                    elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)
    
    # Set x-axis ticks at center of each instance group
    instance_centers = [i * group_width + (n_methods - 1) * bar_width / 2 for i in range(n_instances)]
    ax.set_xticks(instance_centers)
    ax.set_xticklabels(instance_names, rotation=45, ha='right', fontsize=9)
    
    # Create legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Method')
    
    # Labels and title
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('% of Best Average', fontsize=12)
    ax.set_title(f'Method Comparison by Instance ({display_name})', fontsize=14)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout to make room for legend
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
            
            output_path = script_dir / f"set1_lead_time_{lead_time_key}.png"
            create_plot_set1(data, lead_time_key, output_path, display_name)
        else:
            print(f"Warning: Folder not found: {folder_path}")

if __name__ == "__main__":
    main()


