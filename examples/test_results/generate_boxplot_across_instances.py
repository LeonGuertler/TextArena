"""
Across-Instance Performance Variation Plot
Each method has 9-10 data points (one per instance, using mean reward)
Normalized by best average of the 4 methods for each instance
This ensures no percentages exceed 100% and shows pure across-instance variation

Generates 3 charts: Lead Time = 0, Lead Time = 4, Lead Time = Mixed
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

def calculate_95ci(data):
    """Calculate mean and 95% CI for a list of values."""
    n = len(data)
    mean = np.mean(data)
    if n <= 1:
        return mean, 0.0  # No CI for single data point
    std = np.std(data, ddof=1)
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci

def create_across_instances_plot(data, lead_time, output_path, display_name=None):
    """
    Create Across-Instance Performance Variation plot using scatter plot.
    Uses only average (mean) per instance, normalized by best average.
    This ensures no percentages exceed 100% and shows pure across-instance variation.
    
    X-axis: 4 methods
    Each method shows all data points (one per instance) as scatter + mean marker
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
    
    # Collect normalized averages for each method (one per instance)
    method_normalized_avgs = {key: [] for key in METHOD_KEYS}
    
    for instance_name, instance_data in filtered_data.items():
        best_avg = get_best_avg_for_instance(instance_data)
        
        for method_key in METHOD_KEYS:
            if method_key in instance_data['results']:
                method_mean = instance_data['results'][method_key]['mean']
                normalized_avg = (method_mean / best_avg) * 100
                method_normalized_avgs[method_key].append(normalized_avg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # X positions for the 4 methods
    x_positions = np.arange(len(METHOD_KEYS))
    
    # Plot scatter points and mean for each method
    for i, method_key in enumerate(METHOD_KEYS):
        data_points = method_normalized_avgs[method_key]
        if data_points:
            # Add jitter to x positions for better visibility
            jitter = np.random.uniform(-0.15, 0.15, len(data_points))
            x_jittered = x_positions[i] + jitter
            
            # Plot individual data points
            ax.scatter(x_jittered, data_points, 
                      color=METHOD_COLORS[i], alpha=0.6, s=50,
                      edgecolors='white', linewidths=0.5)
            
            # Plot mean as a larger marker with black edge
            mean_val = np.mean(data_points)
            ax.scatter(x_positions[i], mean_val, 
                      color=METHOD_COLORS[i], s=80, marker='D',
                      edgecolors='black', linewidths=1.2, zorder=5)
            
            print(f"  {method_key}: {len(data_points)} instances, mean={mean_val:.2f}%, min={min(data_points):.2f}%, max={max(data_points):.2f}%")
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(METHOD_LABELS, fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('% of Best Average', fontsize=12)
    ax.set_title(f'Across-Instance Performance Variation ({display_name})', fontsize=14)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='100% (Best)')
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis range
    all_points = [p for pts in method_normalized_avgs.values() for p in pts]
    y_min = min(all_points) - 5 if all_points else 0
    y_max = 105  # Fixed at 105% to show 100% line clearly
    ax.set_ylim(max(0, y_min), y_max)
    
    # Add legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='D', color='gray', markersize=10,
                              markeredgecolor='black', markeredgewidth=1.5,
                              linestyle='None', label='Mean')]
    ax.legend(handles=legend_elements, loc='lower right')
    
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
            
            output_path = script_dir / f"across_instances_lead_time_{lead_time_key}.png"
            create_across_instances_plot(data, lead_time_key, output_path, display_name)
        else:
            print(f"Warning: Folder not found: {folder_path}")

if __name__ == "__main__":
    main()

