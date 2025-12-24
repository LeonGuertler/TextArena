"""
Within-Instance Variance Plot
Shows standard deviation (std) across runs for each method within each instance
Uses std values directly from JSON files

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

def load_json_files(folder_path):
    """Load all JSON files from a folder and return a dict keyed by instance_name."""
    data = {}
    for file in Path(folder_path).glob('*.json'):
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            instance_name = content['instance_name']
            data[instance_name] = content
    return data

def create_variance_plot(data, lead_time, output_path, display_name=None):
    """
    Create variance plot showing std across runs for each method within each instance.
    X-axis: instances
    Each instance has 4 points (one per method) showing std value
    """
    if display_name is None:
        display_name = f"Lead Time = {lead_time}"
    
    # Sort instances for consistent ordering
    instance_names = sorted(data.keys())
    n_instances = len(instance_names)
    n_methods = len(METHOD_KEYS)
    
    # Width and spacing
    point_width = 0.18
    group_width = n_methods * point_width + 0.2
    
    # Create figure with extra space on the right for legend
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Plot each method
    for j, method_key in enumerate(METHOD_KEYS):
        std_values = []
        positions = []
        
        for i, instance_name in enumerate(instance_names):
            instance_data = data[instance_name]
            
            if method_key in instance_data['results']:
                std_val = instance_data['results'][method_key]['std']
                std_values.append(std_val)
                positions.append(i * group_width + j * point_width)
        
        # Create label with note for OR (std=0, not visible on log scale)
        if method_key == 'or':
            label = f"{METHOD_LABELS[j]} (std=0, not shown)"
        else:
            label = METHOD_LABELS[j]
        
        # Plot scatter points
        ax.scatter(positions, std_values, 
                  color=METHOD_COLORS[j], s=80, alpha=0.7,
                  edgecolors='black', linewidths=0.5,
                  label=label)
    
    # Set x-axis ticks at center of each instance group
    instance_centers = [i * group_width + (n_methods - 1) * point_width / 2 for i in range(n_instances)]
    ax.set_xticks(instance_centers)
    ax.set_xticklabels(instance_names, rotation=45, ha='right', fontsize=9)
    
    # Create legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Method')
    
    # Labels and title
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Standard Deviation (std)', fontsize=12)
    ax.set_title(f'Within-Instance Variance Across Runs ({display_name})', fontsize=14)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Use log scale for y-axis if std values span a wide range
    # Check if we need log scale
    all_stds = []
    for instance_name, instance_data in data.items():
        for method_key in METHOD_KEYS:
            if method_key in instance_data['results']:
                all_stds.append(instance_data['results'][method_key]['std'])
    
    if all_stds:
        max_std = max(all_stds)
        min_std = min([s for s in all_stds if s > 0])  # Exclude zeros
        if max_std / min_std > 100:  # If range is very wide, use log scale
            ax.set_yscale('log')
            ax.set_ylabel('Standard Deviation (std, log scale)', fontsize=12)
    
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
            
            output_path = script_dir / f"variance_lead_time_{lead_time_key}.png"
            create_variance_plot(data, lead_time_key, output_path, display_name)
        else:
            print(f"Warning: Folder not found: {folder_path}")

if __name__ == "__main__":
    main()

