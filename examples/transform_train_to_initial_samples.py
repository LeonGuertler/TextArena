"""
Generate initial_samples from train.csv files for real_instances.

This script reads the train.csv files (weeks 0-9) and creates a dictionary
of initial demand samples that can be used to initialize OR/LLM agents.
"""

import pandas as pd
from pathlib import Path


def extract_initial_samples(folder_path: Path, item_id: str):
    """
    Extract initial demand samples from train.csv.
    
    Args:
        folder_path: Path to instance folder
        item_id: Item identifier
        
    Returns:
        List of demand values from weeks 1-9 (excluding week 0 which is typically 0)
    """
    train_file = folder_path / "train.csv"
    
    if not train_file.exists():
        print(f"Warning: {train_file} not found")
        return []
    
    df = pd.read_csv(train_file)
    
    # Exclude week 0 (typically initialization week with 0 demand)
    # Use weeks 1-9 for initial samples
    df_samples = df[df['week_number'] >= 1]
    
    samples = df_samples['demand'].tolist()
    
    return samples


def main():
    base_path = Path(__file__).parent / "real_instances_50_weeks"
    
    if not base_path.exists():
        print(f"Error: {base_path} not found!")
        return
    
    print("="*70)
    print("Extracting initial samples from train.csv files")
    print("="*70)
    print()
    
    # Get all instance folders
    instance_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    # Extract samples for each instance
    all_initial_samples = {}
    
    for folder in instance_folders:
        item_id = folder.name
        samples = extract_initial_samples(folder, item_id)
        
        if samples:
            all_initial_samples[item_id] = samples
            print(f"{item_id}:")
            print(f"  Samples (weeks 1-9): {samples}")
            print(f"  Count: {len(samples)}")
            print(f"  Mean: {sum(samples)/len(samples):.1f}")
            print()
    
    print("="*70)
    print("Summary - Python code to use in demo scripts:")
    print("="*70)
    print()
    print("# Initial samples from train.csv (weeks 1-9)")
    print("initial_samples = {")
    for item_id, samples in all_initial_samples.items():
        print(f"    '{item_id}': {samples},")
    print("}")
    print()
    
    # Also show how to use with single instance
    print("# For testing single instance, example:")
    if all_initial_samples:
        first_item = list(all_initial_samples.keys())[0]
        first_samples = all_initial_samples[first_item]
        print(f"item_id = '{first_item}'")
        print(f"initial_samples = {{'{first_item}': {first_samples}}}")
    

if __name__ == "__main__":
    main()
