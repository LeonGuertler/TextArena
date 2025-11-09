"""
Renumber train.csv files from week 0-9 to week 1-10
"""
import pandas as pd
import os
import shutil

# List of all real instances
instances = [
    '1047675', '168927', '168989', '172343', '279137',
    '521818', '527757', '827911', '864511', '938576'
]

base_dir = 'real_instances_50_weeks'

print("="*70)
print("Renumbering train.csv files: week 0-9 → week 1-10")
print("="*70)

for instance in instances:
    train_path = os.path.join(base_dir, instance, 'train.csv')
    train_temp = os.path.join(base_dir, instance, 'train_temp.csv')
    
    if not os.path.exists(train_path):
        print(f"✗ {instance}: train.csv not found")
        continue
    
    try:
        # Read the file
        df = pd.read_csv(train_path)
        
        # Verify current week_number range
        current_min = df['week_number'].min()
        current_max = df['week_number'].max()
        
        if current_min != 0 or current_max != 9:
            print(f"⚠ {instance}: Unexpected week range {current_min}-{current_max}, skipping")
            continue
        
        # Renumber: add 1 to all week_number values
        df['week_number'] = df['week_number'] + 1
        
        # Verify new range
        new_min = df['week_number'].min()
        new_max = df['week_number'].max()
        
        # Save to temporary file
        df.to_csv(train_temp, index=False)
        
        # Replace original file
        shutil.move(train_temp, train_path)
        
        print(f"✓ {instance}: Renumbered {current_min}-{current_max} → {new_min}-{new_max}")
    
    except Exception as e:
        print(f"✗ {instance}: Error - {e}")
        # Clean up temp file if it exists
        if os.path.exists(train_temp):
            os.remove(train_temp)

print("\n" + "="*70)
print("Renumbering completed!")
print("="*70)
