import pandas as pd

instances = ['1047675', '168927', '168989', '172343', '279137', 
             '521818', '527757', '827911', '864511', '938576']

print("\n" + "="*70)
print("Verification: All train.csv files")
print("="*70)

all_ok = True
for instance in instances:
    try:
        df = pd.read_csv(f'real_instances_50_weeks/{instance}/train.csv')
        min_week = df['week_number'].min()
        max_week = df['week_number'].max()
        num_rows = len(df)
        
        status = "✓" if (min_week == 1 and max_week == 10 and num_rows == 10) else "✗"
        if status == "✗":
            all_ok = False
            
        print(f"{status} {instance}: weeks {min_week}-{max_week} ({num_rows} rows)")
    except Exception as e:
        print(f"✗ {instance}: Error - {e}")
        all_ok = False

print("="*70)
if all_ok:
    print("✓ All train.csv files verified successfully!")
else:
    print("⚠ Some files have issues!")
print("="*70)
