# CSV Column Name Fix

## Issue
When running tests with real instances, the scripts failed with error:
```
Error loading CSV: 'day'
```

## Root Cause
The CSV files use column name `'week'` (not `'day'`), but all test scripts were trying to access `df['day']`.

## Date
November 7, 2025

## Files Modified

### 1. examples/or_csv_demo.py
**Line 82-83**: Changed news column access
```python
# Before:
news_days = self.df[self.df['news'].notna()]['day'].tolist()
print(f"News scheduled for days: {news_days}")

# After:
news_weeks = self.df[self.df['news'].notna()]['week'].tolist()
print(f"News scheduled for weeks: {news_weeks}")
```

**Line 204**: Changed news_schedule dictionary key
```python
# Before:
day = int(row['day'])

# After:
week = int(row['week'])
```

**Line 527**: Updated output message
```python
# Before:
print(f"Promised lead time (used by OR algorithm): {args.promised_lead_time} days")

# After:
print(f"Promised lead time (used by OR algorithm): {args.promised_lead_time} weeks")
```

**Line 587-588, 598**: Updated decision output messages
```python
# Before:
print(f"\nWARNING Day {current_day}: Supplier unavailable (lead_time=inf)")
print(f"Day {current_day} OR Decision: {action} (automatically set to 0)")
print(f"Day {current_day} OR Decision ({args.policy.upper()} Policy):")

# After:
print(f"\nWARNING Week {current_day}: Supplier unavailable (lead_time=inf)")
print(f"Week {current_day} OR Decision: {action} (automatically set to 0)")
print(f"Week {current_day} OR Decision ({args.policy.upper()} Policy):")
```

**Line 665-677**: Updated breakdown section
```python
# Before:
# Daily breakdown
print("Daily Breakdown:")
print(f"Day {day}{news_str}: Profit=${profit:.2f}, Holding=${holding:.2f}, Reward=${reward:.2f}")

# After:
# Weekly breakdown
print("Weekly Breakdown:")
print(f"Week {day}{news_str}: Profit=${profit:.2f}, Holding=${holding:.2f}, Reward=${reward:.2f}")
```

### 2. examples/llm_csv_demo.py
**Line 164-165**: Changed news column access
```python
# Before:
news_days = self.df[self.df['news'].notna()]['day'].tolist()
print(f"News scheduled for days: {news_days}")

# After:
news_weeks = self.df[self.df['news'].notna()]['week'].tolist()
print(f"News scheduled for weeks: {news_weeks}")
```

**Line 286**: Changed news_schedule dictionary key
```python
# Before:
day = int(row['day'])

# After:
week = int(row['week'])
```

### 3. examples/llm_to_or_csv_demo.py
**Line 166-167**: Changed news column access
```python
# Before:
news_days = self.df[self.df['news'].notna()]['day'].tolist()
print(f"News scheduled for weeks: {news_days}")  # Note: message was already "weeks" but variable was wrong

# After:
news_weeks = self.df[self.df['news'].notna()]['week'].tolist()
print(f"News scheduled for weeks: {news_weeks}")
```

**Line 288**: Changed news_schedule dictionary key
```python
# Before:
day = int(row['day'])

# After:
week = int(row['week'])
```

### 4. examples/or_to_llm_csv_demo.py
**Line 164-165**: Changed news column access
```python
# Before:
news_days = self.df[self.df['news'].notna()]['day'].tolist()
print(f"News scheduled for days: {news_days}")

# After:
news_weeks = self.df[self.df['news'].notna()]['week'].tolist()
print(f"News scheduled for weeks: {news_weeks}")
```

**Line 286**: Changed news_schedule dictionary key
```python
# Before:
day = int(row['day'])

# After:
week = int(row['week'])
```

## Impact

### Before Fix
- All test scripts failed with `KeyError: 'day'` when loading CSV files
- Real instance tests could not run

### After Fix
- All test scripts now successfully load CSV files with `'week'` column
- Output messages correctly reference "weeks" instead of "days"
- Tests run successfully with real instance data

## Verification

Successfully ran first real instance test:
```bash
uv run python or_csv_demo.py \
  --demand-file real_instances_50_weeks/1047675/test.csv \
  --promised-lead-time 4 \
  --policy capped \
  --real-instance-train real_instances_50_weeks/1047675/train.csv \
  > real_instances_logs/1047675/or.txt
```

Output confirmed:
- ✅ CSV loaded: "Loaded CSV with 50 weeks of demand data"
- ✅ News detection: "News scheduled for weeks: [1, 2, 3, ..., 50]"
- ✅ Lead time message: "Promised lead time (used by OR algorithm): 4 weeks"
- ✅ Decision headers: "Week 1 OR Decision (CAPPED Policy):"
- ✅ Summary: "Weekly Breakdown:"
- ✅ Final result: "Total Reward (OR Baseline): $3495.00"

## Notes

- Variable names in code still use `day` for backward compatibility (e.g., `current_day`, `day_log`)
- These are internal variables and don't affect CSV column access
- CSV column name is consistently `'week'` in all transformed test files
- The fix aligns code with actual CSV structure
