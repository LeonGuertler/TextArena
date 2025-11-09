# Real Instances Re-extraction Summary

## Problem

The original extraction of the 10 real instances used:
- **Training data**: weeks 0-9 from original CSV (10 weeks)
- **Test data**: weeks 10-59 from original CSV, mapped to weeks 1-50 (50 weeks)

**Issue**: Week 0 in the original CSV files was always invalid (demand was 0 or missing data), which meant the training data included bad data that could negatively impact algorithm initialization.

## Solution

Re-extracted all 10 real instances to exclude week 0:
- **Training data**: weeks 1-10 from original CSV (10 weeks) ✅
- **Test data**: weeks 11-60 from original CSV, mapped to weeks 1-50 (50 weeks) ✅

This ensures:
1. Week 0 (invalid data) is completely excluded
2. Training uses 10 weeks of valid data (weeks 1-10)
3. Test data uses 50 weeks of data (weeks 11-60, mapped to 1-50)
4. No overlap between training and test data
5. All data is valid and usable for algorithm training

## Implementation

### Script Created

**`re_extract_real_instances.py`** - Main re-extraction script
- Reads original CSV files from `d:\OR agent docs\favorita data\output_csv\`
- Extracts weeks 1-10 for training
- Extracts weeks 11-60 for testing (mapped to 1-50)
- Applies same transformations as original extraction:
  - Merges holiday + weeks_to_christmas into news column
  - Adds lead_time, profit, holding_cost columns
  - Renames columns with item_id suffix for test data
  
### Verification Scripts

1. **`verify_train_weeks.py`** - Verifies all train.csv files
   - Checks week range is 1-10
   - Confirms 10 rows per file
   
2. **`verify_test_weeks.py`** - Verifies all test.csv files
   - Checks week range is 1-50
   - Confirms 50 rows per file
   - Validates column structure

3. **`verify_extraction_shift.py`** - Shows the data shift comparison
   - Displays sample data from multiple instances
   - Confirms no overlap between train and test

4. **`show_final_data.py`** - Shows final extracted data
   - Displays complete training data
   - Shows sample test data

## Execution Results

All 10 instances successfully re-extracted:

✅ **1047675** - BEVERAGES
✅ **168927** - CLEANING  
✅ **168989** - GROCERY I
✅ **172343** - BREAD/BAKERY
✅ **279137** - PRODUCE
✅ **521818** - DAIRY
✅ **527757** - DELI
✅ **827911** - POULTRY
✅ **864511** - MEATS
✅ **938576** - GROCERY I

### Verification Status

```
✓ All 10 train.csv files: weeks 1-10 (10 rows each)
✓ All 10 test.csv files: weeks 1-50 (50 rows each)
✓ Demo script test successful
```

## Data Format

### Training Data (train.csv)

```csv
week_number,demand,description,holiday,weeks_to_christmas
1,56.0,BEVERAGES,,51
2,63.0,BEVERAGES,,50
...
10,48.0,BEVERAGES,,42
```

- **Rows**: 10 (weeks 1-10)
- **Columns**: week_number, demand, description, holiday, weeks_to_christmas

### Test Data (test.csv)

```csv
week,demand_1047675,description_1047675,lead_time_1047675,profit_1047675,holding_cost_1047675,news
1,44.0,BEVERAGES,4,2.0,1.0,41 weeks to Christmas
2,41.0,BEVERAGES,4,2.0,1.0,40 weeks to Christmas
...
50,46.0,BEVERAGES,4,2.0,1.0,44 weeks to Christmas
```

- **Rows**: 50 (weeks 1-50, from original weeks 11-60)
- **Columns**: week, demand_{item_id}, description_{item_id}, lead_time_{item_id}, profit_{item_id}, holding_cost_{item_id}, news

## Testing

Tested with `or_csv_demo.py`:
```bash
uv run python or_csv_demo.py --demand-file real_instances_50_weeks/1047675/test.csv --real-instance-train 1047675
```

✅ **Result**: Script ran successfully
- Loaded 10 weeks of training data (weeks 1-10)
- Executed 50 weeks of test data (weeks 1-50)
- Total reward: $1490.00

## Impact on Existing Code

✅ **No code changes required** - All test scripts already compatible:
- `or_csv_demo.py` 
- `llm_csv_demo.py`
- `llm_to_or_csv_demo.py`
- `or_to_llm_csv_demo.py`

The filter condition `train_df[train_df['week_number'] >= 1]` already handles the new week range correctly.

## Key Differences: Old vs New

| Aspect | Old Extraction | New Extraction |
|--------|---------------|----------------|
| Train weeks | 0-9 (includes invalid week 0) | 1-10 (excludes invalid week 0) |
| Test weeks | 10-59 (from original) | 11-60 (from original) |
| Test mapped to | 1-50 | 1-50 |
| Valid training data | 9 weeks (week 0 invalid) | 10 weeks (all valid) |

## Next Steps

1. ✅ Re-extraction complete
2. ✅ Verification successful  
3. ✅ Demo script tested
4. Ready for production use

All 10 real instances now have valid training data and are ready for algorithm testing and benchmarking.
