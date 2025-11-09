# Weekly Time Scale Migration

## Overview
Migrated all test scripts from **day-based** to **week-based** time scale. This applies to both synthetic test cases and real retail instances.

## Date
November 7, 2025

## Changes Made

### 1. Core Environment (textarena/envs/VendingMachine/env.py)
**Modified game output format:**
- `Day X concluded:` → `Week X concluded:`
- `ordered on Day Y, lead_time was Z days` → `ordered on Week Y, lead_time was Z weeks`

### 2. Test Scripts - LLM Prompts

#### 2.1 llm_csv_demo.py
**System Prompt Changes:**
- Objective: `daily rewards` → `weekly rewards`
- Reward formula: `Daily reward` → `Weekly reward`
- Lead time: `X days` → `X weeks`
- Time references: `today` → `this week`, `Day 1` → `Week 1`
- Holding cost: `/unit/day` → `/unit/week`
- News timing: `each day` → `each week`

**NEWS INFORMATION Section Added:**
```
- The 'news' column may contain important contextual information:
  1. Holiday/Event information: Special events that may affect demand
     Examples: 'Holiday, National', 'Additional, National; Event, National'
  2. Weeks to Christmas: Countdown to Christmas season
     Format: 'X weeks to Christmas' or 'Holiday, National (X weeks to Christmas)'
- You must analyze whether these events correlate with demand changes
- Christmas proximity may influence consumer buying patterns
- Not all holidays/events necessarily impact demand - use historical data to assess
- If no news is present for a week, the field will be empty
```

**Carry-over Insights Examples:**
- `Day 5` → `Week 5`
- `3+ days` → `3+ weeks`
- `Days 8-12 avg` → `Weeks 8-12 avg`

**Strategy Section:**
- `TODAY'S NEWS` → `THIS WEEK'S NEWS`
- `look for 'lead_time was X days'` → `look for 'lead_time was X weeks'`

**Comments & Messages:**
- `Sort insights by day number` → `Sort insights by week number`
- `Day {num}:` → `Week {num}:`
- `Loaded CSV with X days of demand` → `Loaded CSV with X weeks of demand`
- `News scheduled for days:` → `News scheduled for weeks:`
- Function doc: `Day number` → `Week number`

#### 2.2 llm_to_or_csv_demo.py
**All changes from llm_csv_demo.py PLUS:**

**OR Parameter Strategy Section:**
- `sustained over 3+ days` → `sustained over 3+ weeks`
- `Day ranges` → `Week ranges`
- `current_day - changepoint_day` → `current_week - changepoint_week`
- Examples: `Day 15` → `Week 15`, `current Day 20` → `current Week 20`

**Strategy Section:**
- `Consider today's news` → `Consider this week's news`

**Pattern Matching:**
- Regex pattern: `lead_time was (\d+) day` → `lead_time was (\d+) week`
- Comments: `ordered on Day Z` → `ordered on Week Z`

#### 2.3 or_to_llm_csv_demo.py
**Similar changes to llm_csv_demo.py:**
- All prompt day/week conversions
- NEWS INFORMATION section added
- Carry-over insights examples updated
- Strategy section timing references

**Example Decision Process:**
- `Step 1: lead_time was 2 days` → `lead_time was 2 weeks`
- `Step 3: TODAY'S NEWS` → `THIS WEEK'S NEWS`
- `Step 4: in 2 days (lead_time!)` → `in 2 weeks (lead_time!)`

**OR Algorithm Description:**
- `each day` → `each week`
- `PROMISED lead time (X days)` → `PROMISED lead time (X weeks)`
- `normal days` → `normal weeks`

### 3. Removed is_real_instance Parameter
**Rationale:** Since ALL instances now use weekly time scale, conditional terminology is no longer needed.

**Modified Functions:**
- `make_vm_agent()` in llm_csv_demo.py
- `make_llm_to_or_agent()` in llm_to_or_csv_demo.py
- `make_hybrid_vm_agent()` in or_to_llm_csv_demo.py

### 4. Pattern Constants
**Updated regex patterns:**
- `DAY_CONCLUDED_PATTERN` → `WEEK_CONCLUDED_PATTERN`
- Pattern: `r'^(\s*Day\s+(\d+)\s+concluded:)(.*)$'` → `r'^(\s*Week\s+(\d+)\s+concluded:)(.*)$'`

## Impact

### Test Data
- **Synthetic cases**: Now interpreted as week-based (weeks 1-50 instead of days 1-50)
- **Real instances**: Already week-based, now consistent with prompt terminology

### LLM Behavior
- LLM will now think in terms of weeks instead of days
- News interpretation aligned with weekly cycle
- Lead time reasoning adjusted to weekly intervals

### CSV Format
- Column `day` still used internally (code variable names unchanged)
- Semantic meaning: represents week number
- No CSV file structure changes needed

## Testing Required

Before production use, verify:
1. LLM correctly interprets weekly time scale
2. News events properly influence weekly demand predictions
3. Lead time calculations work correctly (weeks instead of days)
4. Carry-over insights reference correct week numbers
5. All 4 test scripts produce expected output format

## Files Modified

### Core Files (1)
- `textarena/envs/VendingMachine/env.py`

### Test Scripts (3)
- `examples/llm_csv_demo.py`
- `examples/llm_to_or_csv_demo.py`
- `examples/or_to_llm_csv_demo.py`

### Documentation (1)
- `WEEKLY_TIMESCALE_MIGRATION.md` (this file)

## Rollback Plan

If issues arise:
1. Revert environment file: `git checkout textarena/envs/VendingMachine/env.py`
2. Revert test scripts: `git checkout examples/*_csv_demo.py`
3. Restore `is_real_instance` parameter logic if needed

## Notes

- **No changes to or_csv_demo.py**: This script does not use LLM, so prompt modifications not needed
- **Variable names**: Internal code still uses `day` variable names for compatibility
- **Backward compatibility**: Existing CSV files work without modification
- **News format**: Assumes news column format: "Holiday/Event info (X weeks to Christmas)"
