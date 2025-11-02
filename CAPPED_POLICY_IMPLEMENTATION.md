# Capped Base Stock Policy Implementation

## Overview

This document describes the implementation of the **capped base stock policy** across all OR-based scripts in the TextArena project. The capped policy addresses order volatility issues when lead times vary daily.

## What's New

### 1. Capped Base Stock Formula

The capped policy limits order quantities to reduce variance:

```
order = max(min(base_stock - current_inventory, cap), 0)
```

Where the cap is computed as:

```
cap = μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
```

- **μ̂/(1+L)**: Single-period expected demand (dividing back from lead-time scaled value)
- **σ̂/√(1+L)**: Single-period standard deviation (dividing back from lead-time scaled value)
- **Φ^(-1)(0.95)**: 95th percentile z-score (fixed at 0.95, hardcoded)

### 2. Policy Selection

All four scripts now support policy selection via command line:

```bash
--policy vanilla   # Standard base-stock policy (uncapped)
--policy capped    # Capped base-stock policy (default)
```

Default: **capped** (as specified)

## Updated Scripts

### 1. `or_csv_demo.py`
- **ORAgent class**: Added `policy` parameter to `__init__`
- **_calculate_order**: Now returns dict with comprehensive statistics
- **get_action**: Returns tuple `(action_json, statistics_dict)`
- **Main loop**: Displays detailed parameter logs for each day

### 2. `or_to_llm_csv_demo.py`
- **ORAgent class**: Added `policy` parameter for OR recommendations
- **_calculate_order**: Returns dict with full calculation details
- **get_recommendation**: Returns tuple `(recommendations, statistics)`
- **Main loop**: Logs OR statistics before LLM makes final decision

### 3. `llm_to_or_csv_demo.py`
- **Backend computation**: Applies capped/vanilla policy after LLM proposes parameters
- **Order computation**: Shows cap value and uncapped order (when using capped policy)
- **Main loop**: Comprehensive logging of all LLM-proposed and computed values

### 4. `clairvoyant_or_csv.py`
- **ClairvoyantORAgent class**: Added `policy` parameter
- **_calculate_order**: Returns dict with statistics (uses schedule-provided μ̂/σ̂)
- **get_action**: Returns tuple `(action_json, statistics_dict)`
- **Main loop**: Detailed logging of clairvoyant parameters and orders

## Comprehensive Logging

All scripts now log the following parameters **for each item, each day**:

### Common Parameters (all scripts):
- **empirical_mean** / **empirical_std** (or schedule values for clairvoyant)
- **mu_hat (μ̂)**: Scaled expected demand over lead time
- **sigma_hat (σ̂)**: Scaled standard deviation over lead time
- **L**: Lead time used in calculation
- **q**: Critical fractile (profit / (profit + holding_cost))
- **z\***: Corresponding z-score (Φ^(-1)(q))
- **base_stock**: Target inventory position
- **current_inventory**: On-hand + in-transit

### Policy-Specific Logging:

**Vanilla Policy:**
- **Order**: Final order quantity

**Capped Policy:**
- **Cap value**: Upper bound on order quantity
- **Order (capped)**: Final order quantity (after applying cap)
- **Order (uncapped)**: What vanilla policy would have ordered

## Usage Examples

### 1. Run OR baseline with capped policy (default):
```bash
python examples/or_csv_demo.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --promised-lead-time 0
```

### 2. Run OR baseline with vanilla policy:
```bash
python examples/or_csv_demo.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --promised-lead-time 0 \
    --policy vanilla
```

### 3. Run LLM→OR with capped policy:
```bash
python examples/llm_to_or_csv_demo.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --promised-lead-time 0 \
    --policy capped
```

### 4. Run OR→LLM hybrid with capped OR recommendations:
```bash
python examples/or_to_llm_csv_demo.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --promised-lead-time 0 \
    --policy capped
```

### 5. Run clairvoyant baseline with capped policy:
```bash
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --instance 1 \
    --policy capped
```

**Note:** For clairvoyant OR, lead time L is **hard-coded** for each instance:
- Instances 1-4: L = 0 (always)
- Instances 5, 6, 8: L = 4 (always)
- Instance 7: L = 4 (days 1-15), L = 6 (days 16+)

## Key Design Decisions

1. **Cap quantile fixed at 0.95**: As specified, this is hardcoded. Change manually in code if needed.

2. **Default policy is capped**: All scripts default to `--policy capped` to use the smoothed control by default.

3. **Backward compatible**: Vanilla policy preserves original OR behavior exactly.

4. **Comprehensive logging**: Every parameter value is now logged each day for both policies, making it easy to understand agent decisions and debug issues.

5. **Consistent implementation**: All four scripts use identical capped policy formula and logging structure.

## Benefits of Capped Policy

1. **Reduced order volatility**: Limits extreme orders when base stock calculations produce high values
2. **Smoother control**: Less sensitive to sudden changes in lead time or demand estimates
3. **Better for variable lead times**: Particularly effective when actual lead time differs from promised/estimated
4. **Interpretable cap**: Cap represents "reasonable single-period order" based on 95th percentile

## Output Format

### Example Detailed Log (Capped Policy):

```
======================================================================
Day 1 OR Decision (CAPPED Policy):
======================================================================

chips(Regular):
  Empirical mean: 91.70
  Empirical std: 26.18
  Lead time (L): 0
  mu_hat (μ̂): 91.70
  sigma_hat (σ̂): 26.18
  Critical fractile (q): 0.6667
  z*: 0.4307
  Base stock: 103.00
  Current inventory: 0
  Cap value: 135.00
  Order (capped): 103
  Order (uncapped): 103

{
  "action": {
    "chips(Regular)": 103
  },
  "rationale": "OR base-stock (capped): chips(Regular): base_stock=103.0, cap=135.0, inv=0, order=103 (uncapped would be 103)"
}
======================================================================
```

## Testing

Test both policies on scenarios with:
- ✅ Stationary demand (should perform similarly)
- ✅ Variable lead times (capped should be smoother)
- ✅ Sudden demand shifts (capped limits overreaction)
- ✅ High variance demand (capped prevents extreme orders)

All scripts are ready to run and compare vanilla vs capped performance!

