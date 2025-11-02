# Instance 4_1: Variance Decrease to Zero

## Overview

Instance **4_1** is a sub-case of Instance 4, designed to test the capped OR policy's behavior when demand variance decreases to zero (deterministic demand).

## Specification

| Property | Value | Description |
|----------|-------|-------------|
| **Instance ID** | `4_1` | String identifier (sub-case of instance 4) |
| **Lead Time** | L = 0 | Always zero (like instance 4) |
| **Days 1-15** | μ̂ = 100, σ̂ = 25 | Same as instance 4 initially |
| **Days 16+** | μ̂ = 100, σ̂ = 0 | **Variance drops to zero** |

## Mathematical Definition

### Schedule Function

```python
def _schedule_instance4_1(day: int, _item_id: str) -> tuple[float, float]:
    if day <= 15:
        return 100.0, 25.0  # Normal variance
    return 100.0, 0.0       # Zero variance (deterministic)
```

### Comparison with Instance 4

| Metric | Instance 4 | Instance 4_1 |
|--------|-----------|--------------|
| Days 1-15 | μ̂=100, σ̂=25 | μ̂=100, σ̂=25 (same) |
| Days 16+ | μ̂=100, σ̂=50 | μ̂=100, σ̂=0 |
| Change | Variance **doubles** | Variance → **zero** |

## Capped Policy Behavior

### Days 1-15 (Normal Variance)
```python
cap = 100/(1+0) + 1.645 × 25/√(1+0)
cap = 100 + 41.125
cap = 141.1 units
```

### Days 16+ (Zero Variance)
```python
cap = 100/(1+0) + 1.645 × 0/√(1+0)
cap = 100 + 0
cap = 100 units
```

**Key insight:** When σ̂ = 0 (deterministic demand), the cap equals exactly the expected demand. There's no safety stock component since demand is certain.

## Usage

```bash
# Run instance 4_1 with capped policy
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case4_1_variance_decrease_cp15.csv \
    --instance 4_1 \
    --policy capped

# Compare with instance 4 (variance increase)
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case4_normal_to_uniform_cp15.csv \
    --instance 4 \
    --policy capped
```

## Expected Behavior

### Vanilla OR Policy
- **Days 1-15**: Orders based on base-stock with z* safety stock
- **Days 16+**: With σ̂=0, base-stock = μ̂ + z*×0 = 100
  - No safety stock needed (demand is deterministic)
  - Orders should stabilize around 100 units

### Capped OR Policy
- **Days 1-15**: Similar to vanilla (cap = 141.1 is usually non-binding)
- **Days 16+**: Cap = 100 units
  - Prevents any orders above expected demand
  - Optimal for deterministic scenario
  - Should match vanilla policy exactly (both order 100)

## Test Scenario Purpose

Instance 4_1 tests:

1. **Variance decrease handling**: How does OR adapt when uncertainty disappears?
2. **Deterministic limit case**: Performance when σ̂ → 0
3. **Capped policy effectiveness**: Does the cap correctly reflect zero uncertainty?
4. **Comparison with instance 4**: Symmetric test (increase vs. decrease)

## Mathematical Insight

When demand becomes deterministic (σ̂ = 0):

```python
Base stock = μ̂ + z* × 0 = μ̂
```

The optimal policy is simply to order the expected demand. The capped policy's cap:

```python
cap = μ̂/(1+L) + Φ^(-1)(0.95) × 0 = μ̂/(1+L)
```

For L=0: `cap = μ̂ = 100`

This perfectly captures the deterministic nature: no safety stock, just order the known demand.

## Logging Output

Example output for Day 16+ with instance 4_1:

```
Day 20 Clairvoyant OR Decision (CAPPED Policy):

chips(Regular):
  mu_hat (μ̂) from schedule: 100.00
  sigma_hat (σ̂) from schedule: 0.00    ← Zero variance!
  Lead time (L): 0
  Critical fractile (q): 0.6667
  z*: 0.4307
  Base stock: 100.00                    ← No safety stock
  Current inventory: 95
  Cap value: 100.00                     ← Cap = expected demand
  Order (capped): 5
  Order (uncapped): 5                   ← Same (not binding)
```

Notice:
- σ̂ = 0 indicates deterministic demand
- Base stock = 100 (just the mean, no safety stock)
- Cap = 100 (single-period expected demand)
- Orders are very stable (just replenish to expected demand)

## Summary

Instance 4_1 is a clean test case for:
- Handling deterministic demand (σ̂ = 0)
- Testing capped policy in zero-variance scenario
- Comparing variance increase (instance 4) vs. decrease (instance 4_1)
- Validating that OR policies correctly handle the deterministic limit case

The naming "4_1" indicates it's a sub-case of instance 4, making it easy to remember they're related but opposite variance changes! 🎯

