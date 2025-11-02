# Clairvoyant OR: Hard-Coded Lead Time Setup

## Overview

The clairvoyant OR agent now uses **hard-coded lead time values** for each instance, matching the implicit lead times encoded in the clairvoyant schedules. This ensures the capped policy correctly extracts single-period demand characteristics.

## Hard-Coded Lead Time Values

| Instance | Lead Time (L) | Schedule μ̂ | Schedule σ̂ | Description |
|----------|---------------|-------------|-------------|-------------|
| **1** | **L = 0** (always) | 100 | 25 | Stationary |
| **2** | **L = 0** (always) | 100→200 | 25→25√2 | Mean shift at day 16 |
| **3** | **L = 0** (always) | 100·t | 25√t | Increasing trend |
| **4** | **L = 0** (always) | 100→100 | 25→50 | Variance increase at day 16 |
| **4_1** | **L = 0** (always) | 100→100 | 25→0 | Variance decrease to zero at day 16 |
| **5** | **L = 4** (always) | 500 | 25√5 ≈ 55.9 | Long lead time, stationary |
| **6** | **L = 4** (always) | 500 | 25√5 ≈ 55.9 | Random/stochastic lead time |
| **7** | **L = 4** (days 1-15)<br>**L = 6** (days 16+) | 500→700 | 25√5→25√7 | Lead time shift at day 16 |
| **8** | **L = 4** (always) | 511 | 25√5.11 ≈ 56.6 | Intermittent supplier |

## Implementation Details

### Code Location

The lead time determination is in `ClairvoyantORAgent._get_lead_time()`:

```python
def _get_lead_time(self) -> float:
    """Get lead time L for current instance and day."""
    instance = self.schedule.instance_id
    
    if instance in ["1", "2", "3", "4", "4_1"]:
        return 0.0
    elif instance in ["5", "6", "8"]:
        return 4.0
    elif instance == "7":
        # Days 1-15: L=4, Days 16+: L=6
        return 4.0 if self.current_day <= 15 else 6.0
    else:
        return 0.0  # Fallback
```

### Key Changes

1. **Removed `--promised-lead-time` argument**: No longer needed since L is hard-coded
2. **Dynamic L for Instance 7**: Switches from L=4 to L=6 on day 16
3. **Automatic L selection**: Agent automatically uses correct L based on instance and day

## Mathematical Consistency

The hard-coded L values match the implicit lead times in the schedules:

### Example: Instance 5
**Schedule provides:**
```python
μ̂ = 500, σ̂ = 25√5 ≈ 55.9
```

**These represent:**
```python
μ̂ = (1+L) × empirical_mean = (1+4) × 100 = 500  ✓
σ̂ = √(1+L) × empirical_std = √5 × 25 = 55.9    ✓
```

**Capped policy extracts single-period values:**
```python
cap = μ̂/(1+L) + Φ⁻¹(0.95) × σ̂/√(1+L)
cap = 500/5 + 1.645 × 55.9/√5
cap = 100 + 1.645 × 25 = 141.1
```

## Usage

Simply specify the instance identifier - L is handled automatically:

```bash
# Instance 1 (L=0, stationary)
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case1_iid_normal.csv \
    --instance 1 \
    --policy capped

# Instance 4_1 (L=0, variance → 0)
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case4_1_variance_decrease_cp15.csv \
    --instance 4_1 \
    --policy capped

# Instance 5 (L=4, stationary)
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case5_L_const4.csv \
    --instance 5 \
    --policy capped

# Instance 7 (L=4→6 at day 16)
python examples/clairvoyant_or_csv.py \
    --demand-file examples/demand_case7_L_shift_cp15.csv \
    --instance 7 \
    --policy capped
```

## Capped Policy Behavior

### Instances 1-4, 4_1 (L=0)
```python
cap = μ̂/(1+0) + 1.645 × σ̂/√(1+0)
cap = μ̂ + 1.645 × σ̂
```
Since L=0, the cap is based directly on the schedule values.

**Special case for Instance 4_1 after day 15:**
```python
σ̂ = 0  (variance becomes zero)
cap = 100 + 1.645 × 0 = 100
```
When variance is zero, the cap equals the mean demand (deterministic scenario). The capped policy effectively limits orders to the expected demand since there's no uncertainty.

### Instances 5-8 (L=4)
```python
cap = μ̂/5 + 1.645 × σ̂/√5
```
Divides by 5 and √5 to extract single-period estimates.

### Instance 7 (L changes)
- **Days 1-15**: `cap = μ̂/5 + 1.645 × σ̂/√5`
- **Days 16+**: `cap = μ̂/7 + 1.645 × σ̂/√7`

The cap adjusts automatically when L changes, reflecting the new single-period characteristics.

## Logging Output

The agent will display the hard-coded L value:

```
=== Clairvoyant OR Agent Initialized (Policy: CAPPED) ===
Schedule: Instance 7: shift to μ̂=700, σ̂=25√7 after day 15.
Instance 7: Lead time L is hard-coded per instance specification

...

Lead time L is hard-coded for each instance:
  L = 4 (days 1-15), L = 6 (days 16+)
```

Daily logs will show the L value used:

```
Day 16 Clairvoyant OR Decision (CAPPED Policy):

chips(Regular):
  mu_hat (μ̂) from schedule: 700.00
  sigma_hat (σ̂) from schedule: 66.14
  Lead time (L): 6           ← Changed from 4!
  Critical fractile (q): 0.6667
  z*: 0.4307
  Base stock: 728.49
  Current inventory: 450
  Cap value: 140.88          ← Adjusted for L=6
  Order (capped): 140
  Order (uncapped): 279
```

## Benefits

1. **Automatic correctness**: L always matches the schedule's implicit lead time
2. **No user error**: Can't accidentally use wrong L for an instance
3. **Dynamic adaptation**: Instance 7 automatically switches L at day 16
4. **Cleaner interface**: One less argument to specify
5. **Mathematical consistency**: Cap calculation always uses correct divisor

## Summary

The hard-coded lead time setup ensures that:
- Each instance uses the L value that matches its schedule design
- The capped policy correctly extracts single-period demand characteristics
- Instance 7 dynamically adapts when the lead time changes
- Users don't need to remember or specify L values manually

This makes the clairvoyant OR experiments more robust and easier to run! 🎯

