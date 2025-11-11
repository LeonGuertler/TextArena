# Lead Time = Infinity Behavior Update

## Problem Statement

Previously, when `lead_time=inf` (supplier unavailable):
- ❌ The test scripts would **detect** this condition
- ❌ Skip the agent's decision entirely
- ❌ Force `order=0` automatically
- ❌ Print **WARNING** messages telling the agent about the issue
- ❌ The environment would also reject orders and print warnings

This meant the agent **knew** about supply chain disruptions before making decisions.

## New Behavior

Now, when `lead_time=inf`:
- ✅ The agent **does NOT know** about the supply issue
- ✅ The agent can place orders normally
- ✅ Orders are **accepted** but will **never arrive** (silently lost)
- ✅ No warnings are printed to the agent
- ✅ The agent must **discover** the problem from observing that shipments don't arrive

This creates a more realistic scenario: **supply chain disruptions happen without warning**.

## Implementation Changes

### 1. Environment (`textarena/envs/VendingMachine/env.py`)

**Before:**
```python
# Check if lead_time is infinite - cannot restock
if lead_time == float('inf'):
    self.current_day_orders[item_id] = 0  # Force to 0
    if qty > 0:
        print(f"⚠️  Day {self.current_day}: Cannot restock {item_id} (lead_time=inf). Order forced to 0 (requested: {qty})")
```

**After:**
```python
# Record the order (even if lead_time=inf)
self.current_day_orders[item_id] = qty

if qty > 0:
    # If lead_time is infinite, the order is accepted but never arrives
    # (Simulates order getting lost without agent knowing)
    if lead_time == float('inf'):
        # Don't add to pending_orders - order gets lost
        # Don't print warning - agent should not know about the issue
        pass
    else:
        # Normal processing: add to pending_orders
        ...
```

**Key changes:**
- Accept the order value (don't force to 0)
- Don't add to `pending_orders` (order never arrives)
- Don't print warnings (agent unaware)

### 2. Test Scripts

Modified all 3 LLM-based test scripts:
- `llm_csv_demo.py`
- `llm_to_or_csv_demo.py`
- `or_to_llm_csv_demo.py`

**Before:**
```python
has_inf_lead_time = False
for item_id in csv_player.get_item_ids():
    config = csv_player.get_day_item_config(current_day, item_id)
    # ... update config ...
    if config['lead_time'] == float('inf'):
        has_inf_lead_time = True

# If supplier unavailable (lead_time=inf), skip VM decision
if has_inf_lead_time:
    zero_orders = {item_id: 0 for item_id in csv_player.get_item_ids()}
    action = json.dumps({"action": zero_orders}, indent=2)
    print(f"\nWARNING Day {current_day}: Supplier unavailable (lead_time=inf)")
    print("VM did not place order (automatically set to 0)")
    done, _ = env.step(action=action)
    continue

action = agent(observation)  # Only reached if no inf lead time
```

**After:**
```python
for item_id in csv_player.get_item_ids():
    config = csv_player.get_day_item_config(current_day, item_id)
    # ... update config ...
    # No checking for lead_time=inf

# Get agent action (even if lead_time=inf - agent doesn't know about supply issues)
action = agent(observation)
```

**Key changes:**
- Removed `has_inf_lead_time` detection
- Removed skip logic and forced zero orders
- Removed WARNING messages
- Agent always makes decisions normally

## Testing Scenario

**CSV File with lead_time=inf:**
```csv
week,demand_cola,lead_time_cola,...
1,50,4,...
2,60,4,...
3,55,inf,...  ← Supplier unavailable (week 3)
4,65,4,...
5,70,inf,...  ← Supplier unavailable (week 5-6)
6,55,inf,...
7,60,4,...
```

**Agent Experience:**
1. **Week 1-2**: Normal operations, orders arrive as expected
2. **Week 3**: Agent places order, but it **never arrives**
   - Agent observes: inventory depletes, no shipment arrives
   - Agent might think: "Where's my order? Is there a delay?"
3. **Week 4**: Agent can order again (supply restored)
4. **Week 5-6**: Two consecutive weeks of supply disruption
   - Agent must adapt to longer period without resupply

**Agent Discovery Process:**
- Agent must infer supply problems from **observations**
- No explicit warnings or notifications
- Must track: "I ordered X units Y days ago, why haven't they arrived?"
- Can learn to detect patterns or react to prolonged stockouts

## Impact on OR Baseline

The OR baseline (`or_csv_demo.py`) was **NOT modified** because:
- It doesn't check for `lead_time=inf` beforehand
- It simply calculates orders based on historical data
- Orders will be accepted but lost (same as LLM agents)
- Provides fair comparison: all algorithms face same "unknown disruptions"

## Example Output

**Before (with warnings):**
```
WARNING Day 7: Supplier unavailable (lead_time=inf)
VM did not place order (automatically set to 0)
```

**After (silent disruption):**
```
Week 7 OR Decision (CAPPED Policy):
======================================================================
cola:
  Base stock: 72.5
  Current inventory: 45
  Order (capped): 28

{
  "action": {
    "cola": 28
  },
  "rationale": "OR base-stock (capped): cola: base_stock=72.5, cap=99.2, inv=45, order=28"
}
======================================================================

Week 7 Demand: {"action": {"cola": 55}}

=== Week 7 Summary ===
cola: ordered=28, arrived=0, starting inventory=45, demand=55, sold=45, ending inventory=0
```

Notice: **Order was placed (28 units), but arrived=0**. The agent must figure out why!

## Benefits

1. **More Realistic**: Supply disruptions happen without warning in real life
2. **Tests Adaptability**: Can the agent detect and respond to missing shipments?
3. **Fair Testing**: All algorithms (LLM, OR, Hybrid) experience the same disruptions
4. **Learning Opportunity**: LLM agents might develop strategies to detect supply issues

## Files Modified

1. ✅ `textarena/envs/VendingMachine/env.py` - Accept orders but don't fulfill
2. ✅ `examples/llm_csv_demo.py` - Remove detection & warnings
3. ✅ `examples/llm_to_or_csv_demo.py` - Remove detection & warnings  
4. ✅ `examples/or_to_llm_csv_demo.py` - Remove detection & warnings
5. ⏭️ `examples/or_csv_demo.py` - No changes needed (already compatible)

## Testing Recommendation

Run case 8 tests with the new behavior:
```powershell
# Test with lead_time=inf scenario
python examples\or_csv_demo.py --demand-file examples\demand_case8_L_inf_p015.csv --promised-lead-time 4 --policy capped > examples\case8_tests\or.txt 2>&1

python examples\llm_csv_demo.py --demand-file examples\demand_case8_L_inf_p015.csv --promised-lead-time 4 > examples\case8_tests\llm_1.txt 2>&1
```

Look for:
- Orders placed during weeks with `lead_time=inf`
- `arrived=0` in those weeks
- Agent behavior after discovering missing shipments
- Whether agents adapt inventory strategies
