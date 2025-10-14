# Example Human-in-the-Loop Interaction

This document shows what a typical interaction looks like when running the VendingMachine demo with human feedback enabled.

## Running the Demo

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    --human-feedback \
    --guidance-frequency 5
```

## Expected Interaction Flow

### Startup

```
======================================================================
HUMAN-IN-THE-LOOP MODE ACTIVATED
======================================================================
✓ Mode 1: Daily feedback on agent decisions is ENABLED
✓ Mode 2: Strategic guidance every 5 days is ENABLED
======================================================================

Loaded CSV with 50 days of demand data
News scheduled for days: [2, 18, 39]
Set NUM_DAYS to 50 based on CSV
```

### Day 1: First Decision with Mode 1 Feedback

```
======================================================================
Day 1 - Stage 1: Agent is thinking...
======================================================================

======================================================================
AGENT'S INITIAL DECISION (Mode 1 - Daily Feedback)
======================================================================

Agent's Rationale:
----------------------------------------------------------------------
It's Day 1, and I have historical demand data showing:
- chips(Regular): average around 200 units/day
- chips(BBQ): average around 145 units/day

Current inventory shows 5 units each, which is very low. Given the 
lead time of 1 day, I need to order today to have stock tomorrow.

I'll order to match expected demand plus a small buffer to avoid 
stockouts, while being mindful of holding costs.

Strategy: Order 200 for Regular and 145 for BBQ to meet baseline demand.
----------------------------------------------------------------------

Agent's Proposed Action:
{
  "chips(Regular)": 200,
  "chips(BBQ)": 145
}

======================================================================
YOUR TURN: Provide feedback on this decision (or press Enter to accept)
======================================================================
You can:
- Suggest adjustments to the order quantities
- Point out considerations the agent might have missed
- Just press Enter to accept the agent's decision as-is
----------------------------------------------------------------------

Your feedback: [User types: "Good analysis. Let's be slightly more conservative on BBQ to reduce holding costs."]

✓ Feedback recorded. Asking agent to reconsider...

======================================================================
Day 1 - Stage 2: Agent reconsidering with your feedback...
======================================================================

Agent's Revised Decision:
----------------------------------------------------------------------
{
  "rationale": "Considering the human feedback about being conservative on BBQ chips to reduce holding costs, I'll adjust my order. The historical data shows BBQ has more variation (115-176 range), so there's risk in overstocking. I'll reduce BBQ order to 130 units while maintaining Regular at 200 since it has more stable demand.",
  "action": {
    "chips(Regular)": 200,
    "chips(BBQ)": 130
  }
}
----------------------------------------------------------------------
```

### Day 2: News Event

```
Day 2 VM Action:
==============================================================

Agent's Rationale:
----------------------------------------------------------------------
TODAY'S NEWS: "Hit Reality Show to Air Finale The Day After Tomorrow"

This is important! The finale airs on Day 4 (day after tomorrow). Given 
the 1-day lead time, I need to increase orders TODAY (Day 2) to have 
extra stock ready for Day 3, which will see increased demand as people 
prepare for watch parties.

Based on past news impact patterns, I expect 50% increase in demand.
Normal demand: 200 Regular, 145 BBQ
Adjusted for news: 300 Regular, 220 BBQ
----------------------------------------------------------------------

Agent's Proposed Action:
{
  "chips(Regular)": 300,
  "chips(BBQ)": 220
}

Your feedback: [User presses Enter - no feedback]

✓ No feedback. Using agent's initial decision.
```

### Day 5: Strategic Guidance Collection (Mode 2)

```
======================================================================
STRATEGIC GUIDANCE REQUEST (Mode 2 - Day 5)
======================================================================
Provide strategic guidance for the agent to follow in upcoming decisions.
This guidance will be remembered and applied to all future decisions.

Examples:
- 'Be more conservative with ordering to reduce holding costs'
- 'When you see news about events, increase orders by 50%'
- 'Focus on maintaining higher stock levels for chips(Regular)'
----------------------------------------------------------------------

Your strategic guidance: [User types: "When news mentions events, always increase orders by at least 40%. Regular chips should maintain minimum stock of 50 units."]

✓ Guidance recorded and will persist in future decisions.

======================================================================
Day 5 - Stage 1: Agent is thinking...
======================================================================

[Agent now sees guidance at top of observation and incorporates it into decisions]
```

### Day 10: Guidance Appears in Context

```
======================================================================
Day 10 - Stage 1: Agent is thinking...
======================================================================

Agent sees in observation:
----------------------------------------------------------------------
======================================================================
HUMAN STRATEGIC GUIDANCE (apply to your decisions)
======================================================================

Guidance 1: When news mentions events, always increase orders by at 
least 40%. Regular chips should maintain minimum stock of 50 units.

======================================================================
[Rest of game observation...]
----------------------------------------------------------------------

Agent's Rationale:
----------------------------------------------------------------------
Following the strategic guidance to maintain minimum 50 units for Regular 
chips. Current inventory shows 45 units, so I need to order enough to 
restore the minimum stock level plus meet tomorrow's expected demand.

Calculation: 50 (minimum) + 200 (expected demand) - 45 (current) = 205 units
----------------------------------------------------------------------
```

### Day 15: Another Guidance Collection

```
======================================================================
STRATEGIC GUIDANCE REQUEST (Mode 2 - Day 15)
======================================================================

Your strategic guidance: [User types: "If holding costs exceed profit for 2 consecutive days, reduce order quantities by 20%"]

✓ Guidance recorded and will persist in future decisions.

======================================================================
Day 15 - Stage 1: Agent is thinking...
======================================================================

Agent now sees TWO guidance messages:
----------------------------------------------------------------------
======================================================================
HUMAN STRATEGIC GUIDANCE (apply to your decisions)
======================================================================

Guidance 1: When news mentions events, always increase orders by at 
least 40%. Regular chips should maintain minimum stock of 50 units.

Guidance 2: If holding costs exceed profit for 2 consecutive days, 
reduce order quantities by 20%

======================================================================
----------------------------------------------------------------------
```

### Game End

```
=== Final Results ===
======================================================================

Per-Item Statistics:

chips(Regular) (Potato Chips (Regular), 10oz bag):
  Ordered: 9850, Sold: 9800, Ending: 55
  Profit/unit: $2, Holding: $1/unit/day
  Total Profit: $19600

chips(BBQ) (Potato Chips (BBQ), 20oz bag):
  Ordered: 7250, Sold: 7200, Ending: 60
  Profit/unit: $3, Holding: $2/unit/day
  Total Profit: $21600

======================================================================
=== TOTAL SUMMARY ===
======================================================================
Total Profit from Sales: $41200.00
Total Holding Cost: $2500.00

>>> Total Reward: $38700.00 <<<
VM Final Reward: 38700.00
======================================================================
```

## Key Observations

### Mode 1 Benefits (Daily Feedback)
- Immediate tactical corrections
- Human can catch agent's oversights
- Agent learns to incorporate feedback style
- Feedback is surgical - affects only that decision

### Mode 2 Benefits (Strategic Guidance)
- Sets long-term principles
- Accumulates domain knowledge
- Shapes agent's decision-making framework
- Influences all subsequent decisions

### Combined Power
When both modes work together:
1. Strategic guidance provides the framework
2. Daily feedback handles exceptions and edge cases
3. Agent becomes increasingly aligned with human preferences
4. Human can focus on high-level strategy while handling tactical issues as needed

## Tips for Effective Use

### Good Daily Feedback Examples:
- ✓ "Increase Regular chips by 50 to account for lead time"
- ✓ "This is too conservative given the news event in 2 days"
- ✓ "Reduce BBQ to 100 - we have excess inventory"

### Poor Daily Feedback Examples:
- ✗ "Better" (too vague)
- ✗ "I don't know" (not helpful)
- ✗ "Always order 300" (this should be Mode 2 guidance)

### Good Strategic Guidance Examples:
- ✓ "Maintain safety stock of 50 units for Regular chips"
- ✓ "When news mentions events, increase orders by 40%"
- ✓ "Prioritize reducing holding costs when inventory > 100"

### Poor Strategic Guidance Examples:
- ✗ "Order 200 today" (too specific - use Mode 1)
- ✗ "Good job" (not guidance)
- ✗ Repeating what the agent already knows

## Skipping Feedback

You can always press Enter without typing to skip:
- Mode 1: Agent's initial decision is used
- Mode 2: No guidance is added for this interval

This is useful when:
- Agent's decision already looks good
- You're busy and want the game to proceed
- You don't have specific guidance at that moment

