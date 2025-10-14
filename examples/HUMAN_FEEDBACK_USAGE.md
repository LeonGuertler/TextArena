# Human-in-the-Loop VendingMachine Usage Guide

This guide explains how to use the human feedback features added to the VendingMachine demos.

## Overview

Two modes of human feedback are now available:

- **Mode 1 (Daily Feedback)**: Provide feedback on the agent's decision each day
- **Mode 2 (Periodic Guidance)**: Provide strategic guidance every N days that persists

Both modes can be used simultaneously or independently.

## Command-Line Options

### For `vending_csv_demo.py`:

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    [--human-feedback] \
    [--guidance-frequency N]
```

### For `agent_or_csv_demo.py`:

```bash
python examples/agent_or_csv_demo.py \
    --demand-file examples/demand.csv \
    [--human-feedback] \
    [--guidance-frequency N]
```

**Arguments:**
- `--demand-file`: (Required) Path to CSV file with demand data
- `--human-feedback`: (Optional) Enable Mode 1 - daily feedback on agent decisions
- `--guidance-frequency N`: (Optional) Enable Mode 2 - collect strategic guidance every N days (0=disabled)

## Usage Examples

### Example 1: Daily Feedback Only (Mode 1)

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    --human-feedback
```

**What happens:**
1. Each day, the agent provides its initial rationale and decision
2. You see the agent's reasoning and proposed order quantities
3. You can provide feedback (or press Enter to accept)
4. If you provide feedback, the agent reconsiders and provides a revised decision
5. This feedback only affects the current day's decision

**Example interaction:**
```
Day 1 - Stage 1: Agent is thinking...
======================================================================
AGENT'S INITIAL DECISION (Mode 1 - Daily Feedback)
======================================================================

Agent's Rationale:
----------------------------------------------------------------------
Based on historical data showing average demand of 200 for Regular 
chips and 145 for BBQ chips, with lead time of 1 day...
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

Your feedback: I think we should be more conservative since holding costs are high

✓ Feedback recorded. Asking agent to reconsider...
```

### Example 2: Periodic Guidance Only (Mode 2)

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    --guidance-frequency 5
```

**What happens:**
1. Every 5 days, you're prompted to provide strategic guidance
2. This guidance is added to the agent's context and persists for all future decisions
3. You can provide general strategies that the agent should follow

**Example interaction:**
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

Your strategic guidance: When news mentions events, increase orders by 30-40% for both items

✓ Guidance recorded and will persist in future decisions.
```

### Example 3: Both Modes Simultaneously

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    --human-feedback \
    --guidance-frequency 5
```

**What happens:**
1. Every day: You can provide feedback on the agent's decision (Mode 1)
2. Every 5 days: You can also provide strategic guidance (Mode 2)
3. Strategic guidance accumulates and influences all decisions
4. Daily feedback only affects that specific day

### Example 4: Using with Hybrid OR+Agent Demo

```bash
python examples/agent_or_csv_demo.py \
    --demand-file examples/demand.csv \
    --human-feedback \
    --guidance-frequency 10
```

This works the same way but with the hybrid agent that also considers OR algorithm recommendations.

## Tips for Effective Feedback

### Mode 1 - Daily Feedback:
- **Be specific**: "Increase chips(Regular) to 250" is better than "order more"
- **Explain your reasoning**: Help the agent understand your thinking
- **Can be empty**: Press Enter to accept the agent's initial decision
- **One-time only**: This feedback won't affect future days

Examples of good feedback:
- "The news mentions an event in 2 days. Since lead time is 1 day, you should order extra today."
- "You're overstocking BBQ chips. Reduce to 120 to minimize holding costs."
- "Historical demand shows spikes after news. Be prepared for higher demand tomorrow."

### Mode 2 - Periodic Guidance:
- **Be strategic**: Provide high-level principles, not day-specific decisions
- **Think long-term**: This guidance will apply to all future decisions
- **Can be empty**: Press Enter if you don't have guidance at this time
- **Accumulates**: Each guidance message is added to the context

Examples of good guidance:
- "Always maintain a safety stock of at least 50 units for Regular chips"
- "When news mentions sports events or TV shows, increase orders by 50%"
- "Prioritize profit over holding costs - it's okay to have some extra inventory"
- "Learn from past news patterns: major events typically increase demand for 2-3 days"

## Workflow

A typical game session with both modes enabled:

```
Day 1:
  - Agent makes initial decision
  - You provide feedback (or accept)
  - Game continues

Day 2-4:
  - Same as Day 1

Day 5:
  - You provide strategic guidance (Mode 2)
  - Agent makes initial decision
  - You provide feedback (or accept)
  - Game continues

Day 6-9:
  - Agent now uses your accumulated guidance
  - You can still provide daily feedback

Day 10:
  - You provide strategic guidance again (Mode 2)
  - Agent makes initial decision with all guidance
  - You provide feedback (or accept)
  - Pattern continues...
```

## Advanced Usage

### Skip Feedback on Certain Days

Even with `--human-feedback` enabled, you can skip feedback on any day by just pressing Enter:

```
Your feedback: [just press Enter]

✓ No feedback. Using agent's initial decision.
```

### Multiple Strategic Guidance Messages

Strategic guidance accumulates. If you provide guidance on Day 5, Day 10, and Day 15, all three messages will be included in the agent's context for Day 16 and beyond.

### Combining with Different CSV Files

You can use these features with any CSV demand file:

```bash
python examples/vending_csv_demo.py \
    --demand-file path/to/your/custom_demand.csv \
    --human-feedback \
    --guidance-frequency 3
```

## Troubleshooting

**Issue**: Agent doesn't seem to consider my feedback  
**Solution**: Make sure your feedback is clear and specific. The agent has the original rationale and your feedback to work with.

**Issue**: Strategic guidance doesn't appear in later decisions  
**Solution**: Guidance is injected at the top of observations. You can verify it's working by checking if the agent's rationale references your guidance.

**Issue**: Too many prompts for input  
**Solution**: Use Mode 2 only with a larger frequency value (e.g., `--guidance-frequency 10`) or disable Mode 1 by removing `--human-feedback`.

## Output Files

When running with human feedback, consider redirecting output to a file for later review:

```bash
python examples/vending_csv_demo.py \
    --demand-file examples/demand.csv \
    --human-feedback \
    --guidance-frequency 5 \
    | tee output_with_feedback.txt
```

This will show output on screen AND save it to a file.

