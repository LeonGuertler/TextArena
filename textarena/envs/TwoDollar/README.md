# Two Dollar Negotiation Environment

This is an implementation of the classic [Two Dollar](https://ocw.mit.edu/courses/15-667-negotiation-and-conflict-management-spring-2001/pages/lecture-notes/) negotiation game, a widely-used exercise in negotiation research and education. The game simulates a simple yet powerful negotiation scenario where two players must agree on how to split $2.00, with each player having secret role instructions that create different constraints and objectives.

## Game Description

Two Dollar is a turn-based negotiation game where two players negotiate over how to divide a fixed sum of money ($2.00). What makes this game compelling is that each player receives secret role instructions that may include minimum thresholds, behavioral constraints, or strategic guidelines, creating realistic asymmetric information scenarios.

### Key Features

- **Simple yet strategic**: Easy to understand rules with complex negotiation dynamics
- **Role-based gameplay**: 17 different roles with unique constraints and objectives
- **Asymmetric information**: Players have secret instructions unknown to their opponent
- **Time pressure**: Limited rounds create urgency and strategic decision-making
- **Enforceable roles**: Certain roles have minimum acceptable amounts and actions
- **Stylistic roles**: Some roles have prompts on how to behave

### Core Mechanics

Players alternate turns making proposals, accepting, or rejecting offers. The game continues until:
- A proposal is accepted (both players receive their agreed amounts)
- Maximum rounds are reached (both players receive $0.00)
- A player exceeds their error allowance (game ends with penalties)

## Components

- **2 players**: Each with secret role instructions
- **$2.00 total**: Fixed amount to be divided
- **20+ roles**: Diverse set of behavioral and threshold constraints
- **Maximum rounds**: Default 20 rounds with configurable limits
- **Error allowance**: Default 3 invalid moves per player before penalties

## Turn Structure

1. **Propose**: Make an offer specifying how much you want for yourself
2. **Accept**: Accept the current proposal on the table
3. **Reject**: Reject the current proposal and continue negotiating
4. Players can include rationale and persuasion before their bracketed actions

## Rules

### Making Proposals
- Format: `[Propose] $X.XX` where X.XX is the amount you want (0.00 to 2.00)
- The other player automatically gets the remainder ($2.00 - $X.XX)
- Can include persuasive text before the bracketed action
- Example: `I think this is fair because we both contributed equally [Propose] $1.00`

### Accepting/Rejecting Proposals
- `[Accept]` to accept the current proposal and end the game
- `[Reject]` to reject the current proposal and continue negotiating
- Can only accept/reject when there is an active proposal
- Cannot accept or reject your own proposals
- Can include rationale: `This seems reasonable to me [Accept]`

### Invalid Moves
- Players have 3 invalid moves allowed before automatic penalties
- Invalid moves include:
  - No bracketed action (`[Propose]`, `[Accept]`, `[Reject]`)
  - Multiple actions in one turn (`[Accept] and also [Reject]`)
  - Invalid proposal amounts (negative, over $2.00, missing dollar sign)
  - Accepting/rejecting without an active proposal
  - Accepting/rejecting your own proposal
  - Role-specific violations (word limits, concession limits, etc.)

## Role System

The Two Dollar environment includes 17 roles that create diverse negotiation scenarios:

### Enforceable Roles
- **say_little**: Limited to 15 words per message
- **high_tension**: Can only make concessions of $0.01 or less
- **x_rounds**: Must reach a deal within a specific deadline
- **50_cents**: Must receive at least $0.50 or get $0.00
- **80_cents**: Must receive at least $0.80 or get $0.00
- **1_dollar**: Must receive at least $1.00 or get $0.00
- **1_30_dollar**: Must receive at least $1.30 or get $0.00
- **1_60_dollar**: Must receive at least $1.60 or get $0.00

### Non-Enforceable Roles
- **another_chance**: Believes opponent deserves another opportunity
- **battle_ax**: Well-known aggressive competitor who gets every penny
- **dependent**: Needs to maintain long-term relationship with colleague
- **hard_time**: Knows opponent has been having personal difficulties
- **imaginative**: Chosen for creativity and inventiveness in storytelling
- **public_figure**: Well-known figure concerned about reputation for fairness
- **tape_recorder**: Records everything and may share details publicly
- **untrustworthy**: Warned that opponent has been untrustworthy in past games
- **vanilla**: Standard negotiator focused on maximizing their share

### Role Categories

**Enforceable Roles**: System-enforced constraints
- `action_validation`: Behavioral rules enforced during gameplay
- `end_game_check`: Minimum thresholds checked when game ends

**Non-Enforceable Roles**: Behavioral guidelines for players
- Provide strategic context and personality traits
- Create realistic negotiation scenarios
- Encourage diverse negotiation styles

## Winning Conditions

The game ends when:
- **Deal Accepted**: A proposal is accepted by both players
- **Time Limit**: Maximum rounds reached (both get $0.00)
- **Error Limit**: A player exceeds invalid move allowance

**Scoring:**
- Players receive points based on their final dollar amounts (0-100 scale)
- Role compliance is enforced for applicable roles
- Winner is determined by who received more money
- Equal splits result in a draw

## Usage

### Action Format Examples

**Making a proposal with rationale:**
```
I believe we should split this evenly since we both participated equally in this negotiation.
[Propose] $1.00
```

**Accepting a proposal:**
```
That sounds fair to me, I can work with that amount.
[Accept]
```

**Rejecting a proposal:**
```
I'm sorry, but that amount is too low for what I need.
[Reject]
```

### Example Game Flow

1. **Player 0** proposes: `I think we should split evenly [Propose] $1.00`
2. **Player 1** rejects: `I need more than that [Reject]`
3. **Player 1** counter-proposes: `How about this split? [Propose] $0.75`
4. **Player 0** rejects: `That's not enough for me [Reject]`
5. **Player 0** proposes: `Let's compromise [Propose] $1.25`
6. **Player 1** accepts: `I can live with that [Accept]`
7. **Game ends**: Player 0 gets $1.25, Player 1 gets $0.75

## Quick Start Guide

### Initialize the Environment

```python
import textarena as ta

# Create environment with random roles
env = ta.make(env_id="TwoDollar-v0")

# Or specify roles explicitly
env = ta.make(env_id="TwoDollar-v0", player_roles=["dependent", "50_cents"])

# Reset with 2 players
env.reset(num_players=2)
```

### Run a Simple Game

```python
import textarena as ta

# Set up agents
agents = {
    0: ta.agents.HumanAgent(),  # Human player
    1: ta.agents.OpenRouterAgent(model_name="your-model-name"),  # AI player
}

# Initialize the environment
env = ta.make(env_id="TwoDollar-v0")
env.reset(num_players=len(agents))

# Main game loop
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

# Get final results
rewards, game_info = env.close()
print(f"Final amounts: Player 0: ${env.final_amounts[0]:.2f}, Player 1: ${env.final_amounts[1]:.2f}")
print(f"Rewards: {rewards}")
```

### Custom Configuration

```python
# Create environment with custom settings
env = ta.make(env_id="TwoDollar-v0",
              player_roles=["high_tension", "1_dollar"],  # Specific roles
              total_amount=5.00,                          # Different total amount
              max_rounds=30,                              # More rounds
              error_allowance=5)                          # More lenient error handling

env.reset(num_players=2, seed=42)  # Reproducible games
```
## Testing

The environment includes a comprehensive test suite with 35 tests covering:

```bash
# Run all tests
python -m pytest textarena/envs/TwoDollar/test_env.py -v

# Run specific test categories
python -m pytest textarena/envs/TwoDollar/test_env.py::TestTwoDollarValidation -v
python -m pytest textarena/envs/TwoDollar/test_env.py::TestTwoDollarRoles -v
```

### Test Coverage
- **Validation Tests**: Action format, proposal validation, multiple actions
- **Game Flow Tests**: Turn alternation, round counting, game ending
- **Role Tests**: Behavioral constraints, threshold enforcement
- **Integration Tests**: Complete game scenarios, error recovery
- **Edge Cases**: Boundary values, whitespace handling, case sensitivity

## References

- Massachusetts Institute of Technology. (2001). Lecture notes — Negotiation and Conflict Management (Course 15.667, Spring 2001). MIT OpenCourseWare. [https://ocw.mit.edu/courses/15-667-negotiation-and-conflict-management-spring-2001/pages/lecture-notes/](https://ocw.mit.edu/courses/15-667-negotiation-and-conflict-management-spring-2001/pages/lecture-notes/)