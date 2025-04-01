# IteratedRockPaperScissors Environment Documentation

## Overview
**IteratedRockPaperScissors** is a simple yet strategic two-player game environment based on the classic Rock-Paper-Scissors. Unlike the single-round version, this implementation runs for a fixed number of rounds (default: 5), and players try to win as many rounds as possible. The environment includes support for shorthand inputs and keeps track of full match history and scores.

## Action Space
- **Format:** Players must submit their move using square brackets:
  ```plaintext
  [rock], [paper], [scissors]
  ```
  - Shorthand: `[r]`, `[p]`, or `[s]` are also accepted.
- **Examples:**
  - `[rock]`, `[r]` — play Rock
  - `[paper]`, `[p]` — play Paper
  - `[scissors]`, `[s]` — play Scissors
- **Notes:** The rest of the player’s message is ignored. Only the first valid token inside square brackets is parsed.

## Observation Space

### Reset Observations
When the game starts, each player receives a prompt like:

```plaintext
You are Player 0 in a 5-round Rock-Paper-Scissors game.
Your goal is to win as many rounds as possible.
In each round, respond with one of: [rock], [paper], or [scissors].
You may also use [r], [p], or [s] as shorthand.
This is round 1 of 5.
```

### Step Observations
After each round is resolved, all players receive the full history so far:

```plaintext
Previous Rounds:
Round 1: P0 -> rock, P1 -> paper
Round result: Player 1 wins!
```

## Gameplay
- **Players:** 2 (Player 0 and Player 1)
- **Rounds:** Default is 5 (can be adjusted via `num_rounds` argument)
- **Objective:** Win more rounds than your opponent by playing the right moves.

## Key Rules

1. **Valid Moves:**
   - A move must be one of `[rock]`, `[paper]`, `[scissors]`, or their shorthands `[r]`, `[p]`, `[s]`.

2. **Round Outcomes:**
   - Rock beats Scissors
   - Scissors beats Paper
   - Paper beats Rock
   - Same move = Draw

3. **Match End:**
   - After all rounds are played, the player with more wins is declared the match winner.
   - If wins are equal, the match is a draw.

4. **Invalid Moves:**
   - Invalid input (e.g., `[banana]`) results in an invalid move penalty for the player.

## Rewards
| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Win**      | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |

## Env-id
| Env-id                              |
|-------------------------------------|
| `IteratedRockPaperScissors-v0`      |

---

### Contact
For questions or issues with this environment, please reach out to `guertlerlo@cfar.a-star.edu.sg`.