# SimpleTak Environment Documentation

## Overview
**SimpleTak** is a streamlined version of the Tak board game where players place stones ('O' and 'X') on a square grid. The objective is to connect two opposite edges of the board by forming a continuous path of their stones. The environment supports arbitrary board sizes, making it suitable for training agents on various board configurations.

## Action Space
- **Format:** Actions are strings representing the player's chosen cell. The format is:
  ```plaintext
  [cell_number]
  ```
- **Examples:**
  - Place a stone in cell 12: `[12]`
  - Place a stone in cell 0: `[0]`
- **Notes:** The players may include additional text before or after their move. Only the properly formatted cell number in square brackets will be parsed.

## Observation Space

### Reset Observations
On reset, each player receives a prompt describing the rules and the current board. For example:

```plaintext
You are Player 0 in SimpleTak. Your stones appear as 'O' and your opponent's stones appear as 'X'.
On your turn, choose an empty cell (by its number) and place your stone there.
For example, '[12]' places your stone in cell 12.
Your objective is to form a continuous path connecting opposite edges (top-to-bottom or left-to-right).

Current Board:

  +----+----+----+----+
  |  0 |  1 |  2 |  3 |
  +----+----+----+----+
  |  4 |  5 |  6 |  7 |
  +----+----+----+----+
  |  8 |  9 | 10 | 11 |
  +----+----+----+----+
  | 12 | 13 | 14 | 15 |
  +----+----+----+----+

Available Moves: [0], [1], [2], ..., [15]
```

### Step Observations
After each move, players receive the updated board state and available moves.

## Gameplay
- **Players:** 2 (Player 0 and Player 1)
- **Initial Setup:** An empty NxN grid
- **Player Symbols:**
  - Player 0: 'O'
  - Player 1: 'X'
- **Objective:** Form a continuous path connecting two opposite edges.

## Key Rules
1. **Valid Moves:**
   - Players can only place stones in empty cells.
   - Moves are submitted as `[cell_number]`.

2. **Winning Conditions:**
   - A player wins by connecting opposite edges of the board with a continuous line of their stones.

3. **Draws:**
   - If all cells are filled with no winner, the game ends in a draw.

4. **Invalid Moves:**
   - If a player selects an occupied cell, the move is invalid.

## Rewards
| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Win**      | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |



| Env-id                  | board_size       |
|-------------------------|------------------|
| `Tak-v0`                | `4`              |
| `Tak-v0-medium`         | `5`              |
| `Tak-v0-large`          | `6`              |
| `Tak-v0-extra-large`    | `8`              |


### Contact
For questions or issues with this environment, please reach out to `guertlerlo@cfar.a-star.edu.sg`.

