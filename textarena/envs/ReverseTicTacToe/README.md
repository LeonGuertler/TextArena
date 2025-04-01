# ReverseTicTacToe Environment Documentation

## Overview
**ReverseTicTacToe** flips the traditional game of Tic Tac Toe on its head. In this environment, players must **avoid** forming a line of three identical symbols. The first player to accidentally complete a row, column, or diagonal **loses** the game. This simple twist introduces a fresh layer of strategy to a familiar format.

## Action Space
- **Format:** Actions are strings containing the chosen cell number in square brackets:
  ```plaintext
  [cell_number]
  ```
- **Examples:**
  - Place your symbol in cell 0: `[0]`
  - Place your symbol in the center: `[4]`
- **Notes:** Extra text is allowed but ignored. Only the `[number]` inside brackets is parsed.

## Observation Space

### Reset Observations
At the start of the game, each player receives a prompt that explains the rules and the objective:

```plaintext
You are Player 1 in Reverse Tic Tac Toe.
Your symbol is 'X'.
The goal is to avoid getting three in a row (horizontally, vertically, or diagonally).
If you make three in a row, you LOSE.
Submit your move using the format '[4]' to place your symbol in cell 4.
As Player 1, you are 'X' and your opponent is 'O'.
```

### Step Observations
After each move, the updated board and list of valid moves are provided:

```plaintext
[Player 0] [4]

Current Board:

  0 | 1 | 2
 ---+---+---
  3 | O | 5
 ---+---+---
  6 | 7 | 8

Available Moves: [0], [1], [2], [3], [5], [6], [7], [8]
```

## Gameplay
- **Players:** 2 (Player 0 and Player 1)
- **Initial Setup:** Empty 3×3 grid
- **Player Symbols:**
  - Player 0: 'O'
  - Player 1: 'X'
- **Turns:** Players alternate turns
- **Objective:** Avoid forming a line of three of your symbols

## Key Rules
1. **Valid Moves:**
   - Only unoccupied cells can be selected
   - Format must match `[number]`

2. **Loss Condition:**
   - If a player places their symbol to form a complete row, column, or diagonal, they **lose**
   - The opponent is then declared the winner

3. **Draw:**
   - If all cells are filled without either player forming a line, the game ends in a draw

4. **Invalid Moves:**
   - Selecting an already occupied cell or using an incorrect format is penalized as invalid

## Rewards
| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Opponent Loss** | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |

## Example Moves
- Valid: `[2]`
- Invalid: `[9]` (out of range)
- Invalid: `place 5` (wrong format)

| Env-id                    |
|---------------------------|
| `ReverseTicTacToe-v0`     |

### Contact
For questions, issues, or contributions, reach out to `guertlerlo@cfar.a-star.edu.sg`.

