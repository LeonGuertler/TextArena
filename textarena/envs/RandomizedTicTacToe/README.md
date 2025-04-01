# RandomizedTicTacToe Environment Documentation

## Overview
**RandomizedTicTacToe** is a chaotic and unpredictable twist on the classic Tic Tac Toe game. Two players take turns placing their symbol ('X' or 'O') on a 3×3 grid, aiming to get three in a row. However, each turn introduces a random **effect** that alters the game dynamics, requiring players to adapt their strategies on the fly.

This environment extends the standard Tic Tac Toe by adding one of four random effects on each turn:
- **SWAP:** Two filled cells are swapped at random.
- **BLOCK:** A random empty cell is permanently blocked.
- **DOUBLE:** The player gets two turns in a row.
- **WILD:** The player’s symbol ('X' or 'O') is randomized for this move.

## Action Space
- **Format:** Actions are strings that include the chosen cell number in square brackets:
  ```plaintext
  [cell_number]
  ```
- **Examples:**
  - Place your symbol in the top-left cell: `[0]`
  - Place your symbol in the center: `[4]`
- **Notes:** Additional text may be included in the message, but only the number inside square brackets will be parsed and executed.

## Observation Space

### Reset Observations
At the beginning of the game, each player receives a prompt explaining the game rules and the special effects. For example:

```plaintext
You are Player 0 in Randomized Tic Tac Toe.
Your symbol is 'O'.
Each turn, a random effect will apply to the game, modifying your strategy.
The effects include:
- SWAP: Two filled cells will be swapped at random.
- BLOCK: A random empty cell will be permanently blocked with a '#'.
- DOUBLE: You will get to play two turns in a row.
- WILD: Your symbol will be randomly chosen as 'X' or 'O' for this move.
Your goal is to get three in a row (horizontally, vertically, or diagonally).
Submit your move using '[cell]'. For example, '[4]' places your symbol in the center.
Your opponent is 'X'.
```

### Step Observations
After each move, players receive the updated board and information about the current random effect. For example:

```plaintext
[Player 1] [4]
[GAME] Player 1 placed an 'X' in cell 4.

Current Board:

  0 | 1 | 2
 ---+---+---
  3 | X | 5
 ---+---+---
  6 | 7 | 8

Current Effect: SWAP
Available Moves: [0], [1], [2], [3], [5], [6], [7], [8]
```

Blocked cells appear as `#` and cannot be selected by either player.

## Gameplay
- **Players:** 2 (Player 0 and Player 1)
- **Initial Setup:** An empty 3x3 grid
- **Player Symbols:**
  - Player 0: 'O'
  - Player 1: 'X'
- **Turns:** Players alternate turns, unless the **DOUBLE** effect is active.
- **Objective:** Form a line of three of your symbols horizontally, vertically, or diagonally.

## Random Effects
Each round begins with one of the following effects randomly applied:

| Effect   | Description |
|----------|-------------|
| **SWAP** | Two filled cells are randomly swapped. |
| **BLOCK** | A random empty cell is replaced with `#` and becomes unusable. |
| **DOUBLE** | The current player takes an extra turn immediately after their first move. |
| **WILD** | The player’s symbol for that move is randomly chosen to be `'X'` or `'O'`, regardless of their original symbol. |

These effects introduce randomness, requiring players to be strategic and adaptive.

## Key Rules
1. **Valid Moves:**
   - Only empty (non-blocked) cells may be selected.
   - Use the `[cell_number]` format to select a cell.

2. **Invalid Moves:**
   - Selecting a cell that is already filled or blocked is considered invalid.
   - Submitting an incorrectly formatted move will result in a penalty.

3. **Winning Conditions:**
   - A player wins by forming a line of three of their symbol.
   - Symbols created using **WILD** are valid for victory conditions.

4. **Draw:**
   - If the board is filled with no winner, the game ends in a draw.

5. **Effect Resolution Order:**
   - Effects are applied **before** the player's move is registered, except for **DOUBLE**, which affects turn rotation after the move.

## Rewards
| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Win**      | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |

## Example Moves
- Player 0 moves: `[0]`
- Player 1 moves: `[4]`
- Invalid move: `[9]` → Cell out of bounds
- Invalid format: `cell 3` → Does not match `[number]` format

| Env-id                       |
|------------------------------|
| `RandomizedTicTacToe-v0`     |

### Contact
For questions, bug reports, or contributions, please contact `guertlerlo@cfar.a-star.edu.sg`.

