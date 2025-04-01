# Wild TicTacToe Environment Documentation

## Overview
**Wild TicTacToe** is a fun and strategic twist on the classic two-player Tic Tac Toe game. In this variant, players can choose to place **either 'X' or 'O'** on their turn—regardless of what they previously used. The goal remains the same: form a line of **three identical symbols**, but the freedom to switch symbols adds a layer of bluffing, counterplay, and dynamic strategy.

This implementation includes full support for move parsing, validation, win detection (per symbol), and draw conditions.

---

## Action Space

- **Format:** Actions are strings that indicate which symbol the player wants to place, and in which cell. The format is:
  ```plaintext
  [symbol cell_number]
  ```
- **Examples:**
  - Place an 'X' in the center cell: `[X 4]`
  - Place an 'O' in the top-left corner: `[O 0]`

- **Notes:**
  - Symbols are case-insensitive: `[x 1]` and `[X 1]` are treated the same.
  - Additional surrounding text is ignored, only the bracketed move matters.
  - Players are free to switch symbols each turn.

---

## Observation Space

### Reset Observations
At the start of the game, each player receives a message like the following:

```plaintext
You are Player 0 in Wild Tic Tac Toe.
On your turn, you can place either an 'X' or an 'O' in an empty square.
You win by aligning three of the same mark (X or O) in a row.
Choose your move using the format '[X 4]' to place X in the center.
```

### Step Observations
After each move, the current state of the board and the updated list of available moves is shared. For example:

```plaintext
[Player 0] [X 4]
[GAME] Player 0 placed an 'X' in cell 4.
Current Board:

  0 | 1 | 2
 ---+---+---
  3 | X | 5
 ---+---+---
  6 | 7 | 8

Available Moves: [X 0], [O 0], [X 1], [O 1], [X 2], [O 2], [X 3], [O 3], [X 5], [O 5], [X 6], [O 6], [X 7], [O 7], [X 8], [O 8]
```

---

## Gameplay

- **Players:** 2 (Player 0 and Player 1)
- **Initial Setup:** An empty 3x3 grid
- **Symbols:** Players can place either 'X' or 'O' on any turn
- **Turns:** Players alternate turns
- **Objective:** Get **three of the same symbol** in a row, regardless of who placed them

---

## Key Rules

1. **Valid Moves:**
   - A move must target an empty cell.
   - The move must specify which symbol to place and where.
   - Format: `[X 4]`, `[O 7]`, etc.

2. **Winning Conditions:**
   - A player wins if they complete a line (horizontal, vertical, or diagonal) with **three matching symbols** (all Xs or all Os).
   - The player who makes the move that completes the line wins—regardless of who placed the other symbols.

3. **Draws:**
   - The game ends in a draw if all cells are filled without a winning line.

4. **Invalid Moves:**
   - Moves targeting already-occupied cells or using an invalid format are considered invalid.

---

## Rewards

| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Win**      | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |

---

| Env-id                   |
|--------------------------|
| `WildTicTacToe-v0`       |

---

### Contact
For questions or issues with this environment, please reach out to  
📧 `guertlerlo@cfar.a-star.edu.sg`

