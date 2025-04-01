# TicTacToe Environment Documentation

## Overview
**TicTacToe** (also known as Noughts and Crosses) is a classic two-player game played on a 3×3 grid. Players take turns marking cells with their symbol ('X' or 'O'), with the objective of placing three of their marks in a horizontal, vertical, or diagonal row. This implementation provides a complete game environment with move validation, win detection, and draw conditions.

## Action Space
- **Format:** Actions are strings representing the player's chosen cell to place their mark. The format is:
  ```plaintext
  [cell_number]
  ```
- **Examples:**
  - Mark the center cell (cell 4): `[4]`
  - Mark the top-right cell (cell 2): `[2]`
- **Notes:** The players may include additional text before or after their move. Only the properly formatted cell number in square brackets will be parsed.

## Observation Space

### Reset Observations
On reset, each player receives a prompt explaining the rules of the game, including the available cells. For example:

```plaintext
You are Player 0 in TicTacToe. Your stones appear as 'O' and your opponent's stones appear as 'X'.
On your turn, you choose one empty cell by its numbered index and place your stone there.
For example, '[4]' places your stone in the center cell of the board.
Your objective is to form a continuous line of three of your stones in any row, column, or diagonal.

Current Board:

  0 | 1 | 2
 ---+---+---
  3 | 4 | 5
 ---+---+---
  6 | 7 | 8

Available Moves: [0], [1], [2], [3], [4], [5], [6], [7], [8]
```

### Step Observations
After each move, players receive the updated board state and available moves. For example:

```plaintext
[Player 0] [4]
[GAME] Player 0 placed an 'O' in cell 4.
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
- **Initial Setup:** An empty 3x3 grid
- **Player Symbols:**
  - Player 0: 'O'
  - Player 1: 'X'
- **Turns:** Players alternate turns marking one empty cell per turn.
- **Objective:** Form a line of three symbols (horizontally, vertically, or diagonally).

## Key Rules
1. **Valid Moves:**
   - Players can only mark empty cells.
   - Moves are submitted as `[cell_number]`.

2. **Winning Conditions:**
   - A player wins by forming a line of three symbols:
     - Horizontally
     - Vertically
     - Diagonally
   - The game ends in a draw if all cells are filled with no winner.

3. **Draws:**
   - If the board is fully occupied with no winning line, the game ends in a draw.

4. **Invalid Moves:**
   - If a player selects a cell that is already occupied, it is considered an invalid move.

## Rewards
| Outcome      | Reward for Winner | Reward for Loser |
|--------------|:-----------------:|:----------------:|
| **Win**      | `+1`              | `-1`             |
| **Draw**     | `0`               | `0`              |
| **Invalid**  | `-1`              | `0`              |


| Env-id                  |
|-------------------------|
| `TicTacToe-v0`          |


### Contact
For questions or issues with this environment, please reach out to `guertlerlo@cfar.a-star.edu.sg`.