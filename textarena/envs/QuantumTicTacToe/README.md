# Quantum TicTacToe Environment Documentation

## Overview
**Quantum TicTacToe** is a futuristic twist on the classic 3x3 Tic Tac Toe game. In this version, each player places **spooky marks** in a *superposition* of two cells. These quantum moves remain entangled until a **cycle** is formed, triggering a **collapse** into classical marks. Players must carefully manage their entanglements to outmaneuver their opponent and be the first to form three classical marks in a row.


## Action Space
- **Format:** Each move places a spooky mark in two empty cells:
  ```plaintext
  [a,b]
  ```
- **Examples:**
  - Entangle cells 0 and 4: `[0,4]`
  - Entangle cells 2 and 5: `[2,5]`
- **Notes:**
  - `a` and `b` must be different and refer to empty cells.
  - Players may include extra text before/after the brackets; only the [a,b] pattern is parsed.

## Observation Space

### Reset Observations
At the start of the game, each player receives a detailed prompt:
```plaintext
You are Player 0 in Quantum Tic Tac Toe.
Your symbol is 'O', and your move numbers are always even (e.g., O2, O4).

📍 Goal: Win by forming a line of three classical marks (solidified from superpositions).

🌀 How to Play:
- On each turn, place a spooky mark in two different empty squares using the format '[a,b]'.
- These marks are entangled, and labeled like 'X1 / X1' or 'O4 / O4'.
- You cannot place spooky marks in a square that has already collapsed (solidified).

💥 Collapse Rule:
- If your move creates a cycle in the entanglement graph, it collapses automatically.
- Each spooky mark in the cycle turns into a classical mark in one of its two positions.
- Any dependent spooky marks also collapse.

⚖️ Victory:
- The game ends when a player has three classical marks in a row.
- If both players get a line during the same collapse, the one with the lower max move number wins.

Example move: '[0,4]' places a spooky mark in cells 0 and 4.
```

### Step Observations
After each move, players receive the updated board with spooky marks, classical marks, and square indices for reference. Example:
```plaintext
Quantum Tic Tac Toe Board:

 [0]       [1]       [2]     
 0A / 2A   1B         2A

---+----------+----------+---

 [3]       [4]       [5]     
 0A         2A       3B

---+----------+----------+---

 [6]       [7]       [8]     
           3A       1B

Submit your move as '[a,b]' to place a quantum mark in two locations.
```

## Gameplay
- **Players:** 2 (Player 0 and Player 1)
- **Board:** 3x3 grid
- **Marks:**
  - Player 0: O (even move numbers, e.g. O2)
  - Player 1: X (odd move numbers, e.g. X1)
- **Turn Structure:**
  - Each player places a spooky mark into two uncollapsed cells.
  - Entanglement cycles automatically collapse and may trigger chains of collapses.

## Key Rules
1. **Spooky Marks:**
   - Each move entangles two empty cells.
   - Cells can contain multiple spooky marks until collapse.

2. **Collapse:**
   - Cycles in the entanglement graph cause all involved spooky marks to collapse.
   - Collapse picks one of the two entangled cells for each mark, favoring the first available.
   - Chains of dependent marks collapse recursively.

3. **Victory:**
   - A player wins by forming three classical marks in a row.
   - If both players win simultaneously, the player with the **lower maximum move number** (earlier win) gets 1 point, the other gets 0.5.

4. **Invalid Moves:**
   - Trying to place a spooky mark in a collapsed cell or using duplicate indices is invalid.

## Rewards
| Outcome               | Player Wins | Opponent |  Notes                         |
|----------------------|:-----------:|:--------:|--------------------------------|
| **Victory**           | `+1`        | `-1`     | First player to form 3-in-a-row |
| **Simultaneous Win**  | `+1` or `0.5` | `0.5` or `+1` | Based on lower max subscript |
| **Draw**              | `0`         | `0`      | No 3-in-a-row and board full   |
| **Invalid Move**      | `-1`        | `0`      | Invalid action                 |


| Env-id                        |
|-------------------------------|
| `QuantumTicTacToe-v0`         |

### Contact
For questions or issues with this environment, please reach out to `guertlerlo@cfar.a-star.edu.sg`.
