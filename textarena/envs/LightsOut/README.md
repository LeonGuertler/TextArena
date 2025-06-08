# Lights Out Environment Documentation

## Overview

**Lights Out** is a classic single-player puzzle game played on a grid of lights. Each light can be either on or off. When a player presses a light, it toggles its state and the state of its four adjacent neighbors (up, down, left, and right). The objective is to turn all the lights off. 

## Action Space

* **Format:** Actions are strings representing the coordinates of the light to press, in the format `[row, col]`.
* **Examples:**

  * Press the light at row 2, column 3: `[2, 3]`
  * Press the light at the top-left corner: `[0, 0]`
* **Notes:** 

    * Players may include extra text before and after the action command, but only the bracketed coordinates are parsed. 
    * Coordinates are 0-indexed.

## Observation Space

**Reset Observations**

In the first observation, the player receives a prompt containing the game rules and the initial state of the puzzle grid. For example:

```plaintext
[GAME] You are Player 0, playing Lights Out.
The board is a 5x5 grid of lights. '1' means ON, '0' means OFF.
The goal is to turn all the lights OFF.
On your turn, choose a cell to press. Pressing a cell toggles its state and the state of its adjacent (up, down, left, right) neighbors.
Submit your move as [row, col]. For example, [2, 3] to press the light at row 2, column 3.

Initial board state:
    0   1   2   3   4
  +---+---+---+---+---+
0 | 1 | 0 | 1 | 0 | 1 |
  +---+---+---+---+---+
1 | 0 | 1 | 0 | 1 | 0 |
  +---+---+---+---+---+
2 | 1 | 0 | 1 | 0 | 1 |
  +---+---+---+---+---+
3 | 0 | 1 | 0 | 1 | 0 |
  +---+---+---+---+---+
4 | 1 | 0 | 1 | 0 | 1 |
  +---+---+---+---+---+
```

### Step Observations

After each move, the player receives an updated view of the grid. For example:

```plaintext
[Player 0] I will press the light at [2, 2].
[GAME] Player 0 pressed cell [2, 2].
New board:
    0   1   2   3   4
  +---+---+---+---+---+
0 | 1 | 0 | 1 | 0 | 1 |
  +---+---+---+---+---+
1 | 0 | 0 | 1 | 0 | 0 |
  +---+---+---+---+---+
2 | 0 | 1 | 0 | 1 | 0 |
  +---+---+---+---+---+
3 | 0 | 0 | 1 | 0 | 0 |
  +---+---+---+---+---+
4 | 1 | 0 | 1 | 0 | 1 |
  +---+---+---+---+---+
```


## Gameplay

- **Players:** 1 (single-player game)
- **Initial Setup:** A square grid of lights in a random configuration.
- **Turns:** The player presses one light per turn.
- **Objective:** Turn all lights off.

## Key Rules

1. **Move Mechanics:**

   * A move consists of choosing a single cell `[row, col]` to press.
   * Pressing a cell toggles the state of that cell and its four orthogonal neighbors (up, down, left, right).

2. **Valid Moves:**

   * The coordinates `[row, col]` must be within the grid boundaries.

3. **Winning Condition:**

   * The player wins when all lights on the grid are turned off.

4. **Loss Condition:**

   * The player loses if they fail to solve the puzzle within the maximum allowed turns.

5. **Game Termination:**

   * The game concludes when the puzzle is solved or the turn limit is reached.


## Rewards

| Outcome            | Reward for Player |
| ------------------ | ----------------- |
| Win                | `+1`                |
| Loss               | `0`                 | 
| Invalid Move       | `-1`                |

## Parameters

* **`grid_size`** (`int`, default: `5`):

  * **Description:** Sets the height and width of the square grid.
  * **Impact:** Larger grids exponentially increase the complexity of the puzzle.

* **`max_turns`** (`int`, default: `100`):

  * **Description:** Maximum number of turns allowed to complete the puzzle
  * **Impact:** Fewer turns increase pressure on the player to solve quickly

## Variants

| Env-id             | `grid_size` | `max_turns` |
| ------------------ | ---------- | ---------- |
| `LightsOut-v0`       | `5`          | `100`         |
| `LightsOut-v0-small` | `3`          | `100`         |
| `LightsOut-v0-large` | `7`          | `100`        |


