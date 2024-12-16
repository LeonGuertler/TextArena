# Ultimate Tic Tac Toe Environment Documentation

## Overview
**Ultimate Tic Tac Toe** is a strategic two-player game that combines the classic Tic Tac Toe with an added layer of complexity. Players aim to win three micro boards in a row (horizontally, vertically, or diagonally) on the macro board, which tracks the outcomes of individual micro boards. Each move influences the opponent's next playable micro board, creating dynamic and strategic gameplay. This environment implements the full rules of Ultimate Tic Tac Toe, including valid move enforcement, micro board and macro board win detection, and a clear rendering of the board state for agent-based gameplay and experimentation.

## Action Space
- **Format:** Actions are strings representing the player's choice. For example:
- **Example:**
    - Choosing the micro board 0 and marking row 1 col 0: [0 1 0]
    - Choosing the micro board 3 and marking row 0 col 2: [3 0 2]
- **Notes:** The players are free to have additional texts in their replies, so long they provide their action in the correct format of [micro_board row col].

## Observation Space
**Reset Observation**
On reset, each player receives a prompt containing their beginning game instructions. For example:
```plaintext
[GAME] You are Player 0. You are playing Ultimate Tic Tac Toe.
Your goal is to win three micro boards in a row (horizontally, vertically, or diagonally) on the macro board.
Each micro board is a 3x3 Tic Tac Toe grid, and the macro board tracks which player has won each micro board.
On your turn, you can mark an empty cell in a micro board. The position of your move determines which micro board
your opponent must play in next.

Rules to remember:
1. A move must be made in the micro board specified by the previous move.
   For example, if the last move was in the top-left corner of a micro board, the opponent must play in the top-left micro board.
2. If the directed micro board is already won or full, you are free to play in any available micro board.
3. You win a micro board by completing a row, column, or diagonal within that board.
4. You win the game by completing three micro boards in a row on the macro board.
5. The game ends in a draw if all micro boards are filled, and no player has three in a row on the macro board.
6. To submit your move, submit them as [micro_board, row, col], where micro_board is the index of the micro board (0-8), and row and col are the cell coordinates (0-2).
For example, to play in the center cell of the top-left micro board, submit [0 1 1].

As Player 0, you will be 'O', whereas your opponent is 'X'.
Below is the current state of the macro board (tracking micro board wins):
      |       |      
      |       |      
      |       |      
-----------------------
      |       |      
      |       |      
      |       |      
-----------------------
      |       |      
      |       |      
      |       |      
-----------------------
```

**Step Observation**
After each step, the players will receive the latest message from the game environment. For example, here's player 0 making its first move and the environment responds back:
```plaintext
[Player 0] [4 1 1]
[GAME] Player 0 made a move in micro board 4 at row 1, col 1. Player 1 must play in micro board 4. New state of the board:
      |       |      
      |       |      
      |       |      
-----------------------
      |       |      
      |   O   |      
      |       |      
-----------------------
      |       |      
      |       |      
      |       |      
-----------------------
```

## Gameplay
- **Players:** 2
- **Turns:** Players take turns placing their mark ('X' or 'O') in a specific cell of a micro board. Each turn determines the micro board their opponent must play in.
- **Board:** The game features a macro board (3x3 grid) representing the outcomes of nine micro boards (each a 3x3 Tic Tac Toe grid). The macro board is updated as players win micro boards.
- **Objective:** Win the game by completing three micro boards in a row (horizontally, vertically, or diagonally) on the macro board.
- **Winning Condition:** The first player to achieve three micro boards in a row on the macro board wins the game. If all micro boards are filled without a winner on the macro board, the game ends in a draw.

## Key Rules
### Gameplay Mechanics
1. Moves:
- Players select a cell in one of the nine micro boards.
- The selected cell determines the next micro board the opponent must play in.
- If the specified micro board is full or already won, the opponent can play in any available micro board.

2. Micro Board Wins:
- A micro board is won by completing a row, column, or diagonal within that board.
- The winning player's mark ('X' or 'O') is placed in the corresponding cell of the macro board.

3. Macro Board Wins:
- The game is won by completing three micro boards in a row on the macro board (horizontally, vertically, or diagonally).

4. Draws:
- The game ends in a draw if all cells on the macro board are filled and no player achieves three in a row.

### Phases of a Turn

1. Move Phase:
- The current player places their mark in an empty cell of the assigned micro board.
- If the directed micro board is unavailable, the player can choose any available micro board.

2. Board Update Phase:
- The game checks if the current move wins the micro board and updates the macro board accordingly.
- The game also evaluates the macro board to determine if a player has won or if the game is a draw.

3. Observation Phase:
- After the move, the game renders the updated board state (both macro and micro boards) for players to review and strategize.

## Rewards

| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`                |
| **Lose**         | `-1`              | `+1`                |
| **Invalid**      | `-1`              | `0`                 |

## Variants

| Env-id                  |
|-------------------------|
| `UltimateTicTacToe-v0`  |

## Example Usage
```python
import textarena as ta

# Initialize the environment
env = ta.make(env_id="UltimateTicTacToe-v0")

# Wrap the environment for easier observation handling
env = ta.wrappers.LLMObservationWrapper(env=env)

# Wrap the environment for pretty rendering
env = ta.wrappers.PrettyRenderWrapper(env=env)

# initalize agents
agents = {
    0: ta.basic_agents.OpenRouterAgent(model_name="gpt-4o"),
    1: ta.basic_agents.OpenRouterAgent(model_name="gpt-4o-mini")
    }

# reset the environment to start a new game
observations = env.reset(seed=490)

# Game loop
done = False
while not done:

    # Get the current player id and observation
    player_id, observation = env.get_observation()

    # Agent decides on an action based on the observation
    action = agents[player_id](observation)

    # Execute the action in the environment
    rewards, truncated, terminated, info = env.step(action=action)

    # Check if the game has ended
    done = terminated or truncated

    # Optionally render the environment to see the current state
    env.render()

    if done:
        break

# Finally, print the game results
for player_id, agent in agents.items():
    print(f"{agent.model_name}: {rewards[player_id]}")
print(f"Reason: {info['reason']}")
```

## Version History
- **v0**
    - Initial release

### Contact
If you have questions or face issues with this specific environment, please reach out directly to bobby_cheng@i2r.a-star.edu.sg