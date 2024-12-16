# Tak Environment Documentation

## Overview
**Tak** is a two-player abstract strategy board game. Players aim to build a continuous road connecting two opposite edges of the board while blocking their opponent's attempts. The game involves placing and moving pieces of various types—Flat Stones, Standing Stones (Walls), and Capstones—each with unique abilities. This environment simulates the full rules of Tak, including flexible board sizes, road-building mechanics, and dynamic board rendering, offering a rich platform for agent-based gameplay and experimentation.

## Action Space
- **Format:** Actions are strings representing the player's choice. For example:
- **Example"** 
    - As player 0, place a flat stone on row 0 col 1: [place () {(0,1):[F0]}]
    - As player 1, move a standing stone from row 0 col 1 to row 1 col 1: [move (0,1) {(1,1):[W1]}]
    - As player 0, move a series of stones from row 2 col 2 to row 2 col 3 and row 2 col 4: [move (2,2) {(2,3): [F0, F1], (2,4): [F0]}]
- **Notes:** The players are free to have additional texts in their replies, so long the provide their action in the correct format of [action source allocation].

## Observation Space
**Reset Observations**
On reset, each player receives a prompt containing their beginning game instructions. For example:
```plaintext
[GAME] You are Player 0. You are playing the Tak game.
Your goal is to connect two opposite edges of the board with your pieces to form a road while blocking your opponent from doing the same.
You can perform the following actions on your turn:
- Place a piece on an empty square.
- Move a stack of pieces from one square to one or more squares. You can stack your pieces on top of other pieces on the target square. The topmost piece determines ownership of the stack.
- Split a stack of pieces into two or more stacks and distribute them to adjacent squares.
- Flatten a wall stone into a flat stone using your capstone.
- Place a Capstone on an empty square.
- Move a Capstone from one square to one or more squares. A capstone can also flatten a wall stone during its move.

For each move, submit your action using the format:
[ACTION SOURCE ALLOCATION]
- ACTION: The type of move you are making ('place' or 'move').
- SOURCE: The grid coordinates where the stones originate. Use () for 'place'.
- ALLOCATION: A dictionary where keys are target grid coordinates and values are the stones or pieces being moved or placed.

Stone Types and Their Abilities:
- Flat Stone ('F'):
  - Forms part of a road (used to connect edges of the board).
  - Can be stacked on top of other pieces or have other pieces stacked on it.
  - Can be moved as part of a stack or individually.

- Wall Stone ('W'):
  - Blocks roads and prevents opponents from completing their connections.
  - Cannot be part of a road.
  - Can be flattened into a flat stone by a capstone.

- Capstone ('C'):
  - Acts as a flat stone and can form part of a road.
  - Can flatten wall stones, removing their blocking effect.
  - Cannot be covered by other pieces, always remains on top of the stack.
  - Is a powerful tool for both road-building and disrupting your opponent's plans.

The stones will be identified by the player as follows:
- Flat Stone for Player 0: 'F0'
- Wall Stone for Player 1: 'W1'
- Capstone for Player 1: 'C1'

Examples:
- To place a capstone on (3,2):
  [place () {(3,2): [C0]}]
- To move all pieces from (2,2) to (2,3):
  [move (2,2) {(2,3): [F0]}]
- To split a stack of 5 pieces from (2,2) into two squares:
  [move (2,2) {(2,3): [F0, F0], (2,4): [W0, F0, C0]}]
- To move and stack one piece from (2,2) onto an existing stack at (2,3):
  [move (2,2) {(2,3): [F0]}]

When submitting your move, think strategically about your road-building goals and your opponent's potential moves.
Here is the current board:
        0       1       2       3  
     -------------------------------
  0 |       |       |       |       |
     -------------------------------
  1 |       |       |       |       |
     -------------------------------
  2 |       |       |       |       |
     -------------------------------
  3 |       |       |       |       |
     -------------------------------
Note that you have 15 stones and 1 capstones to begin with.
```

**Step Observation**
After each step, the players receive the latest message from the game environment. For example, here's player 0 making its first move and the environment responds back:
```plaintext
[Player 0] [place () {(0,0):[F0]}]
[GAME] Player 0 placed a piece on ([(0, 0)]). New board state:
        0        1        2        3   
     -----------------------------------
  0 | (1) F0 |        |        |        |
     -----------------------------------
  1 |        |        |        |        |
     -----------------------------------
  2 |        |        |        |        |
     -----------------------------------
  3 |        |        |        |        |
     -----------------------------------
```

## Gameplay
- **Players**: 2
- **Turns**: Players alternate turns placing their pieces (Flat Stones, Standing Stones, or Capstones) on the board or moving stacks of pieces to build roads or block their opponent.
- **Board**: The game is played on a square grid of size 4x4, 5x5, or 6x6, depending on the difficulty level. Each player starts with a limited number of stones and one Capstone.
- **Objective**: Connect two opposite edges of the board with a continuous road of Flat Stones or Capstones to win the game.
- **Winning Condition**: A player wins by forming a road that spans between two opposite edges of the board. If no roads are completed and all pieces are placed, the winner is determined by the number of visible Flat Stones on the board.

## Key Rules
### Gameplay Mechanics
1. Piece Placement:
- Players can place a Flat Stone, a Standing Stone (Wall), or a Capstone on any empty square of the board.
- Capstones cannot be covered by other pieces.

2. Stack Movement:
- Players can move stacks of pieces they control. The number of pieces in the stack determines how far the stack can move.
- A stack can be split, with pieces dropped along its path, and the remaining pieces continuing to their destination.

3. Blocking and Flattening:
- Walls (Standing Stones) block roads and prevent opponents from forming connections.
- Capstones can flatten Walls into Flat Stones, enabling road formation.

4. Road Formation:
- A road is a continuous path of connected Flat Stones or Capstones.
- Walls and opponent pieces do not contribute to a player's road.

5. Draws:
- If no player forms a road and all pieces are placed, the game ends in a draw.
- The winner is then determined by the player with the highest number of visible Flat Stones.

### Phases of a Turn
1. Action Phase:
- The current player chooses one of the following actions:
    - Place: Place a Flat Stone, Standing Stone, or Capstone on an empty square.
    - Move: Move a stack of pieces from one square to one or more adjacent squares.

2. Board Update Phase:
- After the action, the game updates the board state, resolving any stacking or flattening effects.
- The game checks for road formation or completion of the winning condition.

3. Observation Phase:
- The updated board state is rendered, showing the current positions of all pieces and stacks for both players to strategize their next moves.

4. Win Condition Check:
- The game evaluates whether a player has completed a road. If so, the game ends with the current player as the winner.


## Rewards

| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`                |
| **Lose**         | `-1`              | `+1`                |
| **Invalid**      | `-1`              | `0`                 |

## Variants

| Env-id                  | difficulty       |
|-------------------------|------------------|
| `Tak-v0-easy`           | `easy`           |
| `Tak-v0-medium`         | `medium`         |
| `Tak-v0-hard`           | `hard`           |

## Example Usage
```python
import textarena as ta

# Initialize the environment
env = ta.make(env_id="Tak-v0-easy")

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