# Spelling Bee Environment

## Overview
**Spelling Bee** is a two-player competitive game where each player attempts to create the longest possible valid English word using a given set of allowed letters. Players submit their words simultaneously, and the player with the longer valid word wins. The game emphasizes vocabulary skills and strategic letter usage.

## Action Space
- **Format:** Actions are strings representing the player's proposed word, wrapped in square brackets.
- **Examples:** 
    - `"[donkey]"`
    - `"I think a good word would be [apples]"`

## Observation Space

### Observations
**Reset Observation:**

On reset, each player receives a prompt containing the allowed letters and game instructions. For example:
```plaintext
[GAME]: You are Player 0 in the Spelling Bee Game.
Allowed Letters: a e l m p s
Create the longest possible English word using only the allowed letters. You may use each letter multiple times.
Please wrap your word in square brackets, e.g., '[example]'.
On your turn, simply type your word.
```

**Step Observation:**
The game provides no step-based observations. This is to make sure that neither player has an unfair advantage. 

## Gameplay
- **Players**: 2
- **Turns**: Each player has one turn to submit their word.
- **Allowed Letters**: A set of unique lowercase letters randomly generated at the start of the game, configurable via the `num_letters` parameter.
- **Objective**: Submit the longest valid English word possible using the allowed letters.
- **Rules**:
    - Words must be composed only of the allowed letters.
    - Each letter can be used zero, once or multiple times.
    - Words must be valid English (british or us) words.
    - Words must be wrapped in square brackets.

## Key Rules
1. Word Composition:
    - Only allowed letters can be used.
    - Each letter can be used multiple times in a word.

2. Word Validation:
    - Words must be recognized as valid English words by standard dictionaries (en_US or en_GB).
3. Submission Format:
    - Words must be enclosed in square brackets, e.g., `[example]`.
4. Game Termination:
    - The game ends after both players have submitted their words and the system has evaluated them

## Rewards
| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`                |
| **Lose**         | `-1`              | `+1`                |
| **Draw**         |  `0`              |  `0`                |
| **Invalid Move** | `-1`              |  `0`                |

## Parameters
- `num_letters` (`int`):
    - **Description**: Number of unique allowed letters in the game.
    - **Impact**: Determines the complexity and difficulty of the game. More letters generally allow for longer and more varied words.


## Variants

| Env-id                   | num_letters |
|--------------------------|:-----------:|
| `SpellingBee-v0`         |     `6`     |
| `SpellingBee-v0-small`   |     `4`     |
| `SpellingBee-v0-large`   |     `10`    |


## Example Usage

```python
import textarena as ta

## initalize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="gpt-4o"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-sonnet"),
}

## initialize the environment
env = ta.make("SpellingBee-v0")

## Wrap the environment for easier observation handling
env = ta.wrappers.LLMObservationWrapper(env=env)

## Wrap the environment for pretty rendering
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o", 1: "Claude-3.5-Sonnet"}
)

## reset the environment to start a new game
env.reset(seed=490)

## Game loop
done = False
while not done:

    # Get player id and observation
    player_id, observation = env.get_observation()

    # Agent decides on an action based on the observation
    action = agents[player_id](observation)

    # Execute the action in the environment
    done, info = env.step(action=action)

rewards = env.close()
```

## Troubleshooting

- **Game Not Terminating Properly:**
    - **Issue:** The game does not end after both players have submitted their words.
    - **Solution:** Check the implementation of the step method to ensure it correctly identifies when both words have been submitted and terminates the game accordingly.


## Version History
- **v0**
  - Initial release 



### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg
