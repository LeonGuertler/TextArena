# Word Chains Environment Documentation

## Overview

**Word Chains** is a turn-based word game where two players take turns providing valid English words that start with the last letter of the previous word. The game enforces rules such as no word repetition and requires valid English words. Players lose if they fail to provide a valid word.

## Action Space

- **Format:** Actions are strings representing a word, enclosed in square brackets.
- **Example:**
    - `[apple]`
    - `[monkey]`
    - `[yellow]`
- **Notes:**
    - Words must be wrapped in square brackets.
    - The word must start with the last letter of the previously played word.
    - Words cannot be repeated.

## Observation Space

### Observations
Players receive a message specifying the last played word, the letter required for the next word, and the history of used words.

**Reset Observation:**
```plaintext
[GAME]: Game started.
The starting word is [banana]. Please provide the next word.
On your turn, simply type your word.
```

**Step Observation:**
```plaintext
Player 0: [apple]
[GAME]: Player 0 played the word: [apple]. Next word must start with 'e'.
Player 1: [event]
[GAME]: Player 1 played the word: [event]. Next word must start with 't'.
```

## Gameplay

- **Players:** 2
- **Turns:** Players alternate turns, providing a word each time.
- **Move Format:** Words must be wrapped in square brackets and start with the correct letter.
- **Objective:** The game continues until a player fails to provide a valid word.
- **Turn Limit:** The game ends in a draw if the maximum number of turns is reached.

## Key Rules

1. **Word Selection:**
    - The word must start with the last letter of the previous word.
    - Example: If the previous word is `[table]`, the next word must start with `e`.

2. **Valid Words:**
    - Words must be in standard English.
    - Words are checked against an English dictionary.

3. **No Repeats:**
    - Words cannot be reused.
    - Example: If `[apple]` was played, it cannot be played again.

4. **Game Termination:**
    - If a player provides an invalid word, they lose.
    - If the turn limit is reached, the game ends in a draw.

## Rewards

| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`                |
| **Lose**         | `-1`              | `+1`                |
| **Draw**         |  `0`              |  `0`                |
| **Invalid Move** | `-1`              |  `0`                |



## Variants

| Env-id                   |
|--------------------------|
| `WordChains-v0`          |




## Example Usage

```python
import textarena as ta

## initalize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="gpt-4o"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-sonnet"),
}

## initialize the environment
env = ta.make("WordChains-v0")

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

- **Invalid Move Format:**
    - **Issue:** Player provides a move that doesn't follow `[word]` format.
    - **Solution:** Ensure that words are enclosed in square brackets.

- **Invalid Words:**
    - **Issue:** Player attempts to use an invalid English word.
    - **Solution:** Use words that exist in standard English dictionaries.

- **Repeated Words:**
    - **Issue:** Player attempts to reuse a previously played word.
    - **Solution:** Ensure that each word is unique during the game.

### Contact
If you have questions or face issues with this specific environment, please reach out to Guertlerlo@cfar.a-star.edu.sg

