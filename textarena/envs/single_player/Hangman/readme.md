# Hangman Environment Documentation

## Overview

The Hangman environment is a single-player word-guessing game where the player attempts to guess a hidden word by suggesting letters or guessing the entire word. The objective is to reveal the correct word before the player runs out of attempts. The environment supports a "hardcore" mode where a more extensive vocabulary is used, adding to the difficulty.

## Action Space
- **Format**: Actions are provided as strings in the format [letter] for guessing a single letter or [WORD] for guessing the entire word.

- **Examples**:
    - Guessing a letter: [A] attempts to reveal all instances of the letter 'A' in the word.
    - Guessing the entire word: [LIGHT] checks if the guessed word matches the hidden word.

- *Notes*:
    - Actions must strictly follow the [letter] or [WORD] format for validation.
    - Additional text can accompany the action, but the environment will only process valid patterns matching these formats.

## Observation Space
**Reset Observation:**
On reset, the observation provides the initial prompt and the state of the Sudoku grid. For example:
```plaintext
You are Player 0. You are playing Hangman.
The objective of the game is to guess the word by providing one letter guesses or the entire word.
Here is the current state of the Hangman grid. Each column is numbered.
The cells that need to be populated with letters are represented by '_'.

Current Hangman Grid:
C00 C01 C02 C03 C04 C05 C06
  _   _   _   _   _   _   _ 

There are two ways you can answer. You can provide one letter guesses in the format of [L], or you can guess the entire word in the format of [LIGHT].
If the given letter is in the word, it will be revealed in the grid.
If the given word is correct, you win.
As you play, the history of your choices will be appended below. Use the information to figure out the word and win.
```

**Step Obervation:**
After each step, the environment returns the action and the updated Sudoku grid as the observation. For example:

```plaintext
Let's start by guessing a letter. I will guess the letter [E]
```

By default, the environment returns observations in the following format:
```python
{
  player_id: int : [
    (sender_id: int, message: str),
    ...
  ]
}
```
where each step can product zero, one or many message tuples.

## Gameplay
- **Word Length:** The length of the word varies depending on the difficulty mode.
- *Turns:* The player guesses one letter or the entire word in each turn.
- **Letter Guessing:** The player suggests a letter, and if the letter exists in the word, it will be revealed in all its positions.
- **Word Guessing:** The player can guess the entire word at once. If correct, the player wins immediately; if incorrect, the guess counts as a failed attempt.
- **Winning Condition:** The game is won when all letters in the word are revealed or the entire word is correctly guessed.
- **Draw Condition:** The player draws if they exceed the maximum number of incorrect attempts (default is 6).

## Key Rules
- **Valid Moves:**

    - The player must enter a single letter in the format [L] (e.g., [A]) or guess the entire word in the format [WORD] (e.g., [LIGHT]).
    - The move must be a valid letter not previously guessed. If it's a word, it must match the hidden word exactly to win.

- **Invalid Moves:**

    - Entering an invalid letter format or a word that doesn't match the hidden word results in a penalty.
    - Repeating a previously guessed letter will also be marked as invalid and will not affect the game state.
    - The player loses one of their allowed attempts for each incorrect guess.

## Rewards
| Outcome          | Reward for Player |
|------------------|:-----------------:|
| **Win**          |       `+1`        |
| **Draw**         |       `0`         |
| **Invalid Move** |       `-1`        |

## Parameters
- `hardcore` (`bool`)
- Description: Sets the difficulty level of the game by determining the vocabulary size from which words are chosen.
- Impact:

    - False (default): The game uses a basic word list (en-basic), making it easier for players with common and shorter words.
    - True: The game uses a larger and more challenging vocabulary (en), featuring less common and longer words, making it suitable for advanced players.

## Variants

| Env-id                | hardcore |
|-----------------------|:--------:|
| `Hangman-v0`          | `False`  |
| `Hangman-v0-hardcore` |  `True`  |

## Example Usage

```python
import textarena as ta

## initializa the environment
env = ta.make("Hangman-v0")

## Wrap the environment for easier observation handling
env = ta.wrappers.LLMObservationWrapper(env=env)

## Wrap the environment for pretty rendering
env = ta.wrappers.PrettyRenderWrapper(env=env)

## initalize agents
agent0  = ta.default_agents.GPTAgent(model="gpt-4o-mini")

## reset the environment to start a new game
observations = env.reset(seed=490)

## Write the game loop
done = False
while not done:
    for player_id, agent in enumerate([agent0]):
        ## get the current observation for the player
        obs = observations[player_id]

        ## Get the agent to use the observation and make an action
        action = agent(obs) 

        ## use the action and execute in the environment
        observation, rewards, truncated, terminated, info = env.step(player_id, action)

        ## render the environment
        env.render()

        ## check if the game has ended
        done = truncated or terminated

## Finally, print the game results
for player_id, agent in enumerate([agent0]):
    print(f"{agent.agent_identifier}: {rewards[player_id]}")
print(f"Reason: {info['reason']}")
```

## Troubleshooting

**TODO**


## Version History
- **v0**
  - Initial release 


### Contact
If you have questions or face issues with this specific environment, please reach out directly to bobby_cheng@i2r.a-star.edu.sg