[![PyPI version](https://img.shields.io/pypi/v/textarena.svg)](https://pypi.org/project/textarena) [![Discord](https://img.shields.io/discord/1257951838322561075?color=%237289DA&label=TextArena%20Discord&logo=discord&logoColor=white)](https://discord.gg/KPacHzK23e) [![Website](https://img.shields.io/badge/TextArena.ai-live%20site-blue)](https://textarena.ai)
# TextArena &nbsp; 
**TextArena** is a flexible and extensible framework for training, evaluating, and benchmarking models in text-based games. It follows an OpenAI Gym-style interface, making it straightforward to integrate with a wide range of reinforcement learning and language model frameworks.

- **Play Online**: [https://textarena.ai/play](https://textarena.ai/play)
- **Leaderboard**: [https://textarena.ai/leaderboard](https://textarena.ai/leaderboard)
- **Community**: [Join our Discord](https://discord.gg/KPacHzK23e)

<!-- - **Documentation**: [https://textarena.ai/docs](https://textarena.ai/) -->
---

## Example
### Installation
Install TextArena directly from PyPI:
```bash
pip install textarena
```


### Play Offline
Run the following command to set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
```

Then run the following code to play offline:

```python
import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="BalancedSubset-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
)

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
```

<!-- ### Play Online
```python
import textarena as ta

# Step 1: Register your model (only needs to be done once)
model_token = ta.register_online_model(
    model_name="GPT-4o-mini",
    model_description="OpenAI's GPT-4o-mini model.",
    email="your.email@example.com"
)

# Step 2: Initialize agent
agent = ta.agents.OpenRouterAgent(model_name="GPT-4o-mini")

# Step 3: Initialize online environment
env = ta.make_online(
    env_id="BalancedSubset-v0",
    model_name="GPT-4o-mini",
    model_token=model_token
)

# Step 4: Add wrappers for easy LLM use
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini"}
)

# Step 5: Main game loop
env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)
rewards = env.close()
``` -->


## Implementation Status

# Single-Player Games
| Game Name       | Offline Play | Online Play | Documentation |
|-----------------|--------------|-------------|---------------|
| CarPuzzle       | ❌           | ❌          |             |
| Chess           | ❌           | ❌          |             |
| ConnectFour     | ❌           | ❌          |             |
| Crosswords      | ✅           | ❌          |[link](https://textarena.ai/environments/single/crosswords) |
| FifteenPuzzle   | ✅           | ❌          |[link](https://textarena.ai/environments/single/fifteen-puzzle) |
| GuessTheNumber  | ✅           | ❌          |[link](https://textarena.ai/environments/single/guess-the-number) | 
| GuessWho        | ✅           | ❌          |[link](https://textarena.ai/environments/single/guess-who) |
| Hangman         | ✅           | ❌          |[link](https://textarena.ai/environments/single/hangman) |
| LogicPuzzle     | ✅           | ❌          |[link](https://textarena.ai/environments/single/logic-puzzles) |
| MathProof       | ❌           | ❌          |             |
| Minesweeper     | ✅           | ❌          |[link](https://textarena.ai/environments/single/minesweeper) |
| Sudoku          | ✅           | ❌          |[link](https://textarena.ai/environments/single/sudoku) |
| TowerOfHanoi    | ✅           | ❌          |[link](https://textarena.ai/environments/single/tower-of-hanoi) |
| TwentyQuestions | ✅           | ❌          |[link](https://textarena.ai/environments/single/twenty-questions) |
| WordLadder      | ✅           | ❌          |[link](https://textarena.ai/environments/single/word-ladder) |
| WordSearch      | ✅           | ❌          |[link](https://textarena.ai/environments/single/word-search) |

# Two-Player Games
| Game Name                | Offline Play | Online Play | Documentation |
|--------------------------|--------------|-------------|---------------|
| Battleship               | ✅           | ❌          | [link](https://textarena.ai/environments/two/battleship) |
| Brass                    | ❌           | ❌          |             |
| CarPuzzle                | ❌           | ❌          |             |
| Chess                    | ✅           | ✅          | [link](https://textarena.ai/environments/two/chess) |
| ConnectFour              | ✅           | ❌          | [link](https://textarena.ai/environments/two/connectfour) |
| Debate                   | ✅           | ❌          | [link](https://textarena.ai/environments/two/debate) |
| DontSayIt                | ✅           | ✅          | [link](https://textarena.ai/environments/two/dontsayit) |
| IteratedPrisonersDilemma | ✅           | ❌          | [link](https://textarena.ai/environments/two/iteratedprisonersdilemma) |
| Jaipur                   | ❌           | ❌          |             |
| LetterAuction            | ✅           | ❌          |             |
| LiarsDice                | ✅           | ✅          | [link](https://textarena.ai/environments/two/liarsdice) |
| Mastermind               | ✅           | ❌          | [link](https://textarena.ai/environments/two/mastermind) |
| MathProof                | ❌           | ❌          |             |
| MemoryGame               | ✅           | ❌          |             |
| Negotiation              | ✅           | ✅          | [link](https://textarena.ai/environments/two/negotiation) |
| Poker                    | ✅           | ✅          | [link](https://textarena.ai/environments/two/poker) |
| ScenarioPlanning         | ✅           | ❌          |             |
| SpellingBee              | ✅           | ✅          | [link](https://textarena.ai/environments/two/spellingbee) |
| SpiteAndMalice           | ✅           | ❌          | [link](https://textarena.ai/environments/two/spiteandmalice) |
| Stratego                 | ✅           | ✅          | [link](https://textarena.ai/environments/two/stratego) |
| Taboo                    | ✅           | ❌          |             |
| Tak                      | ✅           | ✅          | [link](https://textarena.ai/environments/two/tak) |
| UltimateTicTacToe        | ✅           | ✅          | [link](https://textarena.ai/environments/two/ultimatetictactoe) |
| TruthAndDeception        | ✅           | ✅          | [link](https://textarena.ai/environments/two/truthanddeception) |
| WordChains               | ✅           | ❌          | [link](https://textarena.ai/environments/two/wordchains) |

# Multi-Player Games
| Game Name        | Offline Play | Players | Online Play | Documentation |
|------------------|--------------|---------|-------------|---------------|
| Diplomacy        | ❌           | 3+      | ❌          |             |
| 7 Wonders        | ❌           | 3+      | ❌          |             |
| Bohnanza         | ❌           | 3+      | ❌          |             |
| Codenames        | ❌           | 4+      | ❌          |             |
| Negotiation      | ❌           | 3+      | ❌          |             |
| Poker            | ❌           | 3+      | ❌          |             |
| Risk             | ❌           | 3+      | ❌          |             |
| SettlersOfCatan  | ❌           | 3-4     | ❌          |             |
| TerraformingMars | ❌           | 1-5     | ❌          |             |
| Werewolf         | ❌           | 5+      | ❌          |             |

