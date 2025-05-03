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
env = ta.make(env_id="SpellingBee-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
```


### Play Online
If you want to evaluate your model against other submitted models and humans, you can simply change the `.make` to `.make_online`. Please make sure that the model_name is unique and that the email address provided is correct.

```python
import textarena as ta
 
model_name = "Standard GPT-4o LLM"
model_description = "Standard OpenAI GPT-4o model."
email = "guertlerlo@cfar.a-star.edu.sg"


# Initialize agent
agent = ta.agents.OpenRouterAgent(model_name="gpt-4o") 


env = ta.make_online(
    env_id=["SpellingBee-v0", "SimpleNegotiation-v0", "Poker-v0"], 
    model_name=model_name,
    model_description=model_description,
    email=email
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
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

| Game                  | Players  | Offline Play | Online Play | Documentation                                                        |
|-----------------------|----------|--------------|-------------|----------------------------------------------------------------------|
| CarPuzzle             | 1        | ❌           | ❌          | —                                                                    |
| [Crosswords](textarena/envs/Crosswords/README.md)            | 1        | ✅           | ❌          | —                                                                    |
| [FifteenPuzzle](textarena/envs/FifteenPuzzle/README.md)         | 1        | ✅           | ❌          | —                                                                    |
| [GuessTheNumber](textarena/envs/GuessTheNumber/README.md)        | 1        | ✅           | ❌          | —                                                                    |
| [GuessWho](textarena/envs/GuessWho/README.md)              | 1        | ✅           | ❌          | —                                                                    |
| [Hangman](textarena/envs/Hangman/README.md)               | 1        | ✅           | ❌          | —                                                                    |
| [LogicPuzzle](textarena/envs/LogicPuzzle/README.md)           | 1        | ✅           | ❌          | —                                                                    |
| [Mastermind](textarena/envs/Mastermind/README.md)            | 1        | ✅           | ❌          | —                                                                    |
| MathProof             | 1        | ❌           | ❌          | —                                                                    |
| [Minesweeper](textarena/envs/Minesweeper/README.md)           | 1        | ✅           | ❌          | —                                                                    |
| [Sudoku](textarena/envs/Sudoku/README.md)                | 1        | ✅           | ❌          | —                                                                    |
| [TowerOfHanoi](textarena/envs/TowerOfHanoi/README.md)          | 1        | ✅           | ❌          | —                                                                    |
| [TwentyQuestions](textarena/envs/TwentyQuestions/README.md)       | 1        | ✅           | ❌          | —                                                                    |
| [WordLadder](textarena/envs/WordLadder/README.md)            | 1        | ✅           | ❌          | —                                                                    |
| [WordSearch](textarena/envs/WordSearch/README.md)            | 1        | ✅           | ❌          | —                                                                    |
| [Wordle](textarena/envs/Wordle/README.md)                | 1        | ✅           | ❌          | —                                                                    |
| | | |
| AirLandAndSea †       | 2        | ❌           | ❌          | —                                                                    |
| BattleOfSexes ‡       | 2        | ❌           | ❌          | —                                                                    |
| [Battleship](textarena/envs/Battleship/README.md)            | 2        | ✅           | ❌          | —                                                                    |
| Brass                 | 2        | ❌           | ❌          | —                                                                    |
| [Breakthrough](textarena/envs/Breakthrough/README.md) ¶        | 2        | ✅           | ❌          | —                                                                    |
| [Checkers](textarena/envs/Checkers/README.md)              | 2        | ✅           | ❌          | —                                                                    |
| [Chess](textarena/envs/Chess/README.md)                 | 2        | ✅           | ✅          | —                                                                    |
| [ConnectFour](textarena/envs/ConnectFour/README.md)           | 2        | ✅           | ✅          | —                                                                    |
| [Debate](textarena/envs/Debate/README.md)                | 2        | ✅           | ❌          | —                                                                    |
| [DontSayIt](textarena/envs/DontSayIt/README.md)             | 2        | ✅           | ✅          | —                                                                    |
| DracoGame ‡           | 2        | ❌           | ❌          | —                                                                    |
| DuopolisticCompetition ‡| 2      | ❌           | ❌          | —                                                                    |
| EscalationGame ‡      | 2        | ❌           | ❌          | —                                                                    |
| Hive †                | 2        | ❌           | ❌          | —                                                                    |
| HotColdGame ‡         | 2        | ❌           | ❌          | —                                                                    |
| IntegrativeDistributiveNegotiation §| 2 | ❌     | ❌          | —                                                                   |
| [IteratedPrisonersDilemma](textarena/envs/IteratedPrisonersDilemma/README.md) | 2     | ✅           | ❌          | —                                                                    |
| [IteratedRockPaperScissors](textarena/envs/IteratedRockPaperScissors/README.md) | 2     | ✅           | ❌          | —                                                                    |
| Jaipur                | 2        | ❌           | ❌          | —                                                                    |
| [KuhnPoker](textarena/envs/KuhnPoker/README.md) ¶           | 2        | ✅           | ❌          | —                                                                    |
| [LetterAuction](textarena/envs/LetterAuction/README.md)         | 2        | ✅           | ❌          | —                                                                    |
| [MemoryGame](textarena/envs/MemoryGame/README.md)            | 2        | ✅           | ❌          | —                                                                    |
| MonopolyGame ‡        | 2        | ❌           | ❌          | —                                                                    |
| [Nim](textarena/envs/Nim/README.md) ¶                 | 2        | ✅           | ❌          | —                                                                    |
| [Othello](textarena/envs/Othello/README.md) (Reversi)     | 2        | ✅           | ❌          | —                                                                    |
| [PigDice](textarena/envs/PigDice/README.md) ¶             | 2        | ✅           | ❌          | —                                                                    |
| Santorini †           | 2        | ❌           | ❌          | —                                                                    |
| [ScenarioPlanning](textarena/envs/ScenarioPlanning/README.md)      | 2        | ✅           | ❌          | —                                                                    |
| SeaBattle †           | 2        | ❌           | ❌          | —                                                                    |
| [SimpleBlindAuction](textarena/envs/SimpleBlindAuction/README.md) ¶  | 2        | ✅           | ❌          | —                                                                    |
| [SimpleNegotiation](textarena/envs/SimpleNegotiation/README.md)     | 2        | ✅           | ✅          | —                                                                    |
| [SpellingBee](textarena/envs/SpellingBee/README.md)           | 2        | ✅           | ✅          | —                                                                    |
| [SpiteAndMalice](textarena/envs/SpiteAndMalice/README.md)        | 2        | ✅           | ✅          | —                                                                    |
| StagHunt ‡            | 2        | ❌           | ❌          | —                                                                    |
| [Stratego](textarena/envs/Stratego/README.md)              | 2        | ✅           | ✅          | —                                                                    |
| [Taboo](textarena/envs/Taboo/README.md)                 | 2        | ✅           | ❌          | —                                                                    |
| [Tak](textarena/envs/Tak/README.md)                   | 2        | ✅           | ✅          | —                                                                    |
| [SimpleTak](textarena/envs/SimpleTak/README.md)             | 2        | ✅           | ❌          | —                                                                    |
| [TicTacToe](textarena/envs/TicTacToe/README.md)             | 2        | ✅           | ✅          | —                                                                    |
| [ReverseTicTacToe](textarena/envs/ReverseTicTacToe/README.md)      | 2        | ✅           | ❌          | —                                                                    |
| [WildTicTacToe](textarena/envs/WildTicTacToe/README.md)         | 2        | ✅           | ❌          | —                                                                    |
| [QuantumTicTacToe](textarena/envs/QuantumTicTacToe/README.md)      | 2        | ✅           | ❌          | —                                                                    |
| [UltimateTicTacToe](textarena/envs/UltimateTicTacToe/README.md)     | 2        | ✅           | ✅          | —                                                                    |
| TriGame ‡             | 2        | ❌           | ❌          | —                                                                    |
| [TruthAndDeception](textarena/envs/TruthAndDeception/README.md)     | 2        | ✅           | ✅          | —                                                                    |
| WaitGoGame ‡          | 2        | ❌           | ❌          | —                                                                    |
| [WordChains](textarena/envs/WordChains/README.md)            | 2        | ✅           | ✅          | —                                                                    |
| | | |
| [ThreePlayerTicTacToe](textarena/envs/ThreePlayerTicTacToe/README.md)  | 3        | ✅           | ❌          | —                                                                    |
| ArcticScavengers †    | 3+       | ❌           | ❌          | —                                                                    |
| AreYouTheTraitor †    | 3+       | ❌           | ❌          | —                                                                    |
| [BlindAuction](textarena/envs/BlindAuction/README.md)          | 3–15     | ✅           | ❌          | —                                                                    |
| [CharacterConclave](textarena/envs/CharacterConclave/README.md)     | 3–15     | ✅           | ❌          | —                                                                    |
| [Codenames](textarena/envs/Codenames/README.md)             | 4        | ✅           | ❌          | —                                                                    |
| [LiarsDice](textarena/envs/LiarsDice/README.md)             | 2–15     | ✅           | ✅          | —                                                                    |
| [Negotiation](textarena/envs/Negotiation/README.md)           | 3–15     | ✅           | ❌          | —                                                                    |
| Pit †                 | 3+       | ❌           | ❌          | (good for real-time version)                                         |
| [Poker](textarena/envs/Poker/README.md)                 | 2–15     | ✅           | ✅          | —                                                                    |
| [Snake](textarena/envs/Snake/README.md)                 | 2–15     | ✅           | ✅          | —                                                                    |
| [Surround](textarena/envs/Surround/README.md)              | 2–15     | ✅           | ❌          | —                                                                    |
| [TwoRoomsAndABoom](textarena/envs/TwoRoomsAndABoom/README.md) †    | 6+       | ❌           | ❌          | —                                                                    |
| Diplomacy             | 3–7      | ✅           | ❌          | —                                                                    |
| [SecretMafia](textarena/envs/SecretMafia/README.md)           | 5–15     | ✅           | ❌          | —                                                                    |
| 7 Wonders             | 3+       | ❌           | ❌          | —                                                                    |
| Bohnanza              | 3+       | ❌           | ❌          | —                                                                    |
| Risk                  | 3+       | ❌           | ❌          | —                                                                    |
| SettlersOfCatan       | 2–4      | ❌           | ❌          | —                                                                    |
| TerraformingMars      | 1–5      | ❌           | ❌          | —                                                                    |
| Werewolf              | 5+       | ❌           | ❌          | —                                                                    |
| EmojiCharade          | 2-14     | ❌           | ❌          | —                                                                    |

† Games from [GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents](https://arxiv.org/pdf/2406.06613)

‡ Games from [Game-theoretic LLM: Agent Workflow for Negotiation Games](https://arxiv.org/pdf/2411.05990)

§ Games from [Evaluating Language Model Agency Through Negotiations](https://arxiv.org/pdf/2401.04536)

¶ These games were added because they are part of [GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations](https://arxiv.org/pdf/2402.12348)






# How to contribute
Generally speaking the easiest ways of contributing are to either add new environments or complete any of the tasks below:

## Open Tasks

### Adding `create_board_str` functions to 
    - Bullshit
    - Codenames
    - Diplomacy
    - QuantumTicTacToe
    - Stratego
    - Taboo

    - Negotiation
    - LetterAuction
    - Tak
    - SpiteAndMalice



### Other todos
    - complete code for bullshit game
    
