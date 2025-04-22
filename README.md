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
| Crosswords            | 1        | ✅           | ❌          | —                                                                    |
| FifteenPuzzle         | 1        | ✅           | ❌          | —                                                                    |
| GuessTheNumber        | 1        | ✅           | ❌          | —                                                                    |
| GuessWho              | 1        | ✅           | ❌          | —                                                                    |
| Hangman               | 1        | ✅           | ❌          | —                                                                    |
| LogicPuzzle           | 1        | ✅           | ❌          | —                                                                    |
| Mastermind            | 1        | ✅           | ❌          | —                                                                    |
| MathProof             | 1        | ❌           | ❌          | —                                                                    |
| Minesweeper           | 1        | ✅           | ❌          | —                                                                    |
| Sudoku                | 1        | ✅           | ❌          | —                                                                    |
| TowerOfHanoi          | 1        | ✅           | ❌          | —                                                                    |
| TwentyQuestions       | 1        | ✅           | ❌          | —                                                                    |
| WordLadder            | 1        | ✅           | ❌          | —                                                                    |
| WordSearch            | 1        | ✅           | ❌          | —                                                                    |
| Wordle                | 1        | ✅           | ❌          | —                                                                    |
| | | |
| AirLandAndSea †       | 2        | ❌           | ❌          | —                                                                    |
| BattleOfSexes ‡       | 2        | ❌           | ❌          | —                                                                    |
| Battleship            | 2        | ✅           | ❌          | —                                                                    |
| Brass                 | 2        | ❌           | ❌          | —                                                                    |
| Breakthrough ¶        | 2        | ✅           | ❌          | —                                                                    |
| Checkers              | 2        | ✅           | ❌          | —                                                                    |
| Chess                 | 2        | ✅           | ✅          | —                                                                    |
| ConnectFour           | 2        | ✅           | ✅          | —                                                                    |
| Debate                | 2        | ✅           | ❌          | —                                                                    |
| DontSayIt             | 2        | ✅           | ✅          | —                                                                    |
| DracoGame ‡           | 2        | ❌           | ❌          | —                                                                    |
| DuopolisticCompetition ‡| 2      | ❌           | ❌          | —                                                                    |
| EscalationGame ‡      | 2        | ❌           | ❌          | —                                                                    |
| Hive †                | 2        | ❌           | ❌          | —                                                                    |
| HotColdGame ‡         | 2        | ❌           | ❌          | —                                                                    |
| IntegrativeDistributiveNegotiation §| 2 | ❌     | ❌          | —                                                                   |
| IteratedPrisonersDilemma | 2     | ✅           | ❌          | —                                                                    |
| Jaipur                | 2        | ❌           | ❌          | —                                                                    |
| KuhnPoker ¶           | 2        | ✅           | ❌          | —                                                                    |
| LetterAuction         | 2        | ✅           | ❌          | —                                                                    |
| MemoryGame            | 2        | ✅           | ❌          | —                                                                    |
| MonopolyGame ‡        | 2        | ❌           | ❌          | —                                                                    |
| Nim ¶                 | 2        | ✅           | ❌          | —                                                                    |
| Othello (Reversi)     | 2        | ✅           | ❌          | —                                                                    |
| PigDice ¶             | 2        | ✅           | ❌          | —                                                                    |
| PrisonersDilemma ‡    | 2        | ❌           | ❌          | —                                                                    |
| Santorini †           | 2        | ❌           | ❌          | —                                                                    |
| ScenarioPlanning      | 2        | ✅           | ❌          | —                                                                    |
| SeaBattle †           | 2        | ❌           | ❌          | —                                                                    |
| SimpleBlindAuction ¶  | 2        | ✅           | ❌          | —                                                                    |
| SimpleNegotiation     | 2        | ✅           | ✅          | —                                                                    |
| SpellingBee           | 2        | ✅           | ✅          | —                                                                    |
| SpiteAndMalice        | 2        | ✅           | ✅          | —                                                                    |
| StagHunt ‡            | 2        | ❌           | ❌          | —                                                                    |
| Stratego              | 2        | ✅           | ✅          | —                                                                    |
| Taboo                 | 2        | ✅           | ❌          | —                                                                    |
| Tak                   | 2        | ✅           | ✅          | —                                                                    |
| TicTacToe             | 2        | ✅           | ✅          | —                                                                    |
| TriGame ‡             | 2        | ❌           | ❌          | —                                                                    |
| TruthAndDeception     | 2        | ✅           | ✅          | —                                                                    |
| UltimateTicTacToe     | 2        | ✅           | ✅          | —                                                                    |
| WaitGoGame ‡          | 2        | ❌           | ❌          | —                                                                    |
| WordChains            | 2        | ✅           | ✅          | —                                                                    |
| | | |
| ArcticScavengers †    | 3+       | ❌           | ❌          | —                                                                    |
| AreYouTheTraitor †    | 3+       | ❌           | ❌          | —                                                                    |
| BlindAuction          | 3–15     | ✅           | ❌          | —                                                                    |
| CharacterConclave     | 3–15     | ✅           | ❌          | —                                                                    |
| Codenames†            | 4        | ❌           | ❌          | —                                                                    |
| LiarsDice             | 2–15     | ✅           | ✅          | —                                                                    |
| Negotiation           | 3–15     | ✅           | ❌          | —                                                                    |
| Pit †                 | 3+       | ❌           | ❌          | —                                                                    |
| Poker                 | 2–15     | ✅           | ✅          | —                                                                    |
| Snake                 | 2–15     | ✅           | ✅          | —                                                                    |
| Surround              | 2–15     | ✅           | ❌          | —                                                                    |
| TwoRoomsAndABoom †    | 6+       | ❌           | ❌          | —                                                                    |
| Diplomacy             | 3–7      | ✅           | ❌          | —                                                                    |
| 7 Wonders             | 3+       | ❌           | ❌          | —                                                                    |
| Bohnanza              | 3+       | ❌           | ❌          | —                                                                    |
| Codenames             | 4+       | ❌           | ❌          | —                                                                    |
| Risk                  | 3+       | ❌           | ❌          | —                                                                    |
| SettlersOfCatan       | 2–4      | ❌           | ❌          | —                                                                    |
| TerraformingMars      | 1–5      | ❌           | ❌          | —                                                                    |
| Werewolf              | 5+       | ❌           | ❌          | —                                                                    |

† Games from [LLM Arena: Studying the Impact of Domain Expertise and Problem Complexity in LLM Competitions](https://arxiv.org/pdf/2406.06613)

‡ Games from [Language Model Negotiations: Theory-of-Mind vs. Complexity of the Game](https://arxiv.org/pdf/2411.05990)

§ Games from [Negotiating with Humans by LLMs via Strategic Reasoning](https://arxiv.org/pdf/2401.04536)

¶ These games were added because they are part of [Language Models Make Better Players than Solvers in Cooperative Games](https://arxiv.org/pdf/2402.12348)
