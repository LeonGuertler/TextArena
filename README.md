<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/ta_black.svg">
  <img alt="TextArena logo" src="/docs/ta_white.svg" width="25%" height="25%">
</picture>
  
A suite of 100+ {single,two,multi}-Player texted based games for benchmarking and training of LLMs.

<h3>

[Play](https://textarena.ai) | [Leaderboard](https://textarena.ai/leaderboard) | [Games](https://github.com/LeonGuertler/TextArena/blob/main/textarena/envs/README.md) | [Examples](https://github.com/LeonGuertler/TextArena/tree/main/examples)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/LeonGuertler/TextArena)](https://github.com/LeonGuertler/TextArena/stargazers)
[![PyPI Downloads](https://static.pepy.tech/badge/textarena)](https://pepy.tech/projects/textarena)
[![Discord](https://img.shields.io/discord/1257951838322561075?color=%237289DA&label=TextArena%20Discord&logo=discord&logoColor=white)](https://discord.gg/KPacHzK23e)
[![PyPI version](https://img.shields.io/pypi/v/textarena.svg)](https://pypi.org/project/textarena)

</div>

## Updates
* **02/10/2025** Enhanced **VendingMachine** environment with multi-item inventory management, lead times, holding costs, and dynamic news events for complex economic simulations.
* **31/07/2025** We added **SettlersOfCatan** to TextArena!
* **14/07/2025** Announcing **MindGames** a NeurIPS2025 competition for training LLMs on various TextArena games that require theory of mind.
* **01/07/2025** Release of v0.6.9 with **100** games and simplified states, new observation wrappers for training and default wrappers for environments. 
* **01/07/2025** Release of __SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning__ introducing RL via self-play on TextArena games as a potential new training paradigm.
* **22/06/2025** Release of [UnstableBaselines](https://github.com/LeonGuertler/UnstableBaselines) a light weight async online RL library for training LLMs on TextArena games. 
* **16/04/2025** Release of the TextArena paper 
* **14/02/2025** Release of the new, stable version for both pip and the website
* **31/01/2025** Initial demo release highlighted by Andrej Karpathy (crashing all our servers)


## Introduction
**TextArena** is a flexible and extensible framework for training, evaluating, and benchmarking models in text-based games. It follows an OpenAI Gym-style interface, making it straightforward to integrate with a wide range of reinforcement learning and language model frameworks.


## Getting Started

### Installation
Install TextArena directly from PyPI:
```bash
pip install textarena
```

### Offline Play
The only requirement __Agents__ need to fulfill is having a __call__ function that accepts string observations and returns string action. We have implemented a number of basic agents that you can find [here](https://github.com/LeonGuertler/TextArena/blob/main/textarena/agents/basic_agents.py). 

#### Example 1: TicTacToe
In this example, we show how you can let **GPT-4o-mini** play against **anthropic/claude-3.5-haiku** in a game of __TicTacToe__.

We will be using the OpenRouterAgent, so first you need to set you OpenRouter API key:
```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
```

Now we can build the models and let them play:

```python
import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
}

# Initialize the environment
env = ta.make(env_id="TicTacToe-v0")

# wrap it for additional visualizations
env = ta.wrappers.SimpleRenderWrapper(env=env) 

env.reset(num_players=len(agents))

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close()
```

#### Example 2: Multi-Item Vending Machine
TextArena also includes complex economic simulation games. Here's an example of the **VendingMachine** environment with multiple items, lead times, and dynamic news events:

```python
import textarena as ta
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-key-here"

# Initialize agents
agents = {
    0: ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt="You are a VM controller..."),
    1: ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt="You are a customer..."),
}

# Initialize the multi-item vending machine
env = ta.make(env_id="VendingMachine-v0")

# Configure multiple items with different economics
env.add_item(item_id="cola", description="Cola", lead_time=1, price=7, cost=4, holding_cost=0.5)
env.add_item(item_id="chips", description="Chips", lead_time=2, price=5, cost=3, holding_cost=0.3)
env.add_item(item_id="water", description="Water", lead_time=0, price=3, cost=2, holding_cost=0.2)

# Add dynamic news events that affect demand
env.add_news(day=2, news="Weekend Sale: Expect 30% higher demand for all items")
env.add_news(day=6, news="Baseball game: Expect 50% higher demand for popcorn")

env.reset(num_players=2)

# Game loop with custom observation wrapper for context management
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, _ = env.step(action=action)

rewards, game_info = env.close()
print(f"VM Total Reward: ${rewards[0]:.2f}")
```

**Key Features of VendingMachine:**
- **Multi-item inventory management** with different lead times and costs
- **Dynamic news events** that agents can anticipate and plan for
- **Economic complexity** with holding costs, profit margins, and procurement planning
- **Custom observation wrapper** providing complete context history and role-specific visibility
- **Realistic supply chain mechanics** with order pipelines and delivery delays



## Citation [![arXiv](https://img.shields.io/badge/arXiv-2504.11442-b31b1b.svg)](https://arxiv.org/abs/2504.11442)

If you use **TextArena** in your research, please cite:

```bibtex
@misc{guertler2025textarena,
    title={TextArena}, 
    author={Leon Guertler and Bobby Cheng and Simon Yu and Bo Liu and Leshem Choshen and Cheston Tan},
    year={2025},
    eprint={2504.11442},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.11442}, 
}
```



## How to Contribute:
If you have any questions at all, feel free to reach out on discord. The below issues are great starting points if you want to contribute:
- Transfer the 'How to Contribute' from here to individual issues
- Make RushHour board generation algorithmic
- extend Fifteenpuzzel to arbitrary sizes
- Add a nice end-of-game screen to the SimpleRenderWrapper visualizations
