"""
Offline evaluation with parallel episodes.

We play N games of each environment in parallel.
Your model (AsyncVllmAgent) is seat‑randomised each game and
plays against a fixed OpenRouter opponent.

The agent’s a_generate() coroutine is awaited concurrently,
so vLLM can batch tokens across *all* running games.
"""
import os, asyncio
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import textarena as ta

NUM_EPISODES     = 32 # per‑environment
PARALLEL_GAMES   = 16 # how many games to run simultaneously
EVAL_ENV_IDS     = [("TicTacToe-v0", 2)]
OPPONENT_NAME    = "google/gemini-2.0-flash-001"
OUT_CSV          = "eval_results/eval_summary.csv"

# AGENTS 
model = ta.agents.AsyncVllmAgent(model_name="Qwen/Qwen3-4B")
opponent = ta.agents.OpenRouterAgent(model_name=OPPONENT_NAME)

async def call_agent(agent, obs: str) -> str:
    if hasattr(agent, "a_generate"): return await agent.a_generate(obs) # our vLLM agent
    # fallback for blocking agents – run in default thread pool
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, agent, obs)

# SINGLE GAME (async version)
async def run_game(env_id: str, num_players: int, mdl, opp) -> Dict[str, Any]:
    env = ta.make(env_id)
    env.reset(num_players=num_players)
    model_pid = np.random.randint(0, num_players)
    done = False
    while not done:
        pid, obs = env.get_observation()
        if pid == model_pid:    action = await call_agent(mdl, obs)
        else:                   action = await call_agent(opp, obs)
        done, _ = env.step(action=action)

    rewards, game_info = env.close()
    return {
        "model_reward": rewards[model_pid],
        "opponent_reward": np.mean([rewards[i] for i in range(num_players) if i != model_pid]),
        "invalid_move": bool(game_info[model_pid]["invalid_move"]),
        "turn_count": game_info[model_pid]["turn_count"],
    }

# PARALLEL EVALUATION PER ENVIRONMENT 
async def evaluate_env(env_id: str, num_players: int) -> Dict[str, Any]:
    stats = dict(
        wins=0, losses=0, draws=0,
        total_reward_model=0.0,
        total_reward_opponent=0.0,
        total_invalid_moves=0,
        total_turns=0,
    )

    sem = asyncio.Semaphore(PARALLEL_GAMES)  # bound concurrency

    async def one_game(_):
        async with sem:
            return await run_game(env_id, num_players, model, opponent)

    # launch all games up‑front
    tasks = [asyncio.create_task(one_game(i)) for i in range(NUM_EPISODES)]
    async for outcome in tqdm_asyncio.as_completed(tasks, total=NUM_EPISODES, desc=f"{env_id}"):
        # W/L/D bookkeeping
        if outcome["model_reward"] > outcome["opponent_reward"]:    stats["wins"] += 1
        elif outcome["model_reward"] < outcome["opponent_reward"]:  stats["losses"] += 1
        else:                                                       stats["draws"] += 1

        # aggregate counters
        stats["total_reward_model"]     += outcome["model_reward"]
        stats["total_reward_opponent"]  += outcome["opponent_reward"]
        stats["total_invalid_moves"]    += int(outcome["invalid_move"])
        stats["total_turns"]            += outcome["turn_count"]

    return stats

# MAIN 
async def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results = defaultdict(list)

    for env_id, nplayers in EVAL_ENV_IDS:
        stats = await evaluate_env(env_id, nplayers)

        # summarise per‑environment
        results["env_id"].append(env_id)
        games = NUM_EPISODES
        results["win_rate"].append(stats["wins"] / games)
        results["loss_rate"].append(stats["losses"] / games)
        results["draw_rate"].append(stats["draws"] / games)
        results["invalid_rate"].append(stats["total_invalid_moves"] / games)
        results["avg_turns"].append(stats["total_turns"] / games)
        results["avg_model_reward"].append(stats["total_reward_model"] / games)
        results["avg_opponent_reward"].append(stats["total_reward_opponent"] / games)

    df = pd.DataFrame(results)
    print("\n=== Evaluation Summary ===")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
