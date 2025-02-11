import openai
from dotenv import load_dotenv
import textarena as ta
import os

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
env = ta.make(env_id="DontSayIt-v0")

# Wrap the environment for easier observation handling
env = ta.wrappers.LLMObservationWrapper(env=env)

# initalize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="gpt-4o"),
    1: ta.agents.OpenRouterAgent(model_name="gpt-4o-mini")
    }

# reset the environment to start a new game
observations = env.reset(seed=490)

# Game loop
done = False


while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
    print(observation)
rewards = env.close()


# Finally, print the game results
for player_id, agent in agents.items():
    print(f"Player id {player_id}: {rewards[player_id]}")
print(f"Reason: {info['reason']}")
