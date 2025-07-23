# """ A minimal script showing how to run textarena locally """

# import textarena as ta 


# agents = {
#     0: ta.agents.HumanAgent(),
#     1: ta.agents.HumanAgent(),
#     # 2: ta.agents.HumanAgent(),
#     # 3: ta.agents.HumanAgent(),
#     # 1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-001"),
# }

# # initialize the environment
# env = ta.make(env_id="ConnectFour-v0-train")
# # env = ta.wrappers.SimpleRenderWrapper(env=env) #, render_mode="standard")
# env.reset(num_players=len(agents))

# # main game loop
# done = False 
# while not done:
#   player_id, observation = env.get_observation()
#   action = agents[player_id](observation)
#   done, step_info = env.step(action=action)
# rewards, game_info = env.close()
# print(rewards)
# print(game_info)

""" A minimal script showing how to run textarena locally """

import textarena as ta 
import textarena.external_envs
import textarena.external_envs.ARCAGI3
from textarena.external_envs.ARCAGI3.env import ArcAgi3Env

agents = {
    # 0: ta.agents.HumanAgent(),
    0: ta.agents.OpenRouterAgent(model_name="google/gemini-2.5-pro") #qwen/qwen3-235b-a22b-07-25"), #google/gemini-2.0-flash-001"),
}

# initialize the environment
# env = ta.make(env_id="SimpleTak-v0-train")
env = ArcAgi3Env()
env = ta.wrappers.ActionFormattingWrapper(env=env)
env = ta.wrappers.LLMObservationWrapper(env=env)

# env = ta.wrappers.SimpleRenderWrapper(env=env) #, render_mode="standard")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  print(observation)
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()
print(rewards)
print(game_info)



# Calling: https://three.arcprize.org:443/api/cmd/ACTION1 with {'json': {'game_id': 'ls20-016295f7601e', 'guid': '865810c4-42b6-4d70-85e4-a5b3c7dea68b', 'reasoning': {}}}
# Calling: https://three.arcprize.org:443/api/cmd/ACTION1 with {'json': {'game_id': 'ls20-016295f7601e', 'guid': '865810c4-42b6-4d70-85e4-a5b3c7dea68b', 'reasoning': {}}}
# Calling: https://three.arcprize.org:443/api/cmd/ACTION3 with {'json': {'game_id': 'ls20-016295f7601e', 'guid': '865810c4-42b6-4d70-85e4-a5b3c7dea68b', 'reasoning': {}}}