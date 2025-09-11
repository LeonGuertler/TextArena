""" A minimal script showing how to run textarena locally """

import textarena as ta 


agents = {
    0: ta.agents.TTTAgent(),
    1: ta.agents.TTTAgent(),
}

# agents = {
#     0: ta.agents.C4Agent(),
#     1: ta.agents.C4Agent(),
# }

# agents = {
#     0: ta.agents.HumanAgent(),
#     1: ta.agents.HumanAgent(),
# }

# initialize the environment
env = ta.make(env_id="TicTacToe-v0", summary_output_path="ttt_summary.json")
# env = ta.make(env_id="ConnectFour-v0", summary_output_path="c4_summary.json")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()

print(f"Rewards: {rewards}")
print(f"Game Info: {game_info}")