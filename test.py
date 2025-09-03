""" A minimal script showing how to run textarena locally """

import textarena as ta 

def get_full_transcript(full_observations, player_id):
    logs = []
    for from_id, message, obs_type in full_observations[player_id]:
        speaker = "Game" if from_id == -1 else f"Player {from_id}"
        logs.append(f"[{obs_type.name}] {speaker}: {message}")
    return "\n".join(logs)


agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent()
    # 1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-001"),
}

# initialize the environment
env = ta.make(env_id="TicTacToe-v1-raw")
env = ta.wrappers.LLMObservationWrapper(env)
# env = ta.wrappers.SimpleRenderWrapper(env=env) #, render_mode="standard")
env.reset(num_players=len(agents))

# main game loop
done = False 
while not done:
  player_id, observation = env.get_observation()
  action = agents[player_id](observation)
  done, step_info = env.step(action=action)
rewards, game_info = env.close()
env.full_observations[player_id].append((player_id, action, ta.ObservationType.PLAYER_ACTION))
ft = get_full_transcript(env.full_observations, 0)
print(">>>>>>", ft)
print(rewards)
print(game_info)
