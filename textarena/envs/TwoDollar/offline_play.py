""" A minimal script showing how to run the Two Dollar negotiation game locally """

import textarena as ta 
import os 

# Example 1: Human vs Human with random roles
print("=== Two Dollar Negotiation Game Demo ===")
print("Example 1: Human vs Human with random roles")

# agents = {
#     0: ta.agents.AWSBedrockAgent(model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',region_name='us-west-2'),
#     1: ta.agents.AWSBedrockAgent(model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',region_name='us-west-2'),
# }

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent()
}

# Initialize the environment with random roles
env = ta.make(env_id="TwoDollar-v0") #, player_roles=["50_cents", "battle_ax"])
env.reset(num_players=len(agents))

print(f"Player 1 role: {env.player_roles[0]['name']}")
print(f"Player 2 role: {env.player_roles[1]['name']}")
print()

# Main game loop
done = False 
while not done:
    player_id, observation = env.get_observation()
    # print(observation)
    action = agents[player_id](observation) 
    done, step_info = env.step(action=action)

rewards, game_info = env.close()

print(f"Final Rewards: {rewards}")
print(f"Game Info: {game_info}")
print()