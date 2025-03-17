import textarena as ta
from textarena.agents.basic_agents import AWSBedrockAgent
import os

# Initialize AWS Bedrock Agent using environment variables
agents = {
    0: AWSBedrockAgent(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name=os.getenv("AWS_REGION"),
        verbose=True,
    ),
    1: AWSBedrockAgent(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name=os.getenv("AWS_REGION"),
        verbose=True,
    ),
}

# Initialize environment
env = ta.make("Stratego-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Claude 3.5 Sonnet V2", 1: "Claude 3.7 Sonnet"},
)
env.reset()
done = False

# Run the simulation
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)

env.close()
print("Game Over!")