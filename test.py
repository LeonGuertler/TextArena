import textarena as ta
from textarena.envs.two_player.Bomberman.env import TwoPlayerBombermanEnv

# Initialize agents
# agents = {
#     0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
#     1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
# }

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent(),
}


# Initialize environment from subset and wrap it
# env = ta.make(env_id="BalancedSubset-v0")
env = TwoPlayerBombermanEnv(
    width=13,
    height=11,
    wall_density=0.3,
    max_turns=50
)
env = ta.wrappers.LLMObservationWrapper(env=env)
# env = ta.wrappers.ActionFormattingWrapper(env=env)
# env = ta.wrappers.SimpleRenderWrapper(
#     env=env,
#     player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
# )

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()