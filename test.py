import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.wrappers.AnswerTokenAgentWrapper(ta.agents.OpenRouterAgent(model_name="GPT-4o")),
    1: ta.agents.wrappers.AnswerTokenAgentWrapper(ta.agents.OpenRouterAgent(model_name="GPT-4o"))
}
# agents = {
#     0: ta.agents.HumanAgent(),
#     1: ta.agents.wrappers.AnswerTokenAgentWrapper(ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"))
# }

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent(),
}






# Initialize environment from subset and wrap it
env = ta.make(env_id="Snake-v0")

env = ta.wrappers.ActionFormattingWrapper(env=env)
env = ta.wrappers.LLMObservationWrapper(env=env)
# env = ta.wrappers.SimpleRenderWrapper(
#     env=env,
#     player_names={0: "GPT-4o-mini", 1: "also GPT-4o-mini"},
# )

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    # input(observation)
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
print(rewards)