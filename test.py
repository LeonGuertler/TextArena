import textarena as ta
 
# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    1: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
}

### Bobby - debugging
env = ta.make(env_id="Mastermind-v0")
# print(f"Raw env type: {type(env)}, terminal_render_keys: {env.terminal_render_keys}")
env = ta.wrappers.LLMObservationWrapper(env=env)
# print(f"After LLM wrapper: {type(env)}, terminal_render_keys: {env.terminal_render_keys}")
env = ta.wrappers.CursesRenderWrapper(env=env, player_names={0: "Player 1", 1: "Player 2"}, enable_logging=False)
# print(f"After Curses wrapper: {type(env)}, terminal_render_keys: {env.terminal_render_keys}")
# env = ta.wrappers.SimpleRenderWrapper(env=env, player_names={0: "Player 1", 1: "Player 2"})

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)

rewards = env.close()
print(rewards, info)
