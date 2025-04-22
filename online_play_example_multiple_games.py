import textarena as ta
import time 
 
model_name = "MODEL_NAME"
model_description = f"MODEL_DESCRIPTION"
email = "EMAIL"


for _ in range(5):
    try:
        # Initialize agent
        agent = ta.agents.wrappers.AnswerTokenAgentWrapper(ta.agents.OpenRouterAgent(model_name=model_name))


        env = ta.make_online(
            env_id=["all"], 
            model_name=model_name,
            model_description=model_description,
            email=email
        )
        env = ta.wrappers.LLMObservationWrapper(env=env)


        env.reset(num_players=1)

        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = agent(observation)
            done, info = env.step(action=action)
        env.close()
        print(info)
        print(f"DONE - sleeping")
        time.sleep(10)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        time.sleep(30)