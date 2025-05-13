import json
import random
import unittest
import textarena as ta

import gym
import babyai_text


class TestBabyAIByBot(unittest.TestCase):

    def test_baby_ai_text_gold_path(self):
        """
        This function tests whether the mission can be accomplished by following the gold path provided by BabyAI bot?
        """

        # Initialize agents
        agents = {
            0: ta.agents.HumanAgent(),
        }
        seed = random.randint(0, 10000)

        # Initialize environment from subset and wrap it
        env = ta.make(env_id="BabyAiText-v0", seed=seed)
        env = ta.wrappers.LLMObservationWrapper(env=env)
        # Optional render wrapper
        env = ta.wrappers.SimpleRenderWrapper(
            env=env,
            player_names={0: "Bot"},
            render_mode="board"
        )

        env.reset(num_players=len(agents))
        gold_path = list(reversed(env.gold_path()))
        print(f"{seed=}")
        print(f"Goal: {env.baby_ai_text_env.mission}")
        done = False
        while not done:
            env.get_observation()
            action = gold_path.pop()
            done, info = env.step(action=action.strip())
        env.get_observation()
        env.close()
        assert info["reward"] > 0, "No reward was returned from the environment"


if __name__ == '__main__':
    unittest.main()
