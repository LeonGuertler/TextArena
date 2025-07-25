import re
from typing import Any, Optional, Tuple, Dict

import textarena as ta

try: import reasoning_gym
except ImportError: raise ImportError(f"reasoning_gym not installed. Please install it via 'pip install reasoning-gym'.") 

_BRACKETED_ANY = re.compile(r"\[([^\[\]]+)\]")
class ReasoningGymEnv(ta.Env):
    def __init__(self, reasoning_gym_env_id: str = "rush_hour"):
        self.reasoning_gym_env_id = reasoning_gym_env_id

    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed)
        self.data = reasoning_gym.create_dataset(self.reasoning_gym_env_id, seed=seed)
        data_point = next(iter(self.data))
        self.state.reset(game_state={"data_point":data_point, "question":data_point['question'], "answer":data_point['answer']}, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: dict) -> str:
        return game_state["question"]+"\nPlease solve this puzzle in a singe turn by providing all necessary actions within squared breackets. I.e. '[action1] [action2]'."

    def _extract_bracketed_actions(self, text: str, *, as_str: bool = True, sep: str = " ") -> str | list[str]:
        parts = [p.strip() for p in _BRACKETED_ANY.findall(text)]
        return sep.join(parts) if as_str else parts

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        extracted_action = self._extract_bracketed_actions(action)
        print(f"EXTRACTED ACTION: {extracted_action}")
        reward = self.data.score_answer(answer=extracted_action, entry=self.state.game_state["data_point"])
        self.state.set_outcome(reward=reward)
        return self.state.step()



