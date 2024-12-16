import textarena as ta 
from textarena.core import ObservationWrapper, Env, Observations, Info
from typing import Dict, Optional, Tuple, List

__all__ = [
    "LLMObservationWrapper"
]


class LLMObservationWrapper(ObservationWrapper):
    """
    A wrapper for converting environment observations into formatted strings suitable
    for large language models (LLMs). It ensures that duplicate observations are not
    added to the full observations.
    """

    def __init__(self, env: Env):
        """
        Initializes the LLMObservationWrapper.

        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}
        self.state = self.env.state

    def _convert_obs_to_str(self, player_id: int) -> Observations:
        """
        Converts the full observations into formatted strings for each recipient.

        Returns:
            Observations: A dictionary mapping recipient IDs to their formatted observation strings.
        """
        str_observation = ""

        for sender_id, message in self.full_observations[player_id]:
            if sender_id == ta.GAME_ID:
                sender_name = "GAME"
            else:
                sender_name = self.state.role_mapping.get(sender_id, f"Player {sender_id}")
            str_observation += f"\n[{sender_name}] {message}"

        recipient_name = self.state.role_mapping.get(player_id, f"Player {player_id}")
        str_observation += f"\n[{recipient_name}]"

        return str_observation

    def observation(self, player_id: int, observation: Optional[ta.Observations]):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        # Extend the full observations with the current observations without duplicates
        if player_id not in self.full_observations:
            self.full_observations[player_id] = []
            
        for obs in observation:
            if obs not in self.full_observations[player_id]:
                self.full_observations[player_id].append(obs)

        return self._convert_obs_to_str(player_id=player_id)