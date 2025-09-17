import textarena as ta
from textarena.core import ObservationWrapper, Env, Observations
from textarena.agents import OpenRouterAgent
from google.cloud import translate_v2 as translate
from typing import Dict, Any, List, Tuple, Optional
from utils import TRANSLATION_PROMPT
import re

__all__ = ["GoogleTranslationWrapper", "OpenRouterTranslationWrapper"]

class GoogleTranslationWrapper(ObservationWrapper):
    """
    Observation translation wrapper using Google Translate
    """
    def __init__(self, env: Env, target_lang: str, client: any):
        """
        Initializes the wrapper.

        Args:
            env (Env): The TextArena environment to wrap.
            target_lang (str): The language to translate observations into (e.g., 'fr', 'es').
            client (any): An initialized translation client instance.
        """
        super().__init__(env)  
        self.client = translate.Client()
        self.target_lang = target_lang
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}

    def _convert_obs_to_str(self, player_id: int) -> Observations:
        str_observation = ""
        if player_id in self.full_observations:
            for sender_id, message, _ in self.full_observations[player_id]:
                if sender_id == ta.GAME_ID: sender_name = "GAME"
                else:                       sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                str_observation += f"\n[{sender_name}] {message}"
        return str_observation

    def observation(self, player_id: int, observation: Optional[ta.Observations]):
        if observation is None: return self._convert_obs_to_str(player_id=player_id)
        if player_id not in self.full_observations: self.full_observations[player_id] = []

        # translate each incoming observation's message (index 1) and preserve tuple shape
        translated = []
        for obs in observation:
            sender = obs[0]
            msg = "" if obs[1] is None else str(obs[1])
            try:
                res = self.client.translate(msg, target_language=self.target_lang)
                msg_tr = res.get("translatedText", msg)
            except Exception: msg_tr = msg  # on failure, keep original
            rest = tuple(obs[2:]) if len(obs) > 2 else ()
            translated.append((sender, msg_tr) + rest)
        self.full_observations[player_id].extend(translated) # Append translated observations in sequence
        return self._convert_obs_to_str(player_id=player_id)
    
class OpenRouterTranslationWrapper(ObservationWrapper):
    """
    Observation translation wrapper using Open Router
    """
    def __init__(self, env: Env, target_lang: str, model: str = 'google/gemini-2.5-flash'):
        """
        Initializes the wrapper.

        Args:
            env (Env): The TextArena environment to wrap.
            target_lang (str): The language to translate observations into (e.g., 'fr', 'es').
            model (str): The LLM to use via OpenRouter for translation
        """
        super().__init__(env)  
        self.target_lang = target_lang
        self.model = model
        self.translation_agent = OpenRouterAgent(model, TRANSLATION_PROMPT.format(language=target_lang))


    def _convert_obs_to_str(self, player_id: int) -> Observations:
        str_observation = ""
        if player_id in self.full_observations:
            for sender_id, message, _ in self.full_observations[player_id]:
                if sender_id == ta.GAME_ID: sender_name = "GAME"
                else:                       sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                str_observation += f"\n[{sender_name}] {message}"
        return str_observation

    def observation(self, player_id: int, observation: Optional[ta.Observations]):
        if observation is None: return self._convert_obs_to_str(player_id=player_id)
        if player_id not in self.full_observations: self.full_observations[player_id] = []

        # translate each incoming observation's message (index 1) and preserve tuple shape
        translated = []
        for obs in observation:
            sender = obs[0]
            msg = "" if obs[1] is None else str(obs[1])
            
            extracted_text = None
            res = self.translation_agent(observation)
            match = re.search(r'<translation>(.*?)</translation>', res, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
            if extracted_text is None:
                extracted_text = msg

            rest = tuple(obs[2:]) if len(obs) > 2 else ()
            translated.append((sender, extracted_text) + rest)
        self.full_observations[player_id].extend(translated) # Append translated observations in sequence
        return self._convert_obs_to_str(player_id=player_id)