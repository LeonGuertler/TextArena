import os
import json
import requests
from typing import Optional, Tuple, Dict, Any

import textarena as ta


ROOT = f"https://three.arcprize.org:443"
ARC_API_KEY = os.getenv("ARC_API_KEY")

if not ARC_API_KEY:
    raise ValueError(
        "ARC_API_KEY not found in environment – set it before running the env."
    )

HEADERS = {
    "X-API-Key": ARC_API_KEY,
    "Accept":    "application/json",
}



def _request(method: str, path: str, **kwargs) -> Any:
    url = f"{ROOT}{path}"
    with requests.Session() as session:
        session.headers.update(HEADERS)
        r = session.request(method.upper(), url, timeout=30, **kwargs)
    if not r.ok: raise RuntimeError(f"{method} {url} → {r.status_code}: {r.text[:200]}")
    return r.json()


def _get_available_games():
    curl --request GET \
  --url https://three.arcprize.org/api/games \
  --header 'Accept: application/json' \
  --header 'X-API-Key: 123'

[
  {
    "game_id": "ls20-016295f7601e",
    "title": "LS20"
  },
  {
    "game_id": "ft09-16726c5b26ff",
    "title": "FT09"
  }
]

def _open_score_card():
    curl --request POST \
  --url https://three.arcprize.org/api/scorecard/open \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: 123' \
  --data '{}'

{
  "card_id": "8bb3b1b8-4b46-4a29-a13b-ad7850a0f916"
}

def _reset_game(game_id, score_card_id)
curl --request POST \
  --url https://three.arcprize.org/api/cmd/RESET \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: 123' \
  --data '{
  "game_id": "ls20-016295f7601e",
  "card_id": "8bb3b1b8-4b46-4a29-a13b-ad7850a0f916"
}'

{
  "game_id": "ls20-016295f7601e",
  "guid": "2fa5332c-2e55-4825-b5c5-df960d504470",
  "frame": [
    [
      [
        0,
        0,
        0,
        "…"
      ],
      [
        "…"
      ]
    ]
  ],
  "state": "NOT_FINISHED",
  "score": 0,
  "win_score": 254,
  "action_input": {
    "id": 0,
    "data": {}
  }
}


class ArcAgi3Env(ta.Env):

    def __init__(self, game_name: str = "ls20"):
        super().__init__()
        self.game_name = game_name
        self.session_id: Optional[str] = None   # returned by /api/play
        self.latest_obs: Optional[str] = None   # current textual observation

    # ------------------------------------------------------------------ reset
    def reset(self, num_players: int, seed: Optional[int] = None):

        # 1. get available games
        # 2. filter out game_id by game_name (will be "title" in return from the func)
        # 3. open score card
        # 4. reset game
        """
        Creates a *fresh* ARC-AGI-3 session.

        POST /api/play
        {
            "game_id": "<game_name>",
            "agent":   "textarena_env",  # any tag you like
            "seed":    <optional>
        }
        """

        payload = {"game_id": self.game_name, "agent": "textarena_env"}
        data = _request("POST", "/api/play", json=payload)
        # Typical response shape (based on docs & quick‑start examples):
        # {
        #   "session_id": "123abc",
        #   "observation": "You are in a dark room...",
        #   "turn": 0,
        #   "done": false
        # }
        input(data)
        self.session_id = data["session_id"]
        self.latest_obs = data["observation"]

        # Initialise TextArena state
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed)
        self.state.reset(
            game_state={"turn": data["turn"]},
            player_prompt_function=self._player_prompt,
        )
        self.state.add_observation(
            message=self.latest_obs,
            observation_type=ta.ObservationType.INITIAL_OBSERVATION,
        )

    # ----------------------------------------------------------- player prompt
    def _player_prompt(self, player_id: int, _game_state: Dict[str, Any]) -> str:
        return (
            "You are playing an ARC‑AGI‑3 benchmark game. "
            "Respond with your **single action** for this turn. "
            "Actions should be wrapped in square brackets, e.g. `[push lever]`.\n\n"
            "Here is your current observation:\n"
            f"{self.latest_obs}"
        )

    # ------------------------------------------------------------------ _update
    def _update_state_from_server(self, server_response: Dict[str, Any]) -> None:
        """
        Mutate self.state with the latest info from the API.
        """
        self.latest_obs = server_response["observation"]
        self.state.game_state["turn"] = server_response["turn"]

        # Add the environment’s reply as an observation
        self.state.add_observation(
            message=self.latest_obs,
            observation_type=ta.ObservationType.GAME_BOARD,
        )

    # ------------------------------------------------------------------- step
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Sends the agent's action to ARC‑AGI‑3, updates local state,
        and returns the usual `(done, info)` tuple.
        """
        if self.session_id is None:
            raise RuntimeError("Call reset() before step().")

        # Log the player's action
        self.state.add_observation(
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )

        # POST /api/play/<session_id>
        payload = {"action": action}
        data = _request("POST", f"/api/play/{self.session_id}", json=payload)

        # Example response:
        # {
        #   "observation": "...",
        #   "reward": 0.0,
        #   "done": false,
        #   "turn": 1
        # }
        self._update_state_from_server(data)

        done = bool(data["done"])
        if done:
            # Mark outcome within TextArena
            self.state.set_outcome(
                reward=float(data.get("reward", 0.0)),
                reason="Episode finished from server.",
            )
        elif self.state.check_turn_limit():
            # optional local turn limit safety
            self.state.set_outcome(reward=0.0, reason="Turn limit reached.")

        # Hand back the standard TA tuple
        return self.state.step()