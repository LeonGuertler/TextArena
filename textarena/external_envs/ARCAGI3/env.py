import os, json, requests, re
from typing import Optional, Tuple, Dict, Any

import textarena as ta

ROOT = f"https://three.arcprize.org:443"
ARC_API_KEY = os.getenv("ARC_API_KEY")
HEADERS = {"X-API-Key": ARC_API_KEY, "Accept": "application/json"}
if not ARC_API_KEY: raise ValueError("ARC_API_KEY not found in environment - set it before running the env.")

def _request(method: str, path: str, **kwargs) -> Any:
    url = f"{ROOT}{path}"
    with requests.Session() as session:
        session.headers.update(HEADERS)
        r = session.request(method.upper(), url, timeout=30, **kwargs)
    if not r.ok: raise RuntimeError(f"{method} {url} → {r.status_code}: {r.text[:200]}")
    return r.json()

def _get_available_games() -> Dict[str, str]:
    """ Returns a mapping: title (lower-case) -> game_id """
    data = _request("GET", "/api/games")
    return {g["title"].lower(): g["game_id"] for g in data}

def _open_scorecard() -> str:
    """ Opens a fresh score-card and yields its ID. """
    data = _request("POST", "/api/scorecard/open", json={})
    return data["card_id"]

def _reset_game(game_id: str, card_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """ Issues the RESET command and returns the full game payload (frame, guid, score, etc.). """
    payload = {"game_id": game_id, "card_id": card_id}
    return _request("POST", "/api/cmd/RESET", json=payload)

def _execute_action(game_id: str, guid: str, action_num: int) -> Dict[str, Any]:
    path = f"/api/cmd/ACTION{action_num}"
    return _request("POST", path, json={"game_id": game_id, "guid": guid, "reasoning": {}})

def _rb(b): input("\n".join("".join(glyph(c) for c in row) for row in b))

def glyph(v: int) -> str:
    if v < 10: return str(v)
    return chr(ord("A") + v - 10)

class ArcAgi3Env(ta.Env):

    def __init__(self, game_name: str = "ls20"):
        super().__init__()
        self.game_name = game_name

    def _render_board(self):
        return "\n".join("".join(glyph(c) for c in row) for row in self.state.game_state["board"])
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        games = _get_available_games()
        self.game_id = games.get(self.game_name.lower())
        if self.game_id is None: raise ValueError(f"Game '{self.game_name}' not found. Available titles: {', '.join(games.keys())}")
        self.card_id = _open_scorecard()
        # print(f"Score Card id: ", self.card_id)
        payload = _reset_game(self.game_id, self.card_id, seed)
        # print("Response Payload: ", payload)
        self.session_id = payload["guid"] # unique episode handle
        # print(f"SESSION_ID: {self.session_id}")
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"turn": 0, "score": payload["score"], "win_score": payload["win_score"], "board": payload["frame"][0]}, player_prompt_function=self._prompt)
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are participating in the Arc-AGI-3 challenge. You will be presented with a number of tricky environments, "
            f"akin to riddles. You can move in the environment using '[w]', '[a]', '[s]', '[d]'. Try to figure out what you are "
            f"supposed to do to solve each level."
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.session_id is None: raise RuntimeError("Call reset() before step().")
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        m = re.search(r"\[(w|a|s|d)\]", action.lower())
        if not m: self.state.set_invalid_move(reward=0, reason="No [w]/[a]/[s]/[d] found."); return self.state.step()

        token = m.group(1) # w / a / s / d
        dir_map = {"w": "up", "a": "left", "s": "down", "d": "right"}
        direction = dir_map[token]
        action_num = {"up": 1, "down": 2, "left": 3, "right": 4}[direction] # map to ACTION endpoint number

        # print(f"SESSION_ID: {self.session_id}")
        server = _execute_action(self.game_id, self.session_id, action_num)
        # print(server)

        # ---- 3. update local TextArena state ---------------------------
        self.state.game_state["turn"] += 1
        self.state.game_state["score"] = server["score"]
        self.game_id = server["game_id"]
        self.session_id = server["guid"]
        # input(f"SESSION_ID: {self.session_id}")
        # input(len(server["frame"]))
        _rb(b=server["frame"][0])
        self.state.game_state["board"] = server["frame"][-1]  # unwrap outer list
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD) # add fresh board observation
        return self.state.step()

