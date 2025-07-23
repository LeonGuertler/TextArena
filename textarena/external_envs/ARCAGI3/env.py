import re, os, requests
from typing import Any, Optional, Tuple, Dict

import textarena as ta


ROOT = "https://three.arcprize.org"
HEADERS = {"X-API-Key": os.getenv("ARC_API_KEY", ""), "Accept": "application/json"}

class ArcClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{ROOT}{path}"
        r = self.session.request(method.upper(), url, timeout=30, **kwargs)
        if not r.ok: raise RuntimeError(f"{method} {url} → {r.status_code}: {r.text[:200]}")
        return r.json()
    def close(self) -> None: self.session.close()

# def glyph(v: int) -> str:
#     if v < 10: return str(v)
#     return chr(ord("A") + v - 10)

def glyph(cell: int) -> str:
    if cell == 4:  # wall
        return "#"
    if cell == 3:  # walkable floor
        return "."                         # dot is visually light
    if cell == 0:                          # empty / void
        return " "                         # keep it blank
    if 1 <= cell <= 9:                     # single‑digit numbers fit
        return str(cell)
    # 10 → 'a', 11 → 'b', ...
    return chr(ord("a") + cell - 10)


class ArcAgi3Env(ta.Env):
    def __init__(self, game_name: str = "ls20", max_turns: Optional[int] = 30):
        super().__init__()
        self.game_name = game_name
        self.api = ArcClient()
        self.max_turns = max_turns
        self.session_id: Optional[str] = None
        self.game_id: Optional[str] = None
        self.card_id: Optional[str] = None

    def _get_available_games(self) -> dict[str, str]:
        data = self.api.request("GET", "/api/games")
        return {g["title"].lower(): g["game_id"] for g in data}
    def _reset_game(self) -> dict[str, Any]:                        return self.api.request("POST", "/api/cmd/RESET", json={"game_id": self.game_id, "card_id": self.card_id})
    def _close_score_card(self):                                    return self.api.request("POST", "/api/scorecard/close", json={"card_id": self.card_id})
    def _execute_action(self, action_num: int) -> dict[str, Any]:   return self.api.request("POST", f"/api/cmd/ACTION{action_num}", json={"game_id": self.game_id, "guid": self.session_id, "reasoning": {}})

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are participating in the Arc-AGI-3 challenge. You will be presented with a number of tricky environments, "
            f"akin to riddles. You can move in the environment using '[w]', '[a]', '[s]', '[d]'. Try to collect information about the environment first "
            f"without assuming to know how to solve it. Once you understand what your actions do and what the objective is, you can try solving it. There "
            f"is no turn limit at all, but depending on the game, after a number of moves you might re-start from the starting position."
        )

    def reset(self, num_players: int, seed: Optional[int] = None):
        games = self._get_available_games()
        self.game_id = games[self.game_name.lower()]
        self.card_id = "45be354b-54c6-4a13-b755-1eb134514d4e"
        payload = self._reset_game()
        self.session_id = payload["guid"]
        self.state = ta.SinglePlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        self.state.reset(game_state={"turn": 0, "score": payload["score"], "win_score": payload["win_score"], "board": payload["frame"][-1]}, player_prompt_function=self._prompt)
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)

    def _render_board(self):
        return "\n".join("".join(glyph(c) for c in row) for row in self.state.game_state["board"])
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.session_id is None: raise RuntimeError("Call reset() before step().")
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        # m = re.findall(r"\[(w|a|s|d)\]", action.lower())
        # if not m: self.state.set_invalid_move(reward=0, reason="No direction token"); return self.state.step()
        # token = m[-1].group(1)
        # print(f"ACTION TOKEN: {token}")
        m = re.findall(r"\[(w|a|s|d)\]", action.lower())  # → e.g. ['a', 'd']
        if not m:
            self.state.set_invalid_move(reward=0, reason="No direction token")
            return self.state.step()

        token = m[-1]         # last captured letter
        print(f"ACTION TOKEN: {token}")
        dir2num  = {"w": 1, "s": 2, "a": 3, "d": 4}
        action_num = dir2num[token]
        server = self._execute_action(action_num)
        self.state.game_state.update(turn=self.state.game_state["turn"]+1, score=server["score"], board=server["frame"][-1])
        self.session_id = server["guid"]
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)

        # check turn limit
        if self.state.check_turn_limit():
            # somehow get score and return it
            self.state.set_outcome(reward=server["score"])
        return self.state.step()

    def close(self):
        # self._close_score_card() # close scorecard
        self.api.close()
        return self.close()
