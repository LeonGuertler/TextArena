import numpy as np
import math
import re, os, json, requests
from collections import Counter
from typing import Any, Optional, Tuple, Dict

import textarena as ta

"""
TODO
- allow for easier switching between render options
"""

ROOT = "https://three.arcprize.org"
api_key = os.getenv("ARC_API_KEY")
if not api_key: raise ValueError("ARC_API_KEY not found. Please set it via 'export ARC_API_KEY='YOURKEY'.")
HEADERS = {"X-API-Key": api_key, "Accept": "application/json"}
ACTION_RE = re.compile(r"\[A(?P<num>[1-6])(?:\s+(?P<x>-?\d+)\s+(?P<y>-?\d+))?\]", re.I) # [A1]..[A5] or [A6 x y]


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


def glyph_ls20(v: int) -> str:
    if v==3: return " " # render "3" as blank
    if v==4: return "#"
    if 0<=v<=9: return str(v)
    return chr(ord("A") + v - 10)

def glyph_ft09(v: int) -> str:
    match v:
        case 4: return " "
        case 3: return "#"
        case _: return str(v)


def _render_block(mat: np.ndarray, glyph_fn) -> list[str]:
    if mat.size == 0: return []
    rows, cols = mat.shape; out = []
    for r in range(rows): out.append("".join(glyph_fn(int(v)) for v in mat[r]))
    return out

def render_ls20_board(board) -> str:
    arr = np.asarray(board, dtype=int)
    shrunk = mixed_downsample(arr)
    lines = _render_block(shrunk, glyph_fn=glyph_ls20)
    return "\n" + "\n".join(lines)


def render_ft09_board(board) -> str:
    arr = np.asarray(board, dtype=int)
    shrunk = full_downsample(arr)
    lines = _render_block(shrunk, glyph_fn=glyph_ft09)
    W = len(lines[0]) if lines else 0
    top_digits = []
    bot_digits = []
    for c in range(W):
        s = str(c)
        top_digits.append(s[-2] if len(s) >= 2 else " ")  # tens
        bot_digits.append(s[-1])                           # ones
    header_top = "".join(top_digits)
    header_bot = "".join(bot_digits)
    out = []
    out.append("      " + header_top + " ")
    out.append("      " + header_bot + " ")
    out.append("    + " + "-" * W + "+")  # separator line
    for i, line in enumerate(lines): out.append(f" {i:<2} | {line} | {i:<2}")
    return "\n" + "\n".join(out)


def downsample_least(arr: np.ndarray, block: int) -> np.ndarray:
    H, W = arr.shape
    nh, nw = (H + block - 1) // block, (W + block - 1) // block
    out = np.empty((nh, nw), dtype=arr.dtype)
    for i in range(nh):
        for j in range(nw):
            r0, r1 = i * block, min((i + 1) * block, H)
            c0, c1 = j * block, min((j + 1) * block, W)
            vals = arr[r0:r1, c0:c1].ravel().tolist()
            cnt = Counter(vals)
            minc = min(cnt.values())
            candidates = [v for v, c in cnt.items() if c == minc]
            out[i, j] = min(candidates)
    return out

def mixed_downsample(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board)
    H, W = board.shape
    bottom_corner = board[H-11:H, 0:11].copy() # bottom corner
    board[H-11:H, 0:11] = 4 # replace in main 
    board = downsample_least(board, 2) # downsample main
    # downsample bottom_corner
    bottom_corner = bottom_corner[1:9, 1:9] 
    bottom_corner = downsample_least(bottom_corner, 3)
    bottom_corner[bottom_corner==4] = 3
    # insert back into board
    H, W = board.shape
    board[H-4:H-1, 1:4] = bottom_corner
    return board

def full_downsample(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board)
    H, W = board.shape
    board = downsample_least(board, 2)
    return board


def glyph(v: int) -> str:
    if v < 10: return str(v)
    return chr(ord("A") + v - 10)



class ArcAGI3Env(ta.Env):
    def __init__(self, game_name: str = "ls20", max_turns: Optional[int] = 50):
        super().__init__()
        self.game_name = game_name
        self.api = ArcClient()
        self.max_turns = max_turns
        self.session_id: Optional[str] = None
        self.game_id: Optional[str] = None
        self.card_id: Optional[str] = os.getenv("ARC_CARD_ID")
    

    def _get_available_games(self) -> dict[str, str]:
        data = self.api.request("GET", "/api/games")
        return {g["title"].lower(): g["game_id"] for g in data}
    def _reset_game(self) -> dict[str, Any]:                        return self.api.request("POST", "/api/cmd/RESET", json={"game_id": self.game_id, "card_id": self.card_id})
    def _open_score_card(self):                                     return self.api.request("POST", "/api/scorecard/open", json={})
    def _close_score_card(self):                                    return self.api.request("POST", "/api/scorecard/close", json={"card_id": self.card_id})
    # def _execute_action(self, action_num: int) -> dict[str, Any]:   return self.api.request("POST", f"/api/cmd/ACTION{action_num}", json={"game_id": self.game_id, "guid": self.session_id, "reasoning": {}})
    def _execute_action(self, action_num: int, payload: Dict[str, Any]) -> dict[str, Any]:
        body = {"game_id": self.game_id, "guid": self.session_id}
        body.update(payload)
        return self.api.request("POST", f"/api/cmd/ACTION{action_num}", json=body)

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are participating in the Arc-AGI-3 challenge. You will be presented with a number of tricky environments, akin to riddles. You have access to the following six actions:"
            f"\nAction 1:\n\tExplanation: the exact in-game effect depends on the title—for example, it might represent 'move up' or 'select option A'.\n\tExample: '[A1]'."
            f"\nAction 2:\n\tExplanation: the exact in-game effect depends on the title—for example, it might represent 'move down' or 'select option B'.\n\tExample: '[A2]'."
            f"\nAction 3:\n\tExplanation: the exact in-game effect depends on the title—for example, it might represent 'move left' or 'select option C'.\n\tExample: '[A3]'."
            f"\nAction 4:\n\tExplanation: the exact in-game effect depends on the title—for example, it might represent 'move right' or 'select option D'.\n\tExample: '[A4]'."
            f"\nAction 5:\n\tExplanation: the exact in-game effect depends on the title—for example, it might represent 'jump', 'rotate', 'fire' or 'select option E'.\n\tExample: '[A5]'."
            f"\nAction 6:\n\tExplanation: a two-parameter command that supplies explicit X/Y coordinates—to an active game session. Common use-cases include 'click/tap at (x,y)', 'place a tile', or 'shoot a projectile', depending on the game's mechanics.\n\tExample: '[A6 12 25]' (will submit the coordinates x=12 y=25 to the game)."
            f"Keep in mind you may not have to use all of them. Start by testing the simple actions and if they don't suffice work your way to the more complex ones."
            f"without assuming to know how to solve it. Once you understand what your actions do and what the objective is, you can try solving it. There "
            f"is no turn limit at all, but depending on the game, after a number of moves you might re-start from the starting position."
        )

    def _render_board(self):
        if self.game_name == "ls20":    return render_ls20_board(board=self.state.game_state["board"])
        elif self.game_name == "ft09":  return render_ft09_board(board=self.state.game_state["board"]) 
        elif self.game_name == "vc30":  pass
        else:                           raise

    def reset(self, num_players: int, seed: Optional[int] = None):
        games = self._get_available_games()
        self.game_id = games[self.game_name.lower()]
        if not self.card_id:
            print(f"No card_id found (ARC_CARD_ID). Creating new one.")
            self.card_id = self._open_score_card()["card_id"]
            print(f"Your new card-id is: {self.card_id}")
        payload = self._reset_game()
        self.session_id = payload["guid"]
        self.state = ta.SinglePlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        self.state.reset(game_state={"turn": 0, "score": payload["score"], "win_score": payload["win_score"], "board": payload["frame"][-1]}, player_prompt_function=self._prompt)
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)

    def _parse_action(self, text: str) -> Tuple[Optional[int], Dict[str, Any], Optional[str]]:
        matches = ACTION_RE.findall(text.strip())
        if not matches: return None, {}, "No valid [A1-6] token found"
        num_s, x_s, y_s = matches[-1]  # take last match
        num = int(num_s)
        payload: Dict[str, Any] = {}
        if num == 6:
            if not (x_s and y_s):
                return None, {}, "ACTION6 requires coordinates: [A6 x y]"
            payload["x"] = int(x_s)*2 # *2 because it's downfiltered
            payload["y"] = int(y_s)*2 # *2 because it's downfiltered
        return num, payload, None

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.session_id is None: raise RuntimeError("Call reset() before step().")
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        action_num, payload, err = self._parse_action(action)
        if err is not None or action_num is None: self.state.set_invalid_move(reward=0, reason=err or "Invalid action"); return self.state.step()
        try: server = self._execute_action(action_num, payload)
        except Exception as e: self.state.set_invalid_move(reward=0, reason=f"Server error: {e}"); return self.state.step()
        self.state.add_observation(message=f"Executed action: [A{action_num}].", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        self.session_id = server.get("guid", self.session_id)
        score = server.get("score", self.state.game_state["score"])
        self.state.game_state.update(turn=self.state.game_state["turn"] + 1, score=score, board=server.get("frame", [self.state.game_state["board"]])[-1])
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)
        if self.state.check_turn_limit(): self.state.set_outcome(reward=score)
        return self.state.step()

    def close(self):
        self.api.close()
        return self.close()
