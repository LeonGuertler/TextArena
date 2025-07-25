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

# def glyph_ls20(v: int) -> str:
#     if v < 10: return str(v)
#     return chr(ord("A") + v - 10)

# def _replace_tiles(board_section: np.ndarray, from_val: int=4, to_val: int=3) -> np.ndarray:
#     """Return a copy of board_section where from_val is replaced by to_val."""
#     out = board_section.copy()
#     out[out == from_val] = to_val
#     return out

# def crop_bottom_left(arr: np.ndarray, row_from: int = 52, col_to: int = 12, blank_val: int = 4) -> np.ndarray:
#     """ Set bottom-left rectangle (rows >= row_from, cols <= col_to) to 'blank_val' (3 -> space in glyph) """
#     arr = arr.copy()
#     arr[row_from:, :col_to+1] = _replace_tiles(arr[row_from:, :col_to+1])
#     return arr

# def downsample_blocks(arr: np.ndarray, block: int=2, blank_val: int=3, rule: str = "least") -> np.ndarray:
#     H, W = arr.shape
#     new_H = (H + block - 1) // block
#     new_W = (W + block - 1) // block
#     out = np.empty((new_H, new_W), dtype=arr.dtype)
#     for i in range(new_H):
#         for j in range(new_W):
#             r0, r1 = i * block, min((i + 1) * block, H)
#             c0, c1 = j * block, min((j + 1) * block, W)
#             patch = arr[r0:r1, c0:c1].ravel().tolist()
#             vals = [x for x in patch if x != blank_val]
#             if not vals:
#                 out[i, j] = blank_val
#                 continue
#             counts = Counter(vals)
#             if rule == "least":
#                 minc = min(counts.values())
#                 candidates = [v for v, c in counts.items() if c == minc]
#                 out[i, j] = min(candidates)
#             else: out[i, j] = counts.most_common(1)[0][0] # "majority"
#     return out

def _header_strings(cols: int, row_w: int, col_offset: int = 0) -> tuple[str, str]:
    ones = "".join(str((c + col_offset) % 10) for c in range(cols))
    tens = "".join(str(((c + col_offset) // 10) % 10) if (c + col_offset) >= 10 else " " for c in range(cols))
    pad  = " " * (row_w + 1)
    return f"{pad}{ones}", f"{pad}{tens}"

def _render_block(mat: np.ndarray, row_offset: int, col_offset: int, glyph_fn, add_border: bool = True) -> list[str]:
    if mat.size == 0: return []
    rows, cols = mat.shape
    row_w = max(2, len(str(row_offset + rows - 1)))
    h1, h2 = _header_strings(cols, row_w, col_offset=col_offset)
    horiz = "-" * cols
    border = " " * row_w + f"+{horiz}+"
    out = [h1, h2]
    if add_border: out.append(border)
    for r in range(rows):
        row_label = str(row_offset + r).rjust(row_w)
        row_cells = "".join(glyph_fn(int(v)) for v in mat[r])
        if add_border:  out.append(f"{row_label}|{row_cells}|")
        else:           out.append(f"{row_label} {row_cells}")
    if add_border: out.append(border)
    return out


def render_ls20_board(board, split_row: int = 4, block: int = 2, blank_val: int = 3, crop_row_from: int = 52, crop_col_to: int = 20) -> str:
    arr = np.asarray(board, dtype=int)

    # 1) crop the bottom-left corner
    # rest = crop_bottom_left(arr, row_from=crop_row_from, col_to=crop_col_to, blank_val=blank_val)

    # 3) downsample the rest
    # shrunk = downsample_blocks(rest, block=block, blank_val=blank_val) if rest.size else np.empty((0, 0), dtype=int)
    shrunk = mixed_downsample(arr)
    # 4) render
    lines = []
    bot_lines = _render_block(shrunk, row_offset=split_row, col_offset=0, glyph_fn=glyph_ls20, add_border=True)

    lines.append(f"--- downsampled by {block}×{block} from row {split_row} onwards ---")
    lines.extend(bot_lines)

    return "\n" + "\n".join(lines)

# import math
# import math

# def _place_bottom_left_into_base(orig: np.ndarray,
#                                  base_ds: np.ndarray,
#                                  block: int,
#                                  bl_block: int,
#                                  crop_row_from: int,
#                                  crop_col_to: int,
#                                  blank_val: int,
#                                  rule: str,
#                                  glyph_fn,
#                                  skip_first_col: bool = True,
#                                  skip_last_row: bool = True) -> np.ndarray:
#     """
#     Downsample bottom-left region with bl_block, but skip its first column
#     and last row before doing so, then overlay (padded) into the base_ds.
#     """
#     H, W = orig.shape

#     # Full BL region (pre-crop), then map 4 -> 3
#     bl_full = _replace_tiles(orig[crop_row_from:, :crop_col_to+1], from_val=4, to_val=3)

#     # Skip first column / last row if requested
#     r0, r1 = 0, bl_full.shape[0]
#     c0, c1 = 0, bl_full.shape[1]
#     if skip_last_row and r1 > 0:   r1 -= 1
#     if skip_first_col and c1 > 0:  c0 = 1
#     bl = bl_full[r0:r1, c0:c1]

#     # Nothing left? Just return the base unchanged.
#     if bl.size == 0:
#         return base_ds

#     # 3x3 downsample on the reduced region
#     bl_ds = downsample_blocks(bl, block=bl_block, blank_val=blank_val, rule=rule)

#     # Compute where to place it in the 2x2 (block) grid
#     start_row_orig = crop_row_from + r0
#     start_col_orig = c0
#     start_row_ds = start_row_orig // block
#     start_col_ds = start_col_orig // block

#     # Footprint in the 2x2 grid we want to cover
#     target_h = math.ceil(bl.shape[0] / block)
#     target_w = math.ceil(bl.shape[1] / block)

#     # Pad the 3x3 result with blanks to match the footprint
#     padded = np.full((target_h, target_w), blank_val, dtype=int)
#     h_copy = min(target_h, bl_ds.shape[0])
#     w_copy = min(target_w, bl_ds.shape[1])
#     padded[:h_copy, :w_copy] = bl_ds[:h_copy, :w_copy]

#     # Overlay
#     end_row = min(start_row_ds + target_h, base_ds.shape[0])
#     end_col = min(start_col_ds + target_w, base_ds.shape[1])
#     base_ds[start_row_ds:end_row, start_col_ds:end_col] = padded[:end_row - start_row_ds, :end_col - start_col_ds]
#     return base_ds

# def render_ls20_board(board,
#                       block: int = 2,
#                       bl_block: int = 3,
#                       blank_val: int = 3,
#                       crop_row_from: int = 52,
#                       crop_col_to: int = 20,
#                       rule: str = "least") -> str:
#     orig = np.asarray(board, dtype=int)

#     # Crop (4->3) and 2x2 downsample base
#     cropped = crop_bottom_left(orig, row_from=crop_row_from, col_to=crop_col_to, blank_val=blank_val)

#     base_ds = downsample_blocks(cropped, block=block, blank_val=blank_val, rule=rule) if cropped.size else np.empty((0, 0), dtype=int)

#     # Overlay 3x3-downsampled BL region, skipping first col & last row
#     mixed = _place_bottom_left_into_base(
#         orig=orig,
#         base_ds=base_ds.copy(),
#         block=block,
#         bl_block=bl_block,
#         crop_row_from=crop_row_from,
#         crop_col_to=crop_col_to,
#         blank_val=blank_val,
#         rule=rule,
#         glyph_fn=glyph_ls20,
#         skip_first_col=True,
#         skip_last_row=True
#     )

#     lines = []
#     lines.append(f"--- mixed board: global {block}×{block}, bottom-left {bl_block}×{bl_block} (padded, skipped first col & last row) ---")
#     lines.extend(_render_block(mixed, row_offset=0, col_offset=0, glyph_fn=glyph_ls20, add_border=True))
#     return "\n" + "\n".join(lines)
def downsample_least(arr: np.ndarray, block: int) -> np.ndarray:
    """Downsample by taking the least-frequent value in each block (ties → smallest)."""
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

# def mixed_downsample(board: np.ndarray, bl_row_from: int=52, bl_col_to: int=20, g_block: int=2, bl_block: int=3) -> np.ndarray:
#     """
#     - Whole board:     2x2 least-frequent.
#     - Bottom-left ROI: 3x3 least-frequent, skipping first col and last row of that ROI.
#       Then overlaid onto the 2x2 canvas.
#     """
#     board = np.asarray(board)
#     base = downsample_least(board, g_block) # 1) Global 2x2
#     # 2) Bottom-left ROI (skip first col, last row)
#     H, W = board.shape
#     r0, r1 = bl_row_from, max(bl_row_from, H - 1)   # skip last row
#     c0, c1 = 1, min(bl_col_to + 1, W)               # skip first col
#     if r0 >= r1 or c0 >= c1: return base  # nothing to overlay
#     bl = board[r0:r1, c0:c1]
#     # board[r0:r1, c0:c1] = 4
#     bl_ds = downsample_least(bl, bl_block)
#     # 3) Overlay onto the 2x2 canvas at the mapped position
#     sr, sc = r0 // g_block, c0 // g_block
#     er = min(sr + bl_ds.shape[0], base.shape[0])
#     ec = min(sc + bl_ds.shape[1], base.shape[1])
#     base[sr:er, sc:ec] = bl_ds[:er - sr, :ec - sc]
#     return base
12 x 12

def mixed_downsample(board: np.ndarray,
                     bl_row_from: int = 52,
                     bl_col_to:   int = 20,
                     g_block:     int = 2,
                     bl_block:    int = 3) -> np.ndarray:
    """
    - Global board: 2x2 least-frequent on a version where the whole BL ROI was first set to 4.
    - BL ROI overlay: 3x3 least-frequent on the *original* values, skipping first col & last row.
    """
    board = np.asarray(board)

    H, W = board.shape
    # ---- define the full BL ROI you want to "reserve" ----
    full_r0, full_r1 = bl_row_from, H        # full ROI rows
    full_c0, full_c1 = 0, min(bl_col_to + 1, W)  # full ROI cols

    # (A) mask that full ROI with 4s BEFORE the global 2x2
    masked = board.copy()
    masked[full_r0:full_r1, full_c0:full_c1] = 4

    # (B) global 2x2 pass on masked
    base = downsample_least(masked, g_block)

    # (C) now compute the *reduced* ROI we actually downsample 3x3 on
    r0, r1 = bl_row_from, max(bl_row_from, H - 1)   # skip last row
    c0, c1 = 1, min(bl_col_to + 1, W)               # skip first col
    if r0 >= r1 or c0 >= c1:
        return base

    bl = board[r0:r1, c0:c1]
    bl_ds = downsample_least(bl, bl_block)

    # (D) compute the full footprint of the full ROI in the 2x2 canvas
    sr_full = full_r0 // g_block
    sc_full = full_c0 // g_block
    target_h_full = math.ceil((full_r1 - full_r0) / g_block)
    target_w_full = math.ceil((full_c1 - full_c0) / g_block)
    er_full = min(sr_full + target_h_full, base.shape[0])
    ec_full = min(sc_full + target_w_full, base.shape[1])

    print(er_full, ec_full)
    print(sr_full, sc_full)
    input()

    # (E) paint entire footprint with 4s (safety, though masked already helped)
    base[29:, :11] = 4
    # base[r0//g_block:r1//g_block, c0//g_block:c1//g_block] = 4
    # base[sr_full:er_full, sc_full:ec_full] = 4

    # (F) overlay the 3x3-downsampled (reduced) ROI where it belongs
    sr = r0 // g_block
    sc = c0 // g_block
    h_copy = min(base.shape[0] - sr, bl_ds.shape[0])
    w_copy = min(base.shape[1] - sc, bl_ds.shape[1])
    base[sr:sr + h_copy, sc:sc + w_copy] = bl_ds[:h_copy, :w_copy]

    return base





def glyph(v: int) -> str:
    if v < 10: return str(v)
    return chr(ord("A") + v - 10)

# def glyph(cell: int) -> str:
#     if cell == 4:  return "#"
#     if cell == 3:  return "."
#     if cell == 0:  return " "
#     if 1 <= cell <= 9: return str(cell)
#     return chr(ord("a") + cell - 10)


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
            f"\nAction 6:\n\tExplanation: a two-parameter command that supplies explicit X/Y coordinates—to an active game session. Common use-cases include 'click/tap at (x,y)', 'place a tile', or 'shoot a projectile', depending on the game's mechanics.\n\tExample: '[A6 12 63]' (will submit the coordinates x=12 y=63 to the game)."
            f"Keep in mind you may not have to use all of them. Start by testing the simple actions and if they don't suffice work your way to the more complex ones."
            f"without assuming to know how to solve it. Once you understand what your actions do and what the objective is, you can try solving it. There "
            f"is no turn limit at all, but depending on the game, after a number of moves you might re-start from the starting position."
        )

    # def _render_board(self) -> str:
    #     rows = len(self.state.game_state["board"])
    #     cols = len(self.state.game_state["board"][0]) if rows else 0
    #     row_w = max(2, len(str(rows - 1))) # Row label width (so 0..63 prints nicely)
    #     ones = "".join(str(c % 10) for c in range(cols))
    #     tens = "".join(str((c // 10) % 10) if c >= 10 else " " for c in range(cols))
    #     pad = " " * (row_w + 1)     # left padding before column headers
    #     header1 = f"{pad}{ones}"
    #     header2 = f"{pad}{tens}"
    #     horiz = "-" * cols
    #     # top_border    = " " * (row_w + 1) + f"+{horiz}+"
    #     top_border    = " " * (row_w) + f"+{horiz}+"
    #     bottom_border = top_border
    #     lines = [header1, header2, top_border]
    #     for r in range(rows):
    #         row_label = str(r).rjust(row_w)
    #         row_cells = "".join(glyph(cell) for cell in self.state.game_state["board"][r])
    #         lines.append(f"{row_label}|{row_cells}|")
    #     lines.append(bottom_border)
    #     input("\n".join(lines))
    #     return "\n"+"\n".join(lines)

    def _render_board(self):
        # input(self.state.game_state["board"])
        if self.game_name == "ls20":
            # input(mixed_downsample(board=self.state.game_state["board"]))
            input(render_ls20_board(board=self.state.game_state["board"]))
            return render_ls20_board(board=self.state.game_state["board"])

        elif self.game_name == "ft09":
            pass 


        elif self.game_name == "vc30":
            pass

        else:
            raise

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
            payload["x"] = int(x_s)
            payload["y"] = int(y_s)
        return num, payload, None

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.session_id is None: raise RuntimeError("Call reset() before step().")
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        action_num, payload, err = self._parse_action(action)
        if err is not None or action_num is None: self.state.set_invalid_move(reward=0, reason=err or "Invalid action"); return self.state.step()
        try: server = self._execute_action(action_num, payload)
        except Exception as e: self.state.set_invalid_move(reward=0, reason=f"Server error: {e}"); return self.state.step()
        self.session_id = server.get("guid", self.session_id)
        score = server.get("score", self.state.game_state["score"])
        self.state.game_state.update(turn=self.state.game_state["turn"] + 1, score=score, board=server.get("frame", [self.state.game_state["board"]])[-1])
        self.state.add_observation(message=self._render_board(), observation_type=ta.ObservationType.GAME_BOARD)
        if self.state.check_turn_limit(): self.state.set_outcome(reward=score)
        return self.state.step()

    def close(self):
        self.api.close()
        return self.close()
