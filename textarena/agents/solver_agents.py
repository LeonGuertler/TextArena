import random
from textarena.core import Agent
import re
import requests
from typing import List, Tuple, Optional
_ROW = re.compile(r'^\s*[^|]*\|\s*[^|]*\|\s*[^|]*\s*$')
_SEP = re.compile(r'^\s*-{3}\+-{3}\+-{3}\s*$')
_MARK_LINE = re.compile(r"As Player\s+\d+,\s+you will be\s+'([XO])'")

PLAYER_MOVE_RE = re.compile(r"\[Player\s+\d+\]\s*\[(\d+)\]")

def _extract_all_moves_from_observation(obs: str) -> List[int]:
    """
    Returns a chronological list of 0-based column indices found in the
    observation, via `[Player i] [<col>]` lines.
    """
    return [int(m.group(1)) for m in PLAYER_MOVE_RE.finditer(obs)]

def _extract_latest_board(obs: str, num_rows: int = 6) -> Optional[List[List[str]]]:
    """
    Parse the LAST 'Board state' grid from the observation and return
    a 2D list of chars ['.', 'X', 'O'] or None if not found.
    """
    # Find all 'Board state:' blocks and take the last one
    blocks = obs.split("Board state:")
    if len(blocks) < 2:
        return None
    last = blocks[-1].strip().splitlines()

    # Expect layout:
    #   0 1 2 3 4 5 6
    #   -------------
    #   <row0>
    #   <row1>
    #   ...
    #   <row5>
    # Grab the last num_rows lines if present
    if len(last) < 2 + num_rows:
        return None
    grid_lines = last[-num_rows:]
    grid = [row.strip().split() for row in grid_lines]
    return grid

def _legal_moves_from_board(board: List[List[str]]) -> List[int]:
    """
    A column is legal if its top cell (row 0) is '.'.
    """
    if not board:
        return list(range(7))
    num_cols = len(board[0])
    return [c for c in range(num_cols) if board[0][c] == "."]

def _fetch_solver_scores(pos_str: str, solver_url: str) -> List[float]:
    """
    Calls your local solver server, returns list of length 7 with scores.
    """
    try:
        r = requests.get(f"{solver_url}?pos={pos_str}", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            return data.get("score", []) or []
        else:
            print(f"[C4Agent] Solver HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[C4Agent] Solver error: {e}")
    return []

def _argmax_with_tiebreak(scores: List[float], candidates: List[int]) -> int:
    """
    Choose the candidate with highest score. Ties broken by 'closer to center'.
    """
    if not candidates:
        return 3  # fallback to center
    best = None
    best_score = None

    def center_dist(c): return abs(c - 3)

    for c in candidates:
        s = scores[c]
        if (best_score is None) or (s > best_score) or (s == best_score and center_dist(c) < center_dist(best)):
            best = c
            best_score = s
    return best

def _rank_index(col: int, scores: List[float], among: List[int]) -> int:
    """
    Rank of chosen 'col' among 'among' columns (0 = best). Stable tiebreaker = smaller col index first.
    """
    # Sort by (-score, col) and find index of 'col'
    ordered = sorted(among, key=lambda c: (-scores[c], c))
    return ordered.index(col)

    
class C4Agent(Agent):
    """
    A stateful TextArena agent that:
      * Tracks the move history as a pos_str for the solver (1-based columns).
      * Calls the local solver for scores and chooses a top move.
      * Returns a TextArena-compatible action string like: "[col 3]".
    """
    def __init__(self, solver_url: str = "http://localhost:5555/solve"):
        super().__init__()
        self.solver_url = solver_url
        self._pos_str = ""           # e.g. "4351..."
        self._seen_moves: List[int] = []

    def _maybe_reset(self, obs: str):
        """
        Reset tracking if this looks like a new game (empty board & no moves).
        """
        moves = _extract_all_moves_from_observation(obs)
        board = _extract_latest_board(obs) or []
        empty = all(ch == "." for row in board for ch in row) if board else True

        if empty and not moves:
            self._pos_str = ""
            self._seen_moves = []

    def _sync_history(self, obs: str):
        """
        Ensure our pos_str matches what's in the observation.
        """
        moves_now = _extract_all_moves_from_observation(obs)
        if len(moves_now) > len(self._seen_moves):
            new_moves = moves_now[len(self._seen_moves):]
            for m in new_moves:
                self._pos_str += str(m + 1)  # solver expects 1-based
            self._seen_moves = moves_now

    def __call__(self, observation: str) -> str:
        print("\n\n+++ +++ +++")  # visualization marker

        # If new game detected, reset local state
        self._maybe_reset(observation)
        # Bring our local pos_str up-to-date from the observation
        self._sync_history(observation)

        # Parse board & legal moves
        board = _extract_latest_board(observation)
        legal = _legal_moves_from_board(board) if board else list(range(7))

        # Get solver scores for CURRENT position
        scores = _fetch_solver_scores(self._pos_str, self.solver_url)

        # Fallback if solver not reachable
        if len(scores) != 7:
            move = 3 if 3 in legal else (legal[0] if legal else 3)
            action = f"[col {move}]"
            # Update our history immediately since env won’t echo our move before next call
            self._pos_str += str(move + 1)
            self._seen_moves.append(move)
            return action

        # Pick the best legal move
        move = _argmax_with_tiebreak(scores, legal)

        # Update our history immediately (TextArena won't reflect our own move until next turn)
        self._pos_str += str(move + 1)
        self._seen_moves.append(move)

        # Return TextArena-friendly format (your validator accepts “[col x]”)
        return f"[col {move}]"

class TTTAgent(Agent):
    """
    Perfect Tic-Tac-Toe agent for TextArena.

    TextArena board indices are ROW-MAJOR:
        0 | 1 | 2
        3 | 4 | 5
        6 | 7 | 8
    """

    def __init__(self):
        super().__init__()

    def __call__(self, observation: str) -> str:
        # 1) Figure out OUR mark from the observation (“As Player k, you will be 'X'…”)
        my_mark = self._extract_my_mark(observation)
        if my_mark not in ('X', 'O'):
            # Fallback (rare): infer by parity, but TextArena may start with O
            board = self._extract_latest_board(observation)
            my_mark = 'X' if self._count(board, 'X') == self._count(board, 'O') else 'O'

        # 2) Parse the **latest** board in the observation
        board = self._extract_latest_board(observation)

        # 3) Compute best move via minimax (X maximizes, O minimizes)
        move, _scores = self._best_move_with_scores(board, my_mark)
        if move is None:
            move = 0  # graceful fallback

        return f"[{move}]"

    # ---------------- Parsing helpers ----------------
    def _extract_my_mark(self, obs: str):
        m = _MARK_LINE.search(obs)
        return m.group(1) if m else None

    def _extract_latest_board(self, obs: str):
        """
        Find the last 5-line board block:
          row
          sep
          row
          sep
          row
        Return a 3x3 list with 'X','O',' '.
        """
        lines = obs.splitlines()
        last_rows = None
        # Prefer the board block that follows the last "[GAME] Current Board:" marker
        anchors = [i for i, ln in enumerate(lines) if "Current Board" in ln]
        start = anchors[-1] + 1 if anchors else 0

        for i in range(start, len(lines) - 4):
            if (_ROW.match(lines[i]) and _SEP.match(lines[i+1]) and
                _ROW.match(lines[i+2]) and _SEP.match(lines[i+3]) and
                _ROW.match(lines[i+4])):
                last_rows = [lines[i], lines[i+2], lines[i+4]]

        # Fallback: last three row-like lines
        if last_rows is None:
            row_like = [ln for ln in lines if ('|' in ln and '---' not in ln)]
            last_rows = row_like[-3:] if len(row_like) >= 3 else ["   |   |   "]*3

        board = []
        for line in last_rows:
            cells = [tok.strip() for tok in line.split('|')]
            row = [('X' if tok == 'X' else 'O' if tok == 'O' else ' ') for tok in cells[:3]]
            while len(row) < 3:
                row.append(' ')
            board.append(row[:3])
        return board

    # ---------------- Minimax ----------------
    def _best_move_with_scores(self, board, mark):
        legal = self._legal_moves(board)
        if not legal:
            return None, {}

        scores = {}
        for idx in legal:
            r, c = divmod(idx, 3)
            board[r][c] = mark
            # After we place `mark`, opponent moves.
            # If opponent is X => maximizing; if opponent is O => minimizing.
            next_is_maximizing = (mark == 'O')  # opponent is X
            val = self._minimax(board, is_maximizing=next_is_maximizing)
            board[r][c] = ' '
            scores[idx] = val

        if mark == 'X':
            target = max(scores.values())
            bests = [i for i, v in scores.items() if v == target]
        else:  # O chooses to minimize X's outcome
            target = min(scores.values())
            bests = [i for i, v in scores.items() if v == target]

        random.shuffle(bests)
        return bests[0], scores

    def _minimax(self, board, is_maximizing):
        if self._won(board, 'X'):
            return +1
        if self._won(board, 'O'):
            return -1
        moves = self._legal_moves(board)
        if not moves:
            return 0

        if is_maximizing:
            best = -10
            for idx in moves:
                r, c = divmod(idx, 3)
                board[r][c] = 'X'
                best = max(best, self._minimax(board, False))
                board[r][c] = ' '
            return best
        else:
            best = 10
            for idx in moves:
                r, c = divmod(idx, 3)
                board[r][c] = 'O'
                best = min(best, self._minimax(board, True))
                board[r][c] = ' '
            return best

    # ---------------- Board utils ----------------
    def _legal_moves(self, board):
        return [r*3 + c for r in range(3) for c in range(3) if board[r][c] == ' ']

    def _won(self, board, mark):
        b = board
        for i in range(3):
            if b[i][0] == b[i][1] == b[i][2] == mark: return True
            if b[0][i] == b[1][i] == b[2][i] == mark: return True
        if b[0][0] == b[1][1] == b[2][2] == mark: return True
        if b[0][2] == b[1][1] == b[2][0] == mark: return True
        return False

    def _count(self, board, mark):
        return sum(cell == mark for row in board for cell in row)

    # (If you ever need cross-engine mapping)
    @staticmethod
    def col_major_to_row_major(idx_cm):
        r = idx_cm % 3
        c = idx_cm // 3
        return r*3 + c

    @staticmethod
    def row_major_to_col_major(idx_rm):
        r, c = divmod(idx_rm, 3)
        return c*3 + r