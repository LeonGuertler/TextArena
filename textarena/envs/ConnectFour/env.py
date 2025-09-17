import re
from typing import Any, Dict, Optional, Tuple
import json
import os
import textarena as ta 
from textarena.envs.ConnectFour.renderer import create_board_str
from typing import Any, Dict, Optional, Tuple, List
import requests

class ConnectFourEnv(ta.Env):
    def __init__(self, is_open: bool=True, num_rows: int=6, num_cols: int=7, error_allowance: int = 1,solver_url: str="http://localhost:5555/solve",summary_output_path="c4_summary.json"):
        """
        Args:
            is_open (bool): If True, the game state is visible to the players.
            num_rows (int): Number of rows in the game board.
            num_cols (int): Number of columns in the game board.
        """
        self.is_open = is_open 
        self.num_rows = num_rows 
        self.num_cols = num_cols 
        self.error_allowance = error_allowance
        self.solver_url = solver_url
        self.summary_output_path = summary_output_path

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def _legal_moves(self) -> List[int]:
        return [c for c in range(self.num_cols) if self.state.game_state["board"][0][c] == "."]

    def _fetch_solver_scores(self, pos_str: str) -> List[float]:
        r = requests.get(f"{self.solver_url}?pos={pos_str}", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            return data.get("score", []) or []
        else:
            raise Exception(f"Solver request failed with status code {r.status_code}: {r.text}")
        return []

    def _dense_level(self, col: int, scores: List[float], candidates: List[int]) -> Optional[int]:
        """
        Dense ranking level (1 = best). Equal scores share the same level.
        Example: scores among candidates = [10,10,9] -> levels = [1,1,2].
        """
        if not scores or not candidates or col not in candidates:
            return None
        unique_desc = sorted({scores[c] for c in candidates}, reverse=True)
        score_to_level = {s: i+1 for i, s in enumerate(unique_desc)}  # 1-based best
        return score_to_level[scores[col]]

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed, error_allowance=self.error_allowance)
        game_state = {
            "board": [["." for _ in range(self.num_cols)] for _ in range(self.num_rows)],
            "solver_logs": [],     # list of {player, col, pos_str_before, scores, rank_legal, rank_all}
            "pos_str": "",         # sequence of 1-based columns
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        self.state.add_observation(message=(f"Board state:\n{self._render_board()}" if self.is_open else "The game board is not visible to players."), observation_type=ta.ObservationType.GAME_BOARD)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are Player {player_id} in Connect Four.\nYour disc symbol: {'X' if player_id == 0 else 'O'}.\n"
            f"The game board has {self.num_rows} rows and {self.num_cols} columns.\n"
            f"Players take turns dropping their disc into one of the columns (0 to {self.num_cols - 1}).\n"
            "The first to connect (their own) four discs vertically, horizontally, or diagonally wins.\n"
            "On your turn, enter the column number in squared brackets to make your move.\nFor example: '[col 4]' or '[col 1]'."
        ) 
    
    def _render_board(self) -> str:
        column_numbers = " ".join([str(c) for c in range(self.num_cols)])
        separator = "-" * (self.num_cols * 2 - 1)
        board_rows = "\n".join([" ".join(row) for row in self.state.game_state["board"]])
        return f"{column_numbers}\n{separator}\n{board_rows}"

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        is_valid, col, reason = self._validate_action(action=action) # check if the actions is valid 
        if not is_valid:  
            self.state.set_invalid_move(reason=reason)
            return self.state.step()
        pos_before = self.state.game_state["pos_str"]
        scores = self._fetch_solver_scores(pos_before)  # length should be 7
        legal = self._legal_moves()

        level_legal = self._dense_level(col, scores, legal)
        level_all   = self._dense_level(col, scores, list(range(self.num_cols)))

        row = self._get_available_row(col)
        player_symbol = "X" if self.state.current_player_id == 0 else "O"
        self.state.add_observation(
            message=f"Player {self.state.current_player_id} dropped their disk ({player_symbol}) into column {col}.",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        self.state.game_state["board"][row][col] = player_symbol

        self.state.game_state["pos_str"] += str(col + 1)
        self.state.game_state["solver_logs"].append({
            "player": self.state.current_player_id,
            "col": col,
            "pos_str_before": pos_before,
            "scores": scores if scores else None,
            "level_legal": level_legal, 
            "level_all": level_all,
        })

        if self._check_win(row, col):
            self.state.set_winner(
                player_id=self.state.current_player_id,
                reason=f"Player {self.state.current_player_id} wins by connecting four!"
            )
            self._finalize_and_write_json(self.summary_output_path)
        elif self._check_draw():
            self.state.set_draw(reason="Game ended in a draw.")
            self._finalize_and_write_json(self.summary_output_path)
        else:
            if self.is_open:
                self.state.add_observation(
                    message=f"Board state:\n{self._render_board()}",
                    observation_type=ta.ObservationType.GAME_BOARD
                )

        return self.state.step()

    def _validate_action(self, action: str) -> Tuple[bool, Optional[int], Optional[str]]:
        match = re.compile(r'.*\[(?:col\s*)?(\d+)\].*', re.IGNORECASE).search(action)
        if not match: return False, None, f"Player {self.state.current_player_id}, Invalid action format. Expected format: '[col x]'."
        col = int(match.group(1))
        if not (0 <= col < self.num_cols): return False, None, f"Player {self.state.current_player_id}, Invalid action. Column {col} is out of bounds."
        if self.state.game_state["board"][0][col] != ".": return False, None, f"Player {self.state.current_player_id}, Invalid action. Column {col} is full."
        return True, col, None 

    def _get_available_row(self, col: int) -> int:
        for r in range(self.num_rows - 1, -1, -1):
            if self.state.game_state["board"][r][col] == ".":
                return r
        raise Exception("The column should be validated before calling the _get_available_row function.")

    def _check_win(self, row: int, col:int) -> bool:
        for direction in [((0, 1), (0, -1)), ((1, 0), (-1, 0)), ((1, 1), (-1, -1)), ((1, -1), (-1, 1)),]:
            total = 1  # Count the disc just placed
            for delta_row, delta_col in direction:
                total += self._check_direction(self.state.game_state["board"], row, col, delta_row, delta_col, self.state.game_state["board"][row][col])
            if total >= 4: return True
        return False

    def _check_direction(self, board, row, col, delta_row, delta_col, disc) -> int:
        count = 0
        r, c = row + delta_row, col + delta_col
        while 0 <= r < self.num_rows and 0 <= c < self.num_cols and board[r][c] == disc:
            count += 1
            r += delta_row
            c += delta_col
        return count

    def _check_draw(self) -> bool: 
        return all(self.state.game_state["board"][0][c] != "." for c in range(self.num_cols))

    def _finalize_and_write_json(self, path: Optional[str]) -> None:
        logs = self.state.game_state.get("solver_logs", [])
        p0_actions, p0_levels_legal, p0_levels_all = [], [], []
        p1_actions, p1_levels_legal, p1_levels_all = [], [], []

        for item in logs:
            if item["player"] == 0:
                p0_actions.append(item["col"])
                p0_levels_legal.append(item["level_legal"])
                p0_levels_all.append(item["level_all"])
            else:
                p1_actions.append(item["col"])
                p1_levels_legal.append(item["level_legal"])
                p1_levels_all.append(item["level_all"])

        summary = {
            "player_0": {
                "actions": p0_actions,
                "levels": p0_levels_legal,
                "levels_legal": p0_levels_legal,  # 1 = best, dense
                "levels_all": p0_levels_all,      # 1 = best, dense
            },
            "player_1": {
                "actions": p1_actions,
                "levels": p1_levels_legal,
                "levels_legal": p1_levels_legal,
                "levels_all": p1_levels_all,
            },
            "pos_str": self.state.game_state.get("pos_str", ""),
        }

        final_path = path or "c4_summary.json"
        with open(final_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)