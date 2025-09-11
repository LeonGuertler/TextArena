import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.TicTacToe.renderer import create_board_str

class TicTacToeEnv(ta.Env):
    def __init__(self, error_allowance: int = 1, summary_output_path="ttt_summary.json"):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.error_allowance = error_allowance
        self._last_eval = None
        self.summary_output_path = summary_output_path

    def get_board_str(self): return create_board_str(board=self.state.game_state["board"])
    def _render_board(self): return "\n---+---+---\n".join("|".join(f" {self.state.game_state['board'][r][c]} " if self.state.game_state['board'][r][c] else f" {str(r * 3 + c)} " for c in range(3)) for r in range(3))
    
    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed, error_allowance=self.error_allowance)
        self.state.reset(game_state={"board": [['' for _ in range(3)] for _ in range(3)]}, player_prompt_function=self._prompt)
        self.state.game_state["move_scores"] = []
        self.state.game_state["turn_summaries"] = []
        self._observer_current_state()

    def _prompt(self, player_id:int, game_state:Dict[str,Any])-> str:
        return (
            f"You are Player {player_id} in Tic Tac Toe.\n"
            "Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.\n"
            "On your turn, you should select the square number (0-8) you want to put your mark in next.\n"
            "For example, '[4]' places your mark in the center cell of the board.\n\n"
            f"As Player {player_id}, you will be '{'X' if player_id == 1 else 'O'}', "
            f"while your opponent is '{'O' if player_id == 1 else 'X'}'.\n"
        )

    def _observer_current_state(self):
        available_moves = [f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3) if self.state.game_state["board"][r][c] == '']
        self.state.add_observation(message=f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}", observation_type=ta.ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.current_player = 'X' if self.state.current_player_id == 1 else 'O'
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        self._score_current_state()

        submitted_cell = None
        applied_cell = None
        valid = False

        match = re.compile(r"\[\s*(\d+)\s*\]").search(action)
        if match is None:
            self.state.set_invalid_move(reason="The submitted move does not follow the correct format.")
        else:
            submitted_cell = int(match.group(1))
            if submitted_cell not in self.cell_mapping:
                self.state.set_invalid_move(reason=f"{submitted_cell}. Must be between 0 and 8.")
            else:
                row, col = self.cell_mapping[submitted_cell]
                if self.state.game_state["board"][row][col] == '':
                    self.state.game_state["board"][row][col] = self.current_player
                    applied_cell = submitted_cell
                    valid = True
                    self.state.add_observation(
                        message=f"Player {self.state.current_player_id} placed their symbol ({self.current_player}) in cell {submitted_cell}.",
                        observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                    )
                    if self._check_winner():
                        self.state.set_winner(player_id=self.state.current_player_id, reason=f"Player {self.state.current_player_id} has won!")
                    elif all(cell != '' for row in self.state.game_state["board"] for cell in row):
                        self.state.set_draw(reason="The game is a draw!")
                else:
                    self.state.set_invalid_move(reason=f"cell {submitted_cell} is already occupied.")

        level_by_idx = self._last_eval["level_by_idx"] if self._last_eval else {}
        action_level = level_by_idx.get(applied_cell) if valid else None
        action_score = self._last_eval["scores"].get(applied_cell) if (valid and self._last_eval) else None

        self.state.game_state["turn_summaries"].append({
            "player_id": self.state.current_player_id,
            "mark": self.current_player,
            "submitted": submitted_cell,
            "applied": applied_cell,
            "valid": valid,
            "ranked": self._last_eval["ranked"] if self._last_eval else [],
            "scores": self._last_eval["scores"] if self._last_eval else {},
            "level_by_idx": level_by_idx,
            "action_level": action_level, 
            "action_score": action_score,  
        })
        self.save_action_levels_json(self.summary_output_path)

        self._observer_current_state()
        
        return self.state.step()
    def _check_winner(self) -> bool:
        board = self.state.game_state["board"]
        for i in range(3):
            if (board[i][0] == board[i][1] == board[i][2] != '' or board[0][i] == board[1][i] == board[2][i] != ''):    return True
        if (board[0][0] == board[1][1] == board[2][2] != '' or board[0][2] == board[1][1] == board[2][0] != ''):        return True
        return False
        

    # Helpers (inside the env)
    def _rm_legal_moves(self, board):
        return [r*3+c for r in range(3) for c in range(3) if board[r][c] == '']

    def _rm_won(self, board, mark):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == mark: return True
            if board[0][i] == board[1][i] == board[2][i] == mark: return True
        if board[0][0] == board[1][1] == board[2][2] == mark: return True
        if board[0][2] == board[1][1] == board[2][0] == mark: return True
        return False

    def _minimax_eval(self, board, is_maximizing: bool) -> int:
        if self._rm_won(board, 'X'): return +1
        if self._rm_won(board, 'O'): return -1
        if not self._rm_legal_moves(board): return 0
        if is_maximizing:
            best = -10
            for idx in self._rm_legal_moves(board):
                r, c = divmod(idx, 3); board[r][c] = 'X'
                best = max(best, self._minimax_eval(board, False))
                board[r][c] = ''
            return best
        else:
            best = 10
            for idx in self._rm_legal_moves(board):
                r, c = divmod(idx, 3); board[r][c] = 'O'
                best = min(best, self._minimax_eval(board, True))
                board[r][c] = ''
            return best

    
    def _score_current_state(self):
        board = self.state.game_state["board"]
        current_mark = 'X' if self.state.current_player_id == 1 else 'O'
        legal = self._rm_legal_moves(board)
        if not legal:
            self._last_eval = None
            return

        scores = {}
        for idx in legal:
            r, c = divmod(idx, 3)
            board[r][c] = current_mark
            val = self._minimax_eval(board, is_maximizing=(current_mark == 'O'))
            board[r][c] = ''
            scores[idx] = val

        if current_mark == 'X':
            ranked_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            unique_vals = sorted(set(scores.values()), reverse=True)
        else:
            ranked_items = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
            unique_vals = sorted(set(scores.values()))

        # val -> level
        level_of_val = {v: i + 1 for i, v in enumerate(unique_vals)}
        # idx -> level
        level_by_idx = {idx: level_of_val[val] for idx, val in scores.items()}

        ranked = [i for i, _ in ranked_items]
        best_move = ranked[0]
        best_val = scores[best_move]

        self.state.game_state["move_scores"].append({
            "turn_player_id": self.state.current_player_id,
            "mark": current_mark,
            "scores": scores,          # {idx: val}
            "ranked": ranked,          
            "best_move": best_move,
            "best_value": best_val,
            "level_by_idx": level_by_idx, 
            "level_order_vals": unique_vals 
        })

        self._last_eval = {
            "player_id": self.state.current_player_id,
            "mark": current_mark,
            "scores": scores,
            "ranked": ranked,
            "best_move": best_move,
            "best_value": best_val,
            "level_by_idx": level_by_idx,
            "level_order_vals": unique_vals,
        }

        pretty = ", ".join(f"{i}:{scores[i]:+d}(L{level_by_idx[i]})" for i in ranked)
        self.state.add_observation(
            message=f"[Solver] P{self.state.current_player_id}({current_mark}) best {best_move} ({best_val:+d}); {pretty}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
        )

    def save_action_levels_json(self, filepath: str):
        """
        {
        "player_0": { "actions": [0,3,7], "levels": [1,2,1] },
        "player_1": { "actions": [4,2,8], "levels": [1,1,3] }
        }
        """
        p0_actions, p0_levels = [], []
        p1_actions, p1_levels = [], []

        for t in self.state.game_state.get("turn_summaries", []):
            action = t.get("applied")
            level  = t.get("action_level")
            if t["player_id"] == 0:
                p0_actions.append(action)
                p0_levels.append(level)
            else:  # player_id == 1
                p1_actions.append(action)
                p1_levels.append(level)

        out = {
            "player_0": {"actions": p0_actions, "levels": p0_levels},
            "player_1": {"actions": p1_actions, "levels": p1_levels},
        }

        import json, os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)