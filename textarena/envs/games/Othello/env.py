import re
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.games.Othello.renderer import create_board_str


DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
EMPTY, BLACK, WHITE = "", "B", "W"

class OthelloEnv(ta.Env):
    """Two-player Othello (Reversi) environment with arbitrary even board size."""
    def __init__(self, board_size: int = 8, show_valid: bool = True):
        if board_size % 2 or board_size < 4:
            raise ValueError("board_size must be an even integer ≥ 4")
        super().__init__()
        self.N = board_size
        self.show_valid = show_valid

        self.action_space = lambda player_id: re.compile(r"\[\s*(\d+)\s*,?\s*(\d+)\s*\]")

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.N and 0 <= c < self.N

    def _counts(self) -> Tuple[int, int]:
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w

    def get_board_str(self) -> str:
        return create_board_str(self.board)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.board = [[EMPTY for _ in range(self.N)] for _ in range(self.N)] # empty board

        # initial four stones in the middle
        m1, m2 = self.N // 2 - 1, self.N // 2
        self.board[m1][m1] = self.board[m2][m2] = WHITE
        self.board[m1][m2] = self.board[m2][m1] = BLACK

        b_count, w_count = self._counts()
        valid_moves = self._valid_moves(BLACK)

        game_state={"board": self.board, "rendered_board": self._render_board(), "black_count": b_count, "white_count": w_count, "valid_moves": valid_moves}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt, role_mapping={0: "Black", 1: "White"})

        obs = f"Game Board:\n{self.state.game_state['rendered_board']}"
        if self.show_valid:
            # obs += "\nValid moves: " + ", ".join(map(str, valid_moves)) if valid_moves else "\nNo valid moves – you may have to skip."
            obs += "\nValid moves: " + ", ".join([f"'{vm}'" for vm in valid_moves]) if valid_moves else "\nNo valid moves – you may have to skip."
        obs += f"\nCurren Scores - Black: {self.state.game_state['black_count']}, White: {self.state.game_state['white_count']}\n"
        self.state.add_observation(message=obs, observation_type=ta.ObservationType.GAME_BOARD)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        piece, colour = (BLACK, "Black") if player_id == 0 else (WHITE, "White")
        return (
            f"You are Player {player_id} playing {colour} ({piece}) in a game of Othello.\nYour goal is to have more pieces of your color on the board by the end of the game.\n"
            f"On your turn, place a piece such that it flanks one or more of your opponent's pieces-in any direction (horizontal, vertical, or diagonal)-between your new piece and another of your existing pieces. "
            f"All flanked opponent pieces will be flipped to your color.\nProvide your move in the format '[row, col]'."
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        pid = self.state.current_player_id
        piece = BLACK if pid == 0 else WHITE
        opp = BLACK if piece == WHITE else WHITE
        self.state.add_observation(from_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        valid = self._valid_moves(piece)
        if not valid:
            self._handle_skip(pid, piece, opp)
            obs = f"Player {pid} had to skip their turn" #\n\n{self.state.game_state['rendered_board']}"
        else:
            match = self.action_space(self.state.current_player_id).search(action)
            if match is None:
                self.state.set_invalid_move(reason="Move must be of the form [row, col].")
                return self.state.step(rotate_player=False)

            r, c = map(int, match.groups())
            if [r, c] not in valid:
                self.state.set_invalid_move(reason=f"Illegal move. Valid moves: {valid}")
                return self.state.step(rotate_player=False)

            flipped = self._place_and_flip(r, c, piece)
            obs = f"Player {pid} ({piece}) played [{r}, {c}] flipping {flipped} piece(s)"
            
            
        self.state.add_observation(message=obs, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

        next_valid = self._valid_moves(opp)
        self.state.game_state["valid_moves"] = next_valid
        self._push_gamestate()

        obs = f"Game Board:\n{self.state.game_state['rendered_board']}"
        if self.show_valid:
            # obs += "\nValid moves: " + ", ".join(map(str, next_valid)) if next_valid else "\nNo valid moves – you may have to skip."
            obs += "\nValid moves: " + ", ".join([f"'{vm}'" for vm in next_valid]) if next_valid else "\nNo valid moves – you may have to skip."
        obs += f"\nScores - Black: {self.state.game_state['black_count']}, White: {self.state.game_state['white_count']}\n"
        self.state.add_observation(message=obs, observation_type=ta.ObservationType.GAME_BOARD)

        if self._game_over(): self._declare_winner()
        else: self.state.game_state["valid_moves"] = next_valid
        return self.state.step()

    def _handle_skip(self, pid, piece, opp):
        self.state.add_observation(f"Player {pid} ({piece}) has no valid moves and must skip.", observation_type=ta.ObservationType.GAME_MESSAGE)
        if not self._valid_moves(opp):
            self._declare_winner()
        else:
            self.state.game_state["valid_moves"] = self._valid_moves(opp)

    def _valid_moves(self, piece) -> List[List[int]]:
        return [[r, c] for r in range(self.N) for c in range(self.N) if self.board[r][c] == EMPTY and self._would_flip(r, c, piece)]

    def _would_flip(self, r, c, piece) -> bool:
        opp = BLACK if piece == WHITE else WHITE
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not (self._in_bounds(rr, cc) and self.board[rr][cc] == opp):
                continue
            while self._in_bounds(rr, cc) and self.board[rr][cc] == opp:
                rr += dr; cc += dc
            if self._in_bounds(rr, cc) and self.board[rr][cc] == piece:
                return True
        return False

    def _place_and_flip(self, r, c, piece) -> int:
        opp = BLACK if piece == WHITE else WHITE
        self.board[r][c] = piece
        flipped = 0
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            line: List[Tuple[int, int]] = []
            while self._in_bounds(rr, cc) and self.board[rr][cc] == opp:
                line.append((rr, cc))
                rr += dr; cc += dc
            if self._in_bounds(rr, cc) and self.board[rr][cc] == piece:
                for fr, fc in line:
                    self.board[fr][fc] = piece
                flipped += len(line)
        return flipped

    def _game_over(self) -> bool:
        full = all(cell != EMPTY for row in self.board for cell in row)
        return full or (not self._valid_moves(BLACK) and not self._valid_moves(WHITE))

    def _render_board(self) -> str:
        header = "  " + " ".join(map(str, range(self.N)))
        rows = [header]
        for i, row in enumerate(self.board):
            cells = [cell if cell else "." for cell in row]
            rows.append(f"{i}|" + "|".join(cells) + "|")
        return "\n".join(rows)

    def _push_gamestate(self):
        b, w = self._counts()
        self.state.game_state.update({"rendered_board": self._render_board(), "black_count": b, "white_count": w})

    def _declare_winner(self):
        b, w = self._counts()
        if b > w: self.state.set_winner(player_id=0, reason=f"Black wins {b}-{w}.")
        elif w > b: self.state.set_winner(player_id=1, reason=f"White wins {w}-{b}.")
        else: self.state.set_draw(reason=f"Draw {b}-{w}.")

