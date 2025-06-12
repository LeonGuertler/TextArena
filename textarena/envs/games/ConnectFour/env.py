import re
from typing import Any, Dict, Optional, Tuple, List, Callable

import textarena as ta 
from textarena.envs.games.ConnectFour.renderer import create_board_str

class ConnectFourEnv(ta.Env):
    def __init__(self, is_open: bool=True, num_rows: int=6, num_cols: int=7):
        """
        Args:
            is_open (bool): If True, the game state is visible to the players.
            num_rows (int): Number of rows in the game board.
            num_cols (int): Number of columns in the game board.
        """
        self.is_open = is_open 
        self.num_rows = num_rows 
        self.num_cols = num_cols 
        self.action_space = {i: re.compile(r'.*\[(?:col\s*)?(\d+)\].*', re.IGNORECASE) for i in range(2)}

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {"board": [["." for _ in range(self.num_cols)] for _ in range(self.num_rows)]} 
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        self.state.add_observation(message=f"Game started with {self.num_rows} rows and {self.num_cols} columns.", observation_type=ta.ObservationType.GAME_BOARD)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are Player {player_id} in Connect Four.\n"
            f"Your disc symbol: {'X' if player_id == 0 else 'O'}.\n"
            f"The game board has {self.num_rows} rows and {self.num_cols} columns.\n"
            f"Players take turns dropping their disc into one of the columns (0 to {self.num_cols - 1}).\n"
            "The first to connect (their own) four discs vertically, horizontally, or diagonally wins.\n"
            "On your turn, enter the column number in squared brackets to make your move.\n"
            "For example: '[col 4]' or '[col 1]'.\n"
        ) + (f"The game board is visible to both players.\nCurrent Board:\n{self._render_board()}" if self.is_open else "The game board is not visible to players.\n")

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
        else:
            row = self._get_available_row(col) # place the disc
            self.state.game_state["board"][row][col] = "X" if self.state.current_player_id == 0 else "O" # insert disc
            if self._check_win(row, col): # Check for a win
                self.state.set_winner(player_id=self.state.current_player_id, reason=f"Player {self.state.current_player_id} wins by connecting four!")
            elif self._check_draw(): # Check for a draw
                self.state.set_draw(reason="Game ended in a draw.")
            else: # update board state 
                if self.is_open:
                    self.state.add_observation(message=f"Board state:\n{self._render_board()}", observation_type=ta.ObservationType.GAME_BOARD)
        return self.state.step()

    def _validate_action(self, action: str) -> Tuple[bool, Optional[int], Optional[str]]:
        match = self.action_space[self.state.current_player_id].search(action)
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
            if total >= 4:
                return True
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
