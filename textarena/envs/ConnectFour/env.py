import re
from typing import Any, Dict, Optional, Tuple, List, Callable

import textarena as ta 
from textarena.envs.ConnectFour.renderer import create_board_str

class ConnectFourEnv(ta.Env):
    """ Environment for Connect Four Game. """
    def __init__(self, is_open: bool=True, num_rows: int=6, num_cols: int=7):
        """
        Initialize the Connect Four game environment.

        Args:
            is_open (bool): If True, the game state is visible to the players.
            num_rows (int): Number of rows in the game board.
            num_cols (int): Number of columns in the game board.
        """
        self.is_open = is_open 
        self.num_rows = num_rows 
        self.num_cols = num_cols 

    # @property 
    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the game to its initial state """
        # Initialize game state variables
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2)

        # create the game board
        game_board = self._create_game_board()

        # reset game state
        game_state = {"board": game_board, "rendered_board": self._render_board(game_board)}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player """
        prompt = (
            f"You are Player {player_id} in Connect Four.\n"
            f"Your disc symbol: {'X' if player_id == 0 else 'O'}.\n"
            f"The game board has {self.num_rows} rows and {self.num_cols} columns.\n"
            f"Players take turns dropping their disc into one of the columns (0 to {self.num_cols - 1}).\n"
            "The first to connect (their own) four discs vertically, horizontally, or diagonally wins.\n"
            "On your turn, enter the column number in squared brackets to make your move.\n"
            "For example: '[col 4]' or '[col 1]'.\n"
        )
        if self.is_open:
            prompt += (
                "The game board is visible to both players.\n"
                f"Current Board:\n{self._render_board()}"
            )
        else:
            prompt += "The game board is not visible to players.\n"
        return prompt

    def _create_game_board(self) -> List[List[str]]:
        """
        Create an empty game board.

        Returns:
            List[List[str]]: A 2D list representing the game board initialized with '.' in all positions.
        """
        return [["." for _ in range(self.num_cols)] for _ in range(self.num_rows)]

    def _render_board(self, game_board: Optional[List[List[str]]] = None) -> str:
        """
        Return a string representation of the board with column numbers.

        Args:
            game_board (Optional[List[List[str]]]): The game board to render. 
                If None, uses the current game state.

        Returns:
            str: The board as a formatted string.
        """
        if game_board is None:
            game_board = self.state.game_state["board"]

        column_numbers = " ".join([str(c) for c in range(self.num_cols)])
        separator = "-" * (self.num_cols * 2 - 1)
        board_rows = "\n".join([" ".join(row) for row in game_board])
        board_str = f"{column_numbers}\n{separator}\n{board_rows}"
        return board_str

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's move """

        # add action to observations and log 
        self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action)

        # check if the actions is valid 
        is_valid, col, reason = self._validate_action(action=action)

        if not is_valid:
            self.state.set_invalid_move(player_id=self.state.current_player_id, reason=reason)

        else:
            # place the disc
            row = self._get_available_row(col)

            # insert disc
            self.state.game_state["board"][row][col] = "X" if self.state.current_player_id == 0 else "O"
            self.state.game_state["rendered_board"] = self._render_board()

            # Check for a win
            if self._check_win(row, col):
                reason=f"Player {self.state.current_player_id} wins by connecting four!"
                self.state.set_winners(player_ids=[self.state.current_player_id], reason=reason)

            # Check for a draw
            elif self._check_draw():
                self.state.set_draw(reason="Game ended in a draw.")

            else:
                # update board state 
                board_str = self._render_board()
                if self.is_open:
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Board state:\n{board_str}", for_logging=False)
                self.state.game_state["rendered_board"] = board_str

        return self.state.step()



    def _validate_action(self, action: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Validate the player's action.

        Args:
            action (str): The action string provided by the player.

        Returns:
            Tuple[bool, Any]: (is_valid, column number or reason for invalidity)
        """
        # action_pattern = re.compile(r'.*\[col\s*(\d+)\].*', re.IGNORECASE)
        action_pattern = re.compile(r'.*\[(?:col\s*)?(\d+)\].*', re.IGNORECASE)

        match = action_pattern.search(action)
        if not match:
            return False, None, f"Player {self.state.current_player_id}, Invalid action format. Expected format: '[col x]'."

        col = int(match.group(1))
        if not (0 <= col < self.num_cols):
            return False, None, f"Player {self.state.current_player_id}, Invalid action. Column {col} is out of bounds."

        if self.state.game_state["board"][0][col] != ".":
            return False, None, f"Player {self.state.current_player_id}, Invalid action. Column {col} is full."

        return True, col, None 


    def _get_available_row(self, col: int) -> int:
        """
        Get the next available row in the specified column.

        Args:
            col (int): The column number.

        Returns:
            int: The row index where the disc can be placed.
        """
        for r in range(self.num_rows - 1, -1, -1):
            if self.state.game_state["board"][r][col] == ".":
                return r
        raise Exception("The column should be validated before calling the _get_available_row function.")


    def _check_win(self, row: int, col:int) -> bool:
        """
        Check if the latest move at (row, col) wins the game.

        Args:
            row (int): The row index of the latest disc.
            col (int): The column index of the latest disc.

        Returns:
            bool: True if the move results in a win, False otherwise.
        """
        board = self.state.game_state["board"]
        disc = board[row][col]
        directions = [
            ((0, 1), (0, -1)),    # Horizontal
            ((1, 0), (-1, 0)),    # Vertical
            ((1, 1), (-1, -1)),   # Diagonal /
            ((1, -1), (-1, 1)),   # Diagonal \
        ]

        for direction in directions:
            total = 1  # Count the disc just placed
            for delta_row, delta_col in direction:
                total += self._check_direction(board, row, col, delta_row, delta_col, disc)
            if total >= 4:
                return True
        return False

    def _check_direction(self, board, row, col, delta_row, delta_col, disc) -> int:
        """
        Check the number of consecutive discs in a specific direction.

        Args:
            board (List[List[str]]): The game board.
            row (int): Starting row index.
            col (int): Starting column index.
            delta_row (int): Row increment.
            delta_col (int): Column increment.
            disc (str): Disc symbol ('X' or 'O').

        Returns:
            int: Number of consecutive discs in the specified direction.
        """
        count = 0
        r, c = row + delta_row, col + delta_col
        while 0 <= r < self.num_rows and 0 <= c < self.num_cols and board[r][c] == disc:
            count += 1
            r += delta_row
            c += delta_col
        return count

    def _check_draw(self) -> bool:
        """
        Check if the game has resulted in a draw.

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return all(self.state.game_state["board"][0][c] != "." for c in range(self.num_cols))
