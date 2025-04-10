import re
from typing import List, Dict, Tuple, Any, Optional

import textarena as ta
from textarena.envs.Breakthrough.renderer import create_board_str

class BreakthroughEnv(ta.Env):
    """
    Environment for the board game Breakthrough.
    
    Default is an 8×8 board:
      - White pawns (W) occupy rows 0 and 1 at start.
      - Black pawns (B) occupy rows 6 and 7 at start.
    White is player 0, moves "up" from row=0 toward row=7.
    Black is player 1, moves "down" from row=7 toward row=0.
    
    Moves are given in bracketed notation, e.g. [a2a3].
    * 'a2' corresponds to col='a', row='2' (1-based).
    Internally, row=1 in notation means index=0 in code.
    Columns 'a'..'h' map to col=0..7 in code.
    """

    def __init__(self, is_open: bool = True, board_size: int = 8, max_turns: int = 100):
        """
        Args:
            is_open: If True, the board state is revealed after every move to both players.
            board_size: Dimension of the board, default 8x8.
            max_turns: If reached, the game will end in a 'draw' by truncation 
                       (though standard Breakthrough can’t really draw, 
                       we include it for consistency with the base environment).
        """
        self.is_open = is_open
        self.board_size = board_size
        self.max_turns = max_turns

        # For convenience: 'a' -> 0, 'b' -> 1, ...
        self._file_to_col = {chr(ord('a') + i): i for i in range(board_size)}
        self._col_to_file = {v: k for k, v in self._file_to_col.items()}

        # For moves like [a2a3]
        file_pattern = f"[a-{chr(ord('a')+board_size-1)}]"
        rank_pattern = f"[1-{board_size}]"
        # e.g. [a2a3], case-insensitive
        self.move_pattern = re.compile(
            rf"\[{file_pattern}{rank_pattern}{file_pattern}{rank_pattern}\]",
            re.IGNORECASE
        )

    def get_board_str(self):
        return create_board_str(board=self.board, board_size=self.board_size)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """
        Reset the environment to an initial state.

        Args:
            num_players (int): Should be 2 for standard Breakthrough.
            seed (Optional[int]): Optional RNG seed.
        """
        # Create the State object
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=2,
            role_mapping={0: "White", 1: "Black"},
            max_turns=self.max_turns
        )

        # Build and initialize the board
        self._init_board()

        # Build the dictionary for game_state
        game_state = {
            "board_str": self._render_board(),
            "valid_moves": None  # will fill in after first prompt if desired
        }

        # Actually reset the state
        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

    def _init_board(self):
        """
        Create an 8×8 (or user-specified) board with initial Breakthrough setup:
          - White (W) on rows 0 and 1
          - Black (B) on rows board_size-2 and board_size-1
          - blank '' otherwise
        """
        self.board = [["" for _ in range(self.board_size)] for _ in range(self.board_size)]

        # White rows
        for r in range(2):
            for c in range(self.board_size):
                self.board[r][c] = "W"

        # Black rows
        for r in range(self.board_size - 2, self.board_size):
            for c in range(self.board_size):
                self.board[r][c] = "B"

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the prompt for each player on reset or at start-of-game.
        """
        color = "White" if player_id == 0 else "Black"
        direction = "up" if player_id == 0 else "down"
        prompt = (
            f"You are {color} in Breakthrough. You move {direction} on an {self.board_size}x{self.board_size} board.\n"
            "Move a single piece one step forward or diagonally forward.\n"
            "A piece may only move diagonally if an opponent piece is there, and only move into such a preoccupied square in a diagonal step.\n"
            "When stepping into a square with an opponent piece, you capture it and the opponent's piece is removed permanently from the board.\n"
            "Use UCI-like notation in brackets, e.g. [a2a3] to move from a2 to a3.\n"
            "* 'a' corresponds to the leftmost column, '1' is the bottom row (from White's perspective).\n"
            "* White's home row is the top row (row 8 for an 8x8). Black's home row is the bottom row (row 1 for an 8x8).\n"
            "The first player whose piece reaches the opponent's home row wins.\n"
            "If your pieces are all captured, you lose immediately.\n"
        )

        # If the board is open, show it
        if self.is_open:
            prompt += f"\nCurrent board state:\n{self._render_board()}\n"

        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a player's move, check for wins, update the environment.
        """
        player_id = self.state.current_player_id

        # Log the player's raw action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Attempt to parse & execute
        self._execute_player_move(action)

        # Check for victory conditions
        self._check_winner()

        # Add open info to observations if is_open
        self._augment_observations()

        # Update board_str in game_state
        self.state.game_state["board_str"] = self._render_board()

        # Return (done, info)
        return self.state.step()

    def _execute_player_move(self, action: str):
        """Parse the action (UCI-like string [a2a3]) and execute if valid."""
        player_id = self.state.current_player_id
        match = self.move_pattern.search(action.strip())
        if not match:
            # No valid bracketed move found
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid format. Please use bracketed algebraic notation like [a2a3]."
            )
            return

        move_str = match.group(0)  # e.g. '[a2a3]'
        # Remove brackets, e.g. 'a2a3'
        move_str = move_str.strip("[]").lower()

        start_file = move_str[0]   # e.g. 'a'
        start_rank = move_str[1]   # e.g. '2'
        end_file   = move_str[2]   # e.g. 'a'
        end_rank   = move_str[3]   # e.g. '3'

        # Convert to 0-based indices internally
        start_col = self._file_to_col.get(start_file, -1)
        start_row = int(start_rank) - 1  # '2' -> row=1
        end_col   = self._file_to_col.get(end_file, -1)
        end_row   = int(end_rank) - 1

        if not self._is_on_board(start_row, start_col) or not self._is_on_board(end_row, end_col):
            self.state.set_invalid_move(
                player_id=player_id,
                reason="Move is out of board bounds."
            )
            return

        # Verify that the move is valid according to Breakthrough rules
        if self._is_valid_move(player_id, start_row, start_col, end_row, end_col):
            # Execute move on board
            piece = self.board[start_row][start_col]
            self.board[start_row][start_col] = ""
            self.board[end_row][end_col] = piece

            # Publish a game-level message
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Player {player_id} moves {move_str} ({piece})."
            )
        else:
            self.state.set_invalid_move(
                player_id=player_id,
                reason="That move does not follow the Breakthrough rules or is not your piece."
            )

    def _is_valid_move(self,
                       player_id: int,
                       start_row: int,
                       start_col: int,
                       end_row: int,
                       end_col: int) -> bool:
        """
        Check if the selected move is valid for the current player.
        Conditions:
          - Must move a piece belonging to this player ('W' if player=0, 'B' if player=1).
          - Must move one row up (White) or down (Black).
          - Must move at most one column left or right (straight or diagonal).
          - If moving straight forward, target must be empty.
          - If moving diagonally forward, target must contain opponent's piece or be empty 
            (in some variations of Breakthrough, diagonal only if capturing. 
             Standard rules say you can move diagonally only if capturing. 
             We'll assume the official rule: diagonal = capture only.)
        """
        piece = self.board[start_row][start_col]
        if player_id == 0 and piece != "W":
            return False
        if player_id == 1 and piece != "B":
            return False

        row_dir = 1 if player_id == 0 else -1  # White moves row+1, Black row-1
        if end_row - start_row != row_dir:
            return False

        col_diff = abs(end_col - start_col)
        if col_diff > 1:
            return False

        # Next, check occupancy conditions
        dest_piece = self.board[end_row][end_col]
        # diagonal move => capturing an opponent
        if col_diff == 1:
            # Must be capturing: the destination must have an opponent's piece
            if player_id == 0 and dest_piece != "B":
                return False
            if player_id == 1 and dest_piece != "W":
                return False
        else:
            # Straight forward => must be empty
            if dest_piece != "":
                return False

        return True

    def _is_on_board(self, row: int, col: int) -> bool:
        """Check if row,col is a valid board coordinate."""
        if row < 0 or row >= self.board_size:
            return False
        if col < 0 or col >= self.board_size:
            return False
        return True

    def _check_winner(self):
        """
        Check for the following game-ending conditions:
          1) A piece has reached the opposite home row:
             - White is on row = board_size-1
             - Black is on row = 0
          2) All of a player's pieces have been captured.
        """
        # Condition 1: Reached home row
        # White
        for c in range(self.board_size):
            if self.board[self.board_size - 1][c] == "W":
                self.state.set_winners(player_ids=[0], reason="White reached Black's home row.")
                return
        # Black
        for c in range(self.board_size):
            if self.board[0][c] == "B":
                self.state.set_winners(player_ids=[1], reason="Black reached White's home row.")
                return

        # Condition 2: All captured
        white_count = sum(self.board[r][c] == "W"
                          for r in range(self.board_size)
                          for c in range(self.board_size))
        black_count = sum(self.board[r][c] == "B"
                          for r in range(self.board_size)
                          for c in range(self.board_size))

        if white_count == 0:
            self.state.set_winners(player_ids=[1], reason="All White pieces captured.")
            return
        if black_count == 0:
            self.state.set_winners(player_ids=[0], reason="All Black pieces captured.")
            return

    def _augment_observations(self):
        """Optionally broadcast the board state if is_open."""
        if self.is_open and not self.state.done:
            # Provide the board state as a text observation
            board_str = self._render_board()
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=board_str,
                for_logging=False
            )

    def _render_board(self) -> str:
        """
        Convert the board into a user-friendly ASCII representation.
        We'll label rows as 1..N (bottom to top) and columns as a..h, 
        matching the typical chess style from White's perspective.
        """
        lines = []
        size = self.board_size
        for row_index in range(size - 1, -1, -1):
            # Start line with row number
            row_label = str(row_index + 1).rjust(2, " ")
            row_str = [f"{row_label} |"]
            for col_index in range(size):
                piece = self.board[row_index][col_index]
                row_str.append(piece if piece else ".")
            lines.append(" ".join(row_str))

        # Column labels
        col_labels = "    " + " ".join(list(self._col_to_file[i] for i in range(size)))
        lines.append(col_labels)

        return "\n"+"\n".join(lines)
