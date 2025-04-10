import re
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.Checkers.renderer import create_board_str

class CheckersEnv(ta.Env):
    """
    Environment for a simple two-player game of Checkers (Draughts).
    
    NOTE: This is a *minimal* implementation and does not handle all advanced rules 
    such as forced captures, multi-jumps, etc. For demonstration/example purposes only.
    """

    def __init__(self, max_turns: int = 50):
        """
        Initialize the Checkers environment.
        
        Args:
            max_turns (int): Maximum number of turns before the game ends in a draw.
        """
        self.max_turns = max_turns
        
        # Regex pattern to find moves in the format [rowFrom colFrom rowTo colTo]
        # e.g. [2 1 3 2] means moving the piece from (2,1) to (3,2)
        self.move_pattern = re.compile(r"\[\s*(\d)\s+(\d)\s+(\d)\s+(\d)\s*\]")

        # Represent pieces as follows:
        # 'r' = Red piece      'R' = Red King
        # 'b' = Black piece    'B' = Black King
        # '.' = empty square (rendered as dot)
        
        # Players: 
        #   0 -> Red 
        #   1 -> Black

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """
        Reset the environment to its initial state.
        
        Args:
            num_players (int): Number of players (must be 2).
            seed (Optional[int]): Random seed if needed.
        """
        # Create state and ensure exactly 2 players
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=2,
            max_turns=self.max_turns,
            role_mapping={0: "Red", 1: "Black"},
        )

        # Create the initial board layout
        self.board = self._initialize_board()
        
        # Build the initial game_state dictionary
        game_state = {
            "board": self.board,  # 2D list
            "rendered_board": self._render_board()
        }

        # Reset the environment's internal state
        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

    def _initialize_board(self) -> List[List[str]]:
        """
        Create an 8x8 list of lists for the board.
        Place black pieces (b) in rows 0..2, red pieces (r) in rows 5..7,
        on alternating dark squares. Other squares are '.' by default.
        """
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # Black pieces (player_id = 1) in rows 0..2
        for row in range(3):
            for col in range(8):
                # Place black on "dark" squares only
                if (row + col) % 2 == 1:
                    board[row][col] = 'b'
        
        # Red pieces (player_id = 0) in rows 5..7
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 'r'
        
        return board

    def _render_board(self) -> str:
        """
        Render the board in a clean, readable ASCII format with edge separation.
        """
        board_str = "     " + "  ".join(str(col) for col in range(8)) + "\n"
        board_str += "   +" + "-" * 25 + "\n"

        for row in range(8):
            row_str = f" {row} |"
            for col in range(8):
                piece = self.board[row][col]
                row_str += f" {piece} "
            board_str += row_str + "\n"
        
        return board_str


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the prompt for the given player.
        
        Args:
            player_id (int): The ID of the current player.
            game_state (Dict[str, Any]): The current game state.
        """
        color = "Red" if player_id == 0 else "Black"
        return (
            f"You are Player {player_id} playing as {color}.\n"
            "Make your move in the format [rowFrom colFrom rowTo colTo], e.g. [2 1 3 2].\n"
            "You may include extra explanation if you wish, but the move must appear exactly in that format.\n"
            "Basic rules:\n"
            "  • Move diagonally forward by 1 if empty.\n"
            "  • Capture by jumping over an opponent piece.\n"
            "  • A piece is Kinged if it reaches the opposite end.\n"
            "\n"
            "Here is the current board:\n"
            f"{game_state['rendered_board']}"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a single move in the game.
        
        Args:
            action (str): The player's textual action, e.g. [2 1 3 2]
        
        Returns:
            done (bool): Whether the game is finished
            info (Dict[str, Any]): Additional game info
        """
        player_id = self.state.current_player_id

        # Log the player's action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Execute the move (or mark invalid move if the format is wrong/illegal)
        self._execute_player_move(action)
        
        # Update the "rendered_board" in the game state
        self.state.game_state["rendered_board"] = self._render_board()

        # Check if we have a winner or draw
        self._check_gameover()

        # Proceed to the next turn (or finalize if done)
        return self.state.step()

    def _execute_player_move(self, action: str):
        """
        Parse the action to find the requested move. If valid, make the move, 
        otherwise set it as an invalid move.
        """
        player_id = self.state.current_player_id
        match = self.move_pattern.search(action.strip())

        if not match:
            self.state.set_invalid_move(player_id, "No valid move format found.")
            return

        # Extract coordinates
        row_from, col_from, row_to, col_to = map(int, match.groups())

        # Validate and execute the move if possible
        if self._is_valid_move(player_id, row_from, col_from, row_to, col_to):
            self._move_piece(player_id, row_from, col_from, row_to, col_to)
        else:
            self.state.set_invalid_move(player_id, f"Move [{row_from} {col_from} {row_to} {col_to}] is illegal.")

    def _is_valid_move(self, player_id: int, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Check if a move is valid under simplified Checkers rules.
        NOTE: This does not handle forced captures or multi-jumps.
        """
        if not (0 <= r1 < 8 and 0 <= c1 < 8 and 0 <= r2 < 8 and 0 <= c2 < 8):
            return False  # Out of board bounds

        piece = self.board[r1][c1]
        target = self.board[r2][c2]

        # Check piece ownership
        if player_id == 0:
            # Player 0 -> must move 'r' or 'R'
            if piece not in ['r', 'R']:
                return False
        else:
            # Player 1 -> must move 'b' or 'B'
            if piece not in ['b', 'B']:
                return False

        if target != '.':
            return False  # destination must be empty

        # Simple move logic (no forced capture / multi-jump):
        dr = r2 - r1
        dc = abs(c2 - c1)

        # If it's a King, can move diagonally up or down by 1 step
        if piece in ['R', 'B']:
            if abs(dr) == 1 and dc == 1:
                return True
            # Or single capture: move by 2 if jumping over opponent
            if abs(dr) == 2 and dc == 2:
                return self._is_valid_capture(r1, c1, r2, c2)
            return False
        else:
            # Non-king logic: must move forward 1 or capture forward
            direction = -1 if piece == 'r' else 1  # Red moves "up", Black moves "down"

            # Non-capturing move
            if dr == direction and dc == 1:
                return True

            # Capturing move
            if dr == 2 * direction and dc == 2:
                return self._is_valid_capture(r1, c1, r2, c2)

            return False

    def _is_valid_capture(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Check if there is an opponent piece in between (for capturing).
        """
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        moving_piece = self.board[r1][c1]
        jumped_piece = self.board[mid_r][mid_c]

        if moving_piece.lower() == 'r' and jumped_piece.lower() == 'b':
            return True
        if moving_piece.lower() == 'b' and jumped_piece.lower() == 'r':
            return True
        return False

    def _move_piece(self, player_id: int, r1: int, c1: int, r2: int, c2: int):
        """
        Carry out the move on the board (including captures and kinging).
        """
        piece = self.board[r1][c1]
        self.board[r1][c1] = '.'
        self.board[r2][c2] = piece

        # If capturing, remove the jumped piece
        if abs(r2 - r1) == 2:
            mid_r = (r1 + r2) // 2
            mid_c = (c1 + c2) // 2
            self.board[mid_r][mid_c] = '.'

        # Check for kinging
        if piece == 'r' and r2 == 0:
            self.board[r2][c2] = 'R'  # Red becomes King
        elif piece == 'b' and r2 == 7:
            self.board[r2][c2] = 'B'  # Black becomes King

        # Send a summary message
        move_summary = f"Player {player_id} moved ({r1},{c1}) -> ({r2},{c2})."
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=move_summary)

    def _check_gameover(self):
        """
        Check if the game is over by seeing if one side has no pieces or no moves.
        If so, declare a winner or set draw accordingly.
        """
        red_pieces = sum(cell.lower() == 'r' for row in self.board for cell in row)
        black_pieces = sum(cell.lower() == 'b' for row in self.board for cell in row)

        if red_pieces == 0:
            self.state.set_winners([1], reason="Red has no pieces left. Black wins!")
            return
        if black_pieces == 0:
            self.state.set_winners([0], reason="Black has no pieces left. Red wins!")
            return

        # If either player has no legal moves, that player loses.
        current_player = self.state.current_player_id
        if not self._has_legal_move(current_player):
            # The other player wins
            winner = 1 - current_player
            self.state.set_winners([winner], reason=f"Player {current_player} has no moves left.")
            return

        # By default, if max_turns is reached, the environment logic in `State.step()` 
        # will set the outcome to a draw (check `self.check_truncated` in the core `State`).

    def _has_legal_move(self, player_id: int) -> bool:
        """
        Basic check to see if the given player has at least one possible move
        in this simplified scenario (forward or capturing).
        """
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if player_id == 0 and piece in ['r', 'R']:
                    if self._can_piece_move(r, c):
                        return True
                elif player_id == 1 and piece in ['b', 'B']:
                    if self._can_piece_move(r, c):
                        return True
        return False

    def _can_piece_move(self, r: int, c: int) -> bool:
        """
        Returns True if the piece at (r, c) has at least one valid move.
        """
        piece = self.board[r][c]
        if piece == '.':
            return False

        # Determine directions
        if piece in ['R', 'B']:
            # King can move in any diagonal direction
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Normal piece moves
            if piece == 'r':
                directions = [(-1, -1), (-1, 1)]  # Red goes "up"
            else:
                directions = [(1, -1), (1, 1)]   # Black goes "down"

        # Check single-step or capture possibility
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Simple move
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == '.':
                return True

            # Check capture possibility
            nr2, nc2 = r + 2*dr, c + 2*dc
            if 0 <= nr2 < 8 and 0 <= nc2 < 8 and self.board[nr2][nc2] == '.':
                jumped = self.board[r + dr][c + dc]
                if piece.lower() == 'r' and jumped.lower() == 'b':
                    return True
                if piece.lower() == 'b' and jumped.lower() == 'r':
                    return True

        return False
