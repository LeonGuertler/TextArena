import re
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.Othello.renderer import create_board_str

class OthelloEnv(ta.Env):
    """ Environment for a two-player game of Othello (Reversi) """
    def __init__(self, max_turns: Optional[int] = 60, show_valid: bool = True):
        """
        Initialize the Othello game environment.
        
        Args:
            max_turns (Optional[int]): Maximum number of turns before the game is truncated.
            show_valid (bool): If True, players are shown valid moves.
        """
        super().__init__()
        self.max_turns = max_turns
        self.show_valid = show_valid
        
        # Regex pattern for parsing moves
        self.move_pattern = re.compile(r"\[\s*(\d)\s*,?\s*(\d)\s*\]")
        
        # Board directions for checking captures
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),           (0, 1),   # W, E
            (1, -1),  (1, 0),  (1, 1)    # SW, S, SE
        ]
    
    def get_board_str(self):
        return create_board_str(self.board)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the Othello game to its initial state """
        # Initialize game state variables
        self.state = ta.State(
            num_players=num_players,
            min_players=2, 
            max_players=2,
            max_turns=self.max_turns,
            role_mapping={0: "Black", 1: "White"}
        )
        
        # Initialize the board (8x8)
        # Empty: '', Black: 'B', White: 'W'
        self.board = [['' for _ in range(8)] for _ in range(8)]
        
        # Set up initial four pieces in the center
        self.board[3][3] = 'W'
        self.board[3][4] = 'B'
        self.board[4][3] = 'B'
        self.board[4][4] = 'W'
        
        # Set initial counts
        black_count = 2
        white_count = 2
        
        # Calculate valid moves for first player (Black)
        valid_moves = self._get_valid_moves('B')
        
        game_state = {
            "board": self.board,
            "rendered_board": self._render_board(),
            "black_count": black_count,
            "white_count": white_count,
            "valid_moves": valid_moves
        }
        
        self.state.reset(
            seed=seed, 
            game_state=game_state, 
            player_prompt_function=self._generate_player_prompt
        )

    def _render_board(self):
        """
        Render the current state of the Othello board as a string.
        """
        header = "  0 1 2 3 4 5 6 7"
        board_str = [header]
        
        for i, row in enumerate(self.board):
            # Format each cell: Empty as '.', Black as 'B', White as 'W'
            formatted_row = [' ' if cell == '' else cell for cell in row]
            row_str = f"{i}|"
            for cell in formatted_row:
                if cell == ' ':
                    row_str += ".|"
                else:
                    row_str += f"{cell}|"
            board_str.append(row_str)
            
        return '\n'.join(board_str)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for a player """
        piece = 'B' if player_id == 0 else 'W'
        color = "Black" if player_id == 0 else "White"
        
        prompt = (
            f"You are playing {color} pieces ('{piece}') in Othello (Reversi).\n\n"
            "Rules:\n"
            "- On your turn, place one of your pieces on the board to capture opponent pieces.\n"
            "- You must place your piece such that it creates a straight line (horizontal, vertical, or diagonal) between your new piece and another of your pieces, with opponent pieces in between.\n"
            "- All opponent pieces in that line are then flipped to your color.\n"
            "- You must make a move that captures at least one opponent piece.\n"
            "- If you cannot make a valid move, your turn is skipped.\n"
            "- The game ends when neither player can make a valid move.\n"
            "- The player with more pieces on the board wins.\n\n"
            "To submit your move, provide the coordinates as [row, col], where both row and col are between 0 and 7.\n"
            "For example, '[2, 3]' places your piece at row 2, column 3.\n\n"
            f"Current board state:\n{game_state['rendered_board']}\n\n"
            f"Piece count - Black: {game_state['black_count']}, White: {game_state['white_count']}\n"
        )
        
        if self.show_valid and game_state['valid_moves']:
            prompt += f"Valid moves for {color}: " + ", ".join([f"[{r}, {c}]" for r, c in game_state['valid_moves']])
        elif self.show_valid and not game_state['valid_moves']:
            prompt += f"No valid moves for {color}. Your turn will be skipped."
            
        return prompt

    def _get_valid_moves(self, piece: str) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the given piece.
        
        Args:
            piece (str): 'B' for Black or 'W' for White
            
        Returns:
            List[Tuple[int, int]]: List of valid move coordinates as (row, col) tuples
        """
        valid_moves = []
        opponent = 'W' if piece == 'B' else 'B'
        
        for row in range(8):
            for col in range(8):
                # Skip if the cell is not empty
                if self.board[row][col] != '':
                    continue
                
                # Check if placing a piece here would capture any opponent pieces
                if self._would_flip_any(row, col, piece):
                    valid_moves.append((row, col))
                    
        return valid_moves

    def _would_flip_any(self, row: int, col: int, piece: str) -> bool:
        """
        Check if placing a piece at the given position would flip any opponent pieces.
        
        Args:
            row (int): Row coordinate
            col (int): Column coordinate
            piece (str): Player's piece ('B' or 'W')
            
        Returns:
            bool: True if at least one opponent piece would be flipped, False otherwise
        """
        opponent = 'W' if piece == 'B' else 'B'
        
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            # Need at least one opponent piece in this direction
            if not (0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent):
                continue
                
            # Continue in this direction
            r += dr
            c += dc
            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == '':
                    # Empty space, no capture in this direction
                    break
                if self.board[r][c] == piece:
                    # Found our own piece, capture is possible
                    return True
                # Otherwise, it's another opponent piece, continue
                r += dr
                c += dc
                
        return False

    def _flip_pieces(self, row: int, col: int, piece: str) -> int:
        """
        Flip opponent pieces after placing a piece at the given position.
        
        Args:
            row (int): Row coordinate where the piece was placed
            col (int): Column coordinate where the piece was placed
            piece (str): Player's piece ('B' or 'W')
            
        Returns:
            int: Number of opponent pieces flipped
        """
        opponent = 'W' if piece == 'B' else 'B'
        flipped_count = 0
        
        # Place the piece
        self.board[row][col] = piece
        
        # Check each direction for opponent pieces to flip
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            # Find opponent pieces in this direction
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
                
            # If we found our own piece at the end, flip all opponent pieces in between
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == piece and to_flip:
                for flip_r, flip_c in to_flip:
                    self.board[flip_r][flip_c] = piece
                    flipped_count += 1
                    
        return flipped_count

    def _count_pieces(self) -> Tuple[int, int]:
        """
        Count the number of black and white pieces on the board.
        
        Returns:
            Tuple[int, int]: (black_count, white_count)
        """
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        return black_count, white_count

    def _is_game_over(self) -> bool:
        """
        Check if the game is over (no valid moves for either player).
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        return not self._get_valid_moves('B') and not self._get_valid_moves('W')

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's move """
        # Get the current player
        player_id = self.state.current_player_id
        piece = 'B' if player_id == 0 else 'W'
        opponent_piece = 'W' if player_id == 0 else 'B'
        
        # Update the log
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)
        
        # Get valid moves for the current player
        valid_moves = self._get_valid_moves(piece)
        self.state.game_state["valid_moves"] = valid_moves
        
        # If there are no valid moves, skip the player's turn
        if not valid_moves:
            message = f"Player {player_id} ({piece}) has no valid moves and must skip their turn."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            
            # Check if the game is over (if both players have no valid moves)
            if not self._get_valid_moves(opponent_piece):
                self._determine_winner()
            else:
                # Update valid moves for the next player
                next_valid_moves = self._get_valid_moves(opponent_piece)
                self.state.game_state["valid_moves"] = next_valid_moves
                
            return self.state.step()
        
        # Parse the move
        match = self.move_pattern.search(action)
        
        if match is None:
            # Invalid format
            reason = f"Invalid move format. Player {player_id} did not respond with valid coordinates."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            row, col = int(match.group(1)), int(match.group(2))
            
            # Check if the move is valid
            if (row, col) in valid_moves:
                # Make the move and flip pieces
                flipped = self._flip_pieces(row, col, piece)
                
                # Update the board display and piece counts
                black_count, white_count = self._count_pieces()
                self.state.game_state["rendered_board"] = self._render_board()
                self.state.game_state["black_count"] = black_count
                self.state.game_state["white_count"] = white_count
                
                # Notify players
                message = (
                    f"Player {player_id} ({piece}) placed a piece at [{row}, {col}] and flipped {flipped} "
                    f"opponent {opponent_piece} piece(s).\n"
                    f"Current scores - Black: {black_count}, White: {white_count}"
                )
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                
                # Check if the game is over
                if self._is_game_over():
                    self._determine_winner()
                else:
                    # Update valid moves for the next player
                    next_valid_moves = self._get_valid_moves(opponent_piece)
                    self.state.game_state["valid_moves"] = next_valid_moves
                    
                    # Add the updated board state to the observation for the next player
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=1-player_id,  # Next player's ID
                        message=f"Updated board state:\n{self.state.game_state['rendered_board']}\n\n"
                               f"Piece count - Black: {black_count}, White: {white_count}",
                        for_logging=False
                    )
                    
                    # Add valid moves information for the next player
                    if self.show_valid:
                        if next_valid_moves:
                            valid_moves_str = ", ".join([f"[{r}, {c}]" for r, c in next_valid_moves])
                            self.state.add_observation(
                                from_id=ta.GAME_ID,
                                to_id=1-player_id,
                                message=f"Valid moves for {'Black' if 1-player_id == 0 else 'White'}: {valid_moves_str}",
                                for_logging=False
                            )
                        else:
                            self.state.add_observation(
                                from_id=ta.GAME_ID,
                                to_id=1-player_id,
                                message=f"No valid moves for {'Black' if 1-player_id == 0 else 'White'}. Your turn will be skipped.",
                                for_logging=False
                            )
            else:
                # Invalid move
                valid_moves_str = ", ".join([f"[{r}, {c}]" for r, c in valid_moves])
                reason = f"Player {player_id} tried to place a piece at an invalid position. Valid moves are: {valid_moves_str}"
                self.state.set_invalid_move(player_id=player_id, reason=reason)
        
        return self.state.step()

    def _determine_winner(self):
        """ Determine the winner based on the piece count """
        black_count, white_count = self._count_pieces()
        
        if black_count > white_count:
            # Black wins
            self.state.set_winners(player_ids=[0], reason=f"Game over. Black wins with {black_count} pieces to White's {white_count} pieces.")
        elif white_count > black_count:
            # White wins
            self.state.set_winners(player_ids=[1], reason=f"Game over. White wins with {white_count} pieces to Black's {black_count} pieces.")
        else:
            # Draw
            self.state.set_draw(reason=f"Game over. The game ends in a draw with both players having {black_count} pieces.")