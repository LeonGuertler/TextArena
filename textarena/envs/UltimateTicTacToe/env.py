import re, random
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.UltimateTicTacToe.renderer import create_board_str

class UltimateTicTacToeEnv(ta.Env):
    """ Environment for a two-player game of Ultimate Tic Tac Toe """
    def __init__(self):
        """ Initialize the environment """
        ## Initialise the board
        self.board = None

        ## initialise the move history
        self.move_history = []

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the environment to the initial state """
        ## Initialise the game state
        self.state = ta.State(num_players=2, min_players=2, max_players=2, max_turns=None)

        # Initialize the board
        self.board, self.macro_board = self._generate_board()
        self.next_micro_board = None

        # Reset the game state
        self.state.reset(
            seed=seed,
            game_state={
                "board": self.board,
                "rendered_board": self._render_board()
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def _generate_board(self):
        """
        Generate the initial board state.

        Returns:
            board:  A list of 9 micro boards (each 3x3).
            macro_board: The 3x3 board for the macro layout.
        """
        # 9 micro boards (each 3×3), plus a 3×3 macro board
        board = [[[' ' for _ in range(3)] for _ in range(3)] for _ in range(9)]
        macro_board = [[' ' for _ in range(3)] for _ in range(3)]
        return board, macro_board
    
    def _render_board(self):
        """
        Return the current state of the Ultimate Tic Tac Toe board as a string.
        """
        board_str = []
        for macro_row in range(3):
            for micro_row in range(3):
                row = []
                for macro_col in range(3):
                    row.extend(self.board[macro_row * 3 + macro_col][micro_row])
                    row.append('|')
                # Remove the last '|'
                row_render = ' '.join(row[:-1])
                board_str.append(row_render)
            # Separator row between macro rows
            if macro_row < 2:
                board_str.append('-' * 21)
        return '\n'.join(board_str)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the prompt for the current player """
        prompt = (
            f"You are Player {player_id} in Ultimate Tic Tac Toe.\n"
            "Your goal is to win three micro boards in a row (horizontally, vertically, or diagonally) on the macro board.\n"
            "Each micro board is a 3x3 Tic Tac Toe grid, and the macro board tracks which player has won each micro board.\n"
            "On your turn, you mark a row&col position. It both chooses where in your micro board to tick and \n"
            "where in the macro board (what micro board) your opponent would play next.\n"
            "\n"
            "Rules to remember:\n"
            "1. A move must be made in the micro board specified by the previous move.\n"
            "2. If the directed micro board is already won or full, you are free to play in any available micro board.\n"
            "3. You win a micro board by completing a row, column, or diagonal within that board.\n"
            "4. You win the game by completing three micro boards in a row on the macro board.\n"
            "5. The game ends in a draw if all micro boards are filled, and no player has three in a row on the macro board.\n"
            "6. To submit your move, provide [micro_board, row, col], where micro_board is the index of the micro board (0-8),\n"
            "   and row and col are the cell coordinates (0-2).\n"
            "For example, [0 1 1] places a tick on the top left (0) board at its center (1,1) and the next move would be on the central board.\n"
            "\n"
            f"As Player {player_id}, you will be '{'X' if player_id == 1 else 'O'}', while your opponent is '{'O' if player_id == 1 else 'X'}'.\n"
            "Below is the current state of the macro board (tracking micro board wins), followed by the micro boards:\n"
            f"{self._render_board()}\n"
        )
        return prompt
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Take a step in the environment """
        # Set the current player
        player_id = self.state.current_player_id
        self.current_player = 'X' if player_id == 1 else 'O'

        # Record the observation (the action)
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, for_logging=True)

        ## set up the action search pattern
        action_search_pattern = re.compile(r"\[\s*(\d)\s*,?\s*(\d)\s*,?\s*(\d)\s*\]") 
        match = action_search_pattern.search(action)

        if match is None:
            # Invalid format
            reason=f"Invalid move format. Player {player_id} did not respond with a valid move."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            micro_board, row, col = match.groups()
            micro_board, row, col = int(micro_board), int(row), int(col)


            if self._is_valid_move(micro_board, row, col):
            
                # Make the move
                self._make_move(micro_board, row, col)

                # Construct a message about the move
                if self.next_micro_board is None:
                    self.next_micro_board_str = "any micro board"
                else:
                    self.next_micro_board_str = f"micro board {self.next_micro_board}"

                message=(
                    f"Player {player_id} made a move in micro board {micro_board} at row {row}, col {col}. "
                    f"Player {1 - player_id} must play in {self.next_micro_board_str}. "
                    f"New state of the board:\n{self._render_board()}"
                )
                self.state.add_observation(from_id=-1, to_id=-1, message=message, for_logging=False)

            # Check if anyone won the macro board
            if self._check_winner(self.macro_board):
                self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} has won the Ultimate Tic Tac Toe!")
            
            elif self._is_draw():
                # Check for draw
                self.state.set_draw(reason="The game is a draw!")

        # Update the rendered board in game state
        self.state.game_state["rendered_board"] = self._render_board()

        return self.state.step()
    
    def _make_move(self, micro_board: int, row: int, col: int):
        """
        Make a move if valid and update the game state accordingly.
        """
        # 1. Place the current player's marker
        self.board[micro_board][row][col] = self.current_player

        # append to move history
        self.move_history.append((micro_board, row, col))
        
        
        # 2. Check if that micro board is now won
        if self._check_winner(self.board[micro_board]):
            # Mark macro board
            self.macro_board[micro_board // 3][micro_board % 3] = self.current_player
            
            # Fill the entire 3×3 micro board with the winner's symbol
            for r in range(3):
                for c in range(3):
                    self.board[micro_board][r][c] = self.current_player

            # If a micro-grid is won, the opponent is free to pick ANY next board
        self.next_micro_board = row * 3 + col

        # If the board we forced is already won or full, THEN we set self.next_micro_board = None
        if (self.macro_board[self.next_micro_board // 3][self.next_micro_board % 3] != ' '
            or self._board_is_full(self.board[self.next_micro_board])):
            self.next_micro_board = None
                  
                  
    def _check_winner(self, board: List[List[str]]) -> bool:
        """
        Check if a given 3×3 board has a winner.
        """
        # Rows and columns
        for i in range(3):
            # Check rows
            if board[i][0] == board[i][1] == board[i][2] != ' ':
                return True
            # Check columns
            if board[0][i] == board[1][i] == board[2][i] != ' ':
                return True

        # Diagonals
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return True
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return True

        return False

    def _is_draw(self) -> bool:
        """
        Check if the entire macro board is filled and nobody has three in a row.
        """
        # If there's any ' ' in the macro board, it's not a draw
        if any(cell == ' ' for row in self.macro_board for cell in row):
            return False
        return True
    
    def _is_valid_move(self, micro_board, row, col):
        """Check if a move is valid."""
        is_invalid = False

        ## check if the micro_board, row, and col are within the valid range
        if micro_board < 0 or micro_board > 8 or row < 0 or row > 2 or col < 0 or col > 2:
            is_invalid = True
            reason = f"Player {self.state.current_player_id} made an invalid move. The micro_board, row, or col is out of range."
        
        ## check if the cell is empty
        elif self.board[micro_board][row][col] != ' ':
            is_invalid = True
            reason = f"Player {self.state.current_player_id} made an invalid move. The cell is already occupied."
        
        ## check if the next micro board is not won but the player is playing in a different micro board
        elif self.next_micro_board is not None and micro_board != self.next_micro_board:
            is_invalid = True
            reason = f"Player {self.state.current_player_id} made an invalid move. The player must play in the next micro board."
        
        ## check if the micro board is won and the player is still playing in it.
        elif self.macro_board[micro_board // 3][micro_board % 3] != ' ':
            is_invalid = True
            reason = f"Player {self.state.current_player_id} made an invalid move. The micro board is already won."

        if is_invalid:
            self.state.set_invalid_move(player_id=self.state.current_player_id, reason=reason)
            return False
        else:
            return self.board[micro_board][row][col] == ' '

    def _board_is_full(self, board: List[List[str]]) -> bool:
        """
        Check if a 3×3 board is full.
        """
        return all(cell != ' ' for row in board for cell in row)
