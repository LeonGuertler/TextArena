from typing import Optional, Dict, Tuple, List, Any
import textarena as ta
import re

class UltimateTicTacToeEnv(ta.Env):
    """
    Environment for a two-player game of Ultimate Tic Tac Toe.
    """
    def __init__(
        self,
    ):
        """
        Initialize the environment.
        """
        self.environment_name = "Ultimate Tic Tac Toe"

        ## Initialise the game state
        self.state = ta.State(
            num_players=2,
            max_turns=None
        )

        ## attach render objects
        # self.board_state_render = ...
        self.render_keys = ['rendered_board']


    def reset(
        self,
        seed: Optional[int] = None
    ):
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Seed for the random number generator.
        """
        ## Initialise the board
        self.board, self.macro_board = self._generate_board()
        self.next_micro_board = None

        ## Initialise the game state
        self.state.reset(
            game_state={
                "board": self.board,
                "rendered_board": self._render_board()
            },
            player_prompt_function=self._generate_player_prompt,
            seed=seed
        )

    def _generate_board(self):
        """
        Generate the initial board state.
        """
        return [[[' ' for _ in range(3)] for _ in range(3)] for _ in range(9)], [[' ' for _ in range(3)] for _ in range(3)]  # 9 micro boards, macro board
    
    def _render_board(self):
        """Return the current state of the Ultimate Tic Tac Toe board as a string."""
        board_str = []
        for macro_row in range(3):
            for micro_row in range(3):
                row = []
                for macro_col in range(3):
                    row.extend(self.board[macro_row * 3 + macro_col][micro_row])
                    row.append('|')
                board_str.append(' '.join(row[:-1]))
            board_str.append('-' * 23)
        return '\n'.join(board_str)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]):
        """
        Generate the prompt for the current player.
        """
        prompt = (
            f"You are Player {player_id}. You are playing Ultimate Tic Tac Toe.\n"
            "Your goal is to win three micro boards in a row (horizontally, vertically, or diagonally) on the macro board.\n"
            "Each micro board is a 3x3 Tic Tac Toe grid, and the macro board tracks which player has won each micro board.\n"
            "On your turn, you can mark an empty cell in a micro board. The position of your move determines which micro board\n"
            "your opponent must play in next.\n"
            "\n"
            "Rules to remember:\n"
            "1. A move must be made in the micro board specified by the previous move.\n"
            "   For example, if the last move was in the top-left corner of a micro board, the opponent must play in the top-left micro board.\n"
            "2. If the directed micro board is already won or full, you are free to play in any available micro board.\n"
            "3. You win a micro board by completing a row, column, or diagonal within that board.\n"
            "4. You win the game by completing three micro boards in a row on the macro board.\n"
            "5. The game ends in a draw if all micro boards are filled, and no player has three in a row on the macro board.\n"
            "6. To submit your move, submit them as [micro_board, row, col], where micro_board is the index of the micro board (0-8), and row and col are the cell coordinates (0-2).\n"
            "For example, to play in the center cell of the top-left micro board, submit [0 1 1].\n"
            "\n"
            f"As Player {player_id}, you will be '{'X' if player_id == 1 else 'O'}', whereas your opponent is '{'O' if player_id == 1 else 'X'}'.\n"
            "Below is the current state of the macro board (tracking micro board wins):\n"
            f"{self._render_board()}\n"
            "\n"
        )

        return prompt
    
    def step(
        self,
        action: str
    ) -> Tuple[
        Optional[ta.Rewards], 
        bool,
        bool,
        ta.Info
    ]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            rewards: The rewards for each player.
            terminted: Whether the episode is over.
            truncated: Whether the episode was truncated.
            info: Additional information.
        """
        ## set the current player
        player_id = self.state.current_player_id
        self.current_player = 'X' if player_id == 1 else 'O'

        ## validate the action
        self.state.check_action_format(
            action=action
        )

        ## update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        ## set up the action search pattern
        action_search_pattern = re.compile(r"\[(\d) (\d) (\d)\]") # [micro_board row col]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid move format. Player {player_id} did not respond with a valid move."]
            )
        else:
            micro_board, row, col = match.groups()
            micro_board, row, col = int(micro_board), int(row), int(col)

            ## check if the move is valid
            if not self._is_valid_move(micro_board, row, col):
                self.state.set_invalid_move(
                    player_ids=[player_id],
                    reasons=[f"Invalid move. Try again."]
                )
            else:
                self._make_move(micro_board, row, col)
                ## check if the next micro board is already won
                if self.macro_board[row][col] != ' ':
                    self.next_micro_board_str = "any micro board"
                else:
                    self.next_micro_board_str = f"micro board {row * 3 + col}"
                ## check if the micro board is won
                if self._check_winner(self.board[micro_board]):
                    self.macro_board[micro_board // 3][micro_board % 3] = self.current_player

                    self.state.add_observation(
                        from_id=-1,
                        to_id=-1,
                        message=f"Player {player_id} won micro board {micro_board}! Player {1-player_id} must play in {self.next_micro_board_str}. New state of the board:\n{self._render_board()}",
                        for_logging=True
                    )
                else:
                    self.state.add_observation(
                        from_id=-1,
                        to_id=-1,
                        message=f"Player {player_id} made a move in micro board {micro_board} at row {row}, col {col}. Player {1-player_id} must play in {self.next_micro_board_str}. New state of the board:\n{self._render_board()}",
                        for_logging=True
                    )

            ## check if the game is over
            if self._check_winner(self.macro_board):
                self.state.set_winners(
                    player_ids=[player_id],
                    reason=[f"Player {player_id} has won the Ultimate Tic Tac Toe!"]
                )
            elif self._is_draw():
                self.state.set_draw(
                    reason=["The game is a draw!"]
                )

        ## update the rendered board
        self.state.game_state["rendered_board"] = self._render_board()

        return self.state.step()
    
    def _make_move(self, micro_board, row, col):
        """
        Make a move if valid and update the game state.
        """
        ## make move
        self.board[micro_board][row][col] = self.current_player

        ## determine the next micro board
        self.next_micro_board = row * 3 + col
        if self.macro_board[self.next_micro_board // 3][self.next_micro_board % 3] != ' ':
            self.next_micro_board = None


    def _check_winner(self, board):
        """Check if a given board has a winner."""
        for i in range(3):
            # Check rows and columns
            if board[i][0] == board[i][1] == board[i][2] != ' ':
                return True
            if board[0][i] == board[1][i] == board[2][i] != ' ':
                return True

        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return True
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return True

        return False

    def _is_draw(self):
        """Check if the game is a draw."""
        if any(cell == ' ' for row in self.macro_board for cell in row):
            return False

        return True
    
    def _is_valid_move(self, micro_board, row, col):
        """Check if a move is valid."""
        if micro_board < 0 or micro_board > 8 or row < 0 or row > 2 or col < 0 or col > 2:
            return False
        
        if self.board[micro_board][row][col] != ' ':
            return False
        
        if self.next_micro_board is not None and micro_board != self.next_micro_board:
            return False

        return self.board[micro_board][row][col] == ' '
    
    def render(self):
        """
        Render the environment.
        """
        return self.state.game_state["rendered_board"]
            