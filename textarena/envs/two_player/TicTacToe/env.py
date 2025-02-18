from typing import Optional, Dict, Tuple, List, Any
import textarena as ta
import re, random

class TicTacToeEnv(ta.Env):
    """
    Environment for a two-player game of Tic Tac Toe.
    """
    def __init__(self):
        super().__init__()

        self.state = ta.State(
            num_players=2,
            max_turns=None
        )

        self.current_player = None
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.move_history = []
        
        
    @property
    def offline_renderer(self):
        from textarena.envs.two_player.TicTacToe.render.renderer import TicTacToeRenderer
        return TicTacToeRenderer
    
    @property
    def terminal_render_keys(self):
        return ["rendered_board"]
    
    def reset(self,seed: Optional[int]=None):
        """
        Reset the environment to the initial state.

        Args:
            seed: Seed for the random number generator.
        """
        if seed is not None:
            random.seed(seed)

        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.state.reset(
            game_state={
                "board": self.board,
                "rendered_board": self._render_board()
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def _render_board(self):
        """
        Return the current state of the Tic Tac Toe board as a string.
        """
        board_str = []
        for row in self.board:
            board_str.append("|".join(row))
        return "\n".join(board_str)

       

    def _generate_player_prompt(slef,player_id:int,game_state:Dict[str,Any])-> str:
        """
        Generate the prompt for the current player.
        """
        prompt = (
            f"You are Player {player_id} in Tic Tac Toe.\n"
            "Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.\n"
            "On your turn, you mark a row&col position on the 3x3 grid.\n"
            "\n"
            "Rules to remember:\n"
            "1. You can only place your mark in an empty cell.\n" 
            "2. You win by completing three of your marks in a row (horizontally, vertically, or diagonally).\n"
            "3. The game ends in a draw if the board is filled with no winner.\n"
            "4. To submit your move, provide [row, col] where row and col are the cell coordinates (0-2).\n"
            "For example, [1 1] places your mark in the center cell of the board.\n"
            "\n"
            f"As Player {player_id}, you will be '{'X' if player_id == 1 else 'O'}', while your opponent is '{'O' if player_id == 1 else 'X'}'.\n"
            "Below is the current state of the board:\n"
            f"{self._render_board()}\n"
        )
        return prompt

    def step(self,action:str)->Tuple[bool,ta.Info]:
        """
        Take a step in the environment.
        """
        # Set the current player
        player_id = self.state.current_player_id
        self.current_player = 'X' if player_id == 1 else 'O'
        
        # Record the observation (the action)
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        # Set up the action search pattern
        action_search_pattern = re.compile(r"\[\s*(\d)\s*,?\s*(\d)\s*\]")
        match = action_search_pattern.search(action)

        if match is None:
            # Invalid format
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with a valid move."
            )
        else:
            row, col = match.groups()
            row, col = int(row), int(col)

            if self._is_valid_move(row, col):
                # Make the move
                self.board[row][col] = self.current_player
                self.move_history.append((row, col))

                # Construct a message about the move
                self.state.add_observation(
                    from_id=-1,
                    to_id=-1,
                    message=(
                        f"Player {player_id} made a move at row {row}, col {col}.\n"
                        f"New state of the board:\n{self._render_board()}"
                    ),
                    for_logging=False
                )

                if self._check_winner():
                    self.state.set_winners(
                        player_ids=[player_id],
                        reason=f"Player {player_id} has won!"
                    )
                elif self._is_draw():
                    self.state.set_draw(
                        reason="The game is a draw!"
                    )
        self.state.game_state["rendered_board"] = self._render_board()

        return self.state.step()
                    
                

                
    def _is_valid_move(self,row,col)->bool:
        """
        Check if a move is valid.
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        
        if self.board[row][col] != '':
            return False
        
        return True
    
    def _check_winner(self) -> bool:
        """
        Check if a given 3×3 board has a winner.
        """
        # Rows and columns
        for i in range(3):
            # Check rows
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '':
                return True
            # Check columns
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return True
            
        # Diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return True
        
        return False
            

        
    def _is_draw(self) -> bool:
        """
        Check if the game is a draw.
        """
        if any(cell == '' for row in self.board for cell in row):
            return False
        return True
        