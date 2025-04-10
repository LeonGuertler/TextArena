import re, random
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.TicTacToe.renderer import create_board_str

class TicTacToeEnv(ta.Env):
    """ Environment for a two-player game of Tic Tac Toe """
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the environment to the initial state """
        self.state = ta.State(num_players=2, min_players=2, max_players=2)
        self.state.reset(
            game_state={"board": [['' for _ in range(3)] for _ in range(3)]}, 
            player_prompt_function=self._generate_player_prompt
        )
        self._observer_current_state()

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def _render_board(self):
        board = self.state.game_state["board"]
        return "\n---+---+---\n".join(
            "|".join(
                f" {board[r][c]} " if board[r][c] else f" {str(r * 3 + c)} "
                for c in range(3)
            )
            for r in range(3)
        )

    def _observer_current_state(self):
        available_moves = [
            f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3)
            if self.state.game_state["board"][r][c] == ''
        ]
        # Compose a message including the board and the available moves
        message = f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id:int, game_state:Dict[str,Any])-> str:
        """ Generate the prompt for the current player """
        prompt = (
            f"You are Player {player_id} in Tic Tac Toe.\n"
            "Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.\n"
            "On your turn, you should select the square number (0-8) you want to put your mark in next.\n"
            "For example, '[4]' places your mark in the center cell of the board.\n"
            "\n"
            f"As Player {player_id}, you will be '{'X' if player_id == 1 else 'O'}', "
            f"while your opponent is '{'O' if player_id == 1 else 'X'}'.\n"
        )
        return prompt

    def step(self,action:str)->Tuple[bool,ta.Info]:
        """ Take a step in the environment """
        # Set the current player
        player_id = self.state.current_player_id
        self.current_player = 'X' if player_id == 1 else 'O'
        
        # Record the observation (the action)
        self.state.add_observation(from_id=player_id, to_id =-1, message=action)

        # Set up the action search pattern
        action_search_pattern = re.compile(r"\[\s*(\d+)\s*\]")
        match = action_search_pattern.search(action)

        if match is None:
            # Invalid format
            reason=f"Invalid move format. Player {player_id} did not respond with a valid move."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            cell = int(match.group(1))

            # Ensure the cell is within 0–8
            if cell not in self.cell_mapping:
                reason = f"Invalid cell number: {cell}. Must be between 0 and 8."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                row, col = self.cell_mapping[cell]
                
                if self.state.game_state["board"][row][col] == '':
                    # Make the move
                    self.state.game_state["board"][row][col] = self.current_player

                    # Check for winner or draw
                    if self._check_winner():
                        self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} has won!")
                    elif all(cell != '' for row in self.state.game_state["board"] for cell in row):
                        self.state.set_draw(reason="The game is a draw!")
                else:
                    reason = f"Invalid move. Player {player_id} selected cell {cell} which is already occupied."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
        self._observer_current_state()
        return self.state.step()

    def _check_winner(self) -> bool:
        """ Check if a given 3×3 board has a winner """
        board = self.state.game_state["board"]
        # Rows and columns
        for i in range(3):
            # Check rows
            if (board[i][0] == board[i][1] == board[i][2] != '' or
                board[0][i] == board[1][i] == board[2][i] != ''):
                return True
            
        # Diagonals
        if (board[0][0] == board[1][1] == board[2][2] != '' or
            board[0][2] == board[1][1] == board[2][0] != ''):
            return True
        
        return False
 