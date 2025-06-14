import re, chess 
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.games.Chess.renderer import create_board_str


class ChessEnv(ta.Env):
    def __init__(self, is_open: bool=True, max_turns: int=30, show_valid: bool=True):
        """
        Args:
            is_open (bool): If True, both players can see the current board state. If False, players receive minimal information.
            max_turns (int): Maximum number of turns before the game ends.
            show_valid (bool): If True, players can see a list of valid moves.
        """
        self.max_turns = max_turns
        self.is_open = is_open 
        self.show_valid = show_valid 
        self.action_space = lambda player_id: re.compile(r"\[[a-h][1-8][a-h][1-8][qrbn]?\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        board = chess.Board()
        valid_moves = ', '.join([f'[{move.uci()}]' for move in board.legal_moves])
        game_state = {"board": board, "valid_moves": valid_moves}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt, role_mapping={0:"White", 1:"Black"})
        self._agument_observations()

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return (
            f"You are playing {'White' if player_id==0 else 'Black'} in a game of Chess.\n"
            "Make your move in UCI format enclosed in square brackets (e.g., [e2e4]).\n"
        )
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        self._execute_player_move(action=action)
        self._check_gameover()
        self._agument_observations()
        return self.state.step()

    def _execute_player_move(self, action: str):
        """Execute the player's move based on the action string."""
        match = self.action_space(self.state.current_player_id).search(action.strip())
        
        if match is None: # check if a move was provided
            self.state.set_invalid_move(reason=f"Wrong move format.")

        else:
            move_uci = match.group(0).lower().replace("[", "").replace("]", "") # Extract the move from within the brackets
            move = chess.Move.from_uci(move_uci) # Attempt to make the move
            if move in self.state.game_state["board"].legal_moves:
                self.state.game_state["board"].push(move) # execute move
                self.state.add_observation(message=f"Player {self.state.current_player_id} made the following move: {move_uci}", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
            else: # illegal move
                self.state.set_invalid_move(reason=f"Illegal move.")

    def _check_gameover(self):
        """Check if the game has ended and set the appropriate state."""
        if self.state.game_state["board"].is_game_over():
            outcome = self.state.game_state["board"].outcome().result()
            if outcome == "1/2-1/2": # check for draw
                self.state.set_draw(reason=f"Game ended in a draw.")
            else:
                winner_id = 0 if outcome == "1-0" else 1
                self.state.set_winner(player_id=winner_id, reason=f"Player {winner_id} wins the match.")

    def _agument_observations(self):
        """Augment observations with current board state and valid moves."""
        message = ""
        if self.is_open: message+=f"Current board:\n{str(self.state.game_state["board"])}" # display the board state
        if self.show_valid: message+=f"\nValid moves: {', '.join([f'[{move.uci()}]' for move in self.state.game_state["board"].legal_moves])}"# show the valid moves
        self.state.add_observation(message=str(self.state.game_state["board"]), observation_type=ta.ObservationType.GAME_BOARD)



        # if self.is_open: prompt += f"Current board state:\n{game_state['board']}\n"
        # # if player_id == 0: prompt += "Please make the first move."
        # if self.show_valid: prompt += f"Valid moves: {game_state['valid_moves']}"