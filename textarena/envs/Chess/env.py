import re, chess 
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.Chess.renderer import create_board_str


class ChessEnv(ta.Env):
    """Environment for playing the game of Chess."""

    def __init__(self, is_open: bool=True, max_turns: int=30, show_valid: bool=True):
        """
        Initialize the Chess game environment.
        Args:
            is_open (bool): If True, both players can see the current board state.
                            If False, players receive minimal information.
            max_turns (int): Maximum number of turns before the game ends.
            show_valid (bool): If True, players can see a list of valid moves.
        """
        self.max_turns = max_turns
        self.is_open = is_open 
        self.show_valid = show_valid 

        # Regex patterns
        self.move_pattern = re.compile(r"\[[a-h][1-8][a-h][1-8][qrbn]?\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the game to its initial state """
        # Initialize game state variables
        self.state = ta.State(
            num_players=num_players,
            min_players=2, 
            max_players=2,
            max_turns=self.max_turns,
            role_mapping={0: "White", 1: "Black"}
        )

        # Initialize the chess board
        self.board = chess.Board()

        game_state = {
            "board": self.board,
            "valid_moves": ', '.join([f'[{move.uci()}]' for move in self.board.legal_moves])
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player """
        color = "White" if player_id == 0 else "Black"
        prompt = (
            f"You are playing {color} in a game of Chess.\n"
            "Make your move in UCI format enclosed in square brackets (e.g., [e2e4]).\n"
            "You can include additional text in your messages.\n"
        )
        if self.is_open:
            prompt += f"Current board state:\n{self.board}\n"

        if player_id == 0:
            prompt += "Please make the first move."

        if self.show_valid:
            prompt += f"Valid moves: {game_state['valid_moves']}"

        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's move """
        # update the log
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # execute move
        self._execute_player_move(action=action)

        # check from game over
        self._check_gameover()

        # add valid moves / board state to observations as necessary
        self._agument_observations()

        # update the board state string
        self.state.game_state["board"] = self.board

        return self.state.step()

    def _execute_player_move(self, action: str):
        """Execute the player's move based on the action string."""
        player_id = self.state.current_player_id
        match = self.move_pattern.search(action.strip())
        
        # check if a move was provided
        if match is None:
            self.state.set_invalid_move(player_id=player_id, reason=f"Player {player_id} did not provide a move.")

        else:
            # Extract the move from within the brackets
            move_uci = match.group(0).lower().replace("[", "").replace("]", "")

            # Attempt to make the move
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                # execute move
                self.board.push(move)
                message=f"Player {player_id} made the following move: {move_uci}"
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

            else:
                # illegal move
                self.state.set_invalid_move(player_id=player_id, reason=f"Player {player_id} tried making an illegal move.")


    def _check_gameover(self):
        """Check if the game has ended and set the appropriate state."""
        if self.board.is_game_over():
            # get winner
            outcome = self.board.outcome().result()

            # check for draw
            if outcome == "1/2-1/2":
                self.state.set_draw(reason=f"Game ended in a draw.")

            else:
                winner_id = 0 if outcome == "1-0" else 1
                self.state.set_winners(player_ids=[winner_id], reason=f"Player {winner_id} wins the match.")

    def _agument_observations(self):
        """Augment observations with current board state and valid moves."""
        if self.is_open:
            # display the board state
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=str(self.board), for_logging=False)

        if self.show_valid:
            # show the valid moves
            message=f"Valid moves: {', '.join([f'[{move.uci()}]' for move in self.board.legal_moves])}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)