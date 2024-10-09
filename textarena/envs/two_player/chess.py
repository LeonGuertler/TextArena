import random
import re
from typing import Any, Dict, Optional, Tuple

import chess

import textarena as ta


class ChessEnv(ta.Env):
    """Environment for playing the game of Chess."""

    def __init__(
        self,
        is_open: bool = True,
        max_turns: int = 30,
        show_valid: bool = True,
    ):
        """
        Initialize the Chess game environment.
        Args:
            is_open (bool): If True, both players can see the current board state.
                            If False, players receive minimal information.
            max_turns (int): Maximum number of turns before the game ends.
            show_valid (bool): If True, players can see a list of valid moves.
        """
        self.environment_name = "Chess"

        # Initialize game state variables
        self.game_state = {
            "turn": 0,
            "is_open": is_open,
            "show_valid": show_valid,
            "max_turns": max_turns,
            "current_board": None,
            "logs": [],
            "render": ["turn", "max_turns", "current_board"],
        }

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[int, str], Dict[int, Any]]:
        """
        Reset the game to its initial state.
        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.
        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        self.game_state["turn"] = 0  # White starts
        self.game_state["logs"] = []

        # Initialize the chess board
        self.board = chess.Board()

        # Log the reset action
        self.game_state["logs"].append("[GAME] Game has been reset.")

        self.player_color = {0: "White", 1: "Black"}

        # Generate initial prompts for both players
        observations = {
            0: self._generate_player_prompt(0),
            1: self._generate_player_prompt(1),
        }

        info = {}

        return observations, info

    def _generate_player_prompt(self, player_id: int) -> str:
        """
        Generate the initial prompt for a player.
        Args:
            player_id (int): ID of the player (0 for White, 1 for Black).
        Returns:
            str: The initial prompt for the player.
        """
        color = "White" if player_id == 0 else "Black"
        prompt = (
            f"You are playing {color} in a game of Chess.\n"
            "Make your move in UCI format enclosed in square brackets (e.g., [e2e4]).\n"
        )
        if self.game_state["is_open"]:
            prompt += f"Current board state:\n{self.board}\n"

        prompt += "You can also include additional text in your messages.\n"

        return prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        ta.Observation,  # observations
        ta.Reward,  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's move.
        Args:
            player_id (int): The player's ID (0 for White, 1 for Black).
            action (str): The move in UCI notation enclosed in square brackets (e.g., [e2e4]).
        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        terminated = False
        truncated = False
        reward = {0: 0, 1: 0}
        info = {}

        # Log the player's action
        self.game_state["logs"].append((-1, f"Player {player_id}: {action}"))

        # Validate the move format using regex
        # Move must be enclosed in square brackets: [e2e4]
        move_pattern = re.compile(r"\[[a-h][1-8][a-h][1-8][qrbn]?\]", re.IGNORECASE)
        match = move_pattern.search(action.strip())
        if not match:
            info = {
                "reason": "Invalid move format. Moves must be enclosed in square brackets (e.g., [e2e4])."
            }
            reward[player_id] = -1  # Penalize for invalid format
            terminated = True
            self.game_state["logs"].append(
                (
                    -1,
                    f"[GAME] Player {player_id} provided an invalid move format. Game over.",
                )
            )
            return None, reward, truncated, terminated, info

        # Extract the move from within the brackets
        move_uci = match.group(0).lower().replace("[", "").replace("]", "")

        # Attempt to make the move
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)

                # Check for game over conditions
                if self.board.is_game_over():
                    result = self.board.result()
                    input(result)
                    if result == "1-0":
                        reward = {0: 1, 1: -1}
                        info = {"result": "White wins by checkmate."}
                        self.game_state["logs"].append(
                            (-1, "[GAME] White wins by checkmate.")
                        )
                    elif result == "0-1":
                        reward = {0: -1, 1: 1}
                        info = {"result": "Black wins by checkmate."}
                        self.game_state["logs"].append(
                            (-1, "[GAME] Black wins by checkmate.")
                        )
                    else:
                        # Handle draws
                        reward = {0: 0, 1: 0}
                        info = {"result": "The game is a draw."}
                        self.game_state["logs"].append(
                            (-1, "[GAME] The game ended in a draw.")
                        )
                    terminated = True

                elif self.board.fullmove_number > self.game_state["max_turns"]:
                    # Draw by turn limit
                    reward = {0: 0, 1: 0}
                    info = {"reason": "Turn limit reached. The game is a draw."}
                    self.game_state["logs"].append(
                        (-1, "[GAME] Turn limit reached. The game is a draw.")
                    )
                    truncated = True
                else:
                    # Game continues
                    self.game_state["turn"] += 1
                    info = {"info": "Move accepted."}
                    self.game_state["logs"].append(
                        (-1, f"[GAME] Player {player_id} made a move: [{move_uci}].")
                    )

            else:
                # Move is illegal
                info = {"reason": "Illegal move."}
                reward[player_id] = -1  # Penalize for illegal move
                terminated = True
                self.game_state["logs"].append(
                    (
                        -1,
                        f"[GAME] Player {player_id} attempted an illegal move: [{move_uci}]. Game over.",
                    )
                )

        except ValueError:
            # Move parsing failed
            info = {"reason": "Invalid move format."}
            reward[player_id] = -1  # Penalize for invalid format
            terminated = True
            self.game_state["logs"].append(
                (
                    -1,
                    f"[GAME] Player {player_id} provided an invalid move format. Game over.",
                )
            )

        # Prepare observations
        observations = self._get_observations(player_id=player_id, action=action)

        self.game_state["current_board"] = str(self.board)

        # update turn counter
        self.game_state["turn"] = self.board.fullmove_number

        return observations, reward, truncated, terminated, info

    def _get_observations(self, player_id: int, action: str) -> ta.Observation:
        """
        Generate observations for both players based on the game state.
        Args:
            active_player_id (int): The ID of the player who just made a move.
            action (str): The current player action.
        Returns:
            Dict[int, str]: Observations for each player.
        """
        observations = {}

        for player_id_local in [0, 1]:  # only do detail for oponent.
            player_observation = f"[{self.player_color[player_id]}] {action}\n"
            if self.game_state["is_open"]:
                # Provide the full board state
                player_observation += f"Current board state:\n{self.board}\n"

            if self.game_state["show_valid"]:
                # Add the valid actions
                valid_moves = [f"[{move.uci()}]" for move in self.board.legal_moves]
                player_observation += f"Valid moves: {', '.join(valid_moves)}.\n"

            observations[player_id_local] = [(-1, player_observation)]
        return observations

    def render(self):
        """
        Render the current game state.
        This method can be called externally to display the game state.
        """
        print(f"Turn {self.game_state['turn']}/{self.game_state['max_turns']}")
        print("Current Board State:")
        print(self.board)
        print("\nAction Logs:")
        for player_id, log in self.game_state["logs"]:
            if player_id == -1:
                print(f"[GAME] {log}")
            else:
                print(f"Player {player_id}: {log}")
        print("\n")
