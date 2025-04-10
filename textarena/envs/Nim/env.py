import re
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.Nim.renderer import create_board_str

class NimEnv(ta.Env):
    """
    A basic two-player Nim environment.

    Rules (Normal Play):
      - There are several piles (or heaps) of objects.
      - Two players alternate turns.
      - On your turn, you must remove at least one object from exactly one pile.
      - You may remove as many objects as you wish from that single pile.
      - The player who takes the last object(s) is the winner.
    """

    def __init__(self, piles: List[int] = None):
        """
        Initialize the Nim environment.

        Args:
            piles (List[int]): Initial sizes of the piles (e.g. [3, 5, 7]).
                If None, defaults to [3,4,5].
        """
        super().__init__()
        # Default to [3,4,5] if no pile sizes are given
        self.initial_piles = piles if piles is not None else [3, 4, 5]

        # We'll look for actions in the format [pile_index quantity_to_remove], e.g. [1 3].
        # This means remove 3 objects from pile #1.
        self.action_pattern = re.compile(r"\[\s*(\d+)\s+(\d+)\s*\]")

    def get_board_str(self):
        return create_board_str(self.piles)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment with the given number of players """
        # Create State (2 players only)
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2)

        # Piles is just a list of integers
        self.piles = self.initial_piles.copy()

        # Build the initial game_state dictionary
        game_state = {"piles": self.piles, "rendered_piles": self._render_piles()}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._intro_prompt)


    def _intro_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ A short, one-time introduction for each player, shown only at game start """
        return (
            f"Welcome to Nim, Player {player_id}!\n\n"
            "Rules:\n"
            "- On your turn, remove at least one object from exactly one pile.\n"
            "- Remove objects with the format [pile quantity], e.g. [0 3].\n"
            "- Whoever takes the last object(s) wins!\n\n"
            "Good luck!"
        ) + f"\nCurrent Pile:\n{self._render_piles()}"

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process a single move in the game """
        player_id = self.state.current_player_id

        # Log the player's action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Execute the move (or mark invalid if the format is incorrect/illegal)
        self._execute_move(action)

        # Update the "rendered_piles" in the game state
        self.state.game_state["rendered_piles"] = self._render_piles()

        # After the current player moves, send the updated board to the opponent.
        message="Updated piles:\n" + self._render_piles()
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

        # Check if the game is over
        self._check_game_over()

        # Proceed to the next turn (or finalize if done)
        return self.state.step()

    def _execute_move(self, action: str) -> None:
        """
        Parse the move action and update the piles accordingly.
        """
        player_id = self.state.current_player_id
        match = self.action_pattern.search(action.strip())

        if not match:
            self.state.set_invalid_move(player_id, "No valid move format found. Use [pile quantity].")
            return

        # Extract pile index and quantity to remove
        try:
            pile_index, quantity = map(int, match.groups())
        except ValueError:
            self.state.set_invalid_move(player_id, "Action must be two integers: [pile quantity].")
            return

        # Validate the move
        if not (0 <= pile_index < len(self.piles)):
            self.state.set_invalid_move(player_id, f"Pile index {pile_index} is out of range.")
            return

        if quantity <= 0:
            self.state.set_invalid_move(player_id, "Must remove at least 1 object.")
            return

        if self.piles[pile_index] < quantity:
            self.state.set_invalid_move(player_id,
                f"Cannot remove {quantity} from pile {pile_index} (only {self.piles[pile_index]} left).")
            return

        # Perform the removal
        self.piles[pile_index] -= quantity

        # Announce the move
        message=f"Player {player_id} removes {quantity} from pile {pile_index}."
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

    def _check_game_over(self) -> None:
        """
        Check if the game is over (all piles empty -> current player took last object).
        If all piles are empty, the current player is the winner.
        """
        # If all piles are empty, the current player just took the last object(s)
        if all(pile == 0 for pile in self.piles):
            winner = self.state.current_player_id
            self.state.set_winners([winner], reason=f"Player {winner} took the last object(s)!")

    def _render_piles(self) -> str:
        """
        Render the piles in a simple textual form like:
         pile 0: 3
         pile 1: 4
         pile 2: 5
        """
        lines = []
        for i, amt in enumerate(self.piles):
            lines.append(f"  pile {i}: {amt}")
        return "\n".join(lines)