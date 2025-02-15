import random
import re
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta


class LiarsDiceEnv(ta.Env):
    """ Environment for the Liar's Dice Game. """

    def __init__(self, num_dice: Optional[int] = 5):
        """
        Initialize the Liar's Dice game environment.

        Args:
            num_dice (int): Number of dice each player rolls.
        """
        self.num_dice = num_dice

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=None,  # Game ends when a bluff is called
        )

        # Updated regex pattern for parsing actions with new bid format
        self.bid_pattern = re.compile(
            r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]",
            re.IGNORECASE
        )
        self.call_pattern = re.compile(r"\[call\]", re.IGNORECASE)


    @property
    def offline_renderer(self):
        from textarena.envs.two_player.LiarsDice.render.renderer import LiarsDiceRenderer
        return LiarsDiceRenderer 

    @property
    def terminal_render_keys(self):
        return [
                ["dice_rolls", 0],
                ["dice_rolls", 1],
            ]


    def reset(self, seed: Optional[int]=None):
        """
        Reset the Liar's Dice game to its initial state.

        Args:
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.

        Returns:
            Tuple[Dict[int, List[Tuple[int, str]]], Dict[str, Any]]:
                - Observations for each player as a dictionary.
                - Additional information, including all dice rolls.
        """
        if seed is not None:
            random.seed(seed)

        # Each player rolls their dice
        game_state = {
            "current_bid": {"quantity": 0, "face_value": 0},
            "dice_rolls": {
                0: [random.randint(1, 6) for _ in range(self.num_dice)],
                1: [random.randint(1, 6) for _ in range(self.num_dice)],
            },
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the initial prompt for a player.

        Args:
            player_id (int): The player's ID.

        Returns:
            str: The initial prompt for the player.
        """
        dice = game_state["dice_rolls"][player_id]
        prompt = (
            f"You are Player {player_id} in Liar's Dice.\n"
            f"You have rolled {self.num_dice} dice: {', '.join(map(str, dice))}.\n"
            "Players take turns making bids on the total quantity of a face value among all dice.\n"
            "On your turn, you can either make a higher bid or call the opponent's bluff.\n"
            "A higher bid must increase either the quantity or the face value (or both)."
            "You cannot decrease either value.\n"
            "Actions:\n"
            "- To make a bid: '[Bid: <quantity>, <face_value>]', e.g., '[Bid: 3, 4]'\n"
            "- To call a bluff: '[Call]'\n"
            "If you call a bluff, all dice are revealed:\n"
            "- If the actual quantity of the face value is less than the bid, you win.\n"
            "- If the actual quantity meets or exceeds the bid, you lose.\n"
            f"The current bid is: Quantity = 0, Face Value = 0\n"
            "It's your turn. What is your action?"
        )

        return prompt


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Take a step in the environment.

        Args:
            action: The action to take
        
        Returns:
            done: Whether the game has concluded.
            info: Additional information.
        """
        # get the  current player id
        player_id = self.state.current_player_id

        # Update the observations and log the action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1, # Broadcast to all
            message=action,
            for_logging=True
        )

        # Check if the player is making a bid
        bid_match = self.bid_pattern.search(action)
        if bid_match:
            new_quantity = int(bid_match.group(1))
            new_face_value = int(bid_match.group(2))

            # Validate the bid
            current_bid = self.state.game_state["current_bid"]
            is_valid, reason = self._is_valid_bid(new_quantity, new_face_value, current_bid)
            if is_valid:
                # Update the current bid
                self.state.game_state["current_bid"] = {
                    "quantity": new_quantity,
                    "face_value": new_face_value
                }
                # Log the bid update
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,  # Broadcast to all
                    message=f"Player {player_id} increases the bid to Quantity = {new_quantity}, Face Value = {new_face_value}",
                    for_logging=True
                )

            else:
                # Invalid bid
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid bid by Player {player_id}: Quantity = {new_quantity}, Face Value = {new_face_value}. Reason: {reason}"
                )

        elif self.call_pattern.search(action):
            if self.state.turn == 0:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Player {player_id} tried to call without a bid having been made by anybody."
                )
            else:
                current_bid = self.state.game_state["current_bid"]
                total_quantity = sum(
                    dice.count(current_bid["face_value"])
                    for dice in self.state.game_state["dice_rolls"].values()
                )
                bid_quantity = current_bid["quantity"]

                if total_quantity < bid_quantity:
                    # Challenger wins
                    self.state.set_winners(
                        player_ids=[player_id],
                        reason=(
                            f"Bluff called by Player {player_id}. "
                            f"Actual quantity of face value {current_bid['face_value']} is {total_quantity}, "
                            f"which is less than the bid ({bid_quantity}). "
                            f"Player {player_id} wins."
                        )
                    )
                else:
                    # Challenger loses
                    self.state.set_winners(
                        player_ids=[1-player_id],
                        reason=(
                            f"Bluff called by Player {player_id}. "
                            f"Actual quantity of face value {current_bid['face_value']} is {total_quantity}, "
                            f"which meets or exceeds the bid ({bid_quantity}). "
                            f"Player {player_id} loses."
                        )
                    )
        else:
            # check if a call was made in the very first turn
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Player {player_id} did neither bid nor call."
            )

        return self.state.step()

    def _is_valid_bid(self, new_quantity: int, new_face_value: int, current_bid: Dict[str, int]) -> bool:
        """
        Validate whether the new bid is higher than the current bid.

        Args:
            new_quantity (int): Quantity in the new bid.
            new_face_value (int): Face value in the new bid.
            current_bid (Dict[str, int]): Current bid details.

        Returns:
            bool: True if the bid is valid, False otherwise.
        """
        if new_quantity < current_bid["quantity"]:
            return False, f"The quantity was reduced from {current_bid['quantity']} to {new_quantity}."
        if new_face_value < current_bid["face_value"]:
            return False, f"The face value was reduced from {current_bid['face_value']} to {new_face_value}."
        if new_quantity == current_bid["quantity"] and new_face_value == current_bid["face_value"]:
            return False, f"The same bid was submitted twice. Either the quantity or face value mus be increased."
        if not (1 <= new_face_value <= 6):
            return False, f"The face values may not be smaller than 1 or bigger than 6. Submitted face_value {new_face_value}."
        return True, ""

