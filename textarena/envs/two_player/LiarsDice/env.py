import random
import re
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta


class LiarsDiceEnv(ta.Env):
    """
    Environment for the Liar's Dice Game where players lose dice upon losing a round.
    """

    def __init__(self, num_dice: Optional[int] = 5):
        """
        Initialize the Liar's Dice game environment.

        Args:
            num_dice (int): Initial number of dice each player starts with.
        """
        self.initial_num_dice = num_dice

        # Initialize game state variables
        self.state = ta.State(num_players=2, max_turns=None)

        # Updated regex pattern for parsing actions
        self.bid_pattern = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
        self.call_pattern = re.compile(r"\[call\]", re.IGNORECASE)

    @property
    def offline_renderer(self):
        raise NotImplementedError
        # from textarena.envs.two_player.LiarsDice.render.renderer import LiarsDiceRenderer
        # return LiarsDiceRenderer 

    @property
    def terminal_render_keys(self):
        return [
                ["dice_rolls", 0],
                ["dice_rolls", 1],
                ["remaining_dice", 0],
                ["remaining_dice", 1],
            ]

    def reset(self, seed: Optional[int]=None):
        """
        Reset the Liar's Dice game to its initial state.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            Tuple[Dict[int, List[Tuple[int, str]]], Dict[str, Any]]:
                - Observations for each player
                - Additional information including dice rolls
        """
        if seed is not None:
            random.seed(seed)

        # Initialize game state with remaining dice counts
        game_state = {
            "current_bid": {"quantity": 0, "face_value": 0},
            "remaining_dice": {
                0: self.initial_num_dice,
                1: self.initial_num_dice
            },
            "dice_rolls": {
                0: [random.randint(1, 6) for _ in range(self.initial_num_dice)],
                1: [random.randint(1, 6) for _ in range(self.initial_num_dice)],
            },
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the prompt for a player.

        Args:
            player_id (int): The player's ID.
            game_state (Dict[int, Any]): Current game state.

        Returns:
            str: The prompt for the player.
        """
        dice = game_state["dice_rolls"][player_id]
        opponent_dice_count = game_state["remaining_dice"][1 - player_id]
        
        prompt = (
            f"You are Player {player_id} in Liar's Dice.\n"
            f"You have {len(dice)} dice: {', '.join(map(str, dice))}.\n"
            f"Your opponent has {opponent_dice_count} dice.\n"
            "Players take turns making bids on the total quantity of a face value among all dice.\n"
            "On your turn, you can either make a higher bid or call the opponent's bluff.\n"
            "A higher bid must increase either the quantity or the face value (or both).\n"
            "You cannot decrease either value.\n"
            "Actions:\n"
            "- To make a bid: '[Bid: <quantity>, <face_value>]', e.g., '[Bid: 3, 4]'\n"
            "- To call a bluff: '[Call]'\n"
            "If you call a bluff:\n"
            "- If the actual quantity is less than the bid, you win and your opponent loses one die.\n"
            "- If the actual quantity meets or exceeds the bid, you lose and lose one die.\n"
            "The first player to lose all dice loses the game."
            f"The current bid is: Quantity = {game_state['current_bid']['quantity']}, "
            f"Face Value = {game_state['current_bid']['face_value']}\n"
            "It's your turn. What is your action?"
        )
        return prompt


    def _roll_new_dice(self):
        """Roll new dice for both players after a round ends."""
        # Reset bid state
        self.state.game_state["current_bid"] = {"quantity": 0, "face_value": 0}
        
        # Roll new dice
        self.state.game_state["dice_rolls"] = {
            player_id: [random.randint(1, 6) for _ in range(remaining)]
            for player_id, remaining in self.state.game_state["remaining_dice"].items()
        }

        # Send private dice roll information to each player
        for player_id, rolled_dice in self.state.game_state["dice_rolls"].items():
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=f"Your new dice are: {', '.join(map(str, rolled_dice))}",
                for_logging=False
            )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Take a step in the environment.

        Args:
            action: The action to take
        
        Returns:
            done: Whether the game has concluded.
            info: Additional information.
        """
        player_id = self.state.current_player_id

        # Log the action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        # First check if it's a call
        if self.call_pattern.search(action):
            if self.state.game_state["current_bid"] == {"quantity": 0, "face_value": 0}:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Player {player_id} tried to call without a bid having been made."
                )
                return self.state.step()

            current_bid = self.state.game_state["current_bid"]
            total_quantity = sum(
                dice.count(current_bid["face_value"])
                for dice in self.state.game_state["dice_rolls"].values()
            )
            bid_quantity = current_bid["quantity"]

            # Determine round winner and update dice counts
            if total_quantity < bid_quantity:
                # Challenger wins, other player loses a die
                loser_id = 1 - player_id
                self.state.game_state["remaining_dice"][loser_id] -= 1
                message = (
                    f"Bluff called by Player {player_id}. "
                    f"Actual quantity of face value {current_bid['face_value']} is {total_quantity}, "
                    f"which is less than the bid ({bid_quantity}). "
                    f"Player {1-player_id} loses one die."
                )
            else:
                # Challenger loses a die
                loser_id = player_id
                self.state.game_state["remaining_dice"][player_id] -= 1
                message = (
                    f"Bluff called by Player {player_id}. "
                    f"Actual quantity of face value {current_bid['face_value']} is {total_quantity}, "
                    f"which meets or exceeds the bid ({bid_quantity}). "
                    f"Player {player_id} loses one die."
                )

            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message,
                for_logging=True
            )

            # Check if game is over
            if self.state.game_state["remaining_dice"][loser_id] == 0:
                winner_id = 1 - loser_id
                self.state.set_winners(
                    player_ids=[winner_id],
                    reason=f"Player {winner_id} wins the game! Player {loser_id} has lost all their dice."
                )
            else:
                self._roll_new_dice()

            return self.state.step()

        # If it's not a call, check if it's a bid
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
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Player {player_id} increases the bid to Quantity = {new_quantity}, Face Value = {new_face_value}",
                    for_logging=True
                )
            else:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid bid by Player {player_id}: Quantity = {new_quantity}, Face Value = {new_face_value}. Reason: {reason}"
                )
            return self.state.step()

        # If neither call nor bid, invalid move
        self.state.set_invalid_move(
            player_id=player_id,
            reason=f"Player {player_id} did neither bid nor call."
        )
        return self.state.step()

    def _is_valid_bid(self, new_quantity: int, new_face_value: int, current_bid: Dict[str, int]) -> Tuple[bool, str]:
        """
        Validate whether the new bid is higher than the current bid.

        Args:
            new_quantity (int): Quantity in the new bid.
            new_face_value (int): Face value in the new bid.
            current_bid (Dict[str, int]): Current bid details.

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if new_quantity < current_bid["quantity"]:
            return False, f"The quantity was reduced from {current_bid['quantity']} to {new_quantity}."
        if new_face_value < current_bid["face_value"]:
            return False, f"The face value was reduced from {current_bid['face_value']} to {new_face_value}."
        if new_quantity == current_bid["quantity"] and new_face_value == current_bid["face_value"]:
            return False, f"The same bid was submitted twice. Either the quantity or face value must be increased."
        if not (1 <= new_face_value <= 6):
            return False, f"The face values may not be smaller than 1 or bigger than 6. Submitted face_value {new_face_value}."
        return True, ""