import re, random
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.LiarsDice.renderer import create_board_str


class LiarsDiceEnv(ta.Env):
    """
    N-player version of Liar's Dice.
    Each player starts with some number of dice, and loses dice upon losing a round.
    The last player with dice is the winner.
    """

    def __init__(self, num_dice: int = 5):
        """
        Initialize the Liar's Dice game environment.

        Args:
            num_dice (int): Initial number of dice each player starts with.
        """
        self.initial_num_dice = num_dice

        # Regex patterns for parsing actions
        self.bid_pattern = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE)
        self.call_pattern = re.compile(r"\[call\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players:int, seed: Optional[int] = None):
        """ Reset the Liar's Dice game to its initial state """
        # Create the underlying State object
        self.state = ta.State(num_players=num_players, min_players=2, max_players=15)

        # Initialize dice for all players
        remaining_dice = {pid: self.initial_num_dice for pid in range(self.state.num_players)}
        dice_rolls = {
            pid: [random.randint(1, 6) for _ in range(self.initial_num_dice)]
            for pid in range(self.state.num_players)
        }

        # Keep track of the last bidder so we know who loses a die on a successful call
        # We'll store None when no bids have yet been made
        game_state = {
            "current_bid": {"quantity": 0, "face_value": 0},
            "last_bidder_id": None,
            "remaining_dice": remaining_dice,
            "dice_rolls": dice_rolls,
        }

        # Reset the State, providing a function to generate each player's prompt
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the prompt for a given player, showing their dice and the counts of others' dice """
        my_dice = game_state["dice_rolls"][player_id]
        # Build a listing of other players' dice counts
        others_info = []
        for pid in range(self.state.num_players):
            if pid != player_id:
                count = game_state["remaining_dice"][pid]
                others_info.append(f"Player {pid} has {count} dice.")

        others_text = "\n".join(others_info)

        current_quantity = game_state["current_bid"]["quantity"]
        current_face_value = game_state["current_bid"]["face_value"]

        prompt = (
            f"You are Player {player_id} in an N-player Liar's Dice game.\n"
            f"You have {len(my_dice)} dice: {', '.join(map(str, my_dice))}.\n"
            f"{others_text}\n\n"
            "Rules:\n"
            "- On your turn, you may either:\n"
            "  [1] Make a new bid with a higher quantity or higher face (or both),\n"
            "  [2] Call the last bid by typing '[Call]'.\n\n"
            "If you call:\n"
            "  - If the actual count of that face value among all dice is less than the bid, the last bidder loses one die.\n"
            "  - Otherwise, the caller loses one die.\n"
            "A player who reaches 0 dice is eliminated. The last remaining player wins.\n\n"
            f"Current bid: Quantity = {current_quantity}, Face Value = {current_face_value}\n"
            "Your action? (e.g. '[Bid: 3, 4]' or '[Call]')"
        )
        return prompt

    def _roll_new_dice(self):
        """
        Roll new dice for each player *who still has dice* after a round ends,
        and reset the current bid.
        """
        self.state.game_state["current_bid"] = {"quantity": 0, "face_value": 0}
        self.state.game_state["last_bidder_id"] = None

        # Roll new dice only for players still holding dice
        new_dice_rolls = {}
        for pid, count in self.state.game_state["remaining_dice"].items():
            new_dice_rolls[pid] = [random.randint(1, 6) for _ in range(count)]

        self.state.game_state["dice_rolls"] = new_dice_rolls

        # Send each player their new private dice
        for pid, rolled in new_dice_rolls.items():
            # message=(
            #     f"Your new dice are: {', '.join(map(str, rolled))}\n"
            #     f"Remaining dice:\n{'\n'.join([f'Player {p}: {d}' for p,d in self.state.game_state['remaining_dice'].items()])}"
            # )
            message = (
                f"Your new dice are: {', '.join(map(str, rolled))}\n"
                "Remaining dice:\n"
                + "\n".join([f"Player {p}: {d}" for p, d in self.state.game_state["remaining_dice"].items()])
            )


            self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=message, for_logging=False)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process one action from the current player """
        player_id = self.state.current_player_id

        # Log the action for the record
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, for_logging=True)

        # 1. Check if action is '[Call]'
        if self.call_pattern.search(action):
            current_bid = self.state.game_state["current_bid"]
            last_bidder_id = self.state.game_state["last_bidder_id"]

            if last_bidder_id is None or current_bid["quantity"] == 0:
                # No existing bid to call
                self.state.set_invalid_move(player_id=player_id, reason="Call made with no prior bid.")
                return self.state.step(rotate_player=False)

            quantity = current_bid["quantity"]
            face_value = current_bid["face_value"]
            # Count how many dice across all players match face_value
            total_face_count = 0
            for pid, dice_list in self.state.game_state["dice_rolls"].items():
                total_face_count += dice_list.count(face_value)

            # If the actual count is lower, last bidder was bluffing -> last bidder loses a die
            if total_face_count < quantity:
                loser_id = last_bidder_id
                msg = (
                    f"Player {player_id} calls! The actual count of face {face_value} is "
                    f"{total_face_count}, which is LESS than {quantity}.\n"
                    f"Player {loser_id} (the last bidder) loses one die."
                )
            else:
                # Otherwise, the caller loses a die
                loser_id = player_id
                msg = (
                    f"Player {player_id} calls! The actual count of face {face_value} is "
                    f"{total_face_count}, which is >= {quantity}.\n"
                    f"Player {loser_id} (the caller) loses one die."
                )

            self._apply_die_loss(loser_id, msg)
            self._rotate_players()
            return self.state.step(rotate_player=False)

        # 2. Otherwise, check if it is a valid '[Bid: X, Y]'
        bid_match = self.bid_pattern.search(action)
        if bid_match:
            new_quantity = int(bid_match.group(1))
            new_face_value = int(bid_match.group(2))

            current_bid = self.state.game_state["current_bid"]
            # Validate it is strictly higher
            is_valid, reason = self._is_valid_bid(new_quantity, new_face_value, current_bid)
            if is_valid:
                self.state.game_state["current_bid"] = {
                    "quantity": new_quantity,
                    "face_value": new_face_value
                }
                self.state.game_state["last_bidder_id"] = player_id
                message=f"Player {player_id} bids {new_quantity} of face {new_face_value}."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            else:
                self.state.set_invalid_move(player_id=player_id, reason=f"Invalid bid: {reason}")

            self._rotate_players()
            return self.state.step(rotate_player=False)

        # 3. If neither a valid call nor bid, it's invalid
        reason="Action not recognized as either a valid '[Bid: X, Y]' or '[Call]'."
        self.state.set_invalid_move(player_id=player_id, reason=reason)
        return self.state.step(rotate_player=False)


    def _rotate_players(self):
        """ rotate the current player """
        # determine the next player still playing
        current_player_id = self.state.current_player_id
        next_player_id = (current_player_id + 1) % self.state.num_players
        while next_player_id != current_player_id:
            if self.state.game_state["remaining_dice"][next_player_id] > 0:
                self.state.manually_update_current_player(new_player_id=next_player_id)
                break
            next_player_id = (next_player_id + 1) % self.state.num_players

        else:
            self.state.set_winners(player_ids=[current_player_id], reason=f"Player {current_player_id} wins! All other players ran out of dice.")


    def _apply_die_loss(self, loser_id: int, message: str):
        """
        Apply the effect of losing one die to a given loser_id, broadcast a message,
        then check if that player is out. Also check if there's only one player left with dice.
        If more than one remains, roll new dice and continue.
        """
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
        self.state.game_state["remaining_dice"][loser_id] -= 1

        # Check if the loser is out (0 dice)
        if self.state.game_state["remaining_dice"][loser_id] == 0:
            # Check how many players have dice > 0
            players_remaining = [
                pid for pid, count in self.state.game_state["remaining_dice"].items()
                if count > 0
            ]
            # If only one remains, that player is the winner
            if len(players_remaining) == 1:
                winner_id = players_remaining[0]
                reason=f"Player {winner_id} wins! All other players ran out of dice."
                self.state.set_winners(player_ids=[winner_id], reason=reason)
                return

        # If the game is not over yet, roll new dice for the next round
        # (all surviving players keep playing)
        self._roll_new_dice()

    def _is_valid_bid(
        self,
        new_quantity: int,
        new_face_value: int,
        current_bid: Dict[str, int]
    ) -> Tuple[bool, str]:
        """
        Check if the new bid is strictly higher than the current bid,
        and if face_value is between 1 and 6.
        """
        old_quantity = current_bid["quantity"]
        old_face_value = current_bid["face_value"]

        # Standard Liar's Dice rule: new bid must be "higher" in either quantity or face
        # You cannot lower either value, and the new bid can't be exactly the same.
        if new_quantity < old_quantity:
            return False, (
                f"New quantity {new_quantity} is lower than current {old_quantity}."
            )
        if new_face_value < old_face_value:
            return False, (
                f"New face value {new_face_value} is lower than current {old_face_value}."
            )
        if new_quantity == old_quantity and new_face_value == old_face_value:
            return False, "Bid is identical to the current bid."
        if not (1 <= new_face_value <= 6):
            return False, "Face value must be between 1 and 6."

        return True, ""
