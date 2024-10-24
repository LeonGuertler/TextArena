"""
Liar's Dice Game

In this game, two players each roll a set number of dice (usually five), keeping their dice hidden from the other player.

**Gameplay:**

- Each player rolls their dice and keeps them hidden.
- Players take turns making bids on the total quantity of a face value (from 1 to 6) among all dice.
  - A bid consists of a quantity and a face value, e.g., "I bid three 4s", meaning there are at least three dice showing a face value of 4.
- On their turn, a player can:
  - **Make a higher bid**: Increase the quantity or the face value (if increasing the face value, the quantity must be at least the same).
  - **Call the bluff**: If a player believes the previous bid is false, they can challenge it by saying "Call".
- If a bluff is called:
  - All dice are revealed.
  - The actual counts are tallied.
  - If the bid is valid (there are at least as many of the face value as claimed), the bidder wins.
  - If the bid is invalid, the challenger wins.
- The game ends when a bluff is called and resolved.

**Key Rules:**

- Bids must increase in quantity or face value.
  - You can increase the quantity while keeping the face value the same or higher.
  - If you increase the face value, the quantity must be equal to or higher than the previous quantity.
- Players must use the correct action formats:
  - To make a bid: "[Bid] <quantity> <face_value>"
  - To call a bluff: "[Call]"
- Face values range from 1 to 6.

**Parameters:**

- `num_dice`: Number of dice each player rolls at the beginning (default is 5).

**Game Outcomes:**

- A player wins by successfully calling a bluff or by making a bid that the opponent believes and cannot challenge.
- A player loses if their bluff is called and the actual counts do not support their bid.
"""

import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class LiarsDiceEnv(ta.Env):
    def __init__(
        self,
        num_dice: Optional[int] = 5,
    ):
        """
        Initialize the Liar's Dice game environment.

        Args:
            num_dice (int): Number of dice each player rolls.
        """
        self.environment_name = "Liar's Dice"
        self.num_dice = num_dice

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=None,
            render_keys=[
                "turn",
                ["current_bid", "quantity"],
                ["current_bid", "face_value"],
                ["dice_rolls", 0],
                ["dice_rolls", 1],
            ],
        )


    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[ta.Observation], ta.Reward]:
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


        # reset the state
        game_state = {
            "current_bid": {"quantity": 0, "face_value": 0},
            "dice_rolls": {
                0: [random.randint(1, 6) for _ in range(self.num_dice)],
                1: [random.randint(1, 6) for _ in range(self.num_dice)],
            }
        }

        # Generate initial prompts for both players
        observations = {
            0: [(ta.GAME_ID, self._generate_player_prompt(player_id=0, game_state=game_state))],
            1: [(ta.GAME_ID, self._generate_player_prompt(player_id=1, game_state=game_state))],
        }

        info = {
            "dice_rolls": game_state["dice_rolls"],
        }

        self.state.reset(
            game_state=game_state,
            initial_logs=[(ta.GAME_ID, "Game started")]
        )

        return observations, info


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> ta.Message:
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
            "Actions:\n"
            "- To make a bid: '[Bid] <quantity> <face_value>', e.g., '[Bid] 3 4'\n"
            "- To call a bluff: '[Call]'\n"
            "If you call a bluff, all dice are revealed:\n"
            "- If the actual quantity of the face value is less than the bid, you win.\n"
            "- If the actual quantity meets or exceeds the bid, you lose.\n"
            f"The current bid is: Quantity = {game_state['current_bid']['quantity']}, Face Value = {game_state['current_bid']['face_value']}\n"
            "It's your turn. What is your action?"
        )
        return prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[Dict[int, str]],  # observations
        Optional[Dict[int, int]],  # reward
        bool,  # truncated
        bool,  # terminated
        Dict[str, Any],  # info
    ]:
        """
        Process the player's action.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's action.

        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        assert isinstance(
            action, str
        ), f"Actions are required to be strings. Received dtype: {type(action)}"

        assert (
            player_id == self.state.current_player
        ), f"The passed player_id is not as expected. Player id received: {player_id}; Expected: {self.state.current_player}"

        terminated, truncated = False, False
        self.step_logs = []
        observations = {0: [], 1: []}
        info = {}
        reward = None


        # update step logs
        self.step_logs.append((player_id, action))

        observations[player_id].append((player_id, action))
        observations[1-player_id].append((player_id, action))


        action_lower = action.strip().lower()

        # Check if the player is calling a bluff
        if "[call]" in action_lower:
            total_quantity = sum(
                dice.count(self.state.game_state["current_bid"]["face_value"])
                for dice in self.state.game_state["dice_rolls"].values()
            )
            bid_quantity = self.state.game_state["current_bid"]["quantity"]
            if total_quantity < bid_quantity:
                # Challenger wins
                info["reason"] = (
                    f"Actual quantity of face value {self.state.game_state['current_bid']['face_value']} "
                    f"is {total_quantity}, less than the bid ({bid_quantity}). "
                    f"Player {player_id} wins."
                )
                reward = {player_id: 1, 1-player_id: -1}
            else:
                # Challenger loses
                reward = {player_id: -1, 1-player_id: 1}
                info["reason"] = (
                    f"Actual quantity of face value {self.state.game_state['current_bid']['face_value']} "
                    f"is {total_quantity}, meets or exceeds the bid ({bid_quantity}). "
                    f"Player {player_id} loses."
                )
            terminated = True 

        # Check if the player is making a bid
        elif "[bid]" in action_lower:
            # Parse the new bid
            match = re.search(r"\[bid\]\s*(\d+)\s+(\d+)", action_lower)
            if not match:
                # Invalid bid format
                reward = {player_id: -1, 1-player_id: 0}
                info["reason"] = f"Invalid bid format. Player {player_id} did not provide quantity and face value."
                terminated = True 
            else:
                new_quantity = int(match.group(1))
                new_face_value = int(match.group(2))

                # Validate face value
                if not (1 <= new_face_value <= 6):
                    reward = {player_id: -1, 1-player_id: 0}
                    info["reason"] = f"Invalid face value. Player {player_id} provided face value {new_face_value}, which is not between 1 and 6."
                    terminated = True 
                else:
                    # Validate bid increase
                    current_quantity = self.state.game_state["current_bid"]["quantity"]
                    current_face_value = self.state.game_state["current_bid"]["face_value"]
                    if (new_quantity < current_quantity) or (
                        new_quantity == current_quantity
                        and new_face_value <= current_face_value
                    ):
                        terminated = True
                        reward = {player_id: -1, 1-player_id: 0}
                        info["reason"] = "Invalid bid. New bid must have a higher quantity or higher face value."
                    else:
                        # Update the current bid
                        prev_bid = self.state.game_state["current_bid"].copy()
                        self.state.game_state["current_bid"]["quantity"] = new_quantity
                        self.state.game_state["current_bid"]["face_value"] = new_face_value
                        updated_bid_message = (
                            f"Player {player_id} increases the bid to Quantity = {new_quantity}, Face Value = {new_face_value} "
                            f"(previous bid was Quantity = {prev_bid['quantity']}, Face Value = {prev_bid['face_value']})."
                        )
                        self.step_logs.append((ta.GAME_ID, updated_bid_message))
                        observations[player_id].append((ta.GAME_ID, updated_bid_message))
                        observations[1-player_id].append((ta.GAME_ID, updated_bid_message))

        else:
            # Invalid action
            terminated = True 
            reward = {player_id: -1, 1-player_id: 0}
            info["reason"] = f"Invalid action. Player {player_id} must use '[Bid] <quantity> <face_value>' or '[Call]'."



        if "reason" in info:
            self.step_logs.append((ta.GAME_ID, info["reason"]))

        self.state.step(
            logging_messages=self.step_logs,
            game_state_updates=None
        )
        return observations, reward, truncated, terminated, info

    def render(self):
        """
        Render the current game state.
        """
        print(f"Turn: {self.state.turn}")
        current_bid = self.state.game_state["current_bid"]
        print(
            f"Current Bid: Quantity = {current_bid['quantity']}, Face Value = {current_bid['face_value']}"
        )
        print("\nGame Logs:")
        for i, log in self.state.logs:
            if i == -1:
                print(f"[GAME]: {log}")
            else:
                print(f"Player {i}: {log}")
        print("\n")
