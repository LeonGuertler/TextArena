import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class NegotiationEnv(ta.Env):
    def __init__(
        self,
        max_turns: Optional[int] = 10,
    ):
        """
        Initialize the Negotiation Game environment.
        Args:
            max_turns (Optional[int]): Maximum number of turns before the game ends.
        """
        self.environment_name = "Negotiation"

        self.resource_names = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
        self.base_values = {"Wheat": 5, "Wood": 10, "Sheep": 15, "Brick": 25, "Ore": 40}

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=max_turns,
            render_keys = ["inventory_value"]
        )



        # Regex patterns
        self.accept_pattern = re.compile(r"\[Accept\]", re.IGNORECASE)
        self.deny_pattern = re.compile(r"\[Deny\]", re.IGNORECASE)
        self.offer_pattern = re.compile(r"\[Offer\](.*?)[\.]", re.IGNORECASE | re.DOTALL)

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[ta.Observation], ta.Info]:
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

        game_state = {
            "num_trades": 0,
            "current_offer": None,
            "player_resources": {
                0: {resource: random.randint(5, 25) for resource in self.resource_names},
                1: {resource: random.randint(5, 25) for resource in self.resource_names},
            },
            "player_values": {}
        }


        # Generate player-specific values for each resource type (±20% of base value, capped at 5 and 40)
        for player_id in [0, 1]:
            game_state["player_values"][player_id] = {}
            for resource in self.resource_names:
                base_value = self.base_values[resource]
                variation = int(0.2 * base_value)
                min_value = max(base_value - variation, 5)
                max_value = min(base_value + variation, 40)
                value = random.randint(min_value, max_value)
                game_state["player_values"][player_id][resource] = value

        # Keep track of the inventory value (both initial and current)
        game_state["inventory_value"] = {
            0: {
                "initial": self._calculate_player_inventory_value(player_id=0, game_state=game_state),
                "current": self._calculate_player_inventory_value(player_id=0, game_state=game_state),
                "change": 0,
            },
            1: {
                "initial": self._calculate_player_inventory_value(player_id=1, game_state=game_state),
                "current": self._calculate_player_inventory_value(player_id=1, game_state=game_state),
                "change": 0,
            },
        }

        # Generate initial prompts for both players
        observations = {
            0: [(ta.GAME_ID, self._generate_player_prompt(player_id=0, game_state=game_state))],
            1: [(ta.GAME_ID, self._generate_player_prompt(player_id=1, game_state=game_state))],
        }

        info = {
            "player_0_values": game_state["player_values"][0],
            "player_1_values": game_state["player_values"][1],
        }

        self.state.reset(
            game_state=game_state,
            initial_logs=[(ta.GAME_ID, "Game started.")]
        )

        return observations, info

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> ta.Message:
        """
        Generate the initial prompt for a player.
        Args:
            player_id (int): ID of the player (0 or 1).
        Returns:
            str: The initial prompt for the player.
        """
        resources = game_state["player_resources"][player_id]
        resource_values = game_state["player_values"][player_id]
        resource_value_list = "; ".join(
            [
                f"{resources[res]} {res} (Value of each: {resource_values[res]})"
                for res in resources.keys()
            ]
        )
        prompt = (
            f"You are Player {player_id} in the Negotiation Game.\n"
            "You have some resources, and your task is to trade such that the total value of your resources increases.\n"
            f"The resources and associated values you currently have are: {resource_value_list}.\n"
            "At each turn, you can talk to your opponent or make an explicit trade offer.\n"
            "Use the following special tokens for actions:\n"
            "  - [Offer]: To make a trade offer.\n"
            "    Format: [Offer] I give [your resources]; You give [their resources].\n"
            "    Example: [Offer] I give 2 Wheat, 1 Ore; You give 3 Sheep.\n"
            "  - [Accept]: To accept an incoming offer.\n"
            "  - [Deny]: To deny an incoming offer.\n"
            "You can include additional text before or after these tokens.\n"
            "If responding to an offer, ensure your reply contains [Accept] or [Deny] as appropriate.\n"
        )
        if self.state.max_turns:
            prompt += (
                f"The game lasts for {self.state.max_turns} turns in total.\n"
            )
        return prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observation],  # observations
        Optional[ta.Reward],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action.
        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's message or action.
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
        reward = None
        info = {}


        # update step logs
        self.step_logs.append((player_id, action))

        observations[player_id].append((player_id, action))
        observations[1-player_id].append((player_id, action))

        

        # check if Accept/Deny is present and a previous offer was made
        if (
            self.state.game_state["current_offer"]
            and self.state.game_state["current_offer"]["to_player"] == player_id
        ):
            # player must reply with Accept/Deny
            if self.accept_pattern.search(action):
                # accept the offer and execute the trade
                success, trade_result = self._execute_trade(
                    self.state.game_state["current_offer"]["offer"], player_id
                )
                if success:
                    self.state.game_state["num_trades"] += 1
                    self.step_logs.append((ta.GAME_ID, f"Player {player_id} accepted the trade offer."))


                else:
                    # Determine who lacked resources
                    if trade_result == "proposer":
                        # Assign rewards: proposer loses
                        reward = {1-player_id: -1, player_id: 0}
                        terminated = True
                        info["reason"] = "Proposer lacks resources. Game over."

                    elif trade_result == "acceptor":
                        # Assign rewards: acceptor loses
                        reward = {player_id: -1, 1 - player_id: 0}
                        terminated = True
                        info["reason"] = "Acceptor lacks resources. Game over."

            elif self.deny_pattern.search(action):
                # offer denied
                self.step_logs.append((ta.GAME_ID, f"Player {player_id} denied the trade offer."))

            else:
                # Accept/Deny required. Illegal action
                terminated = True
                info["reason"] = f"Player {1-player_id} made an offer and Player {player_id} neither accepted nor denied it."

            self.state.game_state["current_offer"] = None

        # check if offer is proposed
        offer_match = self.offer_pattern.search(action)
        if offer_match:
            offer_text = offer_match.group(1).strip()
            parsed_offer = self._parse_offer(offer_text)
            if parsed_offer:
                self.state.game_state["current_offer"] = {
                    "from_player": player_id,
                    "to_player": 1-player_id,
                    "offer": parsed_offer,
                }
                self.step_logs.append(
                    (ta.GAME_ID, f"Player {player_id} made an offer.")
                )
            else:
                info["reason"] = f"Invalid offer format by Player {player_id}. Game over."
                # Invalid offer; agent receives -1 reward
                reward = {player_id: -1, 1-player_id: 0}
                terminated = True


        # Update the inventory values
        self._update_inventory_values()

        if "reason" in info:
            self.step_logs.append(
                (ta.GAME_ID, info["reason"])
            )


        truncated = self.state.step(
            logging_messages=self.step_logs,
        )

        return observations, reward, truncated, terminated, info

    def _parse_offer(self, offer_str: str) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Parse the offer string to extract the resources being traded.
        Expected format: "I give [resources]; You give [resources]"
        Returns a dictionary with 'my_offer' and 'their_offer' as dictionaries of resources and quantities.
        """
        try:
            # Remove any line breaks and extra spaces for robust parsing
            offer_str = " ".join(offer_str.split())

            # Split by '; You give ' to separate the offers
            offer_parts = re.split(r";\s*You give\s*", offer_str, flags=re.IGNORECASE)
            if len(offer_parts) != 2:
                self.state.logs.append(
                    (-1, "[DEBUG] Offer split into incorrect number of parts.")
                )
                return None
            my_offer_str = re.sub(
                r"^I give\s*", "", offer_parts[0], flags=re.IGNORECASE
            ).strip()
            their_offer_str = (
                offer_parts[1].strip().rstrip(".")
            )  # Remove trailing period if present

            my_offer = self._parse_resource_list(my_offer_str)
            their_offer = self._parse_resource_list(their_offer_str)

            if not my_offer or not their_offer:
                self.step_logs.append(
                    (ta.GAME_ID, "[DEBUG] Parsed offers are invalid.")
                )
                return None

            return {"my_offer": my_offer, "their_offer": their_offer}
        except Exception as e:
            # Log the exception for debugging purposes
            self.step_logs.append((ta.GAME_ID, f"[DEBUG] _parse_offer exception: {e}"))
            return None

    def _parse_resource_list(self, resource_str: str) -> Optional[Dict[str, int]]:
        """
        Parse a string of resources and quantities into a dictionary.
        Example input: "2 Wheat, 1 Ore"
        Returns a dictionary: {'Wheat': 2, 'Ore': 1}
        """
        resource_list = re.split(r",\s*|and\s*", resource_str)
        resources = {}
        for item in resource_list:
            item = item.strip()
            if not item:
                continue
            try:
                qty_str, resource_name = item.split(" ", 1)
                qty = int(qty_str)
                resource_name = (
                    resource_name.strip().title()
                )  # Ensure consistent casing
                if resource_name not in self.resource_names or qty <= 0:
                    self.step_logs.append(
                        (ta.GAME_ID, f"[DEBUG] Invalid resource or quantity: {resource_name}, {qty}")
                    )
                    return None
                resources[resource_name] = qty
            except Exception as e:
                # Log the exception for debugging purposes
                self.step_logs.append(
                    (ta.GAME_ID, f"[DEBUG] _parse_resource_list exception: {e}")
                )
                return None
        return resources

    def _execute_trade(
        self, trade: Dict[str, Dict[str, int]], acceptor_id: int
    ) -> Tuple[bool, str]:
        """
        Execute the trade between players.
        Returns a tuple (success: bool, trade_result: str).
        trade_result can be 'proposer' or 'acceptor' indicating who lacks resources.
        """
        proposer_id = self.state.game_state["current_offer"]["from_player"]
        my_offer = trade["my_offer"]  # Resources proposer gives
        their_offer = trade["their_offer"]  # Resources acceptor gives

        # Check if proposer has enough resources
        for resource, qty in my_offer.items():
            if self.state.game_state["player_resources"][proposer_id].get(resource, 0) < qty:
                return False, "proposer"  # Proposer lacks resources

        # Check if acceptor has enough resources
        for resource, qty in their_offer.items():
            if self.state.game_state["player_resources"][acceptor_id].get(resource, 0) < qty:
                return False, "acceptor"  # Acceptor lacks resources

        # Execute the trade
        for resource, qty in my_offer.items():
            self.state.game_state["player_resources"][proposer_id][resource] -= qty
            self.state.game_state["player_resources"][acceptor_id][resource] += qty
        for resource, qty in their_offer.items():
            self.state.game_state["player_resources"][acceptor_id][resource] -= qty
            self.state.game_state["player_resources"][proposer_id][resource] += qty

        return True, "success"

    def _calculate_player_inventory_value(self, player_id: int, game_state: Dict[str, Any]) -> float:
        """
        Calculate the inventory value of Player player_id.
        Args:
            player_id (int): The player's ID (0 or 1).
        Returns:
            inventory_value (float): The value of the players inventory
        """
        resources = game_state["player_resources"][player_id]
        values = game_state["player_values"][player_id]
        inventory_value = sum([qty * values[res] for res, qty in resources.items()])
        return inventory_value

    def _update_inventory_values(self):
        """
        For both players, update the inventory values.
        """
        for player_id in range(2):
            # get initial inventory value
            initial_inventory_value = self.state.game_state["inventory_value"][player_id][
                "initial"
            ]

            # calculate current inventory value
            current_inventory_value = self._calculate_player_inventory_value(
                player_id=player_id,
                game_state=self.state.game_state
            )

            # update
            self.state.game_state["inventory_value"][player_id][
                "current"
            ] = current_inventory_value
            self.state.game_state["inventory_value"][player_id]["change"] = (
                current_inventory_value - initial_inventory_value
            )

    def _calculate_rewards(self) -> Tuple[ta.Reward, str]:
        """
        Calculate the rewards for both players based on their resource values.
        Returns:
            Tuple[Dict[int, int], str]: The rewards for each player and the reason.
        """
        # Determine rewards
        if self.state.game_state["num_trades"] == 0:
            # No trades made, it's a draw
            rewards = {0: 0, 1: 0}
            reason = "No trades were made. The game is a draw."
        else:
            reason = "Game over. Calculating gains."
            player0_gain = self.state.game_state["inventory_value"][0]["change"]
            player1_gain = self.state.game_state["inventory_value"][1]["change"]
            if player0_gain > player1_gain:
                rewards = {0: 1, 1: -1 if player1_gain < 0 else 0}
                reason += " Player 0 wins by having a higher gain."
            elif player1_gain > player0_gain:
                rewards = {1: 1, 0: -1 if player0_gain < 0 else 0}
                reason += " Player 1 wins by having a higher gain."
            else:
                rewards = {0: 0, 1: 0}
                reason += " Both players have equal gains. It's a draw."
        return rewards, reason

    def render(self):
        """
        Render the current game state.
        This method should be called externally to display the game state.
        """
        turn_info = f"Turn {self.turn}/{self.state.max_turns}"
        print(turn_info)
        print("Player Resources and Values:")
        for player_id in [0, 1]:
            resources = self.state.game_state["player_resources"][player_id]
            values = self.state.game_state["player_values"][player_id]
            resource_list = "; ".join(
                [
                    f"{resources[res]} {res} (Value: {values[res]})"
                    for res in resources.keys()
                ]
            )
            print(f"  Player {player_id}: {resource_list}")

        if self.state.game_state["current_offer"]:
            offer = self.state.game_state["current_offer"]
            print(
                f"\nCurrent offer from Player {offer['from_player']} to Player {offer['to_player']}:"
            )
            my_offer = "; ".join(
                [f"{qty} {res}" for res, qty in offer["offer"]["my_offer"].items()]
            )
            their_offer = "; ".join(
                [f"{qty} {res}" for res, qty in offer["offer"]["their_offer"].items()]
            )
            print(f"  I give: {my_offer}; You give: {their_offer}")

        else:
            print("\nNo trades have been made yet.")

        print("\nAction Logs:")
        for player_id, log in self.state.logs:
            if player_id == -1:
                print(f"[GAME] {log}")
            else:
                print(f"[Player {player_id}] {log}")
        print("\n")
