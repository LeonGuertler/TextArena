import re, random
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.SimpleNegotiation.renderer import create_board_str


class SimpleNegotiationEnv(ta.Env):
    """ Environment for the SimpleNegotiation Game. """
    def __init__(self, max_turns: Optional[int] = 10):
        self.max_turns = max_turns
        self.resource_names = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
        self.base_values = {"Wheat": 5, "Wood": 10, "Sheep": 15, "Brick": 25, "Ore": 40}

        # Final Regex patterns for parsing actions
        self.accept_pattern = re.compile(r"\[Accept\]", re.IGNORECASE)
        self.deny_pattern = re.compile(r"\[Deny\]", re.IGNORECASE)
        self.offer_pattern = re.compile(
            r"\[Offer:\s*(?:I\s+(?:give|offer)\s+)?([^\[\]]+?)\s*\.*\]",  # Handles optional leading phrases and trailing period
            re.IGNORECASE | re.DOTALL
        )

    def get_board_str(self):
        return create_board_str(
            player_resources=self.state.game_state["player_resources"], 
            player_values=self.state.game_state["player_values"], 
            inventory_values=self.state.game_state["inventory_value"], 
            current_offer=self.state.game_state["current_offer"]
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player """
        resource_value_list = "\n\t+ ".join(
            [
                f"{f'[{res}]':{' '}<8}  Qty: {game_state['player_resources'][player_id][res]:{' '}<2}   Value: {game_state['player_values'][player_id][res]}"
                for res in game_state['player_resources'][player_id].keys()
            ]
        )
        prompt = (
            f"You are Player {player_id} in the Negotiation Game.\n"
            "You have some resources, and your task is to trade such that the total value of your resources increases.\n"
            f"The resources and associated values you currently have are:\n\t+ "
            f"{resource_value_list}\n"
            "At each turn, you can talk to your opponent or make a trade offer.\n"
            "Use the following special tokens for actions:\n"
            "  - [Offer]: To make a trade offer.\n"
            "    Format: [Offer: Offered Resources -> Requested Resources]\n"
            "    Example: [Offer: 3 Sheep, 2 Ore -> 5 Brick, 2 Sheep]\n"
            "  - [Accept]: To accept an incoming offer.\n"
            "  - [Deny]: To deny an incoming offer (default).\n"
            "You can include additional text before and/or after these tokens.\n"
        )
        if self.state.max_turns:
            prompt += f"The game lasts for {self.state.max_turns} turns in total.\n"
        else:
            prompt += "The game has no turn limit.\n"
        return prompt

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the Negotiation Game to its initial state """
        # Initialize game state variables
        self.state = ta.State(num_players=2, min_players=2, max_players=2, max_turns=self.max_turns)

        game_state = {
            "current_offer": None,
            "player_resources": {
                0: {resource: random.randint(5, 25) for resource in self.resource_names},
                1: {resource: random.randint(5, 25) for resource in self.resource_names},
            },
            "player_values": {},
            "trade_history": [],
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

        # Keep track of the inventory (both initial and current)
        for player_id in [0, 1]:
            initial_value = self._calculate_player_inventory_value(player_id, game_state)
            game_state.setdefault("inventory_value", {})[player_id] = {
                "initial": initial_value,
                "current": initial_value,
                "change": 0,
            }

        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action """
        # Update the observations and log the action
        self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action)

        # Check if the player is responding to an existing offer
        self._check_and_execute_existing_offer(player_id=self.state.current_player_id, action=action)

        # Check if the player's action contains a new trade offer
        self._check_for_new_offer(player_id=self.state.current_player_id, action=action)

        # If turn limit, determine winner
        if self.state.turn == self.state.max_turns-1:
            self._determine_winner()

        return self.state.step()

    def _check_and_execute_existing_offer(self, player_id: int, action: str) -> None:
        """
        Check if the player accepts or denies the current offer and execute accordingly.

        Args:
            player_id (int): ID of the player responding to the offer.
            action (str): The action string.
        """
        # check if there is a current offer
        current_offer = self.state.game_state.get("current_offer")
        
        # check if an offer exists, and whether it was accepted
        if current_offer and self.accept_pattern.search(action):
            self._attempt_to_execute_trade(player_id=player_id, action=action)
        else:
            # make sure the offer is reset
            self.state.game_state["current_offer"] = None  # Reset


    def _attempt_to_execute_trade(self, player_id: int, action: str) -> None:
        """
        Attempt to execute the trade if both players have sufficient resources.

        Args:
            player_id (int): ID of the player accepting the offer.
            action (str): The action string.
        """
        current_offer = self.state.game_state["current_offer"]
        proposer_id = current_offer["from_player"]
        acceptor_id = player_id

        # # Check if the trade can be executed
        # if proposer_valid and acceptor_valid:
        if self._check_if_sufficient_resources(
            trade_resources=current_offer["requested_resources"],
            player_resources=self.state.game_state["player_resources"][acceptor_id]
        ):
            # Execute the trade
            for resource, qty in current_offer["offered_resources"].items():
                self.state.game_state["player_resources"][proposer_id][resource] -= qty
                self.state.game_state["player_resources"][acceptor_id][resource] += qty
            for resource, qty in current_offer["requested_resources"].items():
                self.state.game_state["player_resources"][acceptor_id][resource] -= qty
                self.state.game_state["player_resources"][proposer_id][resource] += qty

            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,  # Broadcast to all
                message=f"Player {acceptor_id} accepted the trade offer from Player {proposer_id}."
            )

            # Update trade history with outcome
            self.state.game_state["trade_history"].append({
                "from_player": proposer_id,
                "to_player": acceptor_id,
                "offered_resources": current_offer["offered_resources"],
                "requested_resources": current_offer["requested_resources"],
                "outcome": "Accepted"
            })

            # Update player inventory value
            self._update_inventory_values()

            # Reset trade offer
            self.state.game_state["current_offer"] = None

        # If not, throw invalid move
        else:
            reason=f"Player {acceptor_id} tried accepting a trade without having the necessary resources."
            self.state.set_invalid_move(player_id=acceptor_id, reason=reason)


    def _check_if_sufficient_resources(self, trade_resources: Dict[str, int], player_resources: Dict[str, int]) -> bool:
        """
        Check if a player has sufficient resources for a trade.

        Args:
            trade_resources (Dict[str, int]): Resources required for the trade.
            player_resources (Dict[str, int]): Player's current resources.

        Returns:
            bool: True if sufficient, False otherwise.
        """
        for resource, qty in trade_resources.items():
            if player_resources.get(resource, 0) < qty:
                return False
        return True

    def _check_for_new_offer(self, player_id: int, action: str):
        """
        Check if the player's action contains a new trade offer.

        Args:
            player_id (int): ID of the player making the offer.
            action (str): The action string.
        """
        # Check if the game has already done
        if not self.state.done:
            offer_match = self.offer_pattern.search(action)
            if offer_match:
                matched_offer = offer_match.group(1).strip()
                parsed_offer = self._parse_offer(matched_offer)
                if parsed_offer:

                    # check if necessary resourcs
                    if self._check_if_sufficient_resources(
                        trade_resources=parsed_offer["offered_resources"],
                        player_resources=self.state.game_state["player_resources"][player_id]
                    ):

                        # Add the offer to the game state with consistent keys
                        self.state.game_state["current_offer"] = {
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"]
                        }

                        # Update trade history with the new offer
                        self.state.game_state["trade_history"].append({
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"],
                            "outcome": None  # To be updated upon acceptance
                        })

                        message=f"Player {player_id} made the following offer to Player {1 - player_id}: {self._offer_to_str(parsed_offer)}"
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                    else:
                        reason=f"Player {player_id} tried to make a trade offer without having the necessary resources."
                        self.state.set_invalid_move(player_id=player_id, reason=reason)
                else:
                    # Erroneous offer
                    reason=f"Player {player_id} made a trade offer in an incorrect format."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

    def _parse_offer(self, offer_str: str) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Parse a trade offer string into a structured dictionary.

        Args:
            offer_str (str): The offer string extracted from the action.

        Returns:
            Optional[Dict[str, Dict[str, int]]]: Parsed offer details or None if parsing fails.
        """
        try:
            # Remove any line breaks and extra spaces for robust parsing
            offer_str = ' '.join(offer_str.split())

            # Remove trailing punctuation (e.g., period)
            offer_str = re.sub(r'[.,!?]+$', '', offer_str)

            # Remove leading phrases like "I give" or "I offer"
            offer_str = re.sub(r'^(I\s+(?:give|offer)\s+)', '', offer_str, flags=re.IGNORECASE)

            # Split by '->' to separate offered and requested resources
            offer_parts = re.split(r'\s*->\s*', offer_str)
            if len(offer_parts) != 2:
                return None  # Erroneous offer

            offered_items_str = offer_parts[0].strip()
            requested_items_str = offer_parts[1].strip()

            offered_items = self._parse_resource_list(offered_items_str)
            requested_items = self._parse_resource_list(requested_items_str)

            if not offered_items or not requested_items:
                return None  # Erroneous offer


            return {'offered_resources': offered_items, 'requested_resources': requested_items}

        except Exception as e:
            return None

    def _parse_resource_list(self, resource_str: str) -> Optional[Dict[str, int]]:
        """
        Parse a string of resources and quantities into a dictionary.

        Args:
            resource_str (str): String containing resources, e.g., "2 Wheat, 1 Ore".

        Returns:
            Optional[Dict[str, int]]: Parsed resources or None if parsing fails.
        """
        resource_list = re.split(r',\s*|\s+and\s+', resource_str, flags=re.IGNORECASE)
        resources = {}
        for item in resource_list:
            item = item.strip()
            if not item:
                continue
            try:
                match = re.match(r'^(\d+)\s+(.+)$', item)
                if not match:
                    return None
                qty_str, resource_name = match.groups()
                qty = int(qty_str)
                resource_name = resource_name.strip().title()  # Ensure consistent casing
                # Handle resource aliases if any (e.g., 'Sheeps' -> 'Sheep')
                resource_aliases = {
                    "Sheeps": "Sheep",
                    "Woods": "Wood",
                    # Add more aliases as needed
                }
                resource_name = resource_aliases.get(resource_name, resource_name)
                if resource_name not in self.resource_names or qty <= 0:
                    return None
                if resource_name in resources:
                    resources[resource_name] += qty
                else:
                    resources[resource_name] = qty
            except Exception as e:
                return None
        return resources

    def _offer_to_str(self, parsed_offer: Dict[str, Dict[str, int]]) -> str:
        """
        Convert a parsed offer dictionary to a readable string format.

        Args:
            parsed_offer (Dict[str, Dict[str, int]]): Parsed offer details.

        Returns:
            str: Readable string representation of the offer.
        """
        offered = ", ".join(f"{qty} {res}" for res, qty in parsed_offer["offered_resources"].items())
        requested = ", ".join(f"{qty} {res}" for res, qty in parsed_offer["requested_resources"].items())
        return f"Offered items: {offered} -> Requested items: {requested}"

    def _determine_winner(self):
        """ Determine the winner based on the change in inventory values """
        # Check if game is over
        if not self.state.done:
            if self.state.game_state["inventory_value"][0]["change"] == self.state.game_state["inventory_value"][1]["change"]:
                # Draw
                self.state.set_draw(reason=f"Same change in inventory value for all players. Draw.")
            else:
                winner_id = 0 if (
                    self.state.game_state["inventory_value"][0]["change"] > self.state.game_state["inventory_value"][1]["change"]
                ) else 1
                reason=f"Player {winner_id} won by having a larger gain in inventory value."
                self.state.set_winners(player_ids=[winner_id], reason=reason)

    def _update_inventory_values(self):
        """ Update the current inventory values and their changes for both players """
        for player_id in range(self.state.num_players):
            # Calculate current inventory value
            current_inventory_value = self._calculate_player_inventory_value(player_id=player_id, game_state=self.state.game_state)

            # Update
            self.state.game_state["inventory_value"][player_id]["current"] = current_inventory_value
            self.state.game_state["inventory_value"][player_id]["change"] = (
                current_inventory_value - self.state.game_state["inventory_value"][player_id]["initial"]
            )

    def _calculate_player_inventory_value(self, player_id: int, game_state: Dict[str, Any]) -> float:
        """
        Calculate the total inventory value for a player.

        Args:
            player_id (int): ID of the player.
            game_state (Dict[str, Any]): Current game state.

        Returns:
            float: Total inventory value.
        """
        resources = game_state["player_resources"][player_id]
        values = game_state["player_values"][player_id]
        inventory_value = sum([qty * values[res] for res, qty in resources.items()])
        return inventory_value
