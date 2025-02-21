import re, random
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class NegotiationEnv(ta.Env):
    """ Environment for the Negotiation Game. """
    def __init__(self, max_turns: Optional[int] = 10):
        """
        Initialize the Negotiation Game environment.

        Args:
            max_turns (Optional[int]): Maximum number of turns before the game is truncated.
        """
        self.resource_names = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
        self.base_values = {"Wheat": 5, "Wood": 10, "Sheep": 15, "Brick": 25, "Ore": 40}

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=max_turns,
        )

        # Final Regex patterns for parsing actions
        self.accept_pattern = re.compile(r"\[Accept\]", re.IGNORECASE)
        self.deny_pattern = re.compile(r"\[Deny\]", re.IGNORECASE)
        self.offer_pattern = re.compile(
            r"\[Offer:\s*(?:I\s+(?:give|offer)\s+)?([^\[\]]+?)\s*\.*\]",  # Handles optional leading phrases and trailing period
            re.IGNORECASE | re.DOTALL
        )

    @property
    def offline_renderer(self):
        from textarena.envs.two_player.Negotiation.render.renderer import NegotiationRenderer
        return NegotiationRenderer 

    @property
    def terminal_render_keys(self):
        return ["player_resources", "inventory_value"]

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the initial prompt for a player.

        Args:
            player_id (int): ID of the player (0 or 1).

        Returns:
            str: The initial prompt for the player.
        """
        resource_value_list = "\n\t+ ".join(
            [
                f"{f'[{res}]':{' '}<8}  Qty: {game_state['player_resources'][player_id][res]:{' '}<2}   Value: {game_state['player_values'][player_id][res]}"
                for res in game_state['player_resources'][player_id].keys()
            ]
        )
        # {s:{pad_char}^{width}
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

    def _initialize_game_data(self):
        """Initialize the data structure for game statistics tracking."""

        player_names = getattr(self, 'player_names', {0: "Player 0", 1: "Player 1"})

        self.game_data = {
            "turns_data": [],
            "game_metadata": {
                "total_trades": 0,
                "total_successful_trades": 0,
                "total_attempted_trades": 0,
                "total_value_traded": 0,
                "total_economic_surplus": 0,
                "pareto_improvements": 0,
                "zero_sum_trades": 0,
                "value_destroying_trades": 0,
                "price_equilibrium": {},
                "delta_value_player_0": 0,
                "delta_value_player_1": 0,
                "player_names": player_names
            }
        }

    def _is_favorable_trade(self, trade_data: Dict) -> Dict:
        """
        Determine if a trade is favorable based on multiple economic metrics.

        Args:
            trade_data (Dict): Data about the trade

        Returns:
            Dict: Dictionary containing various economic trade metrics
        """
        initiator_id = trade_data["from_player"]
        recipient_id = trade_data["to_player"]

        # Calculate value from initiator's perspective
        offered_value_init = self._calculate_trade_value(trade_data["offered_resources"], initiator_id)
        requested_value_init = self._calculate_trade_value(trade_data["requested_resources"], initiator_id)

        # Calculate value from recipient's perspective
        offered_value_recip = self._calculate_trade_value(trade_data["offered_resources"], recipient_id)
        requested_value_recip = self._calculate_trade_value(trade_data["requested_resources"], recipient_id)

        # Calculate net gains for each party
        initiator_gain = requested_value_init - offered_value_init
        recipient_gain = offered_value_recip - requested_value_recip

        # Calculate total economic surplus
        economic_surplus = initiator_gain + recipient_gain

        # Calculate comparative advantage ratios for resources involved
        comparative_advantages = {}
        for resource in set(
                list(trade_data["offered_resources"].keys()) + list(trade_data["requested_resources"].keys())):
            init_value = self.state.game_state["player_values"][initiator_id].get(resource, 0)
            recip_value = self.state.game_state["player_values"][recipient_id].get(resource, 0)
            if init_value > 0 and recip_value > 0:  # Avoid division by zero
                comparative_advantages[resource] = init_value / recip_value

        # Determine trade efficiency (% of theoretical maximum gains achieved)
        # Theoretical maximum occurs when all resources go to the player who values them most
        theoretical_max = 0
        for resource, qty in trade_data["offered_resources"].items():
            theoretical_max += qty * max(
                self.state.game_state["player_values"][initiator_id][resource],
                self.state.game_state["player_values"][recipient_id][resource]
            )
        for resource, qty in trade_data["requested_resources"].items():
            theoretical_max += qty * max(
                self.state.game_state["player_values"][initiator_id][resource],
                self.state.game_state["player_values"][recipient_id][resource]
            )

        # Actual value after trade
        actual_after_trade = offered_value_recip + requested_value_init

        # Value before trade
        value_before_trade = offered_value_init + requested_value_recip

        # Calculate efficiency percentage
        if theoretical_max > value_before_trade:
            # Only if trade could theoretically increase total value
            trade_efficiency = (actual_after_trade - value_before_trade) / (theoretical_max - value_before_trade) * 100
        else:
            trade_efficiency = 0

        # Calculate opportunity costs (simplified version based on highest value alternatives)
        opportunity_costs = {}
        for resource, qty in trade_data["offered_resources"].items():
            # Opportunity cost is what else this resource could have been traded for
            max_alternative_value = 0
            for other_resource in self.resource_names:
                if other_resource != resource:
                    # How much of other_resource could be obtained for same value
                    alt_qty = (qty * self.state.game_state["player_values"][initiator_id][resource]) / \
                              max(1, self.state.game_state["player_values"][recipient_id][other_resource])
                    alt_value = alt_qty * self.state.game_state["player_values"][initiator_id][other_resource]
                    max_alternative_value = max(max_alternative_value, alt_value)
            opportunity_costs[f"{resource}_offered"] = max_alternative_value

        for resource, qty in trade_data["requested_resources"].items():
            max_alternative_value = 0
            for other_resource in self.resource_names:
                if other_resource != resource:
                    alt_qty = (qty * self.state.game_state["player_values"][recipient_id][resource]) / \
                              max(1, self.state.game_state["player_values"][initiator_id][other_resource])
                    alt_value = alt_qty * self.state.game_state["player_values"][recipient_id][other_resource]
                    max_alternative_value = max(max_alternative_value, alt_value)
            opportunity_costs[f"{resource}_requested"] = max_alternative_value

        return {
            "favorable_to_initiator": initiator_gain > 0,
            "favorable_to_recipient": recipient_gain > 0,
            "mutually_favorable": initiator_gain > 0 and recipient_gain > 0,
            "zero_sum": abs(economic_surplus) < 0.001,  # Near zero with small float tolerance
            "value_destroying": economic_surplus < 0,
            "initiator_gain": initiator_gain,
            "recipient_gain": recipient_gain,
            "economic_surplus": economic_surplus,
            "comparative_advantages": comparative_advantages,
            "trade_efficiency": trade_efficiency,
            "opportunity_costs": opportunity_costs
        }

    def _update_turn_data(self, turn_number: int, trade_data: Optional[Dict] = None):
        """
        Update the data for the current turn, handling multiple offers per turn.

        Args:
            turn_number (int): Current turn number
            trade_data (Optional[Dict]): Data about trades made in this turn
        """
        # Get or create turn data for current turn
        if len(self.game_data["turns_data"]) <= turn_number:
            turn_data = {
                "turn_number": turn_number,
                "trades_data": {
                    "number_of_offers": 0,
                    "successful_trades": 0,
                    "rejected_trades": 0,
                    "trades": []
                }
            }
            self.game_data["turns_data"].append(turn_data)
        else:
            turn_data = self.game_data["turns_data"][turn_number]

        if trade_data:
            # Get the outcome, defaulting to None for new offers
            outcome = trade_data.get("outcome")
            from_player = trade_data["from_player"]
            to_player = trade_data["to_player"]

            # Get economic metrics for this trade
            economic_metrics = self._is_favorable_trade(trade_data)

            # Calculate values from both perspectives
            value_offer_init = self._calculate_trade_value(
                trade_data["offered_resources"],
                from_player
            )
            value_ask_init = self._calculate_trade_value(
                trade_data["requested_resources"],
                from_player
            )

            value_offer_recip = self._calculate_trade_value(
                trade_data["offered_resources"],
                to_player
            )
            value_ask_recip = self._calculate_trade_value(
                trade_data["requested_resources"],
                to_player
            )

            # Calculate exchange ratios for price equilibrium tracking
            exchange_ratios = {}
            for res_offered, qty_offered in trade_data["offered_resources"].items():
                for res_requested, qty_requested in trade_data["requested_resources"].items():
                    ratio_key = f"{res_offered}_per_{res_requested}"
                    ratio_value = qty_offered / max(qty_requested, 1)  # Avoid division by zero
                    exchange_ratios[ratio_key] = ratio_value

                    # Update game-level price equilibrium tracking
                    if ratio_key not in self.game_data["game_metadata"]["price_equilibrium"]:
                        self.game_data["game_metadata"]["price_equilibrium"][ratio_key] = []

                    if outcome == "Accepted":  # Only track successful trades for equilibrium
                        self.game_data["game_metadata"]["price_equilibrium"][ratio_key].append(ratio_value)

            trade_info = {
                "trade_initiated_by": from_player,
                "value_offer_initiator": value_offer_init,
                "value_ask_initiator": value_ask_init,
                "value_offer_recipient": value_offer_recip,
                "value_ask_recipient": value_ask_recip,
                "resources_offered": trade_data["offered_resources"],
                "resources_asked": trade_data["requested_resources"],
                "exchange_ratios": exchange_ratios,
                "economic_metrics": economic_metrics,
                "trade_successful": outcome == "Accepted",
                "text_metadata": {
                    "text_length": len(str(trade_data))
                }
            }

            turn_data["trades_data"]["trades"].append(trade_info)
            turn_data["trades_data"]["number_of_offers"] += 1

            if outcome == "Accepted":
                turn_data["trades_data"]["successful_trades"] += 1
                self.game_data["game_metadata"]["total_successful_trades"] += 1
                self.game_data["game_metadata"]["total_value_traded"] += (
                        value_offer_init + value_ask_init
                )

                # Update economic surplus tracking
                self.game_data["game_metadata"]["total_economic_surplus"] += economic_metrics["economic_surplus"]

                # Update trade type counters
                if economic_metrics["mutually_favorable"]:
                    self.game_data["game_metadata"]["pareto_improvements"] += 1
                elif economic_metrics["zero_sum"]:
                    self.game_data["game_metadata"]["zero_sum_trades"] += 1
                elif economic_metrics["value_destroying"]:
                    self.game_data["game_metadata"]["value_destroying_trades"] += 1

            elif outcome == "Rejected":
                turn_data["trades_data"]["rejected_trades"] += 1

            self.game_data["game_metadata"]["total_attempted_trades"] += 1
            self.game_data["game_metadata"]["total_trades"] += turn_data["trades_data"]["successful_trades"]

    def _check_for_new_offer(self, player_id: int, action: str):
        """
        Check if the player's action contains a new trade offer.

        Args:
            player_id (int): ID of the player making the offer.
            action (str): The action string.
        """
        if not self.state.done:
            offer_match = self.offer_pattern.search(action)
            if offer_match:
                matched_offer = offer_match.group(1).strip()
                parsed_offer = self._parse_offer(matched_offer)
                if parsed_offer:
                    if self._check_if_sufficient_resources(
                            trade_resources=parsed_offer["offered_resources"],
                            player_resources=self.state.game_state["player_resources"][player_id]
                    ):
                        # Add the offer to the game state with consistent keys
                        self.state.game_state["current_offer"] = {
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"],
                            "outcome": None  # Initialize outcome as None
                        }

                        # Display trade metrics to inform players about the economics of the offer
                        economic_metrics = self._is_favorable_trade(self.state.game_state["current_offer"])
                        trade_message = (
                            f"Player {player_id} made the following offer to Player {1 - player_id}: "
                            f"{self._offer_to_str(parsed_offer)}\n"
                        )

                        # Add economic analysis to help players understand the trade
                        if economic_metrics["mutually_favorable"]:
                            trade_message += "This trade would benefit both players."
                        elif economic_metrics["favorable_to_initiator"] and not economic_metrics[
                            "favorable_to_recipient"]:
                            trade_message += "This trade primarily benefits the initiator."
                        elif economic_metrics["favorable_to_recipient"] and not economic_metrics[
                            "favorable_to_initiator"]:
                            trade_message += "This trade primarily benefits the recipient."

                        if economic_metrics["value_destroying"]:
                            trade_message += " Note: This trade would reduce total economic value."

                        self.state.add_observation(
                            from_id=ta.GAME_ID,
                            to_id=-1,  # Broadcast to all
                            message=trade_message
                        )
                    else:
                        self.state.set_invalid_move(
                            player_id=player_id,
                            reason=f"Player {player_id} tried to make a trade offer without having the necessary resources."
                        )
                else:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Player {player_id} made a trade offer in an incorrect format."
                    )

    def _calculate_trade_value(self, resources: Dict[str, int], player_id: int) -> float:
        """
        Calculate the total value of resources in a trade for a specific player.

        Args:
            resources (Dict[str, int]): Resources involved in the trade
            player_id (int): ID of the player whose values to use

        Returns:
            float: Total value of the resources
        """
        return sum(
            qty * self.state.game_state["player_values"][player_id][resource]
            for resource, qty in resources.items()
        )

    def save_game_data(self) -> Dict:
        """
        Finalize and save the game data with enhanced economic metrics.

        Returns:
            Dict: Complete game data structure with economic analysis
        """
        # Update final value changes
        for player_id in [0, 1]:
            self.game_data["game_metadata"][f"delta_value_player_{player_id}"] = (
                self.state.game_state["inventory_value"][player_id]["change"]
            )

        # Add final economic analysis summaries
        if "price_equilibrium" in self.game_data["game_metadata"]:
            for resource_pair, ratios in self.game_data["game_metadata"]["price_equilibrium"].items():
                if ratios:  # Only calculate if there are recorded trades
                    self.game_data["game_metadata"][f"final_{resource_pair}_price"] = sum(ratios) / len(ratios)

        return self.game_data


    def reset(self, seed: Optional[int] = None):
        """
        Reset the Negotiation Game to its initial state.

        Args:
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.

        Returns:
            Optional[ta.Observations]: Initial observations for both players as a dict.
        """
        if seed is not None:
            random.seed(seed)

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

        self._initialize_game_data() # Moved before the return statement.
        return self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process the player's action.

        Args:
            action (str): The player's message or action.

        Returns:
            tuple: (done, info)
        """
        # Update the observations and log the action
        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=-1,  # Broadcast to all
            message=action,
            for_logging=True
        )

        # Get current offer state before processing new action
        previous_offer = self.state.game_state.get("current_offer")

        # Process response to existing offer
        self._check_and_execute_existing_offer(
            player_id=self.state.current_player_id,
            action=action
        )

        # Check for new offer
        self._check_for_new_offer(
            player_id=self.state.current_player_id,
            action=action
        )

        # Update turn data if there was an offer that got processed
        if previous_offer or self.state.game_state.get("current_offer"):
            self._update_turn_data(
                turn_number=self.state.turn,
                trade_data=previous_offer if previous_offer else self.state.game_state.get("current_offer")
            )

        # If turn limit reached, determine winner
        if self.state.turn == self.state.max_turns - 1:
            self._determine_winner()

        # If game is ending, save final game data
        if self.state.done:
            self.state.game_state["final_game_data"] = self.save_game_data()

        return self.state.step()

    def _check_and_execute_existing_offer(self, player_id: int, action: str) -> None:
        """Check if the player accepts or denies the current offer and execute accordingly."""
        current_offer = self.state.game_state.get("current_offer")

        if current_offer:
            if self.accept_pattern.search(action):
                self._attempt_to_execute_trade(
                    player_id=player_id,
                    action=action
                )
            elif self.deny_pattern.search(action):
                # Update trade history for rejected trades
                if self.state.game_state["current_offer"] is not None:
                    self.state.game_state["current_offer"]["outcome"] = "Rejected"
                self.state.game_state["current_offer"] = None

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

            if self.state.game_state["current_offer"] is not None:
                self.state.game_state["current_offer"]["outcome"] = "Accepted"

            # Update player inventory value
            self._update_inventory_values()

            # Reset trade offer
            self.state.game_state["current_offer"] = None

        # If not, throw invalid move
        else:
            self.state.set_invalid_move(
                player_id=acceptor_id,
                reason=f"Player {acceptor_id} tried accepting a trade without having the necessary resources."
            )

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
        if not self.state.done:
            offer_match = self.offer_pattern.search(action)
            if offer_match:
                matched_offer = offer_match.group(1).strip()
                parsed_offer = self._parse_offer(matched_offer)
                if parsed_offer:
                    if self._check_if_sufficient_resources(
                            trade_resources=parsed_offer["offered_resources"],
                            player_resources=self.state.game_state["player_resources"][player_id]
                    ):
                        # Add the offer to the game state with consistent keys
                        self.state.game_state["current_offer"] = {
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"],
                            "outcome": None  # Initialize outcome as None
                        }

                        # DO NOT add to trade_history yet.  Wait for outcome.
                        # self.state.game_state["trade_history"].append(...)  <-- REMOVE THIS

                        self.state.add_observation(
                            from_id=ta.GAME_ID,
                            to_id=-1,  # Broadcast to all
                            message=f"Player {player_id} made the following offer to Player {1 - player_id}: {self._offer_to_str(parsed_offer)}"
                        )
                    else:
                        self.state.set_invalid_move(
                            player_id=player_id,
                            reason=f"Player {player_id} tried to make a trade offer without having the necessary resources."
                        )
                else:
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Player {player_id} made a trade offer in an incorrect format."
                    )

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
        """
        Determine the winner based on the change in inventory values.
        """
        # Check if game is over
        if not self.state.done:
            if self.state.game_state["inventory_value"][0]["change"] == self.state.game_state["inventory_value"][1]["change"]:
                # Draw
                self.state.set_draw(
                    reason=f"Same change in inventory value for all players. Draw."
                )
            else:
                winner_id = 0 if (
                    self.state.game_state["inventory_value"][0]["change"] > self.state.game_state["inventory_value"][1]["change"]
                ) else 1
                self.state.set_winners(
                    player_ids=[winner_id],
                    reason=f"Player {winner_id} won by having a larger gain in inventory value."
                )

    def _update_inventory_values(self):
        """
        Update the current inventory values and their changes for both players.
        """
        for player_id in range(self.state.num_players):
            # Calculate current inventory value
            current_inventory_value = self._calculate_player_inventory_value(
                player_id=player_id,
                game_state=self.state.game_state
            )

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
