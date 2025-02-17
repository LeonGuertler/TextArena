import re
import random
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta


class NegotiationEnv(ta.Env):
    """
    N-player Negotiation Game with the ability to:
      - broadcast messages,
      - send private messages,
      - make multiple offers to specific opponents,
      - accept/deny multiple pending offers.
    """

    # Regex patterns to parse player actions
    broadcast_pattern = re.compile(r"\[Broadcast\s*:?\s*\](.*?)?(?=\[|$)", re.IGNORECASE | re.DOTALL)
    whisper_pattern = re.compile(r"\[Whisper\s+(?:to\s+)?(?:Player\s+)?(\d+)\s*:?\s*\](.*?)?(?=\[|$)", re.IGNORECASE | re.DOTALL)
    offer_pattern = re.compile(r"\[Offer\s+(?:to\s+)?(?:Player\s+)?(\d+)\s*:?\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    accept_pattern = re.compile(r"\[Accept\s*#?\s*(\d+)\]", re.IGNORECASE)
    deny_pattern = re.compile(r"\[Deny\s*#?\s*(\d+)\]", re.IGNORECASE)

    def __init__(self, num_players: int = 3, max_turns: Optional[int] = 25):
        """
        Initialize the N-player Negotiation Game environment.

        Args:
            num_players (int): How many players will participate.
            max_turns (Optional[int]): Maximum number of turns before the game ends.
        """
        self.num_players = num_players
        self.resource_names = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]
        self.base_values = {"Wheat": 5, "Wood": 10, "Sheep": 15, "Brick": 25, "Ore": 40}

        # Create the underlying game state
        self.state = ta.State(num_players=num_players, max_turns=max_turns)

    @property
    def offline_renderer(self):
        raise NotImplementedError


    @property
    def terminal_render_keys(self):
        return ["player_resources", "player_values", "pending_offers"]


    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment to its initial state.
        """
        if seed is not None:
            random.seed(seed)

        # Initialize each player's resources to random amounts
        # and each player's private values for resources
        player_resources = {}
        player_values = {}
        for pid in range(self.num_players):
            # random resource counts
            player_resources[pid] = {
                r: random.randint(5, 25) for r in self.resource_names
            }
            # random personal valuations (±20% around base, but clamp 5..40)
            player_values[pid] = {}
            for r in self.resource_names:
                base = self.base_values[r]
                variation = int(0.2 * base)  # ±20%
                low, high = max(5, base - variation), min(40, base + variation)
                player_values[pid][r] = random.randint(low, high)

        # We’ll track “pending_offers” in a dictionary keyed by offer_id
        #   Each entry is: { from: pid, to: pid, offered: {...}, requested: {...} }
        # We'll increment an offer_id counter for each new offer
        game_state = {
            "player_resources": player_resources,
            "player_values": player_values,
            "pending_offers": {},
            "offer_id_counter": 0,
        }

        # Optionally track some final scoring or changes from initial
        # If you want to track changes in resource value, store initial_value, etc.

        # Reset the State
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the prompt for each player on reset (or new round).
        Display their resource counts and personal valuations.
        """
        resources = game_state["player_resources"][player_id]
        valuations = game_state["player_values"][player_id]
        resource_lines = []
        for r in self.resource_names:
            qty = resources[r]
            val = valuations[r]
            resource_lines.append(f"- {qty} x {r} (value: {val} each)")
        resource_str = "\n".join(resource_lines)

        prompt = (
            f"You are Player {player_id} in a multi-player game of Negotiation.\n"
            f"You have:\n{resource_str}\n\n"
            "You can broadcast messages, privately message someone, or make trade offers.\n"
            "You can also accept or deny any offers you received previously.\n"
            f"Your personal valuations are shown above; your goal is to maximize your total resource value.\n"
            # f"This is a turn-based negotiation with up to {self.num_players} players.\n\n"
            "Available actions:\n"
            "  [Broadcast: Some message] - Send a message to all players\n"
            "  [Whisper to X: Some message] - Send a private message to a specific player\n"
            "  [Offer to X: 2 Wheat -> 3 Wood] - Make a trade offer to a specific player\n"
            "  [Accept <x>], [Deny <x>] - Accept or Deny a trade offer\n"
            "You may combine multiple tokens in a single turn if you like.\n"
        )
        if self.state.max_turns is not None:
            prompt += f"Game ends after {self.state.max_turns} turns.\n"
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a single step from the current player. That action may
        include multiple tokens (broadcast, message, offers, accepts, denies).
        """
        current_pid = self.state.current_player_id
        print("+"*20)
        print(f"PLAYER {current_pid} - {action}")
        print("+"*20)
        # Log the raw action for debugging
        # self.state.add_log(from_id=current_pid, message=action)
        self.state.add_observation(
            from_id=current_pid,
            to_id=current_pid,  # send it back to player
            message=action,
            for_logging=True
        )

        # 1. Parse out all tokens in the player's action
        #    We can do each in sequence so a single "action" can contain many instructions.
        #    For a simpler approach, you might limit to a single instruction per turn.
        self._process_broadcasts(current_pid, action)
        self._process_private_messages(current_pid, action)
        self._process_offers(current_pid, action)
        self._process_accepts_and_denies(current_pid, action)

        # 2. If we've reached turn limit, optionally decide a winner or call it a draw
        if self.state.turn == (self.state.max_turns or 0) - 1:
            self._determine_winner()

        # 3. End the step: rotate to next player
        return self.state.step()

    def _process_broadcasts(self, from_pid: int, action: str):
        """Find all `[Broadcast: ...]` patterns, send them to everyone."""
        matches = self.broadcast_pattern.findall(action)
        for msg_content in matches:
            msg_content = msg_content.strip()
            if msg_content:
                self.state.add_observation(
                    from_id=from_pid,
                    to_id=-1,  # broadcast to all
                    message=f"(Broadcast) Player {from_pid} says: {msg_content}",
                    for_logging=False # already logged anyway
                )

    def _process_private_messages(self, from_pid: int, action: str):
        """
        Find all `[Message to X: ...]` patterns, send them only to X.
        Everyone sees that from_pid performed an action, but not the content itself.
        """
        matches = self.whisper_pattern.findall(action)
        for target_str, msg_content in matches:
            msg_content = msg_content.strip()
            try:
                target_pid = int(target_str)
            except ValueError:
                # If we can't parse the ID, mark invalid
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Invalid private-message target: {target_str}",
                )
                continue

            if target_pid not in range(self.num_players):
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Attempted to message a non-existent player {target_pid}."
                )
                continue

            if msg_content:
                self.state.add_observation(
                    from_id=from_pid,
                    to_id=target_pid,
                    message=f"(Private) Player {from_pid} says: {msg_content}",
                    for_logging=False # already logged
                )
            else:
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason="Empty private message?"
                )

    def _process_offers(self, from_pid: int, action: str):
        """
        Find all `[Offer to X: ... -> ...]` patterns.  
        Each is stored in game_state['pending_offers'] with a new ID.
        """
        game_state = self.state.game_state
        matches = self.offer_pattern.findall(action)
        # Each match returns: (target_pid_str, "2 Wood -> 1 Ore")
        for target_str, offer_str in matches:
            offer_str = offer_str.strip()
            try:
                target_pid = int(target_str)
            except ValueError:
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Invalid offer target: {target_str}"
                )
                continue
            if target_pid not in range(self.num_players):
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Offer made to invalid player ID {target_pid}"
                )
                continue

            # Split "2 Wood -> 1 Ore"
            # A simple approach: "Left -> Right"
            parts = re.split(r"->", offer_str)
            if len(parts) != 2:
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Cannot parse Offer: '{offer_str}'. Must be like '2 Wheat -> 3 Wood'."
                )
                continue

            offered_str = parts[0].strip()
            requested_str = parts[1].strip()

            # Parse resources
            offered_dict = self._parse_resource_list(offered_str)
            requested_dict = self._parse_resource_list(requested_str)
            if offered_dict is None or requested_dict is None:
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=f"Invalid resource format in offer: '{offer_str}'"
                )
                continue

            # Check if the offering player has enough resources to cover what they are offering
            if not self._check_sufficient_resources(
                from_pid, offered_dict, game_state["player_resources"]
            ):
                self.state.set_invalid_move(
                    player_id=from_pid,
                    reason=("You do not hold enough resources to offer "
                            f"{offered_dict} to Player {target_pid}.")
                )
                continue

            # Create a new offer ID
            game_state["offer_id_counter"] += 1
            new_id = game_state["offer_id_counter"]
            game_state["pending_offers"][new_id] = {
                "from": from_pid,
                "to": target_pid,
                "offered_resources": offered_dict,
                "requested_resources": requested_dict
            }

           

            # Also broadcast an FYI that an offer was created (though not the details)
            # or you might prefer partial info, e.g. "Player {from_pid} made an offer to Player {target_pid}."
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Offer #{new_id} created: Player {from_pid} -> Player {target_pid}."
            )
             # Let target know they've received a new offer
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=target_pid,
                message=(
                    f"You have a new offer [ID #{new_id}] from Player {from_pid}: "
                    f"{self._offer_to_str(offered_dict, requested_dict)}\n"
                    f"You can [accept #{new_id}] or [deny #{new_id}] it."
                ),
                for_logging=False # already logged
            )

    def _process_accepts_and_denies(self, current_pid: int, action: str):
        """
        Find `[Accept #ID]` and `[Deny #ID]`.
        - Only the 'to' player of that offer can accept/deny it.
        - On accept, verify that both sides have the resources needed to fulfill.
          If valid, exchange resources and remove the offer from pending_offers.
        - On deny, simply remove the offer from pending_offers.
        """
        accepts = self.accept_pattern.findall(action)  # list of IDs
        denies  = self.deny_pattern.findall(action)    # list of IDs

        for offer_id_str in accepts:
            try:
                offer_id = int(offer_id_str)
            except ValueError:
                self.state.set_invalid_move(
                    player_id=current_pid,
                    reason=f"Invalid offer ID in Accept: {offer_id_str}"
                )
                continue
            self._attempt_accept_offer(current_pid, offer_id)

        for offer_id_str in denies:
            try:
                offer_id = int(offer_id_str)
            except ValueError:
                self.state.set_invalid_move(
                    player_id=current_pid,
                    reason=f"Invalid offer ID in Deny: {offer_id_str}"
                )
                continue
            self._deny_offer(current_pid, offer_id)

    def _attempt_accept_offer(self, current_pid: int, offer_id: int):
        """
        Attempt to accept an offer from game_state['pending_offers'][offer_id].
        Confirm that `current_pid` is the 'to' player. Check resources, then exchange.
        """
        game_state = self.state.game_state
        if offer_id not in game_state["pending_offers"]:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Offer #{offer_id} does not exist."
            )
            return
        off = game_state["pending_offers"][offer_id]
        if off["to"] != current_pid:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Offer #{offer_id} is not addressed to you."
            )
            return

        # Check if from-player still has what they offered
        if not self._check_sufficient_resources(
            off["from"], off["offered_resources"], game_state["player_resources"]
        ):
            # The offering player no longer has enough resources to complete the trade
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=(f"Offer #{offer_id} canceled because Player {off['from']} "
                         "no longer has enough resources to fulfill it.")
            )
            del game_state["pending_offers"][offer_id]
            return

        # Check if to-player has what is requested
        if not self._check_sufficient_resources(
            off["to"], off["requested_resources"], game_state["player_resources"]
        ):
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"You do not have enough resources to fulfill Offer #{offer_id}."
            )
            return

        # Execute the trade
        self._exchange_resources(off["from"], off["to"],
                                 off["offered_resources"], off["requested_resources"])
        # Announce success
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=(f"Player {off['to']} ACCEPTED Offer #{offer_id} from Player {off['from']}: "
                     f"{self._offer_to_str(off['offered_resources'], off['requested_resources'])}")
        )
        # Remove it from pending
        del game_state["pending_offers"][offer_id]

        # (Optional) You might update a "trade history" or track "inventory_value" here.

    def _deny_offer(self, current_pid: int, offer_id: int):
        """
        Deny (remove) the specified offer, if it targets the current player.
        """
        game_state = self.state.game_state
        if offer_id not in game_state["pending_offers"]:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Offer #{offer_id} does not exist."
            )
            return
        off = game_state["pending_offers"][offer_id]
        if off["to"] != current_pid:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Offer #{offer_id} is not addressed to you."
            )
            return

        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {current_pid} DENIED Offer #{offer_id} from Player {off['from']}."
        )
        del game_state["pending_offers"][offer_id]

    def _check_sufficient_resources(self, pid: int, needed: Dict[str, int],
                                    all_resources: Dict[int, Dict[str, int]]) -> bool:
        """
        Return True if player `pid` has at least `needed[resource]` of each resource.
        """
        for resource, qty in needed.items():
            if all_resources[pid].get(resource, 0) < qty:
                return False
        return True

    def _exchange_resources(self, from_pid: int, to_pid: int,
                            offered: Dict[str, int], requested: Dict[str, int]):
        """
        Actually remove `offered` from `from_pid` and add to `to_pid`,
        and remove `requested` from `to_pid` and add to `from_pid`.
        """
        # For brevity. (In real code, always double-check you can do the exchange.)
        res = self.state.game_state["player_resources"]
        # from_pid loses offered
        for r, q in offered.items():
            res[from_pid][r] -= q
            res[to_pid][r]   += q
        # to_pid loses requested
        for r, q in requested.items():
            res[to_pid][r]   -= q
            res[from_pid][r] += q

    def _parse_resource_list(self, resource_str: str) -> Optional[Dict[str, int]]:
        """
        Example of parsing "2 Wheat, 1 Ore" into {"Wheat": 2, "Ore": 1}.
        Returns None on parse error.
        """
        # Split by commas or "and"
        items = re.split(r",\s*|\s+and\s+", resource_str, flags=re.IGNORECASE)
        parsed = {}
        for item in items:
            item = item.strip()
            if not item:
                continue
            # Expect format "<qty> <ResourceName>"
            match = re.match(r"(\d+)\s+(.+)", item)
            if not match:
                return None
            qty_str, rname = match.groups()
            qty = int(qty_str)
            # Attempt to standardize resource name
            rname = rname.strip().title()
            if rname not in self.resource_names:
                return None
            if qty <= 0:
                return None
            parsed[rname] = parsed.get(rname, 0) + qty
        return parsed if parsed else None

    def _offer_to_str(self, offered: Dict[str, int], requested: Dict[str, int]) -> str:
        """
        Return a quick string representation like "2 Wheat -> 3 Wood"
        """
        off_str = ", ".join(f"{q} {r}" for r, q in offered.items())
        req_str = ", ".join(f"{q} {r}" for r, q in requested.items())
        return f"{off_str} -> {req_str}"

    def _determine_winner(self):
        """
        Simple function if your game ends automatically at max_turns.
        You can define scoring logic, or declare a draw, or so on.
        """
        # Example: sum each player's total resource "market value"
        # Compare to find a winner
        game_state = self.state.game_state
        final_values = {}
        for pid in range(self.num_players):
            val = self._calculate_inventory_value(pid, game_state)
            final_values[pid] = val

        best_pid = max(final_values, key=final_values.get)
        best_val = final_values[best_pid]
        # Check if tie
        winners = [p for p, v in final_values.items() if v == best_val]
        if len(winners) == 1:
            self.state.set_winners(
                player_ids=[best_pid],
                reason=f"Player {best_pid} wins with a total value of {best_val}!"
            )
        else:
            self.state.set_draw(
                reason=f"Tie among players {winners} with value {best_val}."
            )

    def _calculate_inventory_value(self, pid: int, game_state: Dict[str, Any]) -> int:
        """
        Summation of (count_of_resource * that_player's_value_for_resource).
        """
        total = 0
        resources = game_state["player_resources"][pid]
        values = game_state["player_values"][pid]
        for r in self.resource_names:
            total += resources[r] * values[r]
        return total
