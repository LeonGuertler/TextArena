import re, random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Poker.renderer import create_board_str


class PokerEnv(ta.Env):
    # Define action patterns
    check_pattern = re.compile(r"\[Check\]", re.IGNORECASE)
    fold_pattern = re.compile(r"\[Fold\]", re.IGNORECASE)
    call_pattern = re.compile(r"\[Call.*\]", re.IGNORECASE)
    bet_pattern = re.compile(r"\[bet (\d+)\]", re.IGNORECASE)
    raise_pattern = re.compile(r"\[raise (\d+)\]", re.IGNORECASE)

    def __init__(
        self,
        num_rounds: int = 10,
        starting_chips: int = 1_000,
        small_blind: int = 10,
        big_blind: int = 20,
    ):
        """
        A simplified multi-player Texas Hold'em Poker environment (no side pots).
        Handles between 2 and 15 players.

        Args:
            num_rounds (int): Number of rounds to play.
            starting_chips (int): Starting chip stack for each player.
            small_blind (int): Small blind amount.
            big_blind (int): Big blind amount.
        """
        self.num_rounds = num_rounds
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Card setup
        self.suits = ["♠", "♥", "♦", "♣"]
        self.ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.rank_values = {rank: idx+2 for idx, rank in enumerate(self.ranks)}
        #  (Mapping e.g. "2" -> 2, ..., "10"->10, "J"->11, "Q"->12, "K"->13, "A"->14)


    def get_board_str(self):
        return create_board_str(
            community_cards=self.state.game_state["visible_community_cards"],
            pot=self.state.game_state["pot"],
            player_chips=self.state.game_state["player_chips"],
            player_hands=self.state.game_state["player_hands"],
            bets=self.state.game_state["player_bets"]
        )


    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the full game to its initial state """
        # Create the underlying state for N players
        self.state = ta.State(num_players=num_players, min_players=2, max_players=15)
        
        # Store the basic structure of the game
        game_state = {
                "round": 1,  # Start at round 1
                "betting_round": 0,  # 0: pre-flop, 1: flop, 2: turn, 3: river
                "player_chips": {pid: self.starting_chips for pid in range(num_players)},
                "player_hands": {pid: [] for pid in range(num_players)},
                "community_cards": [],
                "visible_community_cards": [],
                "pot": 0,
                "current_bet": 0,
                "player_bets": {pid: 0 for pid in range(num_players)},
                "button": 0,  # Dealer button position
                "folded_players": set(),
                "all_in_players": set(),
                "checked_players": set(),
                "round_turn": 0,  # how many actions have taken place in this betting round
                "game_complete": False,
                "last_valid_round": 1,  # Track the last valid round number for error recovery
                "last_valid_betting_round": 0  # Track the last valid betting round for error recovery
            }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt, executable_on_reset=[self._reset_round])
        
        # Announce new game starting
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Starting a new {self.num_rounds}-round Texas Hold'em game with {num_players} players."
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the initial prompt explaining the poker game rules and format to a player.
        This is called once at the start of the game.
        """
        prompt = (
            f"You are Player {player_id} in a {self.state.num_players}-player Texas Hold'em Poker game.\n"
            f"Game Information:\n"
            f"- This is a {self.num_rounds}-round game\n"
            f"- Each player starts with {self.starting_chips} chips\n"
            f"- Small blind is {self.small_blind} chips\n"
            f"- Big blind is {self.big_blind} chips\n"
            f"- Players must call the current bet to stay in the hand\n\n"
            "Available Actions:\n"
            "  '[Check]' - When there's no bet to call\n"
            "  '[Call]' - Match the current bet\n"
            "  '[Fold]' - Give up your hand\n"
            "  '[Bet <amount>]' - Make a new bet, e.g. '[Bet 100]'\n"
            "  '[Raise <amount>]' - Increase the current bet, e.g. '[Raise 200]'\n\n"
            "The Player with the most chips at the end wins"
        )
        return prompt

    def _create_deck(self) -> List[Dict[str, str]]:
        """Create a standard 52-card deck."""
        return [{"rank": rank, "suit": suit} for suit in self.suits for rank in self.ranks]


    def _reset_round(self):
        """
        Initialize a new round (hand):
        - Shuffle and deal hole cards
        - Reset pot, bets, etc.
        - Post small blind and big blind
        - Set next player to act (first active player after big blind)
        - Ensure players with 0 chips are marked as all-in
        """
        gs = self.state.game_state
        num_players = self.state.num_players

        deck = self._create_deck()
        random.shuffle(deck)

        # Deal each player 2 cards
        for pid in range(num_players):
            gs["player_hands"][pid] = [deck.pop(), deck.pop()]
            c1, c2 = gs["player_hands"][pid]
            # Private observation for each player's hole cards
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=pid,
                message=f"Your hole cards: [{c1['rank']}{c1['suit']}, {c2['rank']}{c2['suit']}]",
                for_logging=True
            )

        # Reset community cards
        gs["community_cards"] = [deck.pop() for _ in range(5)]
        gs["visible_community_cards"] = []

        # Reset pot, bets, etc.
        gs["pot"] = 0
        gs["current_bet"] = 0
        gs["player_bets"] = {pid: 0 for pid in range(num_players)}
        gs["folded_players"] = set()
        gs["all_in_players"] = set()
        gs["checked_players"] = set()
        gs["round_turn"] = 0
        
        # Reset tracking for betting rounds
        gs["last_bettor"] = -1
        gs["bet_round_complete"] = False
        
        # Ensure players with 0 chips are properly marked as all-in
        for pid in range(num_players):
            if gs["player_chips"][pid] == 0:
                gs["all_in_players"].add(pid)

        # Post blinds
        button = gs["button"]
        sb_player = (button + 1) % num_players
        bb_player = (button + 2) % num_players

        # Subtract small blind
        sb = min(self.small_blind, gs["player_chips"][sb_player])
        gs["player_chips"][sb_player] -= sb
        gs["player_bets"][sb_player] = sb
        gs["pot"] += sb

        # Subtract big blind
        bb = min(self.big_blind, gs["player_chips"][bb_player])
        gs["player_chips"][bb_player] -= bb
        gs["player_bets"][bb_player] = bb
        gs["pot"] += bb

        gs["current_bet"] = max(sb, bb)

        # Next player to act should be the first active player after the big blind
        # (not just the position after big blind, but the first player who can actually act)
        next_player_id = (bb_player + 1) % num_players
        
        # Find the first player who can actually act (not all-in, not 0 chips)
        while (next_player_id in gs["all_in_players"] or 
            gs["player_chips"][next_player_id] == 0) and next_player_id != bb_player:
            next_player_id = (next_player_id + 1) % num_players
        
        self.state.manually_update_current_player(new_player_id=next_player_id)
        self._observe_current_pot()

    def _observe_current_pot(self):
        """
        Send a formatted observation to the next player showing the game state.
        """
        gs = self.state.game_state
        num_players = self.state.num_players

        # Validate that round number is within bounds
        round_num = max(1, min(gs["round"], self.num_rounds))
        betting_round = max(0, min(gs["betting_round"], 3))
        
        # Header: show round and turn info.
        round_text = f"Round {round_num} of {self.num_rounds}"
        turn_names = {0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River"}
        turn_text = turn_names.get(betting_round, "Unknown Turn")

        # Ensure visible cards match the betting round
        if betting_round == 0:
            expected_visible = []
        elif betting_round == 1:
            expected_visible = gs["community_cards"][:3]
        elif betting_round == 2:
            expected_visible = gs["community_cards"][:4]
        elif betting_round == 3:
            expected_visible = gs["community_cards"][:5]
        
        # Fix visible cards if they don't match expected
        if gs["visible_community_cards"] != expected_visible and not gs["game_complete"]:
            gs["visible_community_cards"] = expected_visible


        # Determine roles based on the dealer button:
        dealer = gs["button"]
        small_blind = (dealer + 1) % num_players
        big_blind = (dealer + 2) % num_players

        # Build the community cards string.
        community_str = ", ".join(
            f"{card['rank']}{card['suit']}" for card in gs["visible_community_cards"]
        )

        # Build player status lines with roles included.
        player_lines = []
        for pid in range(num_players):
            role_labels = []
            if pid == dealer:
                role_labels.append("Dealer")
            if pid == small_blind:
                role_labels.append("Small Blind")
            if pid == big_blind:
                role_labels.append("Big Blind")
            role_str = f" ({'/'.join(role_labels)})" if role_labels else ""

            if pid in gs["folded_players"]:
                status = "folded"
            elif pid in gs["all_in_players"]:
                status = "all-in"
            else:
                status = "active"

            player_lines.append(
                f"Player {pid}{role_str}: {gs['player_chips'][pid]} chips, bet={gs['player_bets'][pid]}, status={status}"
            )

        # Compose the full observation message.
        current_pid = self.state.current_player_id
        hole = gs["player_hands"][current_pid]
        obs_message = (
            f"----- {round_text} - Turn: {turn_text} -----\n"
            f"Visible Community Cards: [{community_str}]\n"
            f"Pot: {gs['pot']}\n"
            + "\n".join(player_lines) +
            f"\nYour hole cards: {hole[0]['rank']}{hole[0]['suit']}, "
            f"{hole[1]['rank']}{hole[1]['suit']}"
            "\n-----------------------------------------"
        )

        # Send the observation only to the current (next) player.
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=current_pid,
            message=obs_message
        )


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the current player's action and advance the game state accordingly"""
        # Check if game is already complete
        if self.state.game_state["game_complete"]:
            reason = "The game is already complete. No further actions can be taken."
            self.state.set_invalid_move(player_id=self.state.current_player_id, reason=reason)
            return False, {"reason": reason}
            
        current_pid = self.state.current_player_id

        # Broadcast the action
        self.state.add_observation(from_id=current_pid, to_id=-1, message=action)

        # Parse and apply the action
        self._process_betting_action(action=action, player_id=current_pid)

        # Safeguard: validate round/betting round consistency
        gs = self.state.game_state
        if gs["round"] < 1 or gs["round"] > self.num_rounds or gs["betting_round"] < 0 or gs["betting_round"] > 3:
            # Log error and fix corrupted state
            err_msg = f"ERROR: Detected invalid game state. Round: {gs['round']}, Betting Round: {gs['betting_round']}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=err_msg)
            # Attempt to repair the state - use most recent valid values
            gs["round"] = max(1, min(gs["round"], self.num_rounds))
            gs["betting_round"] = max(0, min(gs["betting_round"], 3))

        # If the betting round is complete, advance game phase
        if not self.state.game_state["game_complete"]:
            if self._is_betting_round_complete() and not self._is_hand_over():
                self._advance_game_phase()

        # After all that, observe pot again
        self._observe_current_pot()

        return self.state.step(rotate_player=False)

    def _process_betting_action(self, action: str, player_id: int):
        """Parse action, validate, apply, and check if hand ends."""
        action_type, bet_amount = self._parse_action(action)
        gs = self.state.game_state

        if action_type == "invalid":
            reason=(
                    f"Player {player_id} did not provide a valid poker action.\n"
                    "Valid actions: '[Check]', '[Fold]', '[Call]', '[Bet <amount>]', '[Raise <amount>]'."
                )
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return

        # Apply the action
        self._apply_action(player_id, action_type, bet_amount)

        # Check if the hand ends immediately (folds or everyone all-in)
        if self._is_hand_over() and not gs["game_complete"]:
            self._handle_hand_completion()
        else:
            # Move action to next active player (if not already going to next round, etc.)
            next_player = self._get_next_active_player(player_id)
            self.state.manually_update_current_player(new_player_id=next_player)

    def _parse_action(self, action: str) -> Tuple[str, Optional[int]]:
        """Parse the player's action string into (action_type, bet_amount)."""
        gs = self.state.game_state
        gs["round_turn"] += 1

        # Attempt pattern matches
        if self.check_pattern.search(action):
            return ("check", None)
        if self.fold_pattern.search(action):
            return ("fold", None)
        if self.call_pattern.search(action):
            return ("call", None)
        if self.bet_pattern.search(action):
            match = self.bet_pattern.search(action)
            amount = int(match.group(1))
            return ("bet", amount)
        if self.raise_pattern.search(action):
            match = self.raise_pattern.search(action)
            amount = int(match.group(1))
            return ("raise", amount)

        return ("invalid", None)


    def _apply_action(self, player_id: int, action_type: str, bet_amount: Optional[int]):
        """
        Update the game state according to the player's action choice.
        Now with proper tracking of the last bettor and bet round completion.
        """
        gs = self.state.game_state
        
        # Initialize tracking fields if they don't exist
        if "last_bettor" not in gs:
            gs["last_bettor"] = -1
        if "bet_round_complete" not in gs:
            gs["bet_round_complete"] = False
            
        # Ensure players with 0 chips are considered all-in
        if gs["player_chips"][player_id] == 0 and player_id not in gs["all_in_players"]:
            gs["all_in_players"].add(player_id)
            self.state.add_observation(
                from_id=ta.GAME_ID, 
                to_id=-1, 
                message=f"Player {player_id} has 0 chips and is automatically considered all-in."
            )
        
        if action_type == "fold":
            gs["folded_players"].add(player_id)
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Player {player_id} folds.")
            
            # Check if this player was the last bettor
            if player_id == gs["last_bettor"]:
                # Find new last bettor or mark betting complete
                # For simplicity, we'll just mark it complete if the last bettor folds
                gs["bet_round_complete"] = True

        elif action_type == "check":
            if gs["current_bet"] > gs["player_bets"][player_id]:
                self.state.set_invalid_move(player_id=player_id, reason="Cannot check when there's a live bet to call.")
            else:
                gs["checked_players"].add(player_id)
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Player {player_id} checks.")
                
                # If we've gone all the way around to the last bettor (or start of betting)
                if gs["last_bettor"] == -1 or self._next_player_would_be_after_last_bettor(player_id):
                    gs["bet_round_complete"] = True

        elif action_type == "call":
            call_needed = gs["current_bet"] - gs["player_bets"][player_id]
            if call_needed < 0:
                # Player is trying to "call" when they've already matched or exceeded. Treat as check.
                gs["checked_players"].add(player_id)
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Player {player_id} checks (already matched)."
                )
                
                # Check if we've gone around since last bet
                if gs["last_bettor"] == -1 or self._next_player_would_be_after_last_bettor(player_id):
                    gs["bet_round_complete"] = True
                return

            # If they do not have enough chips, they go all in.
            if call_needed >= gs["player_chips"][player_id]:
                all_in_amount = gs["player_chips"][player_id]
                gs["player_bets"][player_id] += all_in_amount
                gs["player_chips"][player_id] = 0
                gs["pot"] += all_in_amount
                gs["all_in_players"].add(player_id)
                message=f"Player {player_id} calls all-in for {all_in_amount}."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                # Also update current bet if needed
                if gs["player_bets"][player_id] > gs["current_bet"]:
                    gs["current_bet"] = gs["player_bets"][player_id]
                    # This is effectively a raise, so update last bettor
                    gs["last_bettor"] = player_id
                    gs["bet_round_complete"] = False
            else:
                # Normal call
                gs["player_bets"][player_id] += call_needed
                gs["player_chips"][player_id] -= call_needed
                gs["pot"] += call_needed
                message=f"Player {player_id} calls {call_needed}."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                
                # Check if we've gone around since last bet
                if gs["last_bettor"] == -1 or self._next_player_would_be_after_last_bettor(player_id):
                    gs["bet_round_complete"] = True

        elif action_type in ("bet", "raise"):
            # Bet or raise resets the betting round
            gs["bet_round_complete"] = False
            
            # Calculate the total amount
            current_player_contribution = gs["player_bets"][player_id]
            if action_type == "bet":
                total_amount = bet_amount
            else:  # raise
                total_amount = gs["current_bet"] + bet_amount

            if total_amount <= gs["current_bet"] and action_type == "raise":
                reason=f"A raise must exceed the current bet of {gs['current_bet']}."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
                return

            # How much do they need to add
            needed = total_amount - current_player_contribution
            if needed < 0:
                reason="Bet/Raise amount is too small or negative."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
                return

            if needed > gs["player_chips"][player_id]:
                # All-in
                needed = gs["player_chips"][player_id]
                gs["all_in_players"].add(player_id)

            gs["player_bets"][player_id] += needed
            gs["player_chips"][player_id] -= needed
            gs["pot"] += needed

            if gs["player_bets"][player_id] > gs["current_bet"]:
                gs["current_bet"] = gs["player_bets"][player_id]
                # Update last bettor
                gs["last_bettor"] = player_id

            message=f"Player {player_id} {action_type}s {needed} (total={gs['player_bets'][player_id]})."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

    def _next_player_would_be_after_last_bettor(self, current_player_id: int) -> bool:
        """Check if the next player would be after the last player who bet/raised."""
        gs = self.state.game_state
        next_player = self._get_next_active_player(current_player_id)
        
        # If the last bettor has no chips, they shouldn't be considered as blocking the round completion
        if gs["last_bettor"] != -1 and gs["player_chips"][gs["last_bettor"]] == 0:
            # Automatically mark players with 0 chips as having acted
            return True
        
        # If next active player is the last bettor (or after them in case they're not active),
        # we've gone around
        if gs["last_bettor"] == -1:
            # No one has bet yet, so we're using position from start of betting
            return next_player == self._get_first_active_player_of_round()
        else:
            return next_player == gs["last_bettor"] or self._player_comes_after(next_player, gs["last_bettor"])

    def _get_first_active_player_of_round(self) -> int:
        """Get the first player who should act in the current betting round."""
        gs = self.state.game_state
        if gs["betting_round"] == 0:  # Pre-flop
            # First player after big blind
            return self._get_next_active_player((gs["button"] + 2) % self.state.num_players)
        else:
            # First player after button
            return self._get_next_active_player((gs["button"] + 1) % self.state.num_players)

    def _player_comes_after(self, player_id: int, reference_player_id: int) -> bool:
        """Check if player_id comes after reference_player_id in the current order of play."""
        # Check if player_id sits after reference_player_id at the table
        # This might need to be adapted based on your specific game flow
        gs = self.state.game_state
        num_players = self.state.num_players
        
        current = reference_player_id
        while True:
            current = (current + 1) % num_players
            if current == player_id:
                return True
            if current == reference_player_id:  # We've gone all the way around
                return False


    def _is_hand_over(self) -> bool:
        """
        Check if the current hand is over because:
          - only one player remains (others folded)
          - or all active players are all in
        """
        gs = self.state.game_state
        num_players = self.state.num_players

        # active = not folded and not out of chips
        active_players = [
            pid for pid in range(num_players)
            if pid not in gs["folded_players"] and gs["player_chips"][pid] > 0
        ]
        # If at any point a player is at 0 chips but hasn't folded, we consider them "all in"
        # so they remain "active" in a sense until showdown, but can't bet further.
        # We'll track that in gs["all_in_players"], but let's be consistent:
        for pid in range(num_players):
            if gs["player_chips"][pid] == 0 and pid not in gs["folded_players"]:
                gs["all_in_players"].add(pid)

        # If only one player is truly active or has not folded, the hand ends
        if len(active_players) <= 1:
            return True

        # If all active players are all_in, the hand ends immediately (skip further betting)
        if all(pid in gs["all_in_players"] for pid in active_players):
            return True

        return False

    def _handle_hand_completion(self):
        """When the hand ends early (folds) or due to all-ins, go directly to showdown, then next round or game end."""
        gs = self.state.game_state

        # Reveal all community cards
        gs["visible_community_cards"] = gs["community_cards"]

        self._handle_showdown()

        # Check if we can start a new hand
        if gs["round"] < self.num_rounds:
            # Check if at least two players still have chips
            players_with_chips = [pid for pid, chips in gs["player_chips"].items() if chips > 0]
            if len(players_with_chips) > 1:
                # New round
                gs["round"] += 1
                gs["betting_round"] = 0
                gs["button"] = (gs["button"] + 1) % self.state.num_players
                
                # Add a clear round transition message
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Starting Round {gs['round']} of {self.num_rounds}"
                )
                
                self._reset_round()
                return
        
        # Game is over - ensure we properly finalize it
        self.determine_winner()
        gs["game_complete"] = True

    def _handle_showdown(self):
        """
        Show all hole cards for the players who haven't folded,
        evaluate their best hands, and award pot to the winner(s).
        This version splits the pot evenly among ties, ignoring side pots.
        """
        gs = self.state.game_state
        num_players = self.state.num_players

        # Which players are still in?
        active_players = [
            pid for pid in range(num_players)
            if pid not in gs["folded_players"]
        ]
        if not active_players:
            # Corner case: everyone folded. Typically the last folder is the last to fold,
            # but in practice you might handle this differently. For simplicity, pot remains
            # unclaimed or we treat the last to fold as winner. We'll pick the last folder’s
            # prior player:
            # But let's handle it simply: no one wins? (unusual corner case)
            message="All players folded simultaneously. No pot awarded (corner case)."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            gs["pot"] = 0
            return

        if len(active_players) == 1:
            # Only one player didn't fold, so they get the pot
            winner = active_players[0]
            gs["player_chips"][winner] += gs["pot"]
            message=f"Player {winner} wins the pot by default (everyone else folded)."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            gs["pot"] = 0
            return

        # Multiple players -> showdown
        # Show each player's hole cards
        details = []
        for pid in active_players:
            hole = gs["player_hands"][pid]
            detail_str = (
                f"Player {pid} hole cards: "
                f"{hole[0]['rank']}{hole[0]['suit']} {hole[1]['rank']}{hole[1]['suit']}"
            )
            details.append(detail_str)
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message="Showdown:\n" + "\n".join(details))

        # Evaluate each player's best 5-card hand
        ranks_dict = {}
        for pid in active_players:
            hole_cards = gs["player_hands"][pid]
            community = gs["community_cards"]
            ranks_dict[pid] = self._evaluate_hand(hole_cards + community)

        # Find the best hand rank
        # hand rank is a tuple (hand_category, tie_breaker_list)
        # We do a max() by comparing these tuples
        best_rank = max(ranks_dict.values())
        # find all players that match that rank
        winners = [pid for pid in active_players if ranks_dict[pid] == best_rank]

        # Award the pot
        if len(winners) == 1:
            w = winners[0]
            gs["player_chips"][w] += gs["pot"]
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Player {w} wins the pot of {gs['pot']}."
            )
            gs["pot"] = 0
        else:
            # tie - split pot evenly
            split_amt = gs["pot"] // len(winners)
            for w in winners:
                gs["player_chips"][w] += split_amt
            remainder = gs["pot"] % len(winners)  # leftover if not divisible
            gs["pot"] = 0
            message=f"Tie among players {winners}. Each gets {split_amt} chips."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            if remainder > 0:
                # If you'd like a random leftover awarding, or skip it
                # We'll just discard it or give it to the first winner:
                gs["player_chips"][winners[0]] += remainder

    def _evaluate_hand(self, cards: List[Dict[str, str]]) -> Tuple[int, List[int]]:
        """
        Evaluate the best 5-card hand from 7 cards (2 hole + 5 community).
        Returns a tuple: (hand_ranking_category, tie_breaker_list),
        where hand_ranking_category (int) is:
           9 = Straight Flush
           8 = Four of a Kind
           7 = Full House
           6 = Flush
           5 = Straight
           4 = Three of a Kind
           3 = Two Pair
           2 = One Pair
           1 = High Card
        The tie_breaker_list is used to compare hands in the same category.
        """
        # Convert to numerical ranks & suits
        ranks = [self.rank_values[c["rank"]] for c in cards]
        suits = [c["suit"] for c in cards]

        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Check for flush
        flush_suit = None
        for s, count in suit_counts.items():
            if count >= 5:
                flush_suit = s
                break

        distinct_ranks = sorted(set(ranks))
        # Check for straight (including ace-low case)
        straight, straight_high = self._check_straight(distinct_ranks)

        # 1) Check Straight Flush
        if flush_suit and straight:
            # collect flush cards
            flush_cards = [r for r, s in zip(ranks, suits) if s == flush_suit]
            flush_cards = sorted(set(flush_cards))
            sf, sf_high = self._check_straight(flush_cards)
            if sf:
                # hand rank 9, tie-break on the top card
                return (9, [sf_high])

        # 2) Four of a Kind
        if 4 in rank_counts.values():
            quad = max(r for r, cnt in rank_counts.items() if cnt == 4)
            kicker = max(r for r in rank_counts if r != quad)
            return (8, [quad, kicker])

        # 3) Full House (3 of a kind + a pair)
        if 3 in rank_counts.values():
            triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
            # see if there's a pair among the others
            pairs = [r for r, cnt in rank_counts.items() if cnt >= 2 and r != triple]
            if pairs:
                pair = max(pairs)
                return (7, [triple, pair])

        # 4) Flush
        if flush_suit:
            flush_cards = [r for r, s in zip(ranks, suits) if s == flush_suit]
            flush_cards.sort(reverse=True)
            return (6, flush_cards[:5])

        # 5) Straight
        if straight:
            return (5, [straight_high])

        # 6) Three of a kind
        if 3 in rank_counts.values():
            triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
            kickers = sorted((r for r in ranks if r != triple), reverse=True)
            return (4, [triple] + kickers[:2])

        # 7) Two Pair
        pairs = [r for r, cnt in rank_counts.items() if cnt == 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            top_two = pairs[:2]
            kicker_candidates = [r for r in ranks if r not in top_two]
            kicker = max(kicker_candidates) if kicker_candidates else 0
            return (3, top_two + [kicker])

        # 8) One Pair
        if len(pairs) == 1:
            p = pairs[0]
            kickers = sorted((r for r in ranks if r != p), reverse=True)
            return (2, [p] + kickers[:3])

        # 9) High card
        sorted_ranks = sorted(ranks, reverse=True)
        return (1, sorted_ranks[:5])

    def _check_straight(self, sorted_ranks: List[int]) -> Tuple[bool, int]:
        """
        Given a sorted list of distinct ranks, determine if there's a straight.
        Returns (is_straight, high_card_of_straight).
        Also checks Ace-low case (A,2,3,4,5).
        """
        if len(sorted_ranks) < 5:
            return False, -1

        # Check Ace-low: ranks might include 14 for Ace
        # if we have [14, 2, 3, 4, 5], treat that as a straight (5-high)
        if {14, 2, 3, 4, 5}.issubset(set(sorted_ranks)):
            return True, 5  # 5 is the high card in A2345

        for i in range(len(sorted_ranks) - 4):
            seq = sorted_ranks[i : i + 5]
            if seq[-1] - seq[0] == 4:
                return True, seq[-1]
        return False, -1

    def _is_betting_round_complete(self):
        """
        Check if the current betting round is complete.
        A betting round is complete when:
        1. There are 0-1 active players left, or
        2. All active players have either:
        a. Checked (when there's no bet), or
        b. Called the current bet AND everyone has had a chance to act since the last bet/raise
        """
        gs = self.state.game_state
        
        # Add a new field to track the last bettor if it doesn't exist
        if "last_bettor" not in gs:
            gs["last_bettor"] = -1  # Initialize to an invalid player ID
        
        # Add a field to track if we've gone around the table since the last bet
        if "bet_round_complete" not in gs:
            gs["bet_round_complete"] = False
            
        # Get truly active players (not folded, not all-in, and have chips)
        active_players = [
            pid for pid in range(self.state.num_players)
            if pid not in gs["folded_players"] and 
            pid not in gs["all_in_players"] and
            gs["player_chips"][pid] > 0
        ]
        
        # If 0-1 active players, round is complete
        if len(active_players) <= 1:
            return True

        current_bet = gs["current_bet"]
        
        if current_bet == 0:
            # If no bet, all active players must have checked
            return all(pid in gs["checked_players"] for pid in active_players)
        else:
            # All players must match the current bet
            all_matched = all(gs["player_bets"][pid] == current_bet for pid in active_players)
            
            # Everyone must have had a chance to act since the last bet/raise
            return all_matched and gs["bet_round_complete"]


    # def _advance_game_phase(self):
    #     gs = self.state.game_state
    #     num_players = self.state.num_players

    #     if gs["betting_round"] < 3:
    #         # Advance to the next betting round
    #         gs["betting_round"] += 1

    #         # Reset betting-specific state
    #         gs["current_bet"] = 0
    #         gs["player_bets"] = {pid: 0 for pid in range(num_players)}
    #         gs["checked_players"] = set()  # Make sure this is cleared
    #         gs["round_turn"] = 0  # Reset turn counter

    #         # Reveal appropriate community cards
    #         if gs["betting_round"] == 1:
    #             # Flop: show first 3 cards
    #             gs["visible_community_cards"] = gs["community_cards"][:3]
    #         elif gs["betting_round"] == 2:
    #             # Turn: show 4 cards 
    #             gs["visible_community_cards"] = gs["community_cards"][:4]
    #         elif gs["betting_round"] == 3:
    #             # River: show all 5 cards
    #             gs["visible_community_cards"] = gs["community_cards"][:5]

    #         # Always restart with player after button
    #         next_player = self._get_next_active_player((gs["button"] + 1) % num_players)
    #         self.state.manually_update_current_player(new_player_id=next_player)
    #     else:
    #         # End of all betting rounds
    #         self._handle_showdown()
            
    #         # Check if we should start new hand
    #         if gs["round"] < self.num_rounds:
    #             remaining_players = [pid for pid, chips in gs["player_chips"].items() if chips > 0]
    #             if len(remaining_players) > 1:
    #                 gs["round"] += 1  # Increment the hand number
    #                 gs["betting_round"] = 0  # Reset betting round
    #                 gs["button"] = (gs["button"] + 1) % num_players  # Move button
    #                 self._reset_round()
    #             else:
    #                 self.determine_winner()
    #                 gs["game_complete"] = True
    #         else:
    #             self.determine_winner()
    #             gs["game_complete"] = True


    def _advance_game_phase(self):
        """Move to the next phase of the game (flop, turn, river, or next hand)"""
        gs = self.state.game_state
        num_players = self.state.num_players

        if gs["betting_round"] < 3:
            # Advance to the next betting round
            gs["betting_round"] += 1

            # Reset betting-specific state
            gs["current_bet"] = 0
            gs["player_bets"] = {pid: 0 for pid in range(num_players)}
            gs["checked_players"] = set()  # Make sure this is cleared
            gs["round_turn"] = 0  # Reset turn counter
            
            # Reset betting round tracking
            gs["last_bettor"] = -1
            gs["bet_round_complete"] = False

            # Reveal appropriate community cards
            if gs["betting_round"] == 1:
                # Flop: show first 3 cards
                gs["visible_community_cards"] = gs["community_cards"][:3]
                card_reveal = ", ".join([f"{c['rank']}{c['suit']}" for c in gs["community_cards"][:3]])
                reveal_message = f"Dealing the Flop: {card_reveal}"
            elif gs["betting_round"] == 2:
                # Turn: show 4 cards 
                gs["visible_community_cards"] = gs["community_cards"][:4]
                card_reveal = f"{gs['community_cards'][3]['rank']}{gs['community_cards'][3]['suit']}"
                reveal_message = f"Dealing the Turn: {card_reveal}"
            elif gs["betting_round"] == 3:
                # River: show all 5 cards
                gs["visible_community_cards"] = gs["community_cards"][:5]
                card_reveal = f"{gs['community_cards'][4]['rank']}{gs['community_cards'][4]['suit']}"
                reveal_message = f"Dealing the River: {card_reveal}"
            
            # Send a message about the newly revealed cards
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=reveal_message
            )

            # Always restart with first active player after the button for post-flop rounds
            next_player = self._get_first_active_player_of_round()
            self.state.manually_update_current_player(new_player_id=next_player)
            
            # After setting the next player, show them the current pot and cards
            # This ensures they see the updated community cards
            self._observe_current_pot()
        else:
            # End of all betting rounds
            self._handle_showdown()
            
            # Check if we should start new hand
            if gs["round"] < self.num_rounds:
                remaining_players = [pid for pid, chips in gs["player_chips"].items() if chips > 0]
                if len(remaining_players) > 1:
                    gs["round"] += 1  # Increment the hand number
                    gs["betting_round"] = 0  # Reset betting round
                    gs["button"] = (gs["button"] + 1) % num_players  # Move button
                    
                    # Add a clear round transition message
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=f"Starting Round {gs['round']} of {self.num_rounds}"
                    )
                    
                    self._reset_round()
                else:
                    self.determine_winner()
                    gs["game_complete"] = True
            else:
                self.determine_winner()
                gs["game_complete"] = True
                

    def _get_next_active_player(self, current_player: int) -> int:
        """
        Return the next player ID that is neither folded nor all-in.
        If all but one are folded/all-in, we can just return that one or
        the same player. (The logic calling this will handle if the hand ends.)
        """
        gs = self.state.game_state
        num_players = self.state.num_players
        candidate = (current_player + 1) % num_players

        while candidate != current_player:
            if (candidate not in gs["folded_players"]
                and candidate not in gs["all_in_players"]):
                return candidate
            candidate = (candidate + 1) % num_players

        # If we circle back, it means all others are folded/all-in,
        # so we return the original player (which might also be folded/all-in).
        return current_player


    def determine_winner(self) -> Tuple[List[int], int]:
        """
        Determine the overall game winner(s) based on who has the most chips.
        If there's a tie, mark them all as winners. 
        Returns (list_of_winner_ids, chips_of_winner).
        """
        gs = self.state.game_state
        final_chips = gs["player_chips"]
        max_chips = max(final_chips.values())

        # Might be multiple
        winners = [pid for pid, amt in final_chips.items() if amt == max_chips]
        if len(winners) == 1:
            w = winners[0]
            gs["winner"] = w
            gs["winning_chips"] = max_chips
            reason = f"Player {w} wins the tournament with {max_chips} chips."
            self.state.set_winners(player_ids=[w], reason=reason)
            # Important: Mark game as complete!
            gs["game_complete"] = True
            
            # Announce final standings
            standings_msg = ", ".join([f"Player {pid} finished with {chips} chips" for pid, chips in final_chips.items()])
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Tournament complete! {standings_msg}"
            )
        else:
            # tie
            tie_msg = f"Players {winners} tie with {max_chips} chips each."
            self.state.set_draw(reason=tie_msg)
            gs["game_complete"] = True
            
        return (winners, max_chips)
