import re, random 
from collections import Counter 
from typing import Any, Dict, List, Optional, Tuple 

import textarena as ta 


class PokerEnv(ta.Env):
    def __init__(
        self,
        num_rounds: int = 10,
        starting_chips: int = 1_000,
        small_blind: int = 10,
        big_blind: int = 20,
    ):
        """
        Initialize the Texas Hold'em Poker game.

        Args:
            num_rounds (int): Number of rounds to play
            starting_chips (int): Starting chips for each player
            small_blind (int): Small blind amount
            big_blind (int): Big blind amount
        """
        # Game configuration
        self.num_rounds = num_rounds 
        self.starting_chips = starting_chips 
        self.small_blind = small_blind 
        self.big_blind = big_blind 

        # Card setup
        self.suits = ["♠", "♥", "♦", "♣"]
        self.ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.rank_values = {rank: idx for idx, rank in enumerate(self.ranks)}
        
        # Initialize game state
        self.state = ta.State(
            num_players=2,
            max_turns=num_rounds, # betting rounds per hand
            check_truncated=False, # no turn limit
        )

        # Define action patterns
        self.check_pattern = re.compile(r"\[Check\]", re.IGNORECASE)
        self.fold_pattern = re.compile(r"\[Fold\]", re.IGNORECASE)
        self.call_pattern = re.compile(r"\[Call.*\]", re.IGNORECASE)
        self.bet_pattern = re.compile(r"\[bet (\d+)\]", re.IGNORECASE)
        self.raise_pattern = re.compile(r"\[raise (\d+)\]", re.IGNORECASE)

    @property
    def offline_renderer(self):
        from textarena.envs.two_player.Poker.render.renderer import PokerRenderer
        return PokerRenderer 

    @property
    def terminal_render_keys(self):
        return [
            ["player_hands", 0],
            ["player_hands", 1],
            "player_chips", 
            "visible_community_cards", 
            "player_bets"
        ]


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the initial prompt explaining the poker game rules and format to a player.
        This is called once at the start of the game.

        Args:
            player_id (int): ID of the player (0 or 1)
            game_state (Dict[str, Any]): Initial game state

        Returns:
            str: The initial prompt explaining the game rules and format
        """
        prompt = (
            f"You are Player {player_id} in Texas Hold'em Poker.\n"
            f"Game Information:\n"
            f"- This is a {self.num_rounds}-round game\n"
            f"- Each player starts with {self.starting_chips} chips\n"
            f"- Small blind is {self.small_blind} chips\n"
            f"- Big blind is {self.big_blind} chips\n"
            f"- Players must call the current bet to stay in the hand\n\n"
            "Available Actions:\n"
            "  [Check] - When there's no bet to call\n"
            "  [Call] - Match the current bet\n"
            "  [Fold] - Give up your hand\n"
            "  [Bet <amount>] - Make a new bet, e.g. [Bet 100]\n"
            "  [Raise <amount>] - Increase the current bet, e.g. [Raise 200]\n\n"
            "The Player with the most chips at the end wins"
        )
        return prompt

    def reset(self, seed: Optional[int] = None):
        """ Reset the full game to its initial state """
        if seed is not None:
            random.seed(seed)

        self.state.reset(
            game_state={
                "round": 1,
                "betting_round": 0,  # 0: pre-flop, 1: flop, 2: turn, 3: river
                "player_chips": {0: self.starting_chips, 1: self.starting_chips},
                "player_hands": {0: [], 1: []},
                "community_cards": [],
                "visible_community_cards": [],
                "pot": 0,
                "current_bet": 0,
                "player_bets": {0: 0, 1: 0},
                "button": 0,  # Dealer button position
                "folded_players": set(),
                "all_in_players": set(),
                "checked_players": set(),  # Track who has checked this betting round
                "round_turn": 0
            },
            player_prompt_function=self._generate_player_prompt,
            executable_on_reset=[self._reset_round]
        )

    def _create_deck(self) -> List[Dict[str, str]]:
        """Create a standard 52-card deck."""
        return [{"rank": rank, "suit": suit} for suit in self.suits for rank in self.ranks]

    def _reset_round(self):
        """ TODO """
        # deal new hand
        deck = self._create_deck()
        random.shuffle(deck)

        for player_id in [0, 1]:
            self.state.game_state["player_hands"][player_id] = [deck.pop() for _ in range(2)]
            card1, card2 = self.state.game_state["player_hands"][player_id]
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=f"Your cards are: [{card1['rank']}{card1['suit']}, {card2['rank']}{card2['suit']}]",
                for_logging=True
            )

        # Deal community cards (face down initially)
        self.state.game_state["community_cards"] = [deck.pop() for _ in range(5)]
        self.state.game_state["visible_community_cards"] = []

        # post the blinds
        sb_player = (self.state.game_state["button"] + 1) % 2
        bb_player = (self.state.game_state["button"] + 2) % 2


        # post small blind
        self.state.game_state["player_chips"][sb_player] -= self.small_blind 
        self.state.game_state["player_bets"][sb_player] = self.small_blind 
        self.state.game_state["pot"] += self.small_blind 

        # post big blind 
        self.state.game_state["player_chips"][bb_player] -= self.big_blind
        self.state.game_state["player_bets"][bb_player] = self.big_blind 
        self.state.game_state["pot"] += self.big_blind 

        self.state.game_state["current_bet"] = self.big_blind 
        self.state.game_state["round_turn"] = 0

        # set current player (bb_player + 1)
        next_player_id = (bb_player + 1) % 2
        self.state.manually_updated_current_player(new_player_id=next_player_id)
        # self.state.current_player_id = (bb_player + 1) % 2

        self._observe_current_pot()


    def _observe_current_pot(self):
        """ TODO """
        community_card_str = ", ".join([f"{card['rank']}{card['suit']}" for card in self.state.game_state["visible_community_cards"]])
        current_pot_message = (
            f"Visible Community Cards: [{community_card_str}]\n"
            f"Current Pot: {self.state.game_state['pot']}\n"
            f"Player 0 chips: {self.state.game_state['player_chips'][0]}; current bet: {self.state.game_state['player_bets'][0]}\n"
            f"Player 1 chips: {self.state.game_state['player_chips'][1]}; current bet: {self.state.game_state['player_bets'][1]}\n"
        )

        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1, # Broadcast to all
            message=current_pot_message
        )

    def step(self,action: str) -> Tuple[bool, ta.Info]:
        """Process a player's action."""
        # add to observations
        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=-1, # Broadcast to all
            message=action
        )

        # process the current betting round
        self._process_betting_action(action=action, player_id=self.state.current_player_id)


        # If betting round is complete, move to next phase
        if self._is_betting_round_complete():
            self._advance_game_phase()

        self._observe_current_pot()
        return self.state.step(rotate_player=False)

    def _parse_action(self, action: str) -> Tuple[str, Optional[int]]:
        """
        Parse the player's action string into action type and bet amount.
        
        Returns:
            Tuple[str, Optional[int]]: Action type and bet amount (if applicable)
        """
        action = action#.strip()

        # try to match the patterns
        check_match = self.check_pattern.search(action)
        fold_match = self.fold_pattern.search(action)
        call_match = self.call_pattern.search(action)
        bet_match = self.bet_pattern.search(action)
        raise_match = self.raise_pattern.search(action)

        self.state.game_state["round_turn"] += 1

        # check patterns in order 
        if check_match:
            return "check", None 
        elif fold_match:
            return "fold", None 
        elif call_match:
            return "call", None 
        elif bet_match:
            amount = int(bet_match.group(1))
            if amount >= self.state.game_state["current_bet"]:
                return "bet", amount
            else:
                return "invalid_bet", None
        elif raise_match:
            amount = int(raise_match.group(1))
            return "raise", amount 
        
        return "invalid", None 

    def _process_betting_action(self, player_id: int, action: str):
        """ TODO """
        # parse and validate the action 
        action_type, bet_amount = self._parse_action(action)

        # check if valid
        if "invalid" in action_type:
            if "invalid_bet" == action_type:
                reason = (
                f"Player {player_id} did not provide a valid poker action.",
                f"A placed bet cannot decrease the current bet."
            )
            else:
                reason=(
                    f"Player {player_id} did not provide a valid poker action.",
                    f"You need to either [check], [fold], [call], [bet <amount>] or [raise <amount]."
                )
            self.state.set_invalid_move(
                player_id=player_id,
                reason=reason
            )
            return

        # apply the action to update chips, pot, etc.
        self._apply_action(player_id, action_type, bet_amount)

        # Check if hand is over (everyone folded or all-in)
        if self._is_hand_over():
            self._handle_hand_completion()

        # move to next player if betting continues
        else:
            next_player = self._get_next_active_player(player_id)
            # self.state.current_player_id = next_player 

            # try setting the next player 
            self.state.manually_updated_current_player(new_player_id=next_player)


    def _is_hand_over(self) -> bool:
        """
        Check if the current hand is over due to folds or all-in situations.
        
        Returns:
            bool: True if hand is over, False otherwise
        """
        active_players = set(range(2)) - self.state.game_state["folded_players"] - self.state.game_state["all_in_players"]
        
        # Hand is over if only one player remains
        if len(active_players) == 1:
            return True
            
        # Hand is over if all remaining players are all-in
        if all(player in self.state.game_state["all_in_players"] 
               for player in active_players):
            return True

        # pot hasn't increased 
            
        return False

    def _handle_hand_completion(self):
        """Handle the completion of a hand, either through showdown or single player remaining."""
        # Reveal all community cards
        self.state.game_state["visible_community_cards"] = self.state.game_state["community_cards"]
        
        # Handle showdown or give pot to last player
        self._handle_showdown()
        
        # Check if game should continue
        if self.state.game_state["round"] < self.num_rounds and all(
            chips > 0 for chips in self.state.game_state["player_chips"].values()
        ):
            # Start new hand
            self.state.game_state["round"] += 1
            self.state.game_state["betting_round"] = 0
            self.state.game_state["button"] = (self.state.game_state["button"] + 1) % 2
            self.state.game_state["folded_players"] = set()
            self.state.game_state["all_in_players"] = set()
            self.state.game_state["checked_players"] = set()
            self._reset_round()
        else:
            # Game is complete, determine winner
            self.determine_winner()
            self.state.game_state["game_complete"] = True
            
            

    def _evaluate_hands(self, active_players: set[int]) -> int:
        """
        Evaluate the poker hands of active players and determine the winner.
        
        Args:
            active_players (set[int]): Set of players still in the hand
        
        Returns:
            int: Player ID of the winner
        """
        # show the cards of all active players
        cards = '\n\t'.join([
            f"Player {p}: " + self.state.game_state["player_hands"][p][0]["rank"]+self.state.game_state["player_hands"][p][0]["suit"]+", "+\
            self.state.game_state["player_hands"][p][1]["rank"]+self.state.game_state["player_hands"][p][1]["suit"]
            for p in active_players
        ])

        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1, # Broadcast to all
            message=f"Showdown. Player cards:\n\t{cards}\n"
        )

        community_cards = self.state.game_state["community_cards"]
        hand_rankings = {}
        
        for player in active_players:
            hole_cards = self.state.game_state["player_hands"][player]
            hand_rankings[player] = self._evaluate_single_hand(hole_cards + community_cards)
        
        # Return player with the best hand
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {max(hand_rankings.items(), key=lambda x: x[1])[0]} wins the round."
        )
        return max(hand_rankings.items(), key=lambda x: x[1])[0]


    def _evaluate_single_hand(self, cards: List[Dict[str, str]]) -> Tuple[int, List[int]]:
        """
        Evaluate a single poker hand (7 cards) and return its ranking.
        
        Returns:
            Tuple[int, List[int]]: (hand_type_rank, [value1, value2, ...])
            where hand_type_rank is:
            9: Straight Flush
            8: Four of a Kind
            7: Full House
            6: Flush
            5: Straight
            4: Three of a Kind
            3: Two Pair
            2: One Pair
            1: High Card
        """
        # Convert cards to ranks and suits
        ranks = [self.rank_values[card["rank"]] for card in cards]
        suits = [card["suit"] for card in cards]
        
        # Check for flush
        suit_counts = Counter(suits)
        flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
        
        # Get rank counts for pairs, trips, etc
        rank_counts = Counter(ranks)
        
        # Check for straight
        distinct_ranks = sorted(set(ranks))
        straight = False
        straight_high = None
        
        # Handle Ace-low straight specially
        if set([14, 2, 3, 4, 5]).issubset(set(ranks)):  # Changed from [0,1,2,3,12] to standard values
            straight = True
            straight_high = 5  # Changed from 3 to 5 for proper high card
        
        # Check normal straights
        for i in range(len(distinct_ranks) - 4):
            if distinct_ranks[i+4] - distinct_ranks[i] == 4:
                straight = True
                straight_high = distinct_ranks[i+4]
        
        # Evaluate hand type from highest to lowest
        # Straight Flush
        if flush_suit and straight:
            flush_cards = [c for c in cards if c["suit"] == flush_suit]
            flush_ranks = sorted([self.rank_values[c["rank"]] for c in flush_cards])
            # Check if flush cards form a straight
            for i in range(len(flush_ranks) - 4):
                if flush_ranks[i+4] - flush_ranks[i] == 4:
                    return (9, [flush_ranks[i+4]])  # Return highest card of straight flush
            # Check Ace-low straight flush
            if set([14, 2, 3, 4, 5]).issubset(set(flush_ranks)):
                return (9, [5])
        
        # Four of a Kind
        if 4 in rank_counts.values():
            quads = max(r for r, count in rank_counts.items() if count == 4)
            kicker = max(r for r in ranks if r != quads)  # Changed to ensure we get highest kicker
            return (8, [quads, kicker])
        
        # Full House
        if 3 in rank_counts.values():
            trips = max(r for r, count in rank_counts.items() if count == 3)
            pairs = [r for r, count in rank_counts.items() if count >= 2 and r != trips]
            if pairs:  # If we have at least one pair
                return (7, [trips, max(pairs)])
        
        # Flush
        if flush_suit:
            flush_ranks = sorted([self.rank_values[c["rank"]] for c in cards if c["suit"] == flush_suit], reverse=True)
            return (6, flush_ranks[:5])
        
        # Straight
        if straight:
            return (5, [straight_high])
        
        # Three of a Kind
        if 3 in rank_counts.values():
            trips = max(r for r, count in rank_counts.items() if count == 3)
            kickers = sorted([r for r in ranks if r != trips], reverse=True)  # Changed to exclude trips value
            return (4, [trips] + kickers[:2])
        
        # Two Pair
        pairs = [r for r, count in rank_counts.items() if count == 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kickers = [r for r in ranks if rank_counts[r] == 1]
            kicker = max(kickers) if kickers else min(pairs)  # Handle case where we might have three pairs
            return (3, pairs[:2] + [kicker])
        
        # One Pair
        if 2 in rank_counts.values():
            pair = max(r for r, count in rank_counts.items() if count == 2)
            kickers = sorted([r for r in ranks if r != pair], reverse=True)  # Changed to exclude pair value
            return (2, [pair] + kickers[:3])
        
        # High Card
        high_cards = sorted(ranks, reverse=True)
        return (1, high_cards[:5])


    def _is_betting_round_complete(self) -> bool:
        """
        Check if the current betting round is complete.
        A betting round is complete when either:
        1. All active players have matched the highest bet
        2. All active players have checked (when there's no bet)
        """
        active_players = set(range(2)) - self.state.game_state["folded_players"]
        current_bet = self.state.game_state["current_bet"]
        
        if current_bet == 0:
            # If no bets, everyone needs to check
            return self.state.game_state["checked_players"] == active_players
        else:
            # If there are bets, everyone needs to match, and if this is the 
            # first round, if the amount is equal to the blind
            # the big-blind needs to have checked
            if all(
                self.state.game_state["player_bets"][pid] == current_bet
                for pid in active_players 
                if pid not in self.state.game_state["all_in_players"]
            ):
                # make sure every player had a turn
                if self.state.game_state["round_turn"] >= 2:
                    return True 
                return False 


    def _advance_game_phase(self):
        """ TODO """
        if self.state.game_state["betting_round"] < 3: # Still in current hand
            # Move to next betting round
            self.state.game_state["betting_round"] += 1

            # Reset betting state
            self.state.game_state["current_bet"] = 0
            self.state.game_state["player_bets"] = {0: 0, 1: 0}
            self.state.game_state["checked_players"] = set()

            # Reveal community cards for the new phase 
            visible_cards = {
                1: 3,  # Flop
                2: 4,  # Turn
                3: 5,  # River
            }.get(self.state.game_state["betting_round"], 0)

            self.state.game_state["visible_community_cards"] = self.state.game_state["community_cards"][:visible_cards]
        
        else: # Hand is complete
            self._handle_showdown()

            # Check if game should continue 
            if self.state.game_state["round"] < self.num_rounds and all(
                chips > 0 for chips in self.state.game_state["player_chips"].values()
            ):
                # Start new hand 
                self.state.game_state["round"] += 1
                self.state.game_state["betting_round"] = 0
                self.state.game_state["button"] = (self.state.game_state["button"] + 1) % 2
                self.state.game_state["folded_players"] = set()
                self.state.game_state["all_in_players"] = set()
                self.state.game_state["checked_players"] = set()
                self.state.game_state["visible_community_cards"] = []
                self._reset_round()
            else:
                # Game is complete
                self.determine_winner()
                self.state.game_state["game_complete"] = True

    def _handle_showdown(self):
        """ TODO """
        active_players = set(range(2)) - self.state.game_state["folded_players"]

        if len(active_players) > 1:
            # evaluate hands and determine winner
            winner = self._evaluate_hands(active_players)
        else:
            # Last player remaining gets pot 
            winner = active_players.pop()

        self.state.game_state["player_chips"][winner] += self.state.game_state["pot"]
        self.state.game_state["pot"] = 0


    def _get_next_active_player(self, current_player_id: int) -> int:
        """ TODO """
        next_player = (current_player_id + 1) % 2

        # Skip players who have folded or are all-in
        while (
            next_player in self.state.game_state["folded_players"] or 
            next_player in self.state.game_state["all_in_players"]
        ):
            next_player = (next_player + 1) % 2

        return next_player 


    def determine_winner(self) -> Tuple[int, int]:
        """
        Determine the overall winner of the game and their final chip count.
        
        Returns:
            Tuple[int, int]: (winner_id, final_chips)
        """

        # Get final chip counts
        final_chips = self.state.game_state["player_chips"]

        chips_player_0 = final_chips[0]
        chips_player_1 = final_chips[1]

        if chips_player_0 == chips_player_1:
            self.state.set_draw(
                reason=f"Both players finished the game with the same number of chips."
            )

        else:
            winner_id = 0 if chips_player_0 > chips_player_1 else 1
            winner_chips = final_chips[winner_id]

            self.state.game_state["game_complete"] = True
            self.state.game_state["winner"] = winner_id
            self.state.game_state["winning_chips"] = winner_chips

            self.state.set_winners(
                player_ids=[winner_id],
                reason=f"Player {winner_id} won by gaining the most chips"
            )
        


    def _apply_action(self, player_id: int, action_type: str, bet_amount: Optional[int]):
        """
        Apply the parsed action to the game state.
        """
        if action_type == "fold":
            self.state.game_state["folded_players"].add(player_id)
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1, # Broadcast to all
                message=f"Player {player_id} has folded."
            )
            
        elif action_type == "check":
            if self.state.game_state["current_bet"] > self.state.game_state["player_bets"][player_id]:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Player {player_id}. Cannot check when there's a bet to call"
                )
            else:
                self.state.game_state["checked_players"].add(player_id)
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Player {player_id} has checked."
                )
            
                
        elif action_type == "call":
            call_amount = self.state.game_state["current_bet"] - self.state.game_state["player_bets"][player_id]

            if call_amount == 0:
                # treat as check
                self.state.game_state["checked_players"].add(player_id)
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Player {player_id} has checked."
                )

            elif call_amount > self.state.game_state["player_chips"][player_id]:
                # Player goes all-in
                call_amount = self.state.game_state["player_chips"][player_id]
                self.state.game_state["all_in_players"].add(player_id)

                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1, # Broadcast to all
                    message=f"Player {player_id} is going all in."
                )
            else:
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1, # Broadcast to all
                    message=f"Player {player_id} has called ({call_amount})."
                )
            
                

                
            self.state.game_state["player_chips"][player_id] -= call_amount
            self.state.game_state["player_bets"][player_id] += call_amount
            self.state.game_state["pot"] += call_amount
            
        elif action_type in ["bet", "raise"]:    
            # Calculate total amount player needs to put in
            total_amount = bet_amount
            if action_type == "raise":
                total_amount += self.state.game_state["current_bet"]


            if total_amount > self.state.game_state["player_chips"][player_id]:
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Player {player_id}. Bet amount exceeds available chips."
                )
            
                
            # Handle the bet
            current_bet = self.state.game_state["player_bets"][player_id]
            amount_to_add = total_amount - current_bet

            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1, # Broadcast to all
                message=f"Player {player_id} has raised/bet ({amount_to_add})."
            )
            
            self.state.game_state["player_chips"][player_id] -= amount_to_add
            self.state.game_state["player_bets"][player_id] = total_amount
            self.state.game_state["pot"] += amount_to_add
            self.state.game_state["current_bet"] = total_amount
            
            # Check for all-in
            if self.state.game_state["player_chips"][player_id] == 0:
                self.state.game_state["all_in_players"].add(player_id)

