import re, random
from typing import Optional, Dict, Any, List, Tuple

import textarena as ta


class BullshitEnv(ta.Env):
    """
    A minimal text-based implementation of the Bullshit (Cheat) card game using textarena.

    Example actions a player might enter:
      - [<declared_rank> <card_idx_1> <card_idx_2> ... ]
      - [BULLSHIT]     -> The player calls Bullshit on the last play.
      - [PASS]         -> The player chooses not to call Bullshit and moves on 
                         (optional, if you want a 'no call' action).
    """

    # Regex patterns to detect user actions
    BULLSHIT_PATTERN = re.compile(r"^\[\s*BULLSHIT\s*\]$", re.IGNORECASE)
    PASS_PATTERN     = re.compile(r"^\[\s*PASS\s*\]$",    re.IGNORECASE)
    PLAY_PATTERN     = re.compile(
        r"^\[\s*"
        r"(?P<rank>\w+)"            # the declared rank (e.g. "3", "A", "K", etc.)
        r"\s+"
        r"(?P<indexes>(?:\d+\s*)+)" # one or more indexes, e.g. "0" or "0 2 3"
        r"\]$",
        re.IGNORECASE
    )

    # Card setup
    suits = ["♠", "♥", "♦", "♣"]  # suit symbols
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

    def __init__(self):
        super().__init__()
        # rank_values: e.g. {"A": 1, "2": 2, ..., "K": 13}
        self.rank_values = {rank: i + 1 for i, rank in enumerate(self.ranks)}

        self.action_space = {i: re.compile(r"^\[\s*BULLSHIT\s*\]$|^\[\s*PASS\s*\]$|^\[\s*(?P<rank>\w+)\s+(?P<indexes>(?:\d+\s*)+)\]$", re.IGNORECASE) for i in range(8)}

    def reset(self, num_players: int, seed: Optional[int] = None) -> None:
        """ Reset the environment for a new Bullshit game """
        self.state = ta.State(num_players=num_players, min_players=2, max_players=8)

        # Create deck (list of dicts), shuffle, and deal
        deck = [{"rank": rank, "suit": suit} for suit in self.suits for rank in self.ranks]
        random.shuffle(deck)

        # Initialize game state
        game_state = {
            "phase": "play", 
            "next_players": [],
            "hands": {}, 
            "pile": [], 
            "last_played_cards": [],
            "last_declared_rank": None, 
            "current_rank": 1 
        }

        # Deal cards to all players
        cpid = 0
        while len(deck) > 0:
            game_state["hands"].setdefault(cpid, [])
            game_state["hands"][cpid].append(deck.pop())
            cpid = (cpid + 1) % num_players

        # Initialize textarena state
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

        # Show each player their initial hand
        self._observer_current_state(broadcast=True)

        # Start with player 0
        self.state.manually_update_current_player(new_player_id=0)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Provide a quick summary / instructions for each player at game start """
        prompt = (
            f"Welcome to Bullshit! You are Player {player_id}.\n\n"
            "OBJECTIVE:\n"
            "  Be the first to get rid of all your cards.\n\n"
            "PLAYFLOW:\n"
            "  1. On your turn, you must place one or more cards face down on the pile.\n"
            "     You will *claim* these cards are of the current rank.\n"
            "  2. The current rank starts at A (Ace) and increments each turn: 2,3,...K.\n"
            "     After K (13), it loops back to A.\n"
            "  3. If the next player thinks you're lying, they may call [BULLSHIT].\n"
            "     - If you lied, you pick up the whole pile.\n"
            "     - If you told the truth, the accuser picks up the pile.\n"
            "  4. The first player to have no cards left wins!\n\n"
            "COMMAND EXAMPLES:\n"
            "  [3 0 2]        => 'I claim these 2 cards are rank 3', using indexes 0 and 2 from my hand.\n"
            "  [BULLSHIT]     => Call bullshit on the previous play.\n"
            "  [PASS]         => Decline to call bullshit (move on).\n\n"
            "Good luck!\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process a single step/action from the current player """
        # Log the player's raw input
        self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action)

        if self.state.game_state["phase"] == "play":
            self._run_play_phase()

        elif self.state.game_state["phase"] == "call":
            self._run_call_phase()

        else:
            raise Exception(f"Unexpected game phase: {self.state.game_state['phase']}")

        # rotate players and game phase as necessary 
        


    def _run_play_phase(self):
        pass 


    def _run_call_phase(self):
        # check if bullshit pattern is present
        if self.BULLSHIT_PATTERN.search(action):

            pass 







        # check the phase 

        act_str = action.strip()

        if self.BULLSHIT_PATTERN.match(act_str):
            self._handle_bullshit(current_pid)
        elif self.PASS_PATTERN.match(act_str):
            # A simple "no action"
            pass
        else:
            # Check if it's a PLAY action
            match = self.PLAY_PATTERN.match(act_str)
            if match:
                declared_rank_str = match.group("rank")  
                indexes_str       = match.group("indexes")  
                self._handle_play(current_pid, declared_rank_str, indexes_str)
            else:
                # Unrecognized command
                self.state.set_invalid_move(
                    player_id=current_pid,
                    reason="Invalid command format. Use [<rank> <indexes>], [BULLSHIT], or [PASS]."
                )

        # After the action is resolved (unless it was invalid):
        if not self.state.prevent_player_change:
            # Check if the current player has 0 cards => they win
            if len(self.state.game_state["hands"][current_pid]) == 0:
                self.state.set_winners(
                    player_ids=[current_pid],
                    reason=f"Player {current_pid} has no cards left and wins!"
                )
            else:
                # Advance turn (and possibly increment rank if we had a valid PLAY)
                self._advance_turn()

            # Show updated state to everyone
            self._observer_current_state(broadcast=True)

        # Return done/info from textarena
        return self.state.step(rotate_player=False)

    # ---------------------------------------------------------------------
    # ACTION HANDLERS
    # ---------------------------------------------------------------------

    def _handle_play(self, current_pid: int, declared_rank_str: str, indexes_str: str):
        """
        Process a player's 'PLAY' action:
         - They claim the rank = declared_rank_str
         - They specify which cards by index from their hand
        """
        # Convert declared rank text -> integer (1..13)
        declared_rank = self._parse_declared_rank(declared_rank_str)
        if declared_rank is None:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Unrecognized rank: {declared_rank_str}. Must be A,2..10,J,Q,K."
            )
            return

        # Parse the card indexes
        try:
            card_indexes = [int(x) for x in indexes_str.split()]
        except ValueError:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Could not parse card indexes: '{indexes_str}'"
            )
            return

        # Validate the indexes exist in this player's hand
        hand = self.state.game_state["hands"][current_pid]
        if any((i < 0 or i >= len(hand)) for i in card_indexes):
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="One or more card indexes are out of range for your hand."
            )
            return

        # Remove the specified cards from the player's hand (from highest to lowest index)
        card_indexes_sorted = sorted(card_indexes, reverse=True)
        played_cards = []
        for i in card_indexes_sorted:
            played_cards.append(hand.pop(i))

        # Update last_played_cards & last_declared_rank
        self.state.game_state["last_played_cards"] = played_cards
        self.state.game_state["last_declared_rank"] = declared_rank

        # Add them to the central pile
        self.state.game_state["pile"].extend(played_cards)

        # Broadcast a message about the claim
        claimed_text = self._int_to_rank(declared_rank)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {current_pid} plays {len(played_cards)} card(s), claiming they are {claimed_text}."
        )

    def _handle_bullshit(self, current_pid: int):
        """
        Process a player's BULLSHIT call. Compare actual ranks of last_played_cards
        vs. last_declared_rank:
          - If they're *all* the declared rank => the caller picks up the pile.
          - Otherwise => the last player picks up the pile.
        """
        last_declared_rank = self.state.game_state["last_declared_rank"]
        last_played_cards  = self.state.game_state["last_played_cards"]

        if not last_declared_rank or not last_played_cards:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="No previous play to call Bullshit on!"
            )
            return

        # Check honesty
        # For each card, see if self.rank_values[card['rank']] == last_declared_rank
        honest = all(
            self.rank_values[card["rank"]] == last_declared_rank
            for card in last_played_cards
        )

        # Typically, last player is the one who actually played the cards
        # but in a strict seat order, that might be (current_pid - 1) mod num_players
        # For clarity, let's store "last_player_id" when playing:
        # but here we replicate your original seat-based approach:
        last_player_id = (current_pid - 1) % self.state.num_players

        if honest:
            # The last play was honest => the caller picks up the pile
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Player {current_pid} calls Bullshit, but the last play was HONEST! Player {current_pid} picks up the pile."
            )
            self._pick_up_pile(current_pid)
        else:
            # The last play was a lie => the last player picks up the pile
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Player {current_pid} calls Bullshit, and they were correct! Player {last_player_id} picks up the pile."
            )
            self._pick_up_pile(last_player_id)

        # Clear last-play info
        self.state.game_state["last_played_cards"]  = []
        self.state.game_state["last_declared_rank"] = None

    # ---------------------------------------------------------------------
    # HELPER METHODS
    # ---------------------------------------------------------------------

    def _pick_up_pile(self, pid: int):
        """
        Move all cards from the table's pile into the specified player's hand.
        """
        pile = self.state.game_state["pile"]
        if pile:
            self.state.game_state["hands"][pid].extend(pile)
            self.state.game_state["pile"].clear()

    def _advance_turn(self):
        """
        Move to the next player and increment the 'current_rank' if there was a PLAY.
        """
        current_pid = self.state.current_player_id
        # If last_played_cards is non-empty, we had a valid PLAY => increment rank
        if self.state.game_state["last_played_cards"]:
            old_rank = self.state.game_state["current_rank"]
            new_rank = old_rank + 1 if old_rank < 13 else 1
            self.state.game_state["current_rank"] = new_rank

        # Move on to next player
        next_pid = (current_pid + 1) % self.state.num_players
        self.state.manually_update_current_player(new_player_id=next_pid)

    def _observer_current_state(self, broadcast: bool = False):
        """
        Provide each player with an updated view: 
          - Their hand (with indexes)
          - Opponents' card counts
          - The current rank to declare
          - Pile size
        """
        gs = self.state.game_state
        current_rank = gs["current_rank"]
        rank_text = self._int_to_rank(current_rank)
        for pid in range(self.state.num_players):
            # This player's hand
            hand = gs["hands"][pid]

            # Build a labeled list of cards: "0: A♣, 1: 2♦, ..."
            indexed_hand = []
            for idx, card in enumerate(hand):
                # card['rank'] is text like "A","2","K"
                # card['suit'] is text like "♠","♥","♦","♣"
                indexed_hand.append(f"{idx}: {card['rank']}{card['suit']}")
            hand_str = ", ".join(indexed_hand)

            # Opponent card counts
            opponents_str = ""
            for opp_pid in range(self.state.num_players):
                if opp_pid != pid:
                    opponents_str += f"  Player {opp_pid}: {len(gs['hands'][opp_pid])} cards\n"

            # Pile size
            pile_size = len(gs["pile"])

            message = (
                f"--- BULLSHIT STATUS ---\n"
                f"Your Hand ({len(hand)} cards): {hand_str}\n\n"
                f"Opponent Card Counts:\n{opponents_str}\n"
                f"Pile has {pile_size} cards.\n"
                f"Current Rank to Declare: {rank_text}\n"
            )
            self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=message)

    def _parse_declared_rank(self, rank_str: str) -> Optional[int]:
        """
        Convert a rank string (e.g. 'A','2','3',...'10','J','Q','K') to integer 1..13.
        Return None if invalid.
        """
        rank_str = rank_str.upper()
        # Simple approach: check if rank_str is in our rank_values keys
        # e.g. rank_values = {"A":1, "2":2, ..., "K":13}
        if rank_str in self.rank_values:
            return self.rank_values[rank_str]
        return None

    def _int_to_rank(self, val: int) -> str:
        """
        Convert an integer rank (1..13) back into a string rank ('A','2'..'K').
        """
        # Because self.ranks[0] = "A", self.ranks[12] = "K"
        # val=1 => index=0 => "A"
        # val=13 => index=12 => "K"
        if 1 <= val <= 13:
            return self.ranks[val - 1]
        return str(val)  # fallback