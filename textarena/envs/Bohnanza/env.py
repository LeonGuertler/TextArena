import re
import random
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Bohnanza.renderer import create_board_str


class BohnanzaEnv(ta.Env):
    """
    Bohnanza bean trading game environment.
    
    A 3-5 player game where players plant, trade, and harvest beans to earn coins.
    Key mechanics:
    - Players cannot rearrange their hand order
    - Active player trades with others using face-up cards
    - All traded beans must be planted immediately
    - Game ends after 3 deck cycles
    """
    
    # Bean types with authentic payouts: {coins_earned: beans_needed}
    BEAN_TYPES = {
        "Blue": {"count": 20, "payouts": {1: 4, 2: 6, 3: 8, 4: 10}},
        "Chili": {"count": 18, "payouts": {1: 3, 2: 6, 3: 8, 4: 9}},
        "Stink": {"count": 16, "payouts": {1: 3, 2: 5, 3: 7, 4: 8}},
        "Green": {"count": 14, "payouts": {1: 3, 2: 5, 3: 6, 4: 7}},
        "Soy": {"count": 12, "payouts": {1: 2, 2: 4, 3: 6, 4: 7}},
        "BlackEyed": {"count": 10, "payouts": {1: 2, 2: 4, 3: 5, 4: 6}},
        "Red": {"count": 8, "payouts": {1: 2, 2: 3, 3: 4, 4: 5}},
        "Garden": {"count": 6, "payouts": {2: 2, 3: 3}}
    }
    
    def __init__(self, max_turns: int = 200, error_allowance: int = 3):
        """
        Initialize the Bohnanza environment.
        
        Args:
            max_turns: Maximum number of turns (high limit for deck-based ending)
            error_allowance: Number of invalid moves allowed per player
        """
        super().__init__()
        self.max_turns = max_turns
        self.error_allowance = error_allowance
        
        # Regex patterns for parsing actions
        self.plant_pattern = re.compile(r"\[Plant\]\s*(\d+)", re.IGNORECASE)
        self.harvest_pattern = re.compile(r"\[Harvest\]\s*(\d+)", re.IGNORECASE)
        self.trade_pattern = re.compile(r"\[Trade\]\s*(.+?)\s*for\s*(.+)", re.IGNORECASE)
        self.accept_pattern = re.compile(r"\[Accept\]\s*Trade(\d+)", re.IGNORECASE)
        self.reject_pattern = re.compile(r"\[Reject\]\s*Trade(\d+)", re.IGNORECASE)
        self.end_trading_pattern = re.compile(r"\[EndTrading\]", re.IGNORECASE)
        self.pass_pattern = re.compile(r"\[Pass\]", re.IGNORECASE)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        if num_players < 3 or num_players > 5:
            raise ValueError("Bohnanza requires 3-5 players")
        
        # Initialize state
        self.state = ta.FFAMultiPlayerState(
            num_players=num_players,
            max_turns=self.max_turns,
            error_allowance=self.error_allowance,
            seed=seed
        )
        
        # Create and shuffle deck
        deck = []
        for bean_type, config in self.BEAN_TYPES.items():
            deck.extend([bean_type] * config["count"])
        random.shuffle(deck)
        
        # Determine fields per player based on player count
        fields_per_player = 3 if num_players == 3 else 2
        
        # Initialize game state
        game_state = {
            "deck": deck,
            "discard_pile": [],
            "deck_cycles": 0,
            "current_phase": "plant",
            "face_up_cards": [],
            "active_trades": {},
            "trade_counter": 0,
            "mandatory_plants": {i: [] for i in range(num_players)},
            "players": {}
        }
        
        # Initialize players
        for i in range(num_players):
            # Deal starting hand of 5 cards
            hand = []
            for _ in range(5):
                if deck:
                    hand.append(deck.pop())
            
            game_state["players"][i] = {
                "hand": hand,
                "fields": [None] * fields_per_player,  # [bean_type, count] or None
                "coins": 0,
                "received_from_trades": []
            }
        
        # Reset state
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the prompt for a player."""
        current_phase = game_state["current_phase"]
        is_active = player_id == self.state.current_player_id
        player = game_state["players"][player_id]
        
        # Base game information
        prompt = f"""You are Player {player_id + 1} in a Bohnanza bean trading game.

GAME RULES:
- Plant, trade, and harvest beans to earn coins
- You CANNOT rearrange your hand - must plant beans in order
- Only the active player can propose trades
- All traded beans must be planted immediately
- Game ends after deck is reshuffled 3 times

BEAN TYPES & PAYOUTS (coins earned : beans needed):"""
        
        for bean_type, config in self.BEAN_TYPES.items():
            payouts_str = ", ".join([f"{coins}:{beans}" for coins, beans in config["payouts"].items()])
            prompt += f"\n  {bean_type}: {payouts_str}"
        
        # Current game state
        prompt += f"\n\nCURRENT STATE:"
        prompt += f"\n- Turn: {self.state.turn + 1}"
        prompt += f"\n- Phase: {current_phase}"
        prompt += f"\n- Active Player: Player {self.state.current_player_id + 1}"
        prompt += f"\n- Deck Cycles: {game_state['deck_cycles']}/3"
        prompt += f"\n- Cards in Deck: {len(game_state['deck'])}"
        
        # Player's status
        prompt += f"\n\nYOUR STATUS:"
        prompt += f"\n- Coins: {player['coins']}"
        prompt += f"\n- Hand ({len(player['hand'])} cards): "
        if player['hand']:
            # Show your complete hand
            prompt += f"[{', '.join(player['hand'])}]"
        else:
            prompt += "Empty"
        
        # Fields
        prompt += f"\n- Fields:"
        for i, field in enumerate(player['fields']):
            if field:
                bean_type, count = field
                prompt += f"\n  Field {i + 1}: {count} {bean_type} beans"
            else:
                prompt += f"\n  Field {i + 1}: Empty"
        
        # Face-up cards (visible to all)
        if game_state["face_up_cards"]:
            prompt += f"\n\nFACE-UP CARDS: {', '.join(game_state['face_up_cards'])}"
        
        # Active trades
        if game_state["active_trades"]:
            prompt += f"\n\nACTIVE TRADES:"
            for trade_id, trade in game_state["active_trades"].items():
                prompt += f"\n  Trade{trade_id}: Player{trade['proposer'] + 1} offers {trade['offer']} for {trade['want']} with Player{trade['target'] + 1}"
        
        # Mandatory plants
        if game_state["mandatory_plants"][player_id]:
            prompt += f"\n\nMUST PLANT: {', '.join(game_state['mandatory_plants'][player_id])}"
        
        # Phase-specific instructions
        prompt += f"\n\nCURRENT PHASE: {current_phase.upper()}"
        
        if current_phase == "plant":
            if is_active:
                prompt += f"\n- MUST plant first card from hand: {player['hand'][0] if player['hand'] else 'None'}"
                prompt += f"\n- MAY plant second card from hand"
                prompt += f"\n- Actions: [Plant] <field_number>, [Pass] (after first plant)"
            else:
                prompt += f"\n- Waiting for Player {self.state.current_player_id + 1} to plant"
        
        elif current_phase == "draw_trade":
            prompt += f"\n- Trading phase: Any player can propose or respond to trades"
            prompt += f"\n- Free text discussion allowed before bracketed actions"
            if is_active:
                prompt += f"\n- You drew face-up cards for trading"
                prompt += f"\n- Actions: [Trade] <offer> for <want>, [EndTrading]"
            else:
                prompt += f"\n- You can propose trades or respond to existing ones"
                prompt += f"\n- Actions: [Trade] <offer> for <want>, [Accept] Trade<X>, [Reject] Trade<X>"
        
        elif current_phase == "plant_mandatory":
            if game_state["mandatory_plants"][player_id]:
                prompt += f"\n- MUST plant all received beans: {', '.join(game_state['mandatory_plants'][player_id])}"
                prompt += f"\n- Actions: [Plant] <field_number>, [Harvest] <field_number> (if needed)"
            else:
                prompt += f"\n- No mandatory plants for you"
                prompt += f"\n- Actions: [Pass]"
        
        elif current_phase == "draw":
            if is_active:
                prompt += f"\n- Drawing 3 cards to your hand"
                prompt += f"\n- Actions: [Pass] (automatic)"
            else:
                prompt += f"\n- Waiting for Player {self.state.current_player_id + 1} to draw"
        
        elif current_phase == "harvest":
            if is_active:
                prompt += f"\n- Optional: Harvest any field for coins"
                prompt += f"\n- Cannot harvest 1-bean field if other fields have 2+ beans"
                prompt += f"\n- Actions: [Harvest] <field_number>, [Pass]"
            else:
                prompt += f"\n- Waiting for Player {self.state.current_player_id + 1} to harvest"
        
        prompt += f"\n\nWrite your reasoning and include your action in brackets."
        
        return prompt
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process a player's action."""
        current_player_id = self.state.current_player_id
        
        # Log the action
        self.state.add_observation(
            from_id=current_player_id,
            to_id=-1,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION
        )
        
        # Process the action based on current phase
        success = self._process_action(current_player_id, action)
        
        if success:
            # Only check for phase transitions on specific actions that end phases
            phase_ended = self._check_if_phase_should_end(current_player_id, action)
            if phase_ended:
                self._check_phase_transition()
            self._check_game_end()
        
        # Store the current player before TextArena tries to advance it
        intended_current_player = self.state.current_player_id
        
        # Let TextArena do its step processing
        done, info = self.state.step()
        
        # Override TextArena's turn advancement - we control turns manually
        if not self.state.done:
            self.state.current_player_id = intended_current_player
        
        return done, info
    
    def _process_action(self, player_id: int, action: str) -> bool:
        """Process a player's action based on current phase."""
        current_phase = self.state.game_state["current_phase"]
        
        if current_phase == "plant":
            return self._process_plant_phase(player_id, action)
        elif current_phase == "draw_trade":
            return self._process_draw_trade_phase(player_id, action)
        elif current_phase == "plant_mandatory":
            return self._process_plant_mandatory_phase(player_id, action)
        elif current_phase == "draw":
            return self._process_draw_phase(player_id, action)
        elif current_phase == "harvest":
            return self._process_harvest_phase(player_id, action)
        else:
            self.state.set_invalid_move("Unknown game phase")
            return False
    
    def _process_plant_phase(self, player_id: int, action: str) -> bool:
        """Process actions during plant phase."""
        if player_id != self.state.current_player_id:
            self.state.set_invalid_move("Not your turn")
            return False
        
        plant_match = self.plant_pattern.search(action)
        pass_match = self.pass_pattern.search(action)
        
        if plant_match:
            field_num = int(plant_match.group(1)) - 1  # Convert to 0-based
            return self._plant_from_hand(player_id, field_num)
        elif pass_match:
            # Can only pass after planting first card
            if not hasattr(self, '_planted_count') or self._planted_count == 0:
                self.state.set_invalid_move("Must plant first card from hand before passing")
                return False
            return True
        else:
            self.state.set_invalid_move("Use [Plant] <field_number> or [Pass]")
            return False
    
    def _process_draw_trade_phase(self, player_id: int, action: str) -> bool:
        """Process actions during draw and trade phase."""
        trade_match = self.trade_pattern.search(action)
        accept_match = self.accept_pattern.search(action)
        reject_match = self.reject_pattern.search(action)
        end_trading_match = self.end_trading_pattern.search(action)
        
        if trade_match:
            # Any player can propose trades during trading phase
            offer = trade_match.group(1).strip()
            want = trade_match.group(2).strip()
            return self._propose_open_trade(player_id, offer, want)
        
        elif accept_match:
            trade_id = int(accept_match.group(1))
            return self._accept_trade(player_id, trade_id)
        
        elif reject_match:
            trade_id = int(reject_match.group(1))
            return self._reject_trade(player_id, trade_id)
        
        elif end_trading_match and player_id == self.state.current_player_id:
            # Only active player can end trading
            return True
        
        else:
            if player_id == self.state.current_player_id:
                self.state.set_invalid_move("Use [Trade] <offer> for <want> with Player<X> or [EndTrading]")
            else:
                self.state.set_invalid_move("Use [Accept] Trade<X> or [Reject] Trade<X>")
            return False
    
    def _process_plant_mandatory_phase(self, player_id: int, action: str) -> bool:
        """Process actions during mandatory planting phase."""
        plant_match = self.plant_pattern.search(action)
        harvest_match = self.harvest_pattern.search(action)
        pass_match = self.pass_pattern.search(action)
        
        if plant_match:
            field_num = int(plant_match.group(1)) - 1
            return self._plant_mandatory(player_id, field_num)
        elif harvest_match:
            field_num = int(harvest_match.group(1)) - 1
            return self._harvest_field(player_id, field_num)
        elif pass_match:
            # Can only pass if no mandatory plants
            if self.state.game_state["mandatory_plants"][player_id]:
                self.state.set_invalid_move("Must plant all received beans first")
                return False
            return True
        else:
            self.state.set_invalid_move("Use [Plant] <field_number>, [Harvest] <field_number>, or [Pass]")
            return False
    
    def _process_draw_phase(self, player_id: int, action: str) -> bool:
        """Process actions during draw phase."""
        if player_id != self.state.current_player_id:
            self.state.set_invalid_move("Not your turn")
            return False
        
        # Automatically draw 3 cards
        self._draw_cards_to_hand(player_id, 3)
        return True
    
    def _process_harvest_phase(self, player_id: int, action: str) -> bool:
        """Process actions during harvest phase."""
        if player_id != self.state.current_player_id:
            self.state.set_invalid_move("Not your turn")
            return False
        
        harvest_match = self.harvest_pattern.search(action)
        pass_match = self.pass_pattern.search(action)
        
        if harvest_match:
            field_num = int(harvest_match.group(1)) - 1
            return self._harvest_field(player_id, field_num)
        elif pass_match:
            return True
        else:
            self.state.set_invalid_move("Use [Harvest] <field_number> or [Pass]")
            return False
    
    def _plant_from_hand(self, player_id: int, field_num: int) -> bool:
        """Plant a bean from hand to a field."""
        player = self.state.game_state["players"][player_id]
        
        if not player["hand"]:
            self.state.set_invalid_move("No cards in hand to plant")
            return False
        
        if field_num < 0 or field_num >= len(player["fields"]):
            self.state.set_invalid_move(f"Invalid field number. Use 1-{len(player['fields'])}")
            return False
        
        # Track planted cards per turn
        if not hasattr(self, '_planted_count'):
            self._planted_count = 0
        
        # Must plant first card in order, then can plant second card
        if self._planted_count == 0:
            # Must plant first card
            bean_to_plant = player["hand"][0]
        elif self._planted_count == 1:
            # Can plant second card (which is now at position 0 after first was removed)
            if len(player["hand"]) == 0:
                self.state.set_invalid_move("No second card to plant")
                return False
            bean_to_plant = player["hand"][0]
        else:
            self.state.set_invalid_move("Already planted maximum cards this turn")
            return False
        
        # Check if field is compatible
        field = player["fields"][field_num]
        if field and field[0] != bean_to_plant:
            self.state.set_invalid_move(f"Field {field_num + 1} has {field[0]} beans, cannot plant {bean_to_plant}")
            return False
        
        # Plant the bean
        if field:
            player["fields"][field_num] = (field[0], field[1] + 1)
        else:
            player["fields"][field_num] = (bean_to_plant, 1)
        
        # Remove from hand (always remove first card since we maintain order)
        player["hand"].pop(0)
        
        # Increment planted count after successful planting
        self._planted_count += 1
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id + 1} planted {bean_to_plant} in field {field_num + 1}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _propose_open_trade(self, proposer_id: int, offer: str, want: str) -> bool:
        """Propose an open trade that any player can accept."""
        # Create trade
        trade_id = self.state.game_state["trade_counter"] + 1
        self.state.game_state["trade_counter"] = trade_id
        
        self.state.game_state["active_trades"][trade_id] = {
            "proposer": proposer_id,
            "target": None,  # Open to any player
            "offer": offer,
            "want": want,
            "status": "pending"
        }
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Trade{trade_id}: Player {proposer_id + 1} offers {offer} for {want} (open to all)",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _propose_trade(self, proposer_id: int, offer: str, want: str, target_id: int) -> bool:
        """Propose a trade between players."""
        if target_id < 0 or target_id >= self.state.num_players:
            self.state.set_invalid_move("Invalid target player")
            return False
        
        if target_id == proposer_id:
            self.state.set_invalid_move("Cannot trade with yourself")
            return False
        
        # Create trade
        trade_id = self.state.game_state["trade_counter"] + 1
        self.state.game_state["trade_counter"] = trade_id
        
        self.state.game_state["active_trades"][trade_id] = {
            "proposer": proposer_id,
            "target": target_id,
            "offer": offer,
            "want": want,
            "status": "pending"
        }
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Trade{trade_id}: Player {proposer_id + 1} offers {offer} for {want} with Player {target_id + 1}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _accept_trade(self, player_id: int, trade_id: int) -> bool:
        """Accept a trade proposal."""
        if trade_id not in self.state.game_state["active_trades"]:
            self.state.set_invalid_move(f"Trade{trade_id} does not exist")
            return False
        
        trade = self.state.game_state["active_trades"][trade_id]
        
        # Check if this is an open trade or targeted trade
        if trade["target"] is not None and trade["target"] != player_id:
            self.state.set_invalid_move("This trade is not for you")
            return False
        
        # Cannot accept your own trade
        if trade["proposer"] == player_id:
            self.state.set_invalid_move("Cannot accept your own trade")
            return False
        
        if trade["status"] != "pending":
            self.state.set_invalid_move("Trade is no longer pending")
            return False
        
        # For open trades, set the target to the accepting player
        if trade["target"] is None:
            trade["target"] = player_id
        
        # Execute the trade
        proposer = self.state.game_state["players"][trade["proposer"]]
        target = self.state.game_state["players"][trade["target"]]
        
        # Parse offer and want (simplified - assume single bean names)
        offer_beans = self._parse_bean_list(trade["offer"])
        want_beans = self._parse_bean_list(trade["want"])
        
        # Add to mandatory plants
        self.state.game_state["mandatory_plants"][trade["proposer"]].extend(want_beans)
        self.state.game_state["mandatory_plants"][trade["target"]].extend(offer_beans)
        
        # Mark trade as accepted
        trade["status"] = "accepted"
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id + 1} accepted Trade{trade_id}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _reject_trade(self, player_id: int, trade_id: int) -> bool:
        """Reject a trade proposal."""
        if trade_id not in self.state.game_state["active_trades"]:
            self.state.set_invalid_move(f"Trade{trade_id} does not exist")
            return False
        
        trade = self.state.game_state["active_trades"][trade_id]
        
        if trade["target"] != player_id:
            self.state.set_invalid_move("This trade is not for you")
            return False
        
        # Remove the trade
        del self.state.game_state["active_trades"][trade_id]
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id + 1} rejected Trade{trade_id}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _plant_mandatory(self, player_id: int, field_num: int) -> bool:
        """Plant a mandatory bean (from trades or face-up cards)."""
        mandatory_plants = self.state.game_state["mandatory_plants"][player_id]
        
        if not mandatory_plants:
            self.state.set_invalid_move("No mandatory beans to plant")
            return False
        
        if field_num < 0 or field_num >= len(self.state.game_state["players"][player_id]["fields"]):
            self.state.set_invalid_move(f"Invalid field number")
            return False
        
        # Plant first mandatory bean
        bean_to_plant = mandatory_plants[0]
        player = self.state.game_state["players"][player_id]
        field = player["fields"][field_num]
        
        # Check compatibility
        if field and field[0] != bean_to_plant:
            self.state.set_invalid_move(f"Field {field_num + 1} has {field[0]} beans, cannot plant {bean_to_plant}")
            return False
        
        # Plant the bean
        if field:
            player["fields"][field_num] = (field[0], field[1] + 1)
        else:
            player["fields"][field_num] = (bean_to_plant, 1)
        
        # Remove from mandatory plants
        mandatory_plants.pop(0)
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id + 1} planted mandatory {bean_to_plant} in field {field_num + 1}",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _harvest_field(self, player_id: int, field_num: int) -> bool:
        """Harvest a field for coins."""
        player = self.state.game_state["players"][player_id]
        
        if field_num < 0 or field_num >= len(player["fields"]):
            self.state.set_invalid_move("Invalid field number")
            return False
        
        field = player["fields"][field_num]
        if not field:
            self.state.set_invalid_move("Field is empty")
            return False
        
        bean_type, bean_count = field
        
        # Check harvest priority rule
        if not self._can_harvest_field(player_id, field_num):
            self.state.set_invalid_move("Cannot harvest 1-bean field when other fields have 2+ beans")
            return False
        
        # Calculate coins earned
        coins_earned = self._calculate_harvest_coins(bean_type, bean_count)
        player["coins"] += coins_earned
        
        # Clear the field
        player["fields"][field_num] = None
        
        # Add beans to discard pile
        self.state.game_state["discard_pile"].extend([bean_type] * bean_count)
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Player {player_id + 1} harvested {bean_count} {bean_type} beans for {coins_earned} coins",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
        
        return True
    
    def _can_harvest_field(self, player_id: int, field_num: int) -> bool:
        """Check if a field can be harvested (priority rule)."""
        player = self.state.game_state["players"][player_id]
        target_field = player["fields"][field_num]
        
        if not target_field:
            return False
        
        target_count = target_field[1]
        
        # If target has only 1 bean, check if other fields have 2+ beans
        if target_count == 1:
            for i, field in enumerate(player["fields"]):
                if i != field_num and field and field[1] > 1:
                    return False
        
        return True
    
    def _calculate_harvest_coins(self, bean_type: str, bean_count: int) -> int:
        """Calculate coins earned from harvesting."""
        payouts = self.BEAN_TYPES[bean_type]["payouts"]
        coins_earned = 0
        
        # Find highest coin value where bean_count >= beans_needed
        for coins, beans_needed in sorted(payouts.items(), reverse=True):
            if bean_count >= beans_needed:
                coins_earned = coins
                break
        
        return coins_earned
    
    def _draw_cards_to_hand(self, player_id: int, num_cards: int):
        """Draw cards from deck to player's hand."""
        player = self.state.game_state["players"][player_id]
        
        for _ in range(num_cards):
            if self.state.game_state["deck"]:
                card = self.state.game_state["deck"].pop()
                player["hand"].append(card)
            elif self.state.game_state["discard_pile"]:
                # Reshuffle deck
                self._reshuffle_deck()
                if self.state.game_state["deck"]:
                    card = self.state.game_state["deck"].pop()
                    player["hand"].append(card)
    
    def _draw_face_up_cards(self, num_cards: int = 2):
        """Draw face-up cards for trading."""
        face_up = []
        
        for _ in range(num_cards):
            if self.state.game_state["deck"]:
                card = self.state.game_state["deck"].pop()
                face_up.append(card)
            elif self.state.game_state["discard_pile"]:
                self._reshuffle_deck()
                if self.state.game_state["deck"]:
                    card = self.state.game_state["deck"].pop()
                    face_up.append(card)
        
        self.state.game_state["face_up_cards"] = face_up
        
        if face_up:
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"Face-up cards: {', '.join(face_up)}",
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
            )
    
    def _reshuffle_deck(self):
        """Reshuffle discard pile into deck."""
        self.state.game_state["deck_cycles"] += 1
        self.state.game_state["deck"] = self.state.game_state["discard_pile"].copy()
        self.state.game_state["discard_pile"] = []
        random.shuffle(self.state.game_state["deck"])
        
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message=f"Deck reshuffled (cycle {self.state.game_state['deck_cycles']}/3)",
            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
        )
    
    def _parse_bean_list(self, bean_str: str) -> List[str]:
        """Parse a string like '2 Coffee Bean' into list of beans."""
        # Simplified parsing - assume format like "2 Coffee Bean" or "Coffee Bean"
        beans = []
        parts = bean_str.strip().split()
        
        if parts and parts[0].isdigit():
            # Format: "2 Coffee Bean"
            count = int(parts[0])
            bean_type = " ".join(parts[1:])
            beans.extend([bean_type] * count)
        else:
            # Format: "Coffee Bean"
            bean_type = bean_str.strip()
            beans.append(bean_type)
        
        return beans
    
    def _check_phase_transition(self):
        """Check if we should transition to the next phase."""
        current_phase = self.state.game_state["current_phase"]
        
        if current_phase == "plant":
            # Move to draw_trade phase
            self.state.game_state["current_phase"] = "draw_trade"
            self._planted_count = 0  # Reset for next turn
            
            # Draw face-up cards for trading
            self._draw_face_up_cards(2)
        
        elif current_phase == "draw_trade":
            # Move to plant_mandatory phase
            self.state.game_state["current_phase"] = "plant_mandatory"
            
            # Add remaining face-up cards to active player's mandatory plants
            active_player = self.state.current_player_id
            remaining_face_up = self.state.game_state["face_up_cards"]
            self.state.game_state["mandatory_plants"][active_player].extend(remaining_face_up)
            self.state.game_state["face_up_cards"] = []
        
        elif current_phase == "plant_mandatory":
            # Check if all players have planted their mandatory beans
            all_planted = all(
                not plants for plants in self.state.game_state["mandatory_plants"].values()
            )
            
            if all_planted:
                self.state.game_state["current_phase"] = "draw"
        
        elif current_phase == "draw":
            # Move to harvest phase
            self.state.game_state["current_phase"] = "harvest"
        
        elif current_phase == "harvest":
            # Move to next player's turn - this is the only place we advance the turn
            self.state.game_state["current_phase"] = "plant"
            
            # Clear accepted trades
            self.state.game_state["active_trades"] = {
                k: v for k, v in self.state.game_state["active_trades"].items()
                if v["status"] == "pending"
            }
            
            # Advance to next player
            self.state.current_player_id = (self.state.current_player_id + 1) % self.state.num_players
    
    def _check_game_end(self):
        """Check if the game should end."""
        # Game ends after 3rd deck cycle
        if self.state.game_state["deck_cycles"] >= 3:
            # If we're in draw_trade phase and deck ran out, complete phases 2 and 3
            current_phase = self.state.game_state["current_phase"]
            if current_phase == "draw_trade":
                # Allow completion of trading and mandatory planting
                return
            
            self._end_game()
    
    def _end_game(self):
        """End the game and determine winner."""
        # All players harvest all remaining fields
        for player_id in range(self.state.num_players):
            player = self.state.game_state["players"][player_id]
            for field_num, field in enumerate(player["fields"]):
                if field:
                    bean_type, bean_count = field
                    coins_earned = self._calculate_harvest_coins(bean_type, bean_count)
                    player["coins"] += coins_earned
                    player["fields"][field_num] = None
                    
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=f"Final harvest: Player {player_id + 1} harvested {bean_count} {bean_type} beans for {coins_earned} coins",
                        observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                    )
        
        # Determine winner
        final_scores = {pid: self.state.game_state["players"][pid]["coins"] for pid in range(self.state.num_players)}
        max_coins = max(final_scores.values())
        winners = [pid for pid, coins in final_scores.items() if coins == max_coins]
        
        if len(winners) == 1:
            winner_id = winners[0]
            self.state.set_winners(
                player_ids=[winner_id],
                reason=f"Player {winner_id + 1} wins with {max_coins} coins!"
            )
        else:
            # Tie-breaking: player furthest clockwise from starting player (player 0)
            # In our case, highest player ID wins
            winner_id = max(winners)
            self.state.set_winners(
                player_ids=[winner_id],
                reason=f"Player {winner_id + 1} wins tie-break with {max_coins} coins (furthest from starting player)"
            )
        
        # Set final scores as rewards
        max_score = max(final_scores.values()) if final_scores else 1
        rewards = {}
        for pid, score in final_scores.items():
            # Normalize to 0-100 range
            normalized_score = int((score / max_score) * 100) if max_score > 0 else 0
            rewards[pid] = max(0, normalized_score)
        
        self.state.rewards = rewards
        self.state.done = True
    
    def get_observation(self):
        """Get observation for current player."""
        player_id = self.state.current_player_id
        observation = self.state.get_current_player_observation()
        
        # Add current game board state
        board_str = create_board_str(self.state.game_state, player_id)
        observation.append((ta.GAME_ID, board_str, ta.ObservationType.GAME_BOARD))
        
        return player_id, observation
    
    def _check_if_phase_should_end(self, player_id: int, action: str) -> bool:
        """Check if the current phase should end based on the action taken."""
        current_phase = self.state.game_state["current_phase"]
        
        if current_phase == "plant":
            # Phase ends when:
            # 1. Active player passes (after planting first card), OR
            # 2. Active player has planted 2 cards (maximum), OR
            # 3. Active player has no more cards to plant
            if player_id == self.state.current_player_id:
                if self.pass_pattern.search(action):
                    return True
                # Check if we've planted maximum cards or no more cards available
                if hasattr(self, '_planted_count'):
                    player = self.state.game_state["players"][player_id]
                    if self._planted_count >= 2 or len(player["hand"]) == 0:
                        return True
            return False
        
        elif current_phase == "draw_trade":
            # Phase ends when active player uses [EndTrading]
            return self.end_trading_pattern.search(action) is not None and player_id == self.state.current_player_id
        
        elif current_phase == "plant_mandatory":
            # Phase ends when all players have no mandatory plants left
            # This is checked in the transition logic, not based on a specific action
            return False
        
        elif current_phase == "draw":
            # Phase ends immediately after active player draws (automatic)
            return player_id == self.state.current_player_id
        
        elif current_phase == "harvest":
            # Phase ends when active player passes or harvests (or any harvest action)
            return (self.pass_pattern.search(action) is not None or 
                   self.harvest_pattern.search(action) is not None) and player_id == self.state.current_player_id
        
        return False
