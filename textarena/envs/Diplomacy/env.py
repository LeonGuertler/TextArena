import re
import random
from typing import Any, Dict, Optional, Tuple, List, Set
from functools import partial
import textarena as ta
from textarena.envs.Diplomacy.game_engine import DiplomacyGameEngine
from textarena.envs.Diplomacy.prompts.prompt import get_state_specific_prompt
import os

class DiplomacyEnv(ta.Env):
    """Environment for Diplomacy with negotiation support"""
    broadcast_pattern = re.compile(
        r"(?:"
        r"\s*\[Broadcast\s*:\s*(.*?)\]"            # [Broadcast: message]
        r"|"
        r"\s*\[Broadcast((?:\s+).*?)\]"            # [Broadcast message]
        r"|"
        r"\s*\[Broadcast\](\s+.*?)(?=\s*\[|$)"     # [Broadcast] message
        r")",
        re.IGNORECASE | re.DOTALL
    )
    
    # Whisper to another player
    whisper_pattern = re.compile(
        r"\s*\[Whisper\s+(?:to\s+)?(?:Player\s+)?(\d+)(?:\s+\(([A-Z]+)\))?\s*:\s*(.*?)\]",
        re.IGNORECASE | re.DOTALL
    )
    
    # Submit orders for the current phase
    submit_orders_pattern = re.compile(
        r"\[Submit\s+Orders\]([\s\S]*?)(?=\[|$)",
        re.IGNORECASE
    )
    
    # Summary pattern to extract summaries from agent responses
    summary_pattern = re.compile(
        r"\[Please provide a brief summary(.*?)(?=\[|$)",
        re.IGNORECASE | re.DOTALL
    )

    def __init__(self, max_turns: int = 30, 
                negotiations_per_phase: int = 3,
                request_summaries: bool = True):
        """
        Initialize the Diplomacy game environment

        Args:
            max_turns (int): Maximum number of game years before ending in a draw
            negotiations_per_phase (int): How many negotiation rounds per game phase
            request_summaries (bool): Whether to request game state summaries from agents
        """
        self.max_turns = max_turns
        self.negotiations_per_phase = negotiations_per_phase
        self.request_summaries = request_summaries
        self.summary_prompt = "\n\n[Please provide a brief summary of the current game state in 3-5 sentences. Include your position, key threats, and opportunities.]"
        self.player_summaries = {}  # Store summaries by player ID
        
        # Game state
        self.engine = None
        self.player_power_map = {}
        self.power_player_map = {}
        self.current_negotiation_round = 0
        self.orders_submitted = set()  # Track which players submitted orders
        self.pending_orders = {}       # Store orders until processing
        self.current_season = None
        self.current_year = None
        self.current_phase = None
        self.chat_history: List[Dict[str, Any]] = []

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment and start a new game """
        self.state = ta.State(num_players=num_players, min_players=3, max_players=7)
        
        # Initialize game engine
        self.engine = DiplomacyGameEngine(max_turns=self.max_turns)
        self.player_power_map = self.engine.setup_game(num_players)
        self.state.role_mapping = self.player_power_map
        self.power_player_map = {power: player for player, power in self.player_power_map.items()}
        
        # Reset game state tracking
        self.current_negotiation_round = 0
        self.orders_submitted = set()
        self.pending_orders = {}
        self.current_season = self.engine.season
        self.current_year = self.engine.year
        self.current_phase = self.engine.phase
        self.offers = {}
        self.next_offer_id = 1
        self.agreements = {}
        self.chat_history: List[Dict[str, Any]] = []

        # Initialize game state for players
        game_state = self.engine.get_state()
        game_state['player_power_map'] = self.player_power_map
        game_state['powers_info'] = {power: {
            'home_centers': self.engine.powers[power].home_centers,
            'controlled_centers': self.engine.powers[power].controlled_centers,
            'units': [str(unit) for unit in self.engine.powers[power].units],
        } for power in self.player_power_map.values()}
        game_state['current_negotiation_round'] = 0
        game_state['total_negotiation_rounds'] = self.negotiations_per_phase

        player_prompt_function = partial(self._generate_player_prompt, player_power_map=self.player_power_map, start_of_game=True)
        # Initialize the state
        self.state.reset(
            game_state=game_state,
            player_prompt_function=player_prompt_function,
            seed=seed,
        )
        
        # Send initial game state announcement to all players
        self._announce_game_state()
        
        return self.state

    def _generate_player_prompt(self, player_power_map: Dict[str, str], player_id: int, game_state: Dict[str, Any], start_of_game: bool = False) -> str:
        """
        Generate a comprehensive prompt for the player at the beginning of the game
        
        Args:
            player_id (int): ID of the player
            game_state (Dict): Current game state
                
        Returns:
            str: Prompt for the player
        """
        power_name = player_power_map.get(player_id)
        if not power_name:
            return "You are not an active player in this game."
            
        # Load the base prompt template
        with open("textarena/envs/Diplomacy/prompts/player_base_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        # Format the template with player-specific information
        prompt = prompt_template.format(
            power_name=power_name,
            player_id=player_id,
            max_turns=self.max_turns,
            negotiations_per_phase=self.negotiations_per_phase,
            state_specific_prompt=get_state_specific_prompt(power_name)
        )
        
        if start_of_game:
            # Add game start information
            start_game_info = [
                "## CURRENT GAME STATE",
                f"It is {game_state['season']} {game_state['year']}, {game_state['phase']} phase.",
                f"This is negotiation round 1 of {game_state['total_negotiation_rounds']}.",
                "",
                "The game has just begun. Use this first negotiation round to establish initial diplomacy.",
                "",
                "Good luck, and may your diplomacy be successful!"
            ]
            prompt += "\n" + "\n".join(start_game_info)
        
        return prompt

    def format_power_units_and_centers(self, player_id: int):
        """
        Show a detailed view of a given power's units and supply centers.
        Includes location expansions and strength summary.
        
        Args:
            player_id (int): The player ID to format information for, -1 mean neutral
            
        Returns:
            str: Formatted string with units and centers information
        """
        if player_id == -1:
            # NEUTRAL POWER
            all_controlled_centers = set()
            for power in self.engine.powers.values():
                all_controlled_centers.update(power.controlled_centers)
            neutral_centers = []
            for center in self.engine.map.get_supply_centers():
                if center not in all_controlled_centers:
                    neutral_centers.append(center)
            return f"{len(neutral_centers)} SUPPLY CENTERS: " + ", ".join(neutral_centers)
        else:
            power_name = self.player_power_map[player_id]
            power = self.engine.powers[power_name]
            
            # Start building the output
            output = []
            power_str = f"{power_name} FORCES"
            # Format units with more detail
            if power.units:
                units_str = f"{len(power.units)} UNITS: "
                unit_details = []
                for unit in power.units:
                    # Assuming unit string format like "A PAR" or "F BRE"
                    unit_str = str(unit)
                    unit_type = unit_str[0]  # 'A' or 'F'
                    location = unit_str[2:]  # Location code
                    # You could add location expansion here if you have a mapping
                    unit_details.append(f"{unit_type} {location}")
                units_str += "(" + ", ".join(unit_details) + ")"
            else:
                units_str = "0 units"
            
            # Format supply centers with more detail
            if power.controlled_centers:
                centers_str = f"{len(power.controlled_centers)} SUPPLY CENTERS: "
                # You could add location expansion here if you have a mapping
                centers_str += "(" + ", ".join(power.controlled_centers) + ")"
            else:
                centers_str = "0 supply centers"
            
            # Add strength summary
            strength_summary = f"Current strength: {len(power.controlled_centers)} centers, {len(power.units)} units"
            
            # Combine all parts
            return f"{power_str}\n{units_str}\n{centers_str}\n{strength_summary}\n\n"
    
    def format_possible_orders(self, possible_orders):
        """
        Format possible orders by categorizing them into strategic groups
        with helpful descriptions.
        
        Args:
            possible_orders (dict): Dictionary of possible orders by location
            
        Returns:
            str: Formatted string of possible orders
        """
        # Categorize orders by strategic purpose
        strategic_orders = {
            "OFFENSIVE": [],  # Orders that can capture centers or threaten enemy units
            "DEFENSIVE": [],  # Orders that protect your centers or units
            "TACTICAL": [],   # Orders that improve position without immediate captures
            "SUPPORT": []     # Support orders
        }
        
        # Get supply centers for context
        supply_centers = set(self.engine.map.get_supply_centers())
        
        # Get current supply center ownership
        power_centers = {}
        for power_name, power in self.engine.powers.items():
            for center in power.controlled_centers:
                power_centers[center] = power_name
        
        # Process each order
        for loc, orders in possible_orders.items():
            for order in orders:
                order_type = None
                
                # Determine order type based on order syntax
                if " H" in order:
                    order_type = "DEFENSIVE"
                elif " S " in order:
                    order_type = "SUPPORT"
                elif " - " in order:
                    # Get destination
                    dest = order.split(" - ")[1].split(" VIA")[0] if " VIA" in order else order.split(" - ")[1]
                    
                    # Check if destination is a supply center
                    if dest in supply_centers:
                        # If center is neutral or enemy-owned, it's offensive
                        if dest not in power_centers or power_centers[dest] != self.player_power_map[self.state.current_player_id]:
                            order_type = "OFFENSIVE"
                        else:
                            order_type = "DEFENSIVE"  # Moving to own supply center
                    else:
                        order_type = "TACTICAL"  # Non-center destination
                elif " C " in order:
                    order_type = "SUPPORT"  # Classify convoy as support
                else:
                    order_type = "TACTICAL"  # Default for other orders
                
                # Add to appropriate category
                if order_type:
                    strategic_orders[order_type].append(order)
        
        # Generate formatted output
        output = "POSSIBLE ORDERS:\n\n"
        
        # Add offensive moves first - these are highest priority
        if strategic_orders["OFFENSIVE"]:
            output += "Offensive Moves (capture territory):\n"
            for order in strategic_orders["OFFENSIVE"]:
                output += f"  {order}\n"
            output += "\n"
        
        # Add defensive moves
        if strategic_orders["DEFENSIVE"]:
            output += "Defensive Moves (protect territory):\n"
            for order in strategic_orders["DEFENSIVE"]:
                output += f"  {order}\n"
            output += "\n"
        
        # Add tactical positioning moves
        if strategic_orders["TACTICAL"]:
            output += "Tactical Moves (improve position):\n"
            for order in strategic_orders["TACTICAL"]:
                output += f"  {order}\n"
            output += "\n"
        
        # Add support moves
        if strategic_orders["SUPPORT"]:
            output += "Support Options (strengthen attacks/defense):\n"
            for order in strategic_orders["SUPPORT"]:
                output += f"  {order}\n"
        
        return output

    def get_prompt(self, player_id: int, history_text: str):
        # 1) Load the template
        template = open("textarena/envs/Diplomacy/prompts/context_prompt.txt", "r").read()

        # 2) Expand the phase info
        phase_info = self.expand_phase_info()

        # 3) Get the game state
        game_state = self.engine.get_state()
        game_state['player_power_map'] = self.player_power_map
        game_state['powers_info'] = {power: {
            'home_centers': self.engine.powers[power].home_centers,
            'controlled_centers': self.engine.powers[power].controlled_centers,
            'units': [str(unit) for unit in self.engine.powers[power].units],
        } for power in self.player_power_map.values()}
        game_state['current_negotiation_round'] = self.current_negotiation_round
        game_state['total_negotiation_rounds'] = self.negotiations_per_phase

        # 4) Get the state summaries
        game_state['our_state_summary_text'] = self.format_power_units_and_centers(player_id)
        
        # 5) Summaries for enemies
        enemies_forces_summary = ""
        for pwr_id, pwr in self.player_power_map.items():
            if pwr_id != player_id:
                enemies_forces_summary += self.format_power_units_and_centers(pwr_id)
        game_state['other_state_summary_text'] = enemies_forces_summary

        # 6) Neutral state summary
        neutral_supply_centers_summary = self.format_power_units_and_centers(-1)
        game_state['neutral_state_summary_text'] = neutral_supply_centers_summary

        # 6) Possible orders
        possible_orders = self.engine.get_possible_orders(self.player_power_map[player_id])
        possible_orders_text = self.format_possible_orders(possible_orders)
        game_state['possible_orders_text'] = possible_orders_text

        # 7) Prompt
        prompt = template.format(
            phase_info=phase_info,
            history_text=history_text,
            our_state_summary_text=game_state['our_state_summary_text'],
            other_state_summary_text=game_state['other_state_summary_text'],
            neutral_state_summary_text=game_state['neutral_state_summary_text'],
            possible_orders_text=game_state['possible_orders_text'],
        )

        game_settings_prompt = self._generate_player_prompt(player_power_map=self.player_power_map, player_id=player_id, game_state=game_state, start_of_game=False)

        return game_settings_prompt + "\n\n" + prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process player actions and extract summaries if present
        
        Args:
            action (str): Player's action as a multi-line string
            
        Returns:
            Tuple[bool, ta.Info]: Game completion status and additional info
        """
        current_pid = self.state.current_player_id
        
        # Extract summary if present and we're requesting summaries
        if self.request_summaries:
            summary_match = self.summary_pattern.search(action)
            if summary_match:
                # Extract the summary text
                summary_text = summary_match.group(1).strip()
                if summary_text:
                    # Find the first line break after the prompt
                    first_line_break = summary_text.find('\n')
                    if first_line_break != -1:
                        summary_text = summary_text[first_line_break:].strip()
                    
                    # Store the summary for this player
                    self.player_summaries[current_pid] = summary_text
                
                # Remove the summary request and response from the action
                action = action[:summary_match.start()] + action[summary_match.end():]
        
        # Continue with the regular step logic
        return super().step(action)

    def _process_player_action(self, player_id: int, power_name: str, action: str) -> Tuple[Dict, bool]:
        """
        Process a player's action string, extracting communications and orders
        
        Args:
            player_id (int): The player ID
            power_name (str): The power name
            action (str): The action string
            
        Returns:
            Tuple[Dict, bool]: Summary of actions taken and whether game state changed
        """
        game_state_changed = False
        action_summary = {
            "broadcasts": [],
            "whispers": [],
            "orders_submitted": False,
            "orders": []
        }
        
        # Process broadcasts
        for match in self.broadcast_pattern.finditer(action):
            message = match.group(1) or match.group(2) or match.group(3)
            if message:
                # broadcast to all 
                self.add_observation(from_id=player_id, to_id=-1, message=message)
                action_summary["broadcasts"].append(message)
        
        # Process whispers
        for match in self.whisper_pattern.finditer(action):
            target_id = int(match.group(1))
            target_power = match.group(2)
            message = match.group(3)
            
            # Validate target_power if provided
            if target_power and self.player_power_map.get(target_id) != target_power:
                continue
            
            # add observation to the whisperee
            self.add_observation(from_id=player_id, to_id=target_id, message=message)
            action_summary["whispers"].append({
                "to_player": target_id,
                "to_power": self.player_power_map.get(target_id),
                "message": message
            })
        
        # Process order submission
        if self.current_negotiation_round == self.negotiations_per_phase - 1:
            for match in self.submit_orders_pattern.finditer(action):
                orders_text = match.group(1).strip()
                if orders_text:
                    self._handle_orders_submission(player_id, power_name, orders_text)
                    action_summary["orders_submitted"] = True
                    action_summary["orders"].append(orders_text)
            
            # Check if all players have submitted orders and it's time to process them
            if len(self.orders_submitted) == len(self.player_power_map):
                game_state_changed = self._process_orders()

        return action_summary, game_state_changed


    def _rotate_players(self):
        """Rotate to the next player"""
        current_player_id = self.state.current_player_id
        next_player_id = (current_player_id + 1) % self.state.num_players
        while next_player_id != current_player_id:
            self.state.manually_update_current_player(new_player_id=next_player_id)
            break
            # if self.state.game_state["remaining_dice"][next_player_id] > 0:
            #     self.state.manually_update_current_player(new_player_id=next_player_id)
            #     break
            # @Leon TODO add the break condition
            # next_player_id = (next_player_id + 1) % self.state.num_players
        # else:
        #     self.state.set_winners(player_ids=[current_player_id], reason=f"Player {current_player_id} wins! All other players ran out of dice.")

    def _handle_orders_submission(self, player_id: int, power_name: str, orders_text: str):
        """Handle order submission from a player"""
        # Parse orders line by line
        orders = [line.strip() for line in orders_text.split('\n') 
                 if line.strip() and not line.strip().startswith('#')]
    
        # Store pending orders and mark player as submitted
        self.pending_orders[power_name] = orders
        self.orders_submitted.add(player_id)
        
        # Notify player their orders were received
        msg = f"[Orders received for {power_name} ({len(orders)} orders)]"
        self.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=msg)

    def _process_orders(self) -> bool:
        """Process all submitted orders"""
        # Submit orders to the engine
        phase_changed, new_state = self.engine.resolve_orders(self.pending_orders)
        
        # Update game state
        new_state['player_power_map'] = self.player_power_map
        self.state.game_state = new_state
        
        # Update phase tracking
        self.current_season = self.engine.season
        self.current_year = self.engine.year
        self.current_phase = self.engine.phase
        
        # Announce results to all players
        self._announce_order_results()
        
        # Reset for next phase
        self.orders_submitted = set()
        self.pending_orders = {}
        self.current_negotiation_round = 0
        
        # Update game state
        self.state.game_state['current_negotiation_round'] = 0
        
        return True

    def _advance_negotiation_round(self):
        """Advance to the next negotiation round"""
        self.current_negotiation_round += 1
        self.state.game_state['current_negotiation_round'] = self.current_negotiation_round
        
        # Notify all players about the new negotiation round
        if self.current_negotiation_round < self.negotiations_per_phase:
            message=(f"[Negotiation Round {self.current_negotiation_round + 1} of "
                       f"{self.negotiations_per_phase} begins]")
            self.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                
            # If this is the final round, notify players to submit orders
            if self.current_negotiation_round == self.negotiations_per_phase - 1:
                message="[Final negotiation round: Please submit your orders]"
                self.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
        else:
            # Force order processing if we've somehow exceeded max rounds
            if self.pending_orders:
                self._process_orders()


    def _announce_game_state(self):
        """Announce the current game state to all players"""
        phase_type = self.engine.phase.value
        season = self.engine.season.value
        year = self.engine.year
        
        # Create announcement message
        announcement = (
            f"===== {season} {year} {phase_type} Phase Begins =====\n"
            f"Current supply center counts:\n"
        )
        
        # Add supply center counts for each power
        for power_name, player_id in self.power_player_map.items():
            centers = len(self.engine.powers[power_name].controlled_centers)
            units = len(self.engine.powers[power_name].units)
            announcement += f"- Player {player_id} ({power_name}): {centers} centers, {units} units\n"
        
        # Send to all players
        for player_id in self.player_power_map.keys():
            self.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=announcement)

    def _announce_order_results(self):
        """Announce order results to all players"""
        # Create basic announcement
        base_announcement = (
            f"===== Order Results: {self.current_season.value} {self.current_year} =====\n"
        )
        
        # Create individualized announcements with detailed results
        for player_id, power_name in self.player_power_map.items():
            power_announcement = base_announcement
            
            # Add own orders and results
            if power_name in self.pending_orders:
                power_announcement += f"\nYour orders ({power_name}):\n"
                for order in self.pending_orders[power_name]:
                    power_announcement += f"- {order}\n"
            
            # Add public information about other powers' orders
            power_announcement += "\nOther powers' orders:\n"
            for other_power, orders in self.pending_orders.items():
                if other_power != power_name:
                    power_announcement += f"\n{other_power}:\n"
                    for order in orders:
                        power_announcement += f"- {order}\n"
            
            # Send to player
            self.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=power_announcement)
        
        # Announce new phase
        self._announce_game_state()

    def _announce_game_result(self):
        """Announce the game result to all players"""
        if self.engine.winners:
            # Victory announcement
            winners_text = ", ".join(self.engine.winners)
            winning_players = [self.power_player_map[power] for power in self.engine.winners]
            winning_players_text = ", ".join(f"Player {pid}" for pid in winning_players)
            
            reason = (
                f"Victory achieved by: {winners_text} ({winning_players_text})\n\n"
                f"Final supply center counts:\n"
            )
            self.state.set_winners(player_ids=winning_players, reason=reason)
        else:
            # Draw announcement
            reason = (
                f"Game ended in a DRAW after {self.engine.turn_number} turns.\n\n"
                f"Final supply center counts:\n"
            )
            self.state.set_draw(reason=reason)

        announcement = "Game Results:\n" + reason
        # Add final supply center counts
        for power_name, player_id in self.power_player_map.items():
            if power_name in self.engine.powers:
                centers = len(self.engine.powers[power_name].controlled_centers)
                announcement += f"- Player {player_id} ({power_name}): {centers} centers\n"
        
        # Add game history summary
        announcement += "\nGame Summary:\n"
        for entry in self.engine.history:
            announcement += f"- {entry['phase']}\n"

        # Send to all players
        for player_id in self.player_power_map.keys():
            self.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=announcement)

    def add_observation(self, from_id: int, to_id: int, message: str):
        """Add an observation to the chat history"""
        self.chat_history.append({
            "turn": self.state.turn,
            "from": from_id,
            "from_power": self.player_power_map.get(from_id),
            "to": to_id,
            "to_power": self.player_power_map.get(to_id),
            "message": message
        })
        self.state.add_observation(from_id=from_id, to_id=to_id, message=message)

    def get_game_state(self):
        game_state = {
            # ===== Game State =====
            "current_season": self.current_season.value,
            "current_year": self.current_year,
            "current_phase": self.current_phase.value,
            "current_negotiation_round": self.current_negotiation_round,
            "total_negotiation_rounds": self.negotiations_per_phase,
            # ===== Players =====
            "Players": [
                {
                    "id": player_id,
                    "power": self.player_power_map[player_id],
                    "controlled_centers": len(self.engine.powers[self.player_power_map[player_id]].controlled_centers),
                    "units": len(self.engine.powers[self.player_power_map[player_id]].units)
                }
                for player_id in self.player_power_map.keys()
            ],
            # ===== Orders =====
            "orders_submitted": list(self.orders_submitted),
            "pending_orders_count": {power: len(orders) for power, orders in self.pending_orders.items()},
            "sc_counts": {power: len(self.engine.powers[power].controlled_centers) for power in self.engine.powers},
            "unit_counts": {power: len(self.engine.powers[power].units) for power in self.engine.powers},
            "is_final_round": self.current_negotiation_round == self.negotiations_per_phase - 1,
        }
        
        return game_state

    def get_conversation_history(self) -> List[List[Dict[str, Any]]]:
        """Log the current conversation history as a list indexed by turn"""
        # Create a list-based conversation history organized by turn
        max_turn = 0
        turn_messages = {}
        
        # Use the dedicated chat_history attribute
        for entry in self.chat_history:
            turn = entry["turn"]
            
            # Process the message to clean up whispers
            message = entry["message"]
            if "[Whisper to" in message:
                # Remove the whisper prefix for cleaner history
                message = re.sub(r"\[Whisper to \d+(?:\s+\([A-Z]+\))?: ", "", message)
                entry["message"] = message
            
            # Organize by turn
            if turn not in turn_messages:
                turn_messages[turn] = []
            
            # Add a cleaned version to the turn messages
            clean_entry = {
                "from": entry["from_power"] if entry["from_power"] else ("GAME" if entry["from"] == ta.GAME_ID else f"Player {entry['from']}"),
                "to": entry["to_power"] if entry["to"] != -1 and entry["to_power"] else ("ALL" if entry["to"] == -1 else f"Player {entry['to']}"),
                "message": message,
                "turn": turn
            }
            turn_messages[turn].append(clean_entry)
            
            max_turn = max(max_turn, turn)
        
        # Convert dict to list with proper indexing
        conversation_history = [[] for _ in range(max_turn + 1)]
        for turn, messages in turn_messages.items():
            conversation_history[turn] = messages
        
        return conversation_history
    
    def get_order_history(self) -> Dict[str, Any]:
        """Get the current order history"""
        return self.engine.order_history
    
    def get_game_state_history(self) -> List[Dict[str, Any]]:
        """Get the game state history"""
        return self.engine.game_state_history
    
    def expand_phase_info(self):
        """
        Convert a phase like 'S1901M' into a more descriptive string:
        'Spring 1901 Movement (early game): Units can move, support, or convoy...'
        This function also references the current year to classify early/mid/late game.
        """
        season = self.engine.season.value
        year = self.engine.year
        phase = self.engine.phase.value
        # Basic mapping of abbreviations
        
        # Approximate game stage
        if year <= 1902:
            stage = "early game"
        elif year <= 1906:
            stage = "mid game"
        else:
            stage = "late game"
        
        # Phase-specific action text
        if phase == 'Movement':
            actions = "Players issue move, support, or convoy orders."
        elif phase == 'Retreats':
            actions = "Dislodged units must retreat or disband."
        elif phase == 'Adjustments':
            actions = "Powers may build new units if they have more centers than units, otherwise disband if fewer."
        else:
            actions = "Unknown phase actions."
        
        return f"{season} {year} {phase} ({stage}): {actions}"

    def generate_phase_summary(self, phase_history):
        # Generate a summary of game state changes, TODO finish this part
        game_state_history = self.get_game_state_history()
        game_state_changes = ""
        if len(game_state_history) >= 2:
            current_state = game_state_history[-1]
            previous_state = game_state_history[-2]
            
            # Compare supply center counts
            sc_changes = []
            for power in current_state["sc_counts"]:
                current_sc = current_state["sc_counts"][power]
                previous_sc = previous_state["sc_counts"].get(power, 0)
                if current_sc != previous_sc:
                    change = current_sc - previous_sc
                    sc_changes.append(f"{power}: {'+' if change > 0 else ''}{change} supply centers")
            
            if sc_changes:
                game_state_changes += "Supply Center Changes:\n" + "\n".join(sc_changes) + "\n\n"
            
            # Compare unit counts
            unit_changes = []
            for power in current_state["unit_counts"]:
                current_units = current_state["unit_counts"][power]
                previous_units = previous_state["unit_counts"].get(power, 0)
                if current_units != previous_units:
                    change = current_units - previous_units
                    unit_changes.append(f"{power}: {'+' if change > 0 else ''}{change} units")
            
            if unit_changes:
                game_state_changes += "Unit Count Changes:\n" + "\n".join(unit_changes) + "\n\n"
        
        # Get the phase summary prompt
        with open(os.path.join(os.path.dirname(__file__), "prompts", "phase_summary_prompt.txt"), "r") as f:
            prompt = f.read()
        
        # Format the prompt with the phase history and game state changes
        prompt = prompt.format(phase_history=phase_history, game_state_changes=game_state_changes)
        
        # ... rest of the function remains the same ...

    def get_observation(self):
        """Get the current player's observation, optionally adding a summary request"""
        player_id, observation = super().get_observation()
        
        # If we're requesting summaries, add the summary prompt
        if self.request_summaries:
            # Convert observation list to string for easier processing
            observation_str = "\n\n".join([f"[{self.state.role_mapping.get(sender_id, f'Player {sender_id}')}] {message}" 
                                          for sender_id, message in observation])
            
            # Add the summary prompt
            observation_str += self.summary_prompt
            
            # Return the modified observation
            return player_id, observation_str
        
        return player_id, observation

    def get_player_summary(self, player_id: int) -> Optional[str]:
        """Get the latest summary provided by a player"""
        return self.player_summaries.get(player_id)
    
    def get_all_summaries(self) -> Dict[int, str]:
        """Get all player summaries"""
        return self.player_summaries