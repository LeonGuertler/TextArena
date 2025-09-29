"""
VendingMachine-specific observation wrapper.

This wrapper provides comprehensive context management for the VendingMachine environment,
maintaining complete historical information while providing role-specific visibility
(VM sees inventory, Demand doesn't).
"""

from typing import Dict, List, Optional, Tuple
import textarena as ta
from textarena.core import ObservationWrapper, Env, ObservationType


class VendingMachineObservationWrapper(ObservationWrapper):
    """
    Custom observation wrapper for VendingMachine environment.
    
    Features:
    - Maintains complete game history for both players
    - Role-specific information visibility (VM sees inventory, Demand doesn't)
    - Detailed descriptive format for historical events
    - Automatic context accumulation and formatting
    """
    
    def __init__(self, env: Env):
        super().__init__(env)
        # Store complete observations for each player
        self.full_observations: Dict[int, List[Tuple[int, str, ObservationType]]] = {}
        # Cache parsed game state information
        self.game_history: List[Dict] = []
        self.current_day = 1
        
    def _extract_game_info_from_observations(self, player_id: int) -> Dict:
        """Extract current game state from observations."""
        game_info = {
            'day': 1,
            'inventory': 0,
            'price': 7,
            'cost': 5,
            'max_days': 10
        }
        
        if player_id not in self.full_observations:
            return game_info
            
        # Look for game board observations to extract current state
        for sender_id, message, obs_type in self.full_observations[player_id]:
            if obs_type == ObservationType.GAME_BOARD:
                # Parse game board message for current state
                lines = message.split('\n')
                for line in lines:
                    if 'DAY' in line and '/' in line:
                        parts = line.split('/')
                        if len(parts) >= 2:
                            try:
                                game_info['day'] = int(parts[0].split()[-1])
                                game_info['max_days'] = int(parts[1].strip())
                            except:
                                pass
                    elif 'Price=$' in line:
                        try:
                            price_part = line.split('Price=$')[1].split(',')[0].strip()
                            game_info['price'] = int(price_part)
                        except:
                            pass
                    elif 'Cost=$' in line:
                        try:
                            cost_part = line.split('Cost=$')[1].strip()
                            game_info['cost'] = int(cost_part)
                        except:
                            pass
                    elif 'Inventory (visible to VM):' in line and player_id == 0:
                        try:
                            inventory_part = line.split('Inventory (visible to VM):')[1].strip()
                            game_info['inventory'] = int(inventory_part)
                        except:
                            pass
        
        return game_info
    
    def _extract_daily_events(self, player_id: int) -> List[str]:
        """Extract and format daily events from game action descriptions."""
        daily_events = []
        
        if player_id not in self.full_observations:
            return daily_events
            
        for sender_id, message, obs_type in self.full_observations[player_id]:
            if obs_type == ObservationType.GAME_ACTION_DESCRIPTION and sender_id == ta.GAME_ID:
                if 'concluded:' in message:
                    # Parse day conclusion message
                    # Format: "Day X concluded: restock=Y, requested=Z, sold=W, stock_end=V."
                    try:
                        day_part = message.split('Day ')[1].split(' concluded:')[0]
                        details_part = message.split('concluded: ')[1].rstrip('.')
                        
                        # Parse the details
                        details = {}
                        for item in details_part.split(', '):
                            key, value = item.split('=')
                            details[key] = int(value)
                        
                        # Format as detailed description
                        event = (f"Day {day_part}: VM restocked {details['restock']} units, "
                                f"Demand requested {details['requested']} units, "
                                f"sold {details['sold']} units, ending stock: {details['stock_end']}")
                        daily_events.append(event)
                    except:
                        # Fallback to original message if parsing fails
                        daily_events.append(message)
        
        return daily_events
    
    def _format_observation_for_player(self, player_id: int) -> str:
        """Format the complete observation string for a specific player."""
        if player_id not in self.full_observations:
            return ""
            
        # Get the initial prompt
        prompt = ""
        for sender_id, message, obs_type in self.full_observations[player_id]:
            if obs_type == ObservationType.PROMPT:
                prompt = message
                break
        
        # Get current game info
        game_info = self._extract_game_info_from_observations(player_id)
        
        # Get historical events
        daily_events = self._extract_daily_events(player_id)
        
        # Build the formatted observation
        observation_parts = []
        
        # Add the initial prompt
        if prompt:
            observation_parts.append(prompt)
        
        # Add current game status
        status_lines = [
            f"=== CURRENT STATUS ===",
            f"Day {game_info['day']} of {game_info['max_days']}",
            f"Price: ${game_info['price']} per unit, Restock cost: ${game_info['cost']} per unit"
        ]
        
        # Add inventory info for VM only
        if player_id == 0:  # VM player
            status_lines.append(f"Current inventory: {game_info['inventory']} units")
        else:  # Demand player
            status_lines.append("Current inventory: Hidden from Demand")
        
        observation_parts.append('\n'.join(status_lines))
        
        # Add game history if any
        if daily_events:
            observation_parts.append("=== GAME HISTORY ===")
            observation_parts.extend(daily_events)
        
        
        return '\n\n'.join(observation_parts)
    
    def observation(self, player_id: int, observation: Optional[List[Tuple[int, str, ObservationType]]]) -> str:
        """
        Process and format observations for the given player.
        
        Args:
            player_id: The ID of the player receiving the observation
            observation: List of new observations, or None to get current state
            
        Returns:
            Formatted observation string with complete context
        """
        if observation is None:
            return self._format_observation_for_player(player_id)
        
        # Initialize player's observation history if needed
        if player_id not in self.full_observations:
            self.full_observations[player_id] = []
        
        # Add new observations to the player's history
        self.full_observations[player_id].extend(observation)
        
        # Return the formatted observation
        return self._format_observation_for_player(player_id)
