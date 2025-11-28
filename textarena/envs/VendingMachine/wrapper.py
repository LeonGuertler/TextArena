"""
VendingMachine-specific observation wrapper.

This wrapper provides comprehensive context management for the VendingMachine environment,
maintaining complete historical information while providing role-specific visibility
(VM sees inventory, Demand doesn't).

Supports multi-item vending machine with lead time and inventory pipeline.
"""

from typing import Dict, List, Optional, Tuple, Any
import textarena as ta
from textarena.core import ObservationWrapper, Env, ObservationType


class VendingMachineObservationWrapper(ObservationWrapper):
    """
    Custom observation wrapper for VendingMachine environment.
    
    Features:
    - Maintains complete game history for both players
    - Role-specific information visibility (VM sees inventory pipeline, Demand doesn't)
    - Detailed descriptive format for historical events
    - Automatic context accumulation and formatting
    - Multi-item support with lead time tracking
    """
    
    def __init__(self, env: Env):
        super().__init__(env)
        # Store complete observations for each player
        self.full_observations: Dict[int, List[Tuple[int, str, ObservationType]]] = {}
        # Cache parsed game state information
        self.game_history: List[Dict[str, Any]] = []
        self.current_day = 1
    
    def reset(self, num_players: int, seed: Optional[int] = None, **kwargs):
        """Reset the wrapper state along with the environment."""
        # Clear wrapper state
        self.full_observations = {}
        self.game_history = []
        self.current_day = 1
        
        # Call parent's reset (which calls env.reset)
        return super().reset(num_players=num_players, seed=seed, **kwargs)
        
    def _extract_game_info_from_observations(self, player_id: int) -> Dict[str, Any]:
        """Extract current game state from observations for multi-item environment."""
        game_info = {
            'day': 1,
            'max_days': 3,
            'items': {}
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
        
        return game_info
    
    def _extract_daily_events(self, player_id: int) -> List[str]:
        """Extract and format daily events from game action descriptions (multi-item)."""
        daily_events = []
        
        if player_id not in self.full_observations:
            return daily_events
            
        for sender_id, message, obs_type in self.full_observations[player_id]:
            if obs_type == ObservationType.GAME_ACTION_DESCRIPTION and sender_id == ta.GAME_ID:
                # Capture day conclusions: visible to both players
                # Note: "VM ordered" messages are removed as ordering info is now in day conclusion
                if 'conclude' in message:
                    daily_events.append(message)
        
        return daily_events
    
    def _format_observation_for_player(self, player_id: int) -> str:
        """Format the complete observation string for a specific player (multi-item)."""
        if player_id not in self.full_observations:
            return ""
            
        # Get the initial prompt
        prompt = ""
        for sender_id, message, obs_type in self.full_observations[player_id]:
            if obs_type == ObservationType.PROMPT:
                prompt = message
                break
        
        # Get the latest game board (env already provides complete item info)
        latest_game_board = ""
        for sender_id, message, obs_type in reversed(self.full_observations[player_id]):
            if obs_type == ObservationType.GAME_BOARD:
                latest_game_board = message
                break
        
        # Get historical events
        daily_events = self._extract_daily_events(player_id)
        
        # Build the formatted observation
        observation_parts = []
        
        # Add the initial prompt if exists
        if prompt:
            observation_parts.append(prompt)
        
        # Add current game board (already formatted by env)
        if latest_game_board:
            observation_parts.append("=== CURRENT STATUS ===")
            observation_parts.append(latest_game_board)
        
        # Add game history if any
        if daily_events:
            observation_parts.append("\n=== GAME HISTORY ===")
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
