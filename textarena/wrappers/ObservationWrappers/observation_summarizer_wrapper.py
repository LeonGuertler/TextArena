from typing import List, Tuple, Optional, Dict, Any
import textarena as ta
from gymnasium.core import ObservationWrapper

class ObservationSummarizer(ObservationWrapper):
    """
    A wrapper that adds a summary request to each observation and extracts the summary from the agent's response.
    Specifically designed for the DiplomacyEnv to provide concise game state summaries.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.latest_summary = None
        self.summary_prompt = "\n\n[Please provide a brief summary of the current game state in 3-5 sentences. Include your position, key threats, and opportunities.]"
    
    def observation(self, observation: str) -> str:
        """Add the summary request to the observation."""
        return observation + self.summary_prompt
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute step in the environment and extract the summary from the agent's response.
        
        Args:
            action: The agent's action, potentially containing a summary
            
        Returns:
            The original step return values from the wrapped environment
        """
        # Extract summary if present
        summary_start = action.find("[Please provide a brief summary")
        if summary_start != -1:
            # Find the response after the prompt
            response_start = action.find("\n", summary_start)
            if response_start != -1:
                # Extract everything from the response start to the next command marker or end
                next_command = action.find("[", response_start + 1)
                if next_command != -1:
                    summary = action[response_start+1:next_command].strip()
                else:
                    summary = action[response_start+1:].strip()
                
                # Store the summary
                self.latest_summary = summary
                
                # Remove the summary request and response from the action
                action = action[:summary_start] + (action[next_command:] if next_command != -1 else "")
        
        # Pass the action to the wrapped environment
        return self.env.step(action) 