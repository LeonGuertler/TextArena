import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def save_game_state(game_id: str, env: Any, agents: Dict[str, Any], 
                    issued_orders: Optional[Dict[str, List[str]]] = None, 
                    accepted_orders: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Saves environment and agent states to a JSON file: <game_id>.json.
    Also appends the current phase's game state, orders issued, and orders accepted
    to a phases list so that every turn is recorded in the same file.
    
    Args:
        game_id: Unique identifier for the game
        env: The game environment
        agents: Dictionary mapping power names to agent objects
        issued_orders: Orders issued by each power
        accepted_orders: Orders accepted by the game engine
    """
    if issued_orders is None:
        issued_orders = {}
    if accepted_orders is None:
        accepted_orders = {}

    # Get game state
    game_state = {}
    if hasattr(env, "get_game_state_dict"):
        game_state = env.get_game_state_dict()
    else:
        # Fallback to get_state if available
        game_state = env.game.get_state() if hasattr(env.game, "get_state") else {}
    
    # Get current phase
    current_phase = env.game.get_current_phase() if hasattr(env.game, "get_current_phase") else "unknown"
    
    # Prepare results dictionary (mapping unit to list of results)
    results = {}
    if hasattr(env, "get_results"):
        results = env.get_results()
    elif hasattr(env.game, "get_results"):
        results = env.game.get_results()
    
    # Get messages for this phase
    messages = []
    if hasattr(env, "get_phase_messages"):
        messages = env.get_phase_messages(current_phase)
    
    # Prepare the current phase record
    current_phase_record = {
        "name": current_phase,
        "state": game_state,
        "orders": issued_orders,
        "results": results,
        "messages": messages,
        "summary": json.dumps({"summary": f"PLACEHOLDER_SUMMARY_{current_phase}"})
    }
    
    # Add phase_summary if available
    if hasattr(env, "get_phase_summary"):
        current_phase_record["phase_summary"] = env.get_phase_summary(current_phase)

    fname = f"{game_id}.json"
    # Load existing data if file exists
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                save_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error reading existing game state from {fname}: {e}")
            save_dict = {
                "id": game_id,
                "map": getattr(env, "map_name", "PLACEHOLDER_MAP"),
                "rules": getattr(env, "rules", []),
                "phases": []
            }
    else:
        save_dict = {
            "id": game_id,
            "map": getattr(env, "map_name", "PLACEHOLDER_MAP"),
            "rules": getattr(env, "rules", []),
            "phases": []
        }

    # Get or initialize phases list
    phases = save_dict.get("phases", [])
    
    # Check if we already have this phase
    phase_exists = False
    for i, phase in enumerate(phases):
        if phase.get("name") == current_phase:
            phases[i] = current_phase_record
            phase_exists = True
            break
    
    # If phase doesn't exist, append it
    if not phase_exists:
        phases.append(current_phase_record)
    
    # Update phases in save_dict
    save_dict["phases"] = phases

    # Save to file
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)
        logger.info(f"Game state saved to {fname}")
    except Exception as e:
        logger.error(f"Error saving game state to {fname}: {e}")

def load_game_state(game_id: str) -> tuple:
    """
    Loads a game state from a JSON file.
    
    Args:
        game_id: Unique identifier for the game
        
    Returns:
        Tuple of (environment, agents) or (None, None) if loading fails
    """
    fname = f"{game_id}.json"
    if not os.path.isfile(fname):
        logger.warning(f"Game state file {fname} not found")
        return None, None

    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loading game state from {fname}...")

        # This part needs to be adapted based on your specific environment and agent classes
        # The following is a placeholder that should be modified
        from textarena.envs.Diplomacy.env import DiplomacyEnv
        
        env = DiplomacyEnv()
        
        # Get the latest phase's state
        if data.get("phases") and len(data["phases"]) > 0:
            latest_phase = data["phases"][-1]
            latest_state = latest_phase.get("state", {})
            
            if hasattr(env.game, "set_state"):
                env.game.set_state(latest_state)
        
        # Set environment attributes
        env.map_name = data.get("map", "PLACEHOLDER_MAP")
        env.rules = data.get("rules", [])
        
        # Load agents - adapt this to your agent class
        agents = {}
        
        # You'll need to implement agent loading based on your specific agent class
        # This is a placeholder
        from textarena.envs.Diplomacy.agent import DiplomacyAgent
        
        # Create agents for each power in the game
        if data.get("phases") and len(data["phases"]) > 0:
            latest_phase = data["phases"][-1]
            latest_state = latest_phase.get("state", {})
            
            for power in latest_state.get("units", {}):
                agent = DiplomacyAgent(power_name=power)
                agents[power] = agent

        return env, agents
    except Exception as e:
        logger.error(f"Error loading game from {fname}: {e}")
        return None, None
