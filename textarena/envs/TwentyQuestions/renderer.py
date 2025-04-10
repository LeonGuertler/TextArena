from typing import Dict, Any

def create_board_str(game_state: Dict[str, Any]) -> str:
    """Render the gamemaster view of the 20 Questions game state."""
    
    target_word = game_state.get("target_word", "???")
    history = game_state.get("history", [])

    lines = []
    lines.append("🎯 Game: 20 Questions")
    lines.append("-" * 40)
    lines.append(f"Target Word: {target_word}")
    lines.append(f"Questions Asked: {len(history)} / 20")
    lines.append("-" * 40)

    if history:
        lines.append("📜 History:")
        for i, (q, a) in enumerate(history, start=1):
            lines.append(f"{i}. Q: {q}")
            lines.append(f"   A: {a}")
    else:
        lines.append("No questions asked yet.")

    return "\n".join(lines)
