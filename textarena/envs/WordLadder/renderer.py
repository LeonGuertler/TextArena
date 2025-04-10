from typing import Dict, Any

def create_board_str(game_state: Dict[str, Any]) -> str:
    history = game_state.get("rendered_text", "")
    start_word = game_state.get("start_word", "?????")
    target_word = game_state.get("target_word", "?????")
    history_words = history.split("Word Ladder History: ")[-1].split(". Target Word")[0].split(" -> ")

    width = max(len(w) for w in history_words + [start_word, target_word])
    border = "+" + "-" * (width + 12) + "+"
    lines = [border]
    lines.append(f"| {'WORD LADDER':^{width + 10}} |")
    lines.append(border)
    lines.append(f"| Start : {start_word:<{width}}   |")
    lines.append(f"| Target: {target_word:<{width}}   |")
    lines.append(border)

    for i, word in enumerate(history_words):
        marker = "→" if i < len(history_words) - 1 else "★"
        lines.append(f"| Step {i+1}: {word:<{width}} {marker} |")

    lines.append(border)
    return "\n".join(lines)
