def create_board_str(game_state: dict) -> str:
    """Render the full Wordle board as ASCII with emoji box feedback."""
    history = game_state.get("guess_history", [])
    target_word = game_state.get("secret_word", "").upper()
    word_length = len(target_word) if target_word else 5

    # Emoji mapping
    color_map = {"G": "🟩", "Y": "🟨", "X": "⬜"}

    lines = []
    lines.append("+" + "-" * 32 + "+")
    lines.append("|{:^32}|".format("WORDLE GAME"))
    lines.append("+" + "-" * 32 + "+")

    if not history:
        lines.append("|{:^32}|".format("No guesses yet."))
        lines.append("+" + "-" * 32 + "+")
    else:
        for i, (word, feedback) in enumerate(history):
            word_str = "   ".join(c.upper() for c in word)
            feedback_str = "  ".join(color_map.get(f, "?") for f in feedback)
            lines.append(f"Guess  #{i+1}: {word_str}")
            lines.append(f"Feedback : {feedback_str}")
            lines.append("")

    lines.append("+" + "-" * 32 + "+")
    lines.append("| Target Word: {:>16}  |".format(target_word))
    lines.append("+" + "-" * 32 + "+")

    return "\n".join(lines)
