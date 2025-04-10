def create_board_str(game_state: dict) -> str:
    allowed_letters = sorted(game_state.get("allowed_letters", []))
    word_history = game_state.get("word_history", [])

    def render_allowed_letters_lines():
        lines = []
        lines.append("┌─ ALLOWED LETTERS ──────────────────┐")
        lines.append("│                                    │")

        n = len(allowed_letters)
        if n <= 4:
            rows = [allowed_letters]
        elif n <= 6:
            rows = [allowed_letters[:2], allowed_letters[2:4], allowed_letters[4:]]
        else:
            top = allowed_letters[:2]
            middle = allowed_letters[2:5]
            bottom = allowed_letters[5:]
            rows = [top, middle, bottom]

        for row in rows:
            padding = (6 - len(row)) * 2
            line_top = " " * padding
            line_mid = " " * padding
            line_bot = " " * padding
            for ch in row:
                line_top += f"┌───┐ "
                line_mid += f"│ {ch.upper()} │ "
                line_bot += f"└───┘ "
            lines.append(f"│{line_top.rstrip():<36}│")
            lines.append(f"│{line_mid.rstrip():<36}│")
            lines.append(f"│{line_bot.rstrip():<36}│")
            lines.append("│                                    │")

        lines.append("└────────────────────────────────────┘")
        return lines

    def render_word_history_lines():
        lines = []
        lines.append("┌─ GAME HISTORY ───────────────────┐")
        lines.append("│                                  │")
        for i, word in enumerate(word_history):
            player = f"P{i % 2}"
            entry = f"{player}: {word.upper()} ({len(word):<2} letters)   "
            lines.append(f"│  {entry.ljust(32)}│")
        lines.append("│                                  │")
        lines.append("└──────────────────────────────────┘")
        return lines

    # Combine allowed letters and game history side by side
    left_box = render_allowed_letters_lines()
    right_box = render_word_history_lines()
    max_lines = max(len(left_box), len(right_box))
    left_box += [" " * len(left_box[0])] * (max_lines - len(left_box))
    right_box += [" " * len(right_box[0])] * (max_lines - len(right_box))

    combined = [
        f"{l:<50} {r}" for l, r in zip(left_box, right_box)
    ]

    return "\n".join(combined)
