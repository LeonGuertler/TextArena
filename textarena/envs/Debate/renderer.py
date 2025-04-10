import textwrap

def create_board_str(game_state: dict) -> str:
    topic = game_state.get("topic", "Unknown Topic")
    sides = game_state.get("sides", {})
    votes = game_state.get("votes", {
        "pre-debate": {"Affirmative": 0, "Negative": 0},
        "post-debate": {"Affirmative": 0, "Negative": 0}
    })

    # Wrap topic to max 75 chars per line, max 2 lines
    wrapped_topic = textwrap.wrap(topic, width=75)[:2]
    if len(wrapped_topic) < 2:
        wrapped_topic.append("")  # Ensure we always have two lines

    player0_side = sides.get(0, "Unassigned")
    player1_side = sides.get(1, "Unassigned")

    lines = []
    lines.append(f"┌─ DEBATE TOPIC ─────────────────────────────────────────────────────────────┐")
    for line in wrapped_topic:
        lines.append(f"│ {line.ljust(75)}│")
    lines.append(f"├────────────────────────────────────────────────────────────────────────────┤")
    lines.append(f"│ Player 0: {player0_side.ljust(65)}│")
    lines.append(f"│ Player 1: {player1_side.ljust(65)}│")
    lines.append(f"├────────────────────────────────────────────────────────────────────────────┤")
    lines.append(f"│ Pre-debate Votes:  Affirmative: {votes['pre-debate']['Affirmative']:.2f}    "
                 f"Negative: {votes['pre-debate']['Negative']:.2f}                     │")
    lines.append(f"│ Post-debate Votes: Affirmative: {votes['post-debate']['Affirmative']:.2f}    "
                 f"Negative: {votes['post-debate']['Negative']:.2f}                     │")
    lines.append(f"└────────────────────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)
