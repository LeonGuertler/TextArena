import textwrap

def create_board_str(game_state: dict) -> str:
    scenario = game_state.get("scenario", "Unknown Scenario")
    strategies = game_state.get("strategies", {})
    votes = game_state.get("votes", {0: {"Votes": 0}, 1: {"Votes": 0}})

    # Wrap scenario to max 75 chars per line, max 3 lines
    wrapped_scenario = textwrap.wrap(scenario, width=75)[:5]
    while len(wrapped_scenario) < 5:
        wrapped_scenario.append("")

    lines = []
    lines.append("┌─ SCENARIO ─────────────────────────────────────────────────────────────────┐")
    for line in wrapped_scenario:
        lines.append(f"│ {line.ljust(75)}│")
    lines.append("├────────────────────────────────────────────────────────────────────────────┤")

    lines.append("│ 🗳️  Judge Votes:")
    lines.append(f"│   Player 0: {votes[0]['Votes']} votes".ljust(75) + "  │")
    lines.append(f"│   Player 1: {votes[1]['Votes']} votes".ljust(75) + "  │")
    lines.append("└────────────────────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)
