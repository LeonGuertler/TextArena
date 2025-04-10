def create_board_str(game_state: dict) -> str:
    phase = game_state.get("phase", "unknown")
    budget = game_state.get("budget_remaining", {})
    votes = game_state.get("votes", {})

    lines = []

    if phase == "discussion":
        lines.append("┌─ CHARACTER CONCLAVE ──────── Phase: Discussion ──────────────────────────┐")
        lines.append("│ Player ID │ Characters Remaining                                         │")
        lines.append("├───────────┼──────────────────────────────────────────────────────────────┤")
        for pid in sorted(budget.keys()):
            remaining = budget[pid]
            lines.append(f"│    {pid:<7}│ {remaining:<61}│")
        lines.append("└──────────────────────────────────────────────────────────────────────────┘")

    elif phase == "voting":
        lines.append("┌─ CHARACTER CONCLAVE ──────────── Phase: Voting ──────────────────────────┐")
        lines.append("│ Players who have voted:                                                  │")
        voted = set(votes.keys())
        for pid in sorted(budget.keys()):
            mark = "✔" if pid in voted else " "
            lines.append(f"│   [{mark}] Player {pid:<2}                                                          │")
        lines.append("├──────────────────────────────────────────────────────────────────────────┤")

        # If all players have voted, show final vote tally
        if len(votes) == len(budget):
            tally = {}
            for voter, voted_for in votes.items():
                tally[voted_for] = tally.get(voted_for, 0) + 1

            lines.append("│ Final Vote Count:                                                       │")
            for pid in sorted(tally.keys()):
                lines.append(f"│   Player {pid}: {tally[pid]} vote(s)                                                    │")
            lines.append("└──────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("│ Waiting for all players to submit their votes...                         │")
            lines.append("└──────────────────────────────────────────────────────────────────────────┘")

    else:
        lines.append("┌─ CHARACTER CONCLAVE ─────────────────────────────────────────────────────┐")
        lines.append("│ Unknown game phase.                                                      │")
        lines.append("└──────────────────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)
