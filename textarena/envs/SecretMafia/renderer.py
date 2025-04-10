def create_board_str(game_state: dict) -> str:
    phase = game_state.get("phase", "Unknown")
    day_number = game_state.get("day_number", 0)
    alive_players = set(game_state.get("alive_players", []))
    player_roles = game_state.get("player_roles", {})
    votes = game_state.get("votes", {})
    to_be_eliminated = game_state.get("to_be_eliminated", None)

    role_icons = {
        "Villager": "👨‍🌾 Villager ",
        "Mafia":    "😈 Mafia    ",
        "Doctor":   "🧑‍⚕️ Doctor   ",
        "Detective":"🕵️ Detective"
    }

    # Header
    lines = []
    lines.append(f"┌─ SECRET MAFIA ─────────────── Phase: {phase:<15} | Day: {day_number} ──────────────┐")
    lines.append("│ Player Status                                                               │")
    lines.append("├────────────┬──────────────┬─────────────────────────────────────────────────┤")
    lines.append("│ Player ID  │   Status     │   Role                                          │")
    lines.append("├────────────┼──────────────┼─────────────────────────────────────────────────┤")

    for pid in sorted(player_roles):
        alive = pid in alive_players
        status = "🟢 Alive" if alive else "⚫️ Dead "
        role = role_icons.get(player_roles[pid]) #, player_roles[pid])
        lines.append(f"│ Player {pid:<3} │ {status:<12} │ {role:<42}    │")

    lines.append("└────────────┴──────────────┴──────────────────────────────────────────────┘")

    # Vote summary
    if phase in {"Day-Voting", "Night-Mafia"}:
        lines.append("\n🗳️ VOTES")
        if votes:
            for voter, target in sorted(votes.items()):
                lines.append(f" - Player {voter} ➜ Player {target}")
        else:
            lines.append(" - No votes have been cast yet.")

    # Pending elimination
    if to_be_eliminated is not None:
        lines.append(f"\n🪦 Player {to_be_eliminated} is marked for elimination.")

    # Team count summary
    mafia_alive = sum(1 for pid in alive_players if player_roles[pid] == "Mafia")
    village_alive = sum(1 for pid in alive_players if player_roles[pid] != "Mafia")
    lines.append(f"\n🔍 Team Breakdown: 😈 Mafia: {mafia_alive} | 🧑‍🌾 Villagers: {village_alive}")

    return "\n".join(lines)
