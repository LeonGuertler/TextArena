# textarena/envs/IteratedPrisonersDilemma/renderer.py

def create_board_str(game_state: dict) -> str:
    current_round = game_state.get("current_round", 1)
    comm_turn = game_state.get("current_comm_turn", 0)
    is_decision_phase = game_state.get("is_decision_phase", False)
    scores = game_state.get("scores", {})
    history = game_state.get("history", [])
    round_decisions = game_state.get("round_decisions", {})

    lines = []

    # Phase display
    phase_str = "🗣️ Communication Phase" if not is_decision_phase else "🎯 Decision Phase"
    lines.append(f"╭── Iterated Prisoner's Dilemma ─────────────────────╮")
    lines.append(f"│ Round: {current_round:<2} | Turn: {comm_turn:<2} | Phase: {phase_str:<20}│")
    lines.append(f"╰────────────────────────────────────────────────────╯")

    # Scoreboard
    lines.append("┌─ 📊 SCORES ─────────────────────┐")
    lines.append(f"│ Player 0: {scores.get(0, 0):<3} pts               │")
    lines.append(f"│ Player 1: {scores.get(1, 0):<3} pts               │")
    lines.append("└─────────────────────────────────┘")

    # Round decision history
    if history:
        lines.append("┌─ 📜 ROUND HISTORY ────────────────────────────────────┐")
        lines.append("│ Round │ Player 0     │ Player 1     │ Outcome         │")
        lines.append("├───────┼──────────────┼──────────────┼─────────────────┤")
        for round_info in history:
            r = round_info.get("round", "?")
            d0 = round_info["decisions"].get(0, "?").capitalize()
            d1 = round_info["decisions"].get(1, "?").capitalize()
            if d0 == d1 == "Cooperate":
                outcome = "Both Cooperated"
            elif d0 == d1 == "Defect":
                outcome = "Both Defected"
            elif d0 == "Cooperate" and d1 == "Defect":
                outcome = "P0 Sucker"
            elif d0 == "Defect" and d1 == "Cooperate":
                outcome = "P1 Sucker"
            else:
                outcome = "Unknown"

            lines.append(f"│ {r:^5} │ {d0:<12} │ {d1:<12} │ {outcome:<8} │")
        lines.append("└───────────────────────────────────────────────────────┘")

    return "\n".join(lines)
