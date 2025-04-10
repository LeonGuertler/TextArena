def create_board_str(game_state: dict) -> str:
    item_names = game_state.get("item_names", [])
    base_values = game_state.get("base_item_values", [])
    player_values = game_state.get("player_item_values", {})
    remaining_capital = game_state.get("remaining_capital", {})
    player_bids = game_state.get("player_bids", {})
    phase = game_state.get("phase", "")
    round_num = game_state.get("round", 0)

    num_items = len(item_names)
    lines = []

    # Phase Header
    lines.append(f"╭── SIMPLE BLIND AUCTION ───── Phase: {phase.capitalize():<10} | Round: {round_num} ──────────────────────╮")
    lines.append("│ Item │ Item Name                                                     │ Base Value │")
    lines.append("├──────┼───────────────────────────────────────────────────────────────┼────────────┤")
    for idx, (name, val) in enumerate(zip(item_names, base_values)):
        lines.append(f"│ {idx:^4} │ {name:<61} │ {val:>10} │")
    lines.append("╰───────────────────────────────────────────────────────────────────────────────────╯")

    # Player-specific subjective values
    lines.append("┌─ PLAYER-SPECIFIC ITEM VALUES ────────────────────────────────────────────────────┐")
    header_str = " ".join([f"I{idx:<2}" for idx in range(num_items)])
    lines.append("│ Player │ " + f"{header_str:<72}" + "│")
    lines.append("├────────┼─────────────────────────────────────────────────────────────────────────┤")
    for pid in [0, 1]:
        value_str = " ".join(f"{player_values[pid][i]:<4}" for i in range(num_items))
        lines.append(f"│   {pid:<4} │ {value_str:<72}│")
    lines.append("└──────────────────────────────────────────────────────────────────────────────────┘")

    # Remaining Capital
    lines.append("┌─ PLAYER CAPITAL ────────────┐")
    for pid in [0, 1]:
        cap = remaining_capital.get(pid, 0)
        lines.append(f"│ Player {pid}: {cap:<6} coins      │")
    lines.append("└─────────────────────────────┘")

    # Show bids if in bidding phase
    if phase == "bidding":
        lines.append("┌──── PLAYER BIDS ───────────────────────────────────┐")
        header_str = " ".join([f"I{idx:<2}" for idx in range(num_items)])
        lines.append("│ Player │ " + f"{header_str:<42}" + "│")
        lines.append("├────────┼───────────────────────────────────────────┤")
        for pid in [0, 1]:
            bids = player_bids.get(pid, {})
            bid_str = " ".join(f"{bids.get(i, 0):<4}" for i in range(num_items))
            lines.append(f"│   {pid:<4} │ {bid_str:<42}│")
        lines.append("└────────────────────────────────────────────────────┘")

    return "\n".join(lines)
