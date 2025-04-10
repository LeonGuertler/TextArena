from typing import Dict, Any, List, Tuple

def create_board_str(game_state: Dict[str, Any]) -> str:
    """
    Render a Liar's Dice game board in ASCII format.
    - game_state["dice_rolls"]: Dict[int, List[int]]
    - game_state["remaining_dice"]: Dict[int, int]
    - game_state["current_bid"]: {"quantity": int, "face_value": int}
    """

    dice_rolls = game_state.get("dice_rolls", {})
    remaining_dice = game_state.get("remaining_dice", {})
    current_bid = game_state.get("current_bid", {"quantity": 0, "face_value": 0})

    num_players = len(remaining_dice)

    def draw_dice_row(dice: List[int]) -> Tuple[str, str, str]:
        # Build one line per layer
        tops = "   " + "  ".join("┌───┐" for _ in dice)
        faces = "   " + "  ".join(f"│ {d} │" for d in dice)
        bottoms = "   " + "  ".join("└───┘" for _ in dice)
        return tops, faces, bottoms

    def draw_player_box(player_id: int, dice: List[int]) -> str:
        top_border = f"┌─ PLAYER {player_id} ─{'─' * 26}┐"
        empty_line = "│" + " " * (len(top_border) - 2) + "│"
        tops, faces, bottoms = draw_dice_row(dice)
        return "\n".join([
            top_border,
            empty_line,
            f"│ {tops:<{len(top_border)-4}} │",
            f"│ {faces:<{len(top_border)-4}} │",
            f"│ {bottoms:<{len(top_border)-4}} │",
            empty_line,
            f"└{'─' * (len(top_border) - 2)}┘"
        ])

    lines = []

    # DISPLAY STYLE SPLIT
    if num_players <= 5:
        for pid in range(num_players):
            if remaining_dice.get(pid, 0) == 0:
                continue
            dice = dice_rolls.get(pid, [])
            lines.append(draw_player_box(pid, dice))
            lines.append("")
    else:
        # Compact summary for > 5 players
        lines.append("Players & Dice:")
        for pid in range(num_players):
            dice = dice_rolls.get(pid, [])
            dice_str = " ".join(str(d) for d in dice)
            lines.append(f"  Player {pid} ({remaining_dice[pid]} dice): {dice_str}")
        lines.append("")

    # BID BOX
    q = current_bid.get("quantity", 0)
    f = current_bid.get("face_value", 0)
    bid_box = f"""
┌─────────────────────────┐
│ Current Bid: {q} × face {f:<2}│
└─────────────────────────┘
""".strip()
    lines.append(bid_box)

    return "\n".join(lines)
