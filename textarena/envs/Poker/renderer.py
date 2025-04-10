from typing import List, Dict, Tuple

def card_block(rank: str, suit: str) -> List[str]:
    """Return an ASCII art block of a card as a list of strings."""
    return [
        "┌─────────┐",
        f"│ {rank:<2}      │",
        "│         │",
        f"│    {suit}    │",
        "│         │",
        f"│      {rank:>2} │",
        "└─────────┘",
    ]

def render_card_row(cards: List[Tuple[str, str]]) -> List[str]:
    """Render a row of cards."""
    lines = ["", "", "", "", "", "", ""]
    for rank, suit in cards:
        block = card_block(rank, suit)
        for i in range(7):
            lines[i] += f"{block[i]} "
    return lines

def create_board_str(
    community_cards: List[Dict[str, str]], 
    pot: int, 
    player_chips: Dict[int, int], 
    player_hands: Dict[int, List[Dict[str, str]]],
    bets: Dict[int, float]) -> str:
    output = []

    # Community cards block
    output.append("┌─ COMMUNITY CARDS " + "─" * 52 + "┐")
    output.append(f"│   Pot: {pot:<3}" + " " * 59 + "│")
    output.append("│" + " " * 70 + "│")

    card_slots = community_cards + [{"rank": "?", "suit": "?"}] * (5 - len(community_cards))
    community_row = render_card_row([(card["rank"], card["suit"]) for card in card_slots])
    for line in community_row:
        output.append(f"│     {line:<65}│")

    output.append("│" + " " * 70 + "│")
    output.append("└" + "─" * 70 + "┘\n")

    # Player rows (up to 5 per row)
    players = list(player_chips.items())
    rows = [players[i:i + 5] for i in range(0, len(players), 5)]

    for row in rows:
        header = ""
        chips_line = ""
        empty_line = ""
        card_lines = [""] * 7
        footer = ""

        for pid, chips in row:
            header += f"┌─ PLAYER {pid} " + "─" * 14 + "┐     "
            # chips_line += f"│ Chips: {chips:<5}" + " " * 12 + "│     "
            chips_line += f"│ Chips: {chips:<5}" + f"   Bet: {bets[pid]:<4}" + "│     "
            empty_line += "│" + " " * 25 + "│     "

            hand = player_hands[pid]
            cards = render_card_row([(hand[0]['rank'], hand[0]['suit']), (hand[1]['rank'], hand[1]['suit'])])
            for i in range(7):
                card_lines[i] += f"│ {cards[i]}│     "

            footer += "└" + "─" * 25 + "┘     "

        output.append(header.rstrip())
        output.append(chips_line.rstrip())
        output.append(empty_line.rstrip())
        for line in card_lines:
            output.append(line.rstrip())
        output.append(empty_line.rstrip())
        output.append(footer.rstrip())

    return "\n".join(output)
