from typing import Dict, List, Tuple, Any

def create_board_str(width: int, height: int, game_state: Dict[str, Any]) -> str:
    """Create a board string for the Surround game using a similar style to Snake rendering."""
    board = [['.' for _ in range(width)] for _ in range(height)]

    # Fill in trails
    trail_data = game_state["board"]
    for y in range(height):
        for x in range(width):
            if trail_data[y][x] is not None:
                board[y][x] = '#'

    # Fill in live player heads
    for pid, pdata in game_state["players"].items():
        if pdata["alive"]:
            px, py = pdata["position"]
            board[py][px] = format(pid, 'X')  # Hex digit for player ID

    # Create visual representation
    lines = []
    content_width = width * 2 - 1
    border_line = "+" + "-" * (content_width + 2) + "+"
    lines.append(border_line)

    for row_idx in range(height - 1, -1, -1):
        row_str = " ".join(board[row_idx])
        lines.append(f"| {row_str} |")

    lines.append(border_line)
    return "\n".join(lines)
