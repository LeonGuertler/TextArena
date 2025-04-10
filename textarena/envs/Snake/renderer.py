from typing import Dict, List, Tuple, Any


def create_board_str(width: int, height: int, snakes: Dict[int, Any], apples: List[Tuple[int, int]]) -> str:
    """ Create an ASCII board representation. Top row is printed last """
    board = [['.' for _ in range(width)] for _ in range(height)]

    # Place apples
    for (ax, ay) in apples:
        board[ay][ax] = 'A'

    # Place snake body and head
    for pid, snake in snakes.items():
        if not snake.alive:
            continue
        for idx, (x, y) in enumerate(snake.positions):
            if idx == 0:
                board[y][x] = format(pid, 'X')  # Use hex digit for player ID
            else:
                board[y][x] = '#'

    lines = []
    content_width = width * 2 - 1
    border_line = "+" + "-" * (content_width + 2) + "+"
    lines.append(border_line)

    # Draw board from top (height-1) to bottom (0)
    for row_idx in range(height - 1, -1, -1):
        row_str = " ".join(board[row_idx])
        lines.append(f"| {row_str} |")

    lines.append(border_line)
    return "\n".join(lines)
