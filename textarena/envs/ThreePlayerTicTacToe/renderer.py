from typing import Dict, Any

def create_board_str(game_state: Dict[str, Any]) -> str:
    board = game_state["board"]
    board_size = len(board)
    max_index = board_size * board_size - 1
    cell_width = max(2, len(str(max_index)))

    def cell_display(r: int, c: int) -> str:
        val = board[r][c]
        return val if val else " "

    def horizontal_line() -> str:
        return "+" + "+".join(["-" * (cell_width + 2) for _ in range(board_size)]) + "+"

    lines = [horizontal_line()]
    for r in range(board_size):
        row = "|".join(f" {cell_display(r, c):^{cell_width}} " for c in range(board_size))
        lines.append(f"|{row}|")
        lines.append(horizontal_line())

    return "\n".join(lines)
