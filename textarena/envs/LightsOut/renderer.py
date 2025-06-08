from typing import Dict, Any, List

def create_board_str(game_state: Dict[str, Any]) -> str:
    """
    Render the Lights Out board with lit cells shaded
    """
    board: List[List[int]] = game_state.get("board", [])
    if not board:
        return "Board not available."

    grid_size = len(board)
    on_symbol = "███"  
    off_symbol = "   "  

    def cell_display(r: int, c: int) -> str:
        """Returns the 3-character representation for a cell"""
        return on_symbol if board[r][c] == 1 else off_symbol

    def horizontal_line(left: str, mid: str, right: str) -> str:
        """Creates a horizontal line for the grid border"""
        return "  " + left + ("─" * 3 + mid) * (grid_size - 1) + "─" * 3 + right

    header = "    " + "   ".join(f"{c:^1}" for c in range(grid_size))
    lines = [header]
    lines.append(horizontal_line("┌", "┬", "┐"))

    for r in range(grid_size):
        row_content = "│".join(cell_display(r, c) for c in range(grid_size))
        lines.append(f"{r} │{row_content}│")
        if r < grid_size - 1:
            lines.append(horizontal_line("├", "┼", "┤"))

    lines.append(horizontal_line("└", "┴", "┘"))

    return "\n".join(lines)