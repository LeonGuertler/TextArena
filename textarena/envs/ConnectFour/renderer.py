from typing import Optional 


def create_board_str(board, player_id: Optional[int]=None) -> str:

    # Dimensions
    num_rows = len(board)        # e.g., 6
    num_cols = len(board[0])     # e.g., 7

    # Helper to convert '.' to a blank space, or keep 'X'/'O' as is (centered in 3 spaces).
    def cell_str(value: str) -> str:
        return "   " if value == "." else f" {value} "

    # 1. Header
    lines = []
    lines.append("  "+"   ".join(str(col) for col in range(num_cols)))

    # 3. Top border (e.g. "                                ┌───┬───┬───┬───┬───┬───┬───┐")
    #    Generate it dynamically based on num_cols
    def top_border(cols: int) -> str:
        return "┌" + "───┬" * (cols - 1) + "───┐"

    # 4. Middle border (between rows)
    def mid_border(cols: int) -> str:
        return "├" + "───┼" * (cols - 1) + "───┤"

    # 5. Bottom border
    def bottom_border(cols: int) -> str:
        return "└" + "───┴" * (cols - 1) + "───┘"

    lines.append(top_border(num_cols))

    # 6. Rows (note that row 0 is at the *top* in this environment)
    for r in range(num_rows):
        row_values = board[r]

        # Build the line for this row
        # e.g. "│ X │ O │   │ ... │"
        row_line = "│" + "│".join(cell_str(val) for val in row_values) + "│"
        lines.append(row_line)

        # If this is not the last row, add a middle separator;
        # otherwise close off with a bottom border.
        if r < num_rows - 1:
            lines.append(mid_border(num_cols))
        else:
            lines.append(bottom_border(num_cols))

    # Combine into one string
    return "\n".join(lines)
