def create_board_str(board, board_size) -> str:
    max_cell_num = board_size * board_size - 1
    digit_count = len(str(max_cell_num))
    def cell_str(r: int, c: int) -> str:
        return " " if board[r][c] == '' else board[r][c] 

    def build_hline() -> str:
        line_parts = []
        for _ in range(board_size):
            line_parts.append("-" * 3) 
        return "+" + "+".join(line_parts) + "+"

    lines = []
    lines.append(build_hline())
    for r in range(board_size):
        row_cells = []
        for c in range(board_size):
            text = cell_str(r, c)
            text_centered = f" {text:^{1}} "
            row_cells.append(text_centered)
        row_line = "|" + "|".join(row_cells) + "|"
        lines.append(row_line)
        lines.append(build_hline())
    return "\n".join(lines)