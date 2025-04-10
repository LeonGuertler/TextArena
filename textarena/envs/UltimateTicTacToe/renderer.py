from typing import List

def create_board_str(board: List[List[List[str]]]) -> str:
    lines = []
    for macro_row in range(3):
        for micro_row in range(3):
            row_line = []
            for macro_col in range(3):
                row_line.extend(board[macro_row * 3 + macro_col][micro_row])
                row_line.append('|')
            # Remove the last '|'
            row_render = ' '.join(row_line[:-1])
            lines.append(row_render)
        # Separator row between macro rows
        if macro_row < 2:
            lines.append('-' * 21)
    return "\n".join(lines)

