from typing import List, Optional

def create_board_str(game_state) -> str:
    board: List[List[Optional[int]]] = game_state["board"]
    header = "     " + "   ".join([f"C{i}" for i in range(4)])
    lines = [header]
    lines.append("   +" + "----+" * 4)
    for i, row in enumerate(board):
        row_str = f"R{i} |"
        for tile in row:
            cell = f"{tile:<2}" if tile is not None else "__"
            row_str += f" {cell} |"
        lines.append(row_str)
        lines.append("   +" + "----+" * 4)
    return "\n".join(lines)
