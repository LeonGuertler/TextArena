from typing import List

def create_board_str(board: List[List[str]]) -> str:
    return "\n---+---+---\n".join(
            "|".join(
                f" {board[r][c]} " if board[r][c] else f"   "
                for c in range(3)
            )
            for r in range(3)
        )
