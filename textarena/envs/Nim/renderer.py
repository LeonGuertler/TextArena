from typing import List 

def create_board_str(piles: List[int]) -> str:
    max_tokens = max(piles) if piles else 0
    lines = []
    # Each row
    for i, count in enumerate(piles):
        token_cells = [f"│ {'●' if j < count else ' '}" for j in range(max_tokens)]
        token_row = " ".join(token_cells) + " │"
        border_row = "       "+"└───" * max_tokens + "┘"
        lines.append(f"Row {i + 1}: {token_row}")
        lines.append(border_row + "\n")

    return "\n".join(lines).rstrip()
