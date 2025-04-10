from typing import Dict, Any

def create_board_str(game_state: Dict[str, Any]) -> str:
    """
    Render the entire logic puzzle board.

    This function assumes the game_state contains a 'board' key with
    a structure like {grid_name: {row: {col: mark}}}.
    
    Returns:
        str: Rendered board as a string.
    """
    board = game_state.get("board", {})
    
    if not board:
        return "The board is empty or not initialized."
    
    # Dynamically calculate the rendering
    output = []

    for grid_name, grid_data in board.items():
        # Get all columns
        columns = list(next(iter(grid_data.values())).keys())
        row_names = list(grid_data.keys())

        # Calculate layout widths
        max_row_width = max(len(r) for r in row_names) + 2
        max_col_width = max(len(c) for c in columns) + 2

        # Header
        output.append(f"\n{'=' * (max_row_width + len(columns) * (max_col_width + 3))}")
        output.append(f"{grid_name.upper().center(max_row_width + len(columns) * (max_col_width + 3))}")
        output.append(f"{'=' * (max_row_width + len(columns) * (max_col_width + 3))}")

        # Column headers
        header = " " * max_row_width + " | ".join(f"{col:^{max_col_width}}" for col in columns) + " |"
        output.append(header)
        output.append("-" * len(header))

        # Rows
        for row in row_names:
            line = f"{row:<{max_row_width}}" + " | ".join(
                f"{grid_data[row][col] if grid_data[row][col] else ' ':^{max_col_width}}" for col in columns
            ) + " |"
            output.append(line)

        output.append("=" * len(header))

    return "\n".join(output)
