'''
Renderer for the Chess environment.
'''

def create_board_str(board_str, grid_size=(3, 3)):
    """
    Generate a string representation of the chess board with pieces using Unicode symbols.
    """

    # Color codes referenced from the CursesRenderWrapper
    COORD_COLOR = 38
    ICON_COLOR = 39
    LIGHT_SQUARE = 52
    DARK_SQUARE = 53

    # Split the board string into rows (8 rows of 8 squares)
    rows = board_str.strip().split('\n')
    
    # Chess pieces mapped to Unicode symbols
    piece_symbols = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        '.': ' '  # Empty square
    }
    
    # Unpack grid size (rows, columns)
    grid_rows, grid_cols = grid_size
    # Total width per cell, including padding
    cell_width = grid_cols
    # Total line width: left rank (1 cell) + 8 board cells + right rank (1 cell)
    total_width = 1 + (8 * cell_width) + 1
    # Add left padding for visual centering (adjust this value to match your margin, e.g., 10 spaces)
    left_padding = " " * 7  # You can change 10 to match your desired margin
    
    # Resulting lines for the grid representation
    output_lines = []
    
    # Add top coordinates (a-h) with padding
    letters = 'abcdefgh'
    top_coord = [f"[fg={COORD_COLOR},bg=0]{letter.center(cell_width)}[/color]" for letter in letters]  # Light Cream
    output_lines.append(left_padding + ''.join(top_coord).ljust(total_width, ' '))
    # Add empty row below top coordinates with padding
    output_lines.append(left_padding + f"[fg=0,bg=0]{' ' * total_width}[/color]")
    
    # Process each of the 8 rows
    for row_idx in range(8):
        # Split the row into individual squares (space-separated)
        squares = rows[row_idx].split()
        
        # Generate lines for the current row based on grid_rows
        for grid_row in range(grid_rows):
            current_line = []
            
            # Add left rank number (8 down to 1) without padding (longest rows)
            if grid_row == grid_rows // 2:  # Center row
                rank = str(8 - row_idx)
                current_line.append(f"[fg={COORD_COLOR},bg=0]{rank.center(cell_width)}[/color]")  # Light Cream
            else:
                current_line.append('' * cell_width)
            
            # Process each of the 8 columns
            for col_idx in range(8):
                piece = piece_symbols[squares[col_idx]]
                # Checkerboard pattern: bg=53 for light (Light Cream), bg=54 for dark (Forest Green)
                bg_color = LIGHT_SQUARE if (row_idx + col_idx) % 2 == 0 else DARK_SQUARE
                
                # Determine content for this grid cell based on row
                if grid_row == grid_rows // 2:  # Center row for piece
                    cell_content = piece.center(cell_width)
                else:
                    cell_content = ' ' * cell_width
                
                current_line.append(f"[fg={ICON_COLOR},bg={bg_color}]{cell_content}[/color]")
            
            # Add right rank number without padding
            if grid_row == grid_rows // 2:
                rank = str(8 - row_idx)
                current_line.append(f"[fg={COORD_COLOR},bg=0]{rank.center(cell_width)}[/color]")  # Light Cream
            
            # Join the line and append to output
            full_line = ''.join(current_line)
            if len(full_line) < total_width:
                full_line += ' ' * (total_width - len(full_line))
            # Only add padding to non-digit rows (coordinate and empty spacer rows)
            if grid_row != grid_rows // 2:  # Non-center rows (empty or coordinate rows)
                output_lines.append(left_padding + full_line)
            else:
                output_lines.append(full_line)  # No padding for rows with digits
    
    # Add empty row above bottom coordinates with padding
    output_lines.append(left_padding + f"[fg=0,bg=0]{' ' * total_width}[/color]")
    # Add bottom coordinates (a-h) with padding
    output_lines.append(left_padding + ''.join(top_coord).ljust(total_width, ' '))
    
    # Combine all lines with newlines
    return '\n'.join(output_lines)