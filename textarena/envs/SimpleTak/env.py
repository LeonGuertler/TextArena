import re
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta

class SimpleTakEnv(ta.Env):
    """
    A simplified version of Tak with a single stone type:
      - Two players alternate turns placing a stone on an empty cell.
      - Player 0 places 'O', Player 1 places 'X'.
      - The goal is to form a continuous path connecting two opposite edges 
        (top↔bottom or left↔right).
      - Action space is similar to TicTacToe, where moves are written as [cell],
        and cell ranges from 0..(board_size^2 - 1).
    """

    def __init__(self, board_size: int = 5):
        """
        Initialize SimpleTak with a given board size.
        
        Args:
            board_size (int): The size of the NxN board (default 5).
        """
        super().__init__()
        self.board_size = board_size
        # Create a mapping from a single integer action to (row, col)
        # e.g., cell 0 -> (0,0), cell 1 -> (0,1), ...
        self.cell_mapping = {
            i: (i // board_size, i % board_size)
            for i in range(board_size * board_size)
        }

    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """ Reset the environment to the initial state """
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2)

        # NxN board initialized with empty strings
        empty_board = [['' for _ in range(self.board_size)] for _ in range(self.board_size)]

        self.state.reset(
            game_state={"board": empty_board},
            player_prompt_function=self._generate_player_prompt,
            seed=seed
        )

        # Provide the initial board observation to all players
        self._observe_current_state()

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate a prompt for the current player, telling them how to make a move.
        """
        symbol = 'O' if player_id == 0 else 'X'
        opponent_symbol = 'X' if symbol == 'O' else 'O'

        prompt = (
            f"You are Player {player_id} in SimpleTak.\n"
            f"On the board, your stones appear as '{symbol}' and "
            f"your opponent's stones appear as '{opponent_symbol}'.\n\n"
            "On your turn, choose one empty cell (by its numbered index) and place your stone there.\n"
            "For example, '[12]' places your stone in cell 12.\n\n"
            "Your objective is to form a continuous path of your stones that connects two opposite edges of the board "
            "(top-to-bottom or left-to-right). If you do this, you win immediately.\n"
        )
        return prompt

    def _observe_current_state(self) -> None:
        """
        Broadcast the current state of the board to all players,
        listing empty cells as possible moves.
        """
        board_str = self._render_board()

        # Gather all empty cells as possible moves
        board = self.state.game_state["board"]
        available_moves = []
        for i in range(self.board_size * self.board_size):
            r, c = self.cell_mapping[i]
            if board[r][c] == '':
                available_moves.append(f"[{i}]")

        message = (
            "Current Board:\n\n"
            f"{board_str}\n"
            "Available Moves: " + ", ".join(available_moves)
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _render_board(self) -> str:
        """
        Render the board with uniform cell widths, including top/bottom 
        and left/right framing lines, e.g.:

           +----+----+----+----+
           |  0 |  1 |  2 |  3 |
           +----+----+----+----+
           |  4 |  5 |  6 |  7 |
           +----+----+----+----+
           |  8 |  9 | 10 | 11 |
           +----+----+----+----+
           | 12 | 13 | 14 | 15 |
           +----+----+----+----+
        """

        board = self.state.game_state["board"]

        # Determine cell width based on the largest cell number
        # e.g., for board_size=4, largest cell_num=15 => 2 digits
        max_cell_num = self.board_size * self.board_size - 1
        digit_count = len(str(max_cell_num))
        cell_width = max(digit_count, 2)  # at least 2 for occupant symbols

        # Helper: content for cell (row,col) either occupant 'X'/'O' or the cell number
        def cell_str(r: int, c: int) -> str:
            if board[r][c] == '':
                # If empty, show cell number
                return str(r * self.board_size + c)
            else:
                # Occupied by 'O' or 'X'
                return board[r][c]

        # Build a horizontal separator line that frames each row
        # We want something like: +----+----+ for each column
        def build_hline() -> str:
            line_parts = []
            for _ in range(self.board_size):
                line_parts.append("-" * (cell_width + 2))  # +2 for spacing around content
            # Join them with '+' and add a leading+trailing '+'
            return "+" + "+".join(line_parts) + "+"

        lines = []
        # Add top framing line first
        lines.append(build_hline())

        for r in range(self.board_size):
            # For each row, build the row of cells
            row_cells = []
            for c in range(self.board_size):
                text = cell_str(r, c)
                # Center the text in cell_width
                text_centered = f" {text:^{cell_width}} "
                row_cells.append(text_centered)
            # Join with '|'
            row_line = "|" + "|".join(row_cells) + "|"
            lines.append(row_line)

            # Add a horizontal line below each row
            lines.append(build_hline())

        # Join all lines
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a player's move, which must be in the form [cell_number].

        1. Validate the format.
        2. Validate the chosen cell index (in range, and not occupied).
        3. Place the player's stone (O or X).
        4. Check win condition or draw.
        5. Return step result to textarena.
        """
        player_id = self.state.current_player_id
        symbol = 'O' if player_id == 0 else 'X'

        # Record the player's raw action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Regex to parse moves like [12]
        pattern = re.compile(r"\[\s*(\d+)\s*\]")
        match = pattern.search(action)

        if match is None:
            reason = f"Invalid move format. Player {player_id} must submit a move like [cell_num]."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            cell_num = int(match.group(1))
            
            # Check if cell_num in valid range
            if cell_num not in self.cell_mapping:
                reason = (
                    f"Invalid cell number {cell_num}. "
                    f"Must be between 0 and {self.board_size**2 - 1}."
                )
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                row, col = self.cell_mapping[cell_num]
                board = self.state.game_state["board"]
                if board[row][col] == '':
                    # Place the stone
                    board[row][col] = symbol
                    # Check for a winning path
                    if self._check_win(symbol):
                        self.state.set_winners(
                            player_ids=[player_id],
                            reason=f"Player {player_id} ('{symbol}') connected two opposite edges!"
                        )
                    else:
                        # If board is fully occupied and no winner => draw
                        if all(board[r][c] != '' 
                               for r in range(self.board_size) 
                               for c in range(self.board_size)):
                            self.state.set_draw(reason="The board is full. It's a draw!")
                else:
                    reason = f"Cell {cell_num} is already occupied. Choose an empty cell."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

        # Update the board display for all players
        self._observe_current_state()
        return self.state.step()

    def _check_win(self, symbol: str) -> bool:
        """
        Check if the current player (symbol='O' or 'X') has formed a connected path
        between two opposite edges (top<->bottom or left<->right).
        
        We'll do a DFS approach: if a contiguous region of 'symbol' touches both
        top and bottom edges, or both left and right edges, that's a win.
        """
        board = self.state.game_state["board"]
        visited = set()
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        n = self.board_size

        def valid(r, c):
            return 0 <= r < n and 0 <= c < n and board[r][c] == symbol

        def dfs(r, c, edges):
            if (r, c) in visited:
                return False
            visited.add((r, c))

            # Check which edges we're on
            if r == 0: edges.add("top")
            if r == n - 1: edges.add("bottom")
            if c == 0: edges.add("left")
            if c == n - 1: edges.add("right")

            # If we connect top↔bottom or left↔right, it's a win
            if ("top" in edges and "bottom" in edges) or ("left" in edges and "right" in edges):
                return True

            # Continue DFS
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if valid(nr, nc):
                    if dfs(nr, nc, edges.copy()):
                        return True
            return False

        # Start DFS from every cell that has the player's symbol
        for row in range(n):
            for col in range(n):
                if board[row][col] == symbol:
                    if dfs(row, col, set()):
                        return True
        return False
