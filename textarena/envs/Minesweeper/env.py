import re, random
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import textarena as ta
from textarena.envs.Minesweeper.renderer import create_board_str

class MinesweeperEnv(ta.Env):
    """ Minesweeper environment """
    
    def __init__(self, rows: int=8, cols: int=8, num_mines: int=10, max_turns: int=100):
        """
        Initialize the Minesweeper environment.
        
        Args:
            rows (int): the number of rows
            cols (int): the number of columns
            num_mines (int): the number of mines
        """
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.max_turns = max_turns

    def get_board_str(self):
        return create_board_str(self.grid, self.revealed, self.flags)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment to its initial state """
        ## initliaze the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## initialize the game state
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.revealed = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.first_move = True # Track if it's the first move to ensure playability

        ## reset the game state
        game_state = {"grid": self.grid, "rendered_board": self._render_board()}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the player prompt """
        prompt = (
            f"You are Player {player_id}. You are playing the Minesweeper game.\n"
            "The objective of the game is to reveal all cells that do not contain mines.\n"
            "To make a move, you can either reveal a cell or place a flag on a suspected mine location using one of the following commands:\n"
            "- 'reveal': Reveal the contents of a specific cell.\n"
            "- 'flag': Place or remove a flag on a specific cell to mark it as a potential mine.\n"
            "To submit your move, type the command followed by the row and column in square brackets.\n"
            "For example:\n"
            "- [reveal 3 2] to reveal the cell in Row 3, Column 2.\n"
            "- [flag 5 6] to place or remove a flag on the cell in Row 5, Column 6.\n"
            "On your first move, you will reveal an area around the cell you choose to ensure a safe start.\n"
            "The current board layout is shown below. Cells that are unrevealed are represented by a dot ('.'), revealed numbers show the count of adjacent mines, and flagged cells are marked with an 'F'.\n"
            "Use logic and deduction to avoid revealing cells with mines!\n"
            "Be mindful not to choose revealed or flagged cells.\n"
            "Here is the current board layout:\n"
        )

        prompt += self.state.game_state["rendered_board"]
        return prompt
    
    def _render_board(self) -> str:
        """
        Render the game board.

        Returns:
            str: The rendered game board.
        """
        board_str = "   " + " ".join([str(c).rjust(2) for c in range(self.cols)]) + "\n"
        for r in range(self.rows):
            row_str = f"{r:2} "
            for c in range(self.cols):
                if self.revealed[r][c]:
                    if self.grid[r][c] == -1:
                        row_str += " * "
                    else:
                        row_str += f" {self.grid[r][c]} "
                elif self.flags[r][c]:
                    row_str += " F "
                else:
                    row_str += " . "
            board_str += row_str + "\n"
        return board_str
        
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Take a step in the environment """
        player_id = self.state.current_player_id
        ## Update the observation
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## Validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\s(\d+)\s(\d+)\]") # e.g. [reveal 3 2]
        match = action_search_pattern.search(action)

        if match is None:
            reason=f"Invalid move format. Player {player_id} did not respond with a valid action and coordinates in square brackets."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            action, row, col = match.group(1).lower(), int(match.group(2)), int(match.group(3))
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                reason=f"Invalid move. The specified row and column coordinates are out of bounds."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                if action == "reveal":
                    if self.revealed[row][col] or self.flags[row][col]:
                        reason=f"Invalid move. The cell at ({row}, {col}) has already been revealed or flagged."                    
                        self.state.set_invalid_move(player_id=player_id, reason=reason)
                    ## Handle the first move
                    if self.first_move:
                        self.clear_all_flags()
                        self.setup_mines(row, col)
                        self.first_move = False
                    
                    queue = deque([(row, col)])  # Start with the initial cell in the queue
                    self.revealed[row][col] = True  # Mark the initial cell as revealed immediately

                    while queue:
                        current_row, current_col = queue.popleft()

                        # Check if it's a mine
                        if self.grid[current_row][current_col] == -1:
                            reason=f"Game over! Player {player_id} hit a mine at ({current_row}, {current_col})."
                            self.state.set_invalid_move(player_id=player_id, reason=reason)

                        # If the cell has no adjacent mines, add its neighbors to the queue
                        if self.grid[current_row][current_col] == 0:
                            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                                neighbor_row, neighbor_col = current_row + dr, current_col + dc
                                # Only add to the queue if within bounds and not revealed or flagged
                                if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols:
                                    if not self.revealed[neighbor_row][neighbor_col] and not self.flags[neighbor_row][neighbor_col]:
                                        self.revealed[neighbor_row][neighbor_col] = True  # Mark as revealed when adding to queue
                                        queue.append((neighbor_row, neighbor_col))

                    message=f"Game Board:\n{self._render_board()}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message, for_logging=False)
                    
                elif action == "flag":
                    if not self.revealed[row][col]:
                        self.flags[row][col] = not self.flags[row][col]
                        # print(f"Flag {'placed' if self.flags[row][col] else 'removed'} at ({row}, {col})")

                    message=f"Game Board:\n{self._render_board()}"
                    self.state.add_observation(from_id=ta.GAME_ID,to_id=player_id, message=message, for_logging=False)

                else:
                    reason=f"Invalid move format. Player {player_id} did not respond with a valid action in square brackets."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

        ## Update the rendered board
        self.state.game_state["rendered_board"] = self._render_board()

        ## Check if the game is terminated
        if self._is_solved():
            reason=f"Congratulations! Player {player_id} has successfully cleared the Minesweeper board."
            self.state.set_winners(player_ids=[player_id], reason=reason)
        
        return self.state.step()
    
    def setup_mines(self, safe_row: int, safe_col: int):
        mines = set()
        while len(mines) < self.num_mines:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            # Avoid placing mines in the safe zone
            if (r, c) not in mines and (r < safe_row - 1 or r > safe_row + 1 or c < safe_col - 1 or c > safe_col + 1):
                mines.add((r, c))
                self.grid[r][c] = -1  # -1 represents a mine
        self.calculate_adjacent_numbers()

    def calculate_adjacent_numbers(self):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                mine_count = sum((0 <= r + dr < self.rows and 0 <= c + dc < self.cols and self.grid[r + dr][c + dc] == -1)
                                 for dr, dc in directions)
                self.grid[r][c] = mine_count

    def clear_all_flags(self):
        """Clear all flags on the board."""
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def _is_solved(self) -> bool:
        """
        Check if the board is in a solved state.

        Returns:
            bool: True if the board is in a solved state, False otherwise.
        """
        return all(
            (self.grid[r][c] == -1 and self.flags[r][c]) or (self.grid[r][c] != -1 and self.revealed[r][c])
            for r in range(self.rows) for c in range(self.cols)
        )
    