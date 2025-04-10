import re, random, copy
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.Sudoku.renderer import create_board_str

class SudokuEnv(ta.Env):
    """ Sudoku Game Environment """

    def __init__(self, clues: int= 30, max_turns: Optional[int] = 100):
        """
        Initialise the Sudoku Environment.
        
        Args:
            clues (str): The number of clues.
            max_turns (int): The maximum number of moves allowed.
        """
        self.clues = clues
        self.max_turns = max_turns

    
    def get_board_str(self):
        return create_board_str(board=self.game_board)

    def _generate_board(self) -> List[List[int]]:
        """
        Generate a Sudoku puzzle based on the difficulty level.
        
        Returns:
            List[List[int]]: The Sudoku puzzle grid.
        """
        ## generate a full grid
        full_grid = self._generate_full_grid()

        ## remove cells to create puzzle
        puzzle_grid = self._remove_cells(full_grid, self.clues)

        return full_grid, puzzle_grid
    
    def _generate_full_grid(self) -> List[List[int]]:
        """
        Generate a fully solvable Sudiku grid using backtracking.
        
        Returns:
            List[List[int]]: The Sudoku grid.
        """
        grid = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_grid(grid)
        return grid

    def _fill_grid(self, grid: List[List[int]]) -> bool:
        """
        Recursively fills the Sudoku grid using backtracking.

        Args:
            grid (List[List[int]]): The Sudoku grid to fill.

        Returns:
            bool: True if the grid is successfully filled, False otherwise.
        """
        empty = self._find_empty(grid)
        if not empty:
            return True  # Grid is complete
        row, col = empty

        numbers = list(range(1, 10))
        random.shuffle(numbers)
        for num in numbers:
            if self.is_safe(grid, row, col, num):
                grid[row][col] = num
                if self._fill_grid(grid):
                    return True
                grid[row][col] = 0
        return False
    
    def _find_empty(self, grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """
        Finds an empty cell in the grid.

        Args:
            grid (List[List[int]]): The Sudoku grid.

        Returns:
            Optional[Tuple[int, int]]: The row and column of an empty cell, or None if full.
        """
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def is_safe(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """
        Checks if it's safe to place a number in a given cell.

        Args:
            grid (List[List[int]]): The Sudoku grid.
            row (int): Row index.
            col (int): Column index.
            num (int): Number to place.

        Returns:
            bool: True if safe, False otherwise.
        """
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in [grid[i][col] for i in range(9)]:
            return False
        # Check subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == num:
                    return False
        return True

    def _remove_cells(self, grid: List[List[int]], clues: int) -> List[List[int]]:
        """
        Removes cells from the full grid to create a puzzle, ensuring a unique solution.

        Args:
            grid (List[List[int]]): A fully solved Sudoku grid.
            clues (int): Number of clues (filled cells) to retain.

        Returns:
            List[List[int]]: The resulting Sudoku puzzle grid.
        """
        puzzle = copy.deepcopy(grid)
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)

        while len(cells) > (81 - clues):
            row, col = cells.pop()
            removed = puzzle[row][col]
            puzzle[row][col] = 0

            # Make a copy to check for uniqueness
            grid_copy = copy.deepcopy(puzzle)
            solutions = []
            self._count_solutions(grid_copy, solutions)
            if len(solutions) != 1:
                # Not unique, revert the removal
                puzzle[row][col] = removed

        return puzzle

    def _solve_sudoku(self, grid: List[List[int]]) -> bool:
        """
        Solves the Sudoku puzzle using backtracking. Modifies the grid in-place.

        Args:
            grid (List[List[int]]): The Sudoku grid to solve.

        Returns:
            bool: True if solvable, False otherwise.
        """
        empty = self._find_empty(grid)
        if not empty:
            return True  # Solved
        row, col = empty

        for num in range(1, 10):
            if self.is_safe(grid, row, col, num):
                grid[row][col] = num
                if self._solve_sudoku(grid):
                    return True
                grid[row][col] = 0
        return False

    def _count_solutions(self, grid: List[List[int]], solutions: List[List[List[int]]], limit: int = 2) -> int:
        """
        Counts the number of solutions for a given Sudoku puzzle.

        Args:
            grid (List[List[int]]): The Sudoku grid.
            solutions (List[List[List[int]]]): A list to store found solutions.
            limit (int): The maximum number of solutions to find.

        Returns:
            int: The number of solutions found.
        """
        if len(solutions) >= limit:
            return len(solutions)

        empty = self._find_empty(grid)
        if not empty:
            solutions.append(copy.deepcopy(grid))
            return len(solutions)
        row, col = empty

        for num in range(1, 10):
            if self.is_safe(grid, row, col, num):
                grid[row][col] = num
                self._count_solutions(grid, solutions, limit)
                grid[row][col] = 0
        return len(solutions)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the game environment """
        ## initialise the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## load the puzzle
        self.full_grid, self.game_board = self._generate_board()

        game_state={
            "board": copy.deepcopy(self.game_board),
            "rendered_board": self._get_grid_string_with_indices(self.game_board),
            "completed": False,
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for the player, providing them with the Sudoku grid and instructions """
        prompt = (
            f"You are Player {player_id}. You are playing Sudoku.\n"
            "Here is the current state of the Sudoku grid. Each row is numbered from 1 to 9, and each column is also numbered from 1 to 9.\n"
            "Empty cells are represented by '.', and pre-filled cells contain digits from 1 to 9.\n\n"
            "Current Sudoku Grid:\n"
        )    
        # Include the grid with row and column indices for clarity
        grid_str = self._get_grid_string_with_indices()
        prompt += f"{grid_str}\n\n"
        
        prompt += (
            "Your objective is to fill the empty cells in the 9x9 grid with digits from 1 to 9 such that:\n"
            "1. Each row contains all digits from 1 to 9 without repetition.\n"
            "2. Each column contains all digits from 1 to 9 without repetition.\n"
            "3. Each of the nine 3x3 subgrids contains all digits from 1 to 9 without repetition.\n\n"
            "Rules and Instructions:\n"
            "1. **Do not overwrite** the initial numbers provided in the grid.\n"
            "2. **Only fill** empty cells represented by '.'.\n"
            "3. You may respond in any manner you prefer, but ensure that your response includes the format of '[row column number]'.\n"
            "4. **Ensure** that your move does not violate Sudoku rules. Invalid moves will result in penalties.\n"
            "Examples:\n"
            "- **Valid Move**:\n"
            "  - Grid Snippet Before Move:\n"
            "  \n"
            "  - Move: `[5 3 7]`\n"
            "  - Explanation: Placing 7 at row 5, column 3 does not violate any Sudoku rules.\n\n"
            "- **Invalid Move** (Overwriting a pre-filled cell):\n"
            "  - Grid Snippet Before Move:\n"
            "  \n"
            "  - Move: `[1 1 9]`\n"
            "  - Explanation: Cell (1,1) is already filled with 5. You cannot overwrite it.\n\n"
            "- **Invalid Move** (Violating Sudoku rules):\n"
            "  - Grid Snippet Before Move:\n"
            "  \n"
            "  - Move: `[1 3 5]`\n"
            "  - Explanation: Placing 5 in row 1, column 3 violates the rule since 5 already exists in row 1.\n\n"
            "The history of your moves and thoughts will be appended as you play more rounds. Use the history of your move to improve your decision making by avoiding the moves you have tried. Good luck!\n\n"
        )
        return prompt


    def step(self, action: str) -> Tuple[bool, ta.Info ]:
        """ Take a step in the game environment given an action from a player """
        player_id = self.state.current_player_id
        
        ## update the observations
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## validate the actions
        ## extract the format [row column number] from the action
        action_search_pattern = re.compile(r"\[(\d+)\s(\d+)\s(\d+)\]")
        match = action_search_pattern.search(action)

        if not match:
            reason=f"Invalid move format. Player {player_id} did not respond with valid 'row column number'."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            row, col, num = map(int, match.groups())
            if row < 1 or row > 9 or col < 1 or col > 9 or num < 1 or num > 9:
                reason=f"Invalid move. Player {player_id} attempted to place {num} at ({row}, {col}), which is out of bounds."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                row_idx, col_idx = row - 1, col - 1
                ## check if the cell is already filled in the initial grid
                if self.game_board[row_idx][col_idx] != 0:
                    reason=f"Invalid move. Player {player_id} attempted to overwrite a pre-filled cell ({row}, {col})."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                elif self._is_move_correct(row_idx, col_idx, num):
                    ## update the grid
                    self.state.game_state["board"][row_idx][col_idx] = num
                    message=f"Board state: \n{self._get_grid_string_with_indices()}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                else:
                    reason=f"Invalid move. Player {player_id} attempted to place {num} at ({row}, {col}), which violates Sudoku rules."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

                ## check if the game is completed
                if self._is_puzzle_complete():
                    self.state.game_state["completed"] = True
                    reason=f"Congratulations! Player {player_id} completed the Sudoku puzzle."
                    self.state.set_winners(player_ids=[player_id], reason=reason)

                ## update game board
                self.state.game_state["rendered_board"] = self._get_grid_string_with_indices(self.state.game_state["board"])

        return self.state.step()
            
    def _get_grid_string_with_indices(self, game_board: Optional[List[int]] = None) -> str:
        """
        Converts the current grid to a formatted string with row and column indices.

        Returns:
            str: Formatted grid string with indices.
        """
        if game_board is None:
            game_board = self.state.game_state["board"]
        header = "   " + " ".join([f"C{j+1}" + ("  " if (j + 1) % 3 == 0 else "") for j in range(9)])  # Column headers
        lines = [header]
        for i, row in enumerate(game_board):
            row_str = f"R{i+1} "  # Row header
            for j, num in enumerate(row):
                cell = str(num) if num != 0 else "."
                row_str += f" {cell} "
                if (j + 1) % 3 == 0 and j < 8:
                    row_str += "| "
            lines.append(row_str.strip())
            if (i + 1) % 3 == 0 and i < 8:
                lines.append("   " + "- " * 16)

        return "\n".join(lines)
    
    def _is_move_correct(self, row: int, col: int, num: int) -> bool:
        """Check if move is correct based on the full solution grid."""
        return self.full_grid[row][col] == num

    def _is_puzzle_complete(self) -> bool:
        """
        Checks if the puzzle is completely and correctly filled.

        Returns:
            bool: True if complete, False otherwise.
        """
        for i in range(9):
            for j in range(9):
                num = self.state.game_state["board"][i][j]
                if num == 0 or not self._is_move_correct_complete(i, j, num):
                    return False
        return True
    
    def _is_move_correct_complete(self, row: int, col: int, num: int) -> bool:
        """
        Checks if the current move is still valid in the completed puzzle.

        Args:
            row (int): Row index (0-based).
            col (int): Column index (0-based).
            num (int): Number to place.

        Returns:
            bool: True if the move is correct, False otherwise.
        """
        # Temporarily remove the number to check for duplicates
        self.state.game_state["board"][row][col] = 0
        correct = self._is_move_correct(row, col, num)
        self.state.game_state["board"][row][col] = num
        return correct
