import re, random, copy, json, os
import importlib.resources
from typing import Any, Dict, List, Tuple, Optional

import textarena as ta
from textarena.envs.LogicPuzzle.renderer import create_board_str

class LogicPuzzleEnv(ta.Env):
    """ Logic Puzzle environment """

    def __init__(self, difficulty: Optional[str] = "easy"):
        """
        Initialize the Logic Puzzle environment with the specified difficulty level.

        Args:
            difficulty (str): The difficulty level of the puzzle (e.g., "easy", "medium", "hard").
        """
        super().__init__()
        self.difficulty = difficulty
        
        # Load the puzzle data
        self.game_board_data = self._load_puzzle_data()
        
    def _load_puzzle_data(self, puzzle_path: Optional[str] = None):
        """
        Load puzzle data from a JSONL file.
        
        The JSONL file must have each line as a JSON object with at least a 'difficulty' field.
        
        Args:
            puzzle_path (str, optional): Path to the JSONL file containing puzzle data.
            
        Returns:
            list: A list of puzzle data objects filtered by the current difficulty level.
            
        Raises:
            FileNotFoundError: If the `puzzle_path` does not exist.
            ValueError: If the JSONL file has an invalid format or no matching puzzles are found.
        """
        try:
            if puzzle_path is not None:
                # Use provided path
                if not os.path.exists(puzzle_path):
                    raise FileNotFoundError(f"Puzzle data file not found at: {puzzle_path}")
                with open(puzzle_path, "r", encoding="utf-8") as file:
                    game_board_data = file.readlines()
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.LogicPuzzle').joinpath('game_board_clues.jsonl').open('r') as file:
                    game_board_data = file.readlines()
                    
            filtered_data = [json.loads(line.lower()) for line in game_board_data 
                            if json.loads(line)["difficulty"] == self.difficulty]
            
            if not filtered_data:
                raise ValueError(f"No puzzles found matching difficulty '{self.difficulty}'.")
                
            return filtered_data
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load puzzle data: {str(e)}")


    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment to its initial state """
        ## initialize the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1)

        
        ## load the game board
        self.game_board, self.game_board_solution, self.clues = self._load_game_board()

        ## reset the game state
        game_state={
            "board": copy.deepcopy(self.game_board),
            "rendered_board": self._render_board(self.game_board)
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the player prompt with clear instructions for making moves """
        prompt = (
            f"You are Player {player_id} in the Logic Puzzle game.\n"
            "Your goal is to solve the puzzle by correctly assigning items to categories based on the clues provided.\n"
            "\n"
            "To make a move, specify the row and column for each item in the shown tables, followed by the mark ('X' or 'O').\n"
            "Use the format: '[row col X]' or '[row col O]', where:\n"
            "- 'O' indicates the item is assigned to the category.\n"
            "- 'X' indicates the item is not assigned to the category.\n"
            "\n"
            "Example: To mark an item in the 'people_locations' grid, enter '[park Alice X]' or '[park Alice O]'.\n"
            "Only items shown in the current grids can be marked, and you can update a cell if needed.\n"
            "\n"
            "Note:\n"
            "- You may revisit and update previously marked cells as your understanding evolves. As long as the update is a mark that is different from the previous.\n"
            "- Each move will be recorded in the history.\n"
            "\n"
            "Here are the clues to assist you:\n"
            f"{self._return_clues()}\n"
            "\n"
            "Current state of the puzzle:\n"
            f"{self._render_board(self.game_board)}"
            "You may only submit one move at a time."
        )
        return prompt


    
    def _load_game_board(self):
        """
        Load the game board data for the logic puzzle.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: The game board data.
            Dict[str, Dict[str, Dict[str, str]]]: The game board solution data.
            List[str]: The list of clues for the puzzle.
        """
        selected_game_board = random.choice(self.game_board_data)
        solution = selected_game_board["solution"]
        clues = selected_game_board["clue"]
        game_board, game_board_solution = self._create_game_board(solution)
        return game_board, game_board_solution, clues
    
    def _create_game_board(self, solution: Dict[str, List[str]]):
        """
        Create the game board for the logic puzzle based on the solution.

        Args:
            solution (Dict[str, List[str]]): The solution for the logic puzzle.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: The game board data.
            Dict[str, Dict[str, Dict[str, str]]]: The game board solution data.
        """
        game_board = {}
        game_board_solution = {}

        categories = list(solution.keys())
        index = random.choice(categories)

        for category in categories:
            if category != index:
                shuffled_items = solution[category][:]
                random.shuffle(shuffled_items)

                game_board[f"{index}_{category}"] = {
                    name: {item: None for item in shuffled_items} for name in solution[index]
                }

                game_board_solution[f"{index}_{category}"] = {
                    name: {item: "O" if item == solution[category][solution[index].index(name)] else "X" for item in shuffled_items} for name in solution[index]
                }
        
        return game_board, game_board_solution
    
    def _render_board(self, game_board: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """
        Render the game board as a string.

        Args:
            game_board (Dict[str, Dict[str, Dict[str, Any]]]): The game board data.

        Returns:
            str: The rendered game board.

        """
        output = []

        for grid_name, grid_data in game_board.items():
            ## get the column headers
            items = list(next(iter(grid_data.values())).keys())
            ## calculate the maximum width needed for headers and each row name
            max_name_width = max(len(name) for name in grid_data.keys()) + 2
            max_col_width = max(len(item) for item in items) + 2

            ## add grid name with a separator
            output.append(f"\n{'=' * (max_name_width + max_col_width * len(items) + len(items) + 5)}")
            output.append(f"{grid_name.center(max_name_width + max_col_width * len(items) + len(items) + 5)}")
            output.append(f"{'=' * (max_name_width + max_col_width * len(items) + len(items) + 5)}")

            ## add column headers with border
            output.append(" " * max_name_width + " | ".join(f"{item:^{max_col_width}}" for item in items) + " |")
            output.append("-" * (max_name_width + len(items) * (max_col_width + 3) - 1))

            ## add each row with values and borders
            for name, marks in grid_data.items():
                row = f"{name:<{max_name_width}}" + " | ".join(
                    f"{marks[item] if marks[item] else ' ':^{max_col_width}}" for item in items
                )
                output.append(f"{row} |")

            ## add a separator after each grid for clarity
            output.append("=" * (max_name_width + len(items) * (max_col_width + 3) - 1))

        return "\n".join(output)
    
    def _return_clues(self):
        """
        Return the clues in a formatted string.

        Returns:
            str: The formatted clues.
        """
        return "\n".join([f"- {clue}" for clue in self.clues])
    
    def step(
        self,
        action: str
    ) -> Tuple[
        Optional[ta.Observations], # observations
        Optional[ta.Rewards], # reward
        bool, # truncated
        bool, # terminated
        ta.Info # info
    ]:
        """
        Take a step in the environment based on the player's action.

        Args:
            player_id (int): The ID of the player making the move.
            action (str): The action taken by the player.

        Returns:
            Observations: Observations for the player after the action.
            Rewards: Rewards for the player after the action.
            bool: Whether the game was truncated.
            bool: Whether the game is terminated.
            Info: Additional information about the game state
        """

        player_id = self.state.current_player_id
        
        ## update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True
        )

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\s([a-zA-Z]+)\s([XOxo])\]") # e.g. [Alice park X]
        matches = action_search_pattern.findall(action) ## should this be search, or find all?
        matches = set(matches)

        if not matches:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with a valid move in square brackets."
            )
        else:
            for match in matches:
                row, col, mark = match
                row = row.lower()
                col = col.lower()
                mark = mark.upper()
                if not self._is_within_bounds(row, col):
                    ## the item is not within the bounds of the grid (i.e., invalid move)
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move. The item is not within the bounds of the grid."
                    )
                    break
                elif self._is_repeated_mark(row, col, mark):
                    ## the item is in a valid format, but it is already marked with the same value in the grid (i.e., repeated move)
                    self.state.set_invalid_move(
                        player_id=player_id,
                        reason=f"Invalid move. The item has already been marked with the same value."
                    )
                    break
                else:
                    ## update the rendered board
                    self._mark_item(row, col, mark)
                    self.state.game_state["rendered_board"] = self._render_board(self.game_board)
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=f"[{row} {col} {mark}] is valid. Game Board:\n{self._render_board(self.game_board)}",
                        for_logging=False
                    )

            if self._is_solved():
                self.state.set_winners(
                    player_ids=[player_id],
                    reason=[f"Congratulations! Player {player_id} has solved the logic puzzle!"]
                )
        
        return self.state.step()
    
    def _is_within_bounds(self, row: str, col: str) -> bool:
        """
        Check if the specified item is within the bounds of the game board.

        Args:
            row (str): The row of the item.
            col (str): The column of the item.

        Returns:
            bool: True if the item is within the bounds, False otherwise.

        """
        for grid_name, grid_data in self.game_board.items():
            if row in grid_data:
                if col in grid_data[row]:
                    return True
        return False
    
    def _is_repeated_mark(self, row: str, col: str, mark: str) -> bool:
        """
        Check if the specified item in the game board is already marked with the same value.

        Args:
            row (str): The row of the item.
            col (str): The column of the item.
            mark (str): The mark to check for.

        Returns:
            bool: True if the item is already marked with the same value, False otherwise.

        """
        for grid_name, grid_data in self.game_board.items():
            if row in grid_data:
                if col in grid_data[row]:
                    if grid_data[row][col] == mark:
                        return True
        return False
    
    def _mark_item(self, row: str, col: str, mark: str):
        """
        Mark the specified item in the game board.

        Args:
            row (str): The row of the item.
            col (str): The column of the item.
            mark (str): The mark to assign to the item.

        """
        for grid_name, grid_data in self.game_board.items():
            if row in grid_data:
                if col in grid_data[row]:
                    grid_data[row][col] = mark
    
    def _is_solved(self) -> bool:
        """
        Compares grids with grids_solution to check if they are the same.
        
        Returns:
            bool: True if the grids match, False otherwise.
        """
        for grid_name, grid_data in self.game_board.items():
            solution_data = self.game_board_solution.get(grid_name)
            if solution_data is None:
                return False

            for name, items in grid_data.items():
                solution_items = solution_data.get(name)
                if solution_items is None:
                    return False

                for item, value in items.items():
                    solution_value = solution_items.get(item)
                    if solution_value is None:
                        return False
                    if value != solution_value:
                        return False

        return True
    
