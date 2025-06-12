import re, random, json, copy, importlib
from typing import Any, Dict, Optional, Tuple, Union

import textarena as ta
from textarena.envs.games.Crosswords.renderer import create_board_str


class CrosswordsEnv(ta.Env):
    def __init__(self, hardcore: Optional[bool]=False, max_turns: Optional[int]=100, num_words: Optional[int]=5):
        """
        Args:
            hardcore (Optional[bool]): Whether to use hardcore mode.
            max_turns (Optional[int]): Maximum number of turns allowed.
            num_words (Optional[int]): Number of words to use in the game.
        """
        super().__init__()
        self.hardcore = hardcore
        self.max_turns = max_turns
        self.num_words = num_words
        
        # with open("textarena/envs/games/Crosswords/words_clues.jsonl", "r") as f: ## load the word list
        #     word_data = f.readlines()
        # self.word_data = [json.loads(x) for x in word_data if json.loads(x)["hardcore"]==hardcore]
        self._load_words(hardcore=hardcore)

        self.action_space = {0: re.compile(r"\[(\d+)\s(\d+)\s([a-zA-Z])\]")}

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _load_words(self, words_path: Optional[str]=None, hardcore: bool=False):
        try:
            if words_path is not None:
                if not os.path.exists(words_path): # Use provided path
                    raise FileNotFoundError(f"Words data file not found at: {words_path}")
                with open(words_path, "r", encoding="utf-8") as file:
                    word_data = file.readlines()
            else: # Use package resource
                with importlib.resources.files('textarena.envs.games.Crosswords').joinpath('words_clues.jsonl').open('r') as file:
                    word_data = file.readlines()
            self.word_data = [json.loads(x) for x in word_data if json.loads(x)["hardcore"] == hardcore]
            if not self.word_data:
                raise ValueError(f"No words found matching hardcore={hardcore} criteria.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load words data: {str(e)}")
    
    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed, max_turns=self.max_turns) ## initialise the game_state
        game_board, placed_words, clues = self._generate_board() ## generate the game board and the placed words for the clues
        game_state={"solution": copy.copy(game_board), "board": self._hide_letters(game_board), "clues": clues, "placed_words": placed_words} # reset the state
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        prompt = (
            f"You are playing Crosswords.\nHere is the current state of the Crosswords grid. Each row and column are numbered.\n"
            "The cells that need to be populated with letters are represented by '_', and those that do not need words are represented by '.'.\n\n"
            "Current Crosswords Grid:\n"
        )
        grid_str = self._render_board(game_state['board'], show_letters=False)
        prompt += grid_str
        prompt += "\n\nHere are the clues for the words you need to find:\n"
        prompt += self._clue_generator()
        prompt += ("\n\nYou can only provide one response per turn. Hence, plan your approach and risk appetite. Only guesses in the format of [row column letter] will be fetched from your response, e.g. [0 0 d], [1 2 G].\n"
                   "As you play, the history of your choices will be appended below. Use the information to complete the game.\n")
        return prompt


    def _generate_board(self):
        ## init the sampled words, their directions and their clues
        sampled_word_data = random.sample(self.word_data, self.num_words)
        sampled_word_data_sorted = sorted(sampled_word_data, key=lambda x: len(x["word"]), reverse=True)
        words = [x["word"] for x in sampled_word_data_sorted]
        directions = {x["word"]: random.choice(["across", "down"]) for x in sampled_word_data_sorted}
        clues = {x["word"]: random.sample(list(x["clues"].values()), 1)[0] for x in sampled_word_data_sorted}

        ## generate the crossword grid
        grid_size = self._determine_initial_grid_size(words)
        grid = self._create_empty_grid(grid_size)

        placed_words = {}  # word: (row, col), where 0 is the starting index

        for word in words:
            placed = False
            if not placed_words:  # First word
                # Place the first word in the center of the grid
                if directions[word] == "across":
                    row = grid_size // 2
                    col = (grid_size - len(word)) // 2
                else:
                    row = (grid_size - len(word)) // 2
                    col = grid_size // 2

                if self._can_place_word(grid, word, directions[word], row, col):
                    self._place_word_on_grid(grid, word, directions[word], row, col)
                    placed_words[word] = (row, col, directions[word])
                    placed = True
            
            else:
                # Attempt to find overlaps
                possible_positions = self._find_overlaps(word, grid, placed_words, directions)
                random.shuffle(possible_positions)  # Randomize to add variability
                for pos in possible_positions:
                    row, col, direction = pos
                    if self._can_place_word(grid, word, direction, row, col):
                        self._place_word_on_grid(grid, word, direction, row, col)
                        placed_words[word] = (row, col, direction)
                        placed = True
                        break

            if not placed:
                # If no overlap placement is possible, try placing the word in any free position
                for row in range(grid_size):
                    for col in range(grid_size):
                        if self._can_place_word(grid, word, directions[word], row, col):
                            self._place_word_on_grid(grid, word, directions[word], row, col)
                            placed_words[word] = (row, col, directions[word])
                            placed = True
                            break
                    if placed:
                        break

        return grid, placed_words, clues

    def _determine_initial_grid_size(self, words):
        """ Determine the initial size of the grid based on the length of the longest word """
        max_length = max(len(word) for word in words)
        return round(max_length * 1.5)  # Ensures that the grid size is larger than the longest word to allow placement

    def _create_empty_grid(self, size):
        """ Create an empty grid of the specified size """
        return [["." for _ in range(size)] for _ in range(size)]

    def _can_place_word(self, grid, word, direction, row, col):
        """
        Check if a word can be placed on the grid at the specified position.

        Args:
            grid (List[List[str]]): The crossword grid.
            word (str): The word to be placed.
            direction (str): The direction in which the word is to be placed ("across" or "down").
            row (int): The row in which the word is to be placed.
            col (int): The column in which the word is to be placed.
        """
        if direction == "across":
            if col + len(word) > len(grid[0]):
                return False
            for i, letter in enumerate(word):
                current_cell = grid[row][col + i]
                if current_cell != "." and current_cell != letter:
                    return False
        else:  # "down"
            if row + len(word) > len(grid):
                return False
            for i, letter in enumerate(word):
                current_cell = grid[row + i][col]
                if current_cell != "." and current_cell != letter:
                    return False

        return True

    def _place_word_on_grid(self, grid, word, direction, row, col):
        """
        Place a word on the grid at the specified position.

        Args:
            grid (List[List[str]]): The crossword grid.
            word (str): The word to be placed.
            direction (str): The direction in which the word is to be placed ("across" or "down").
            row (int): The row in which the word is to be placed.
            col (int): The column in which the word is to be placed.
        """
        if direction == "across":
            for i, letter in enumerate(word):
                grid[row][col + i] = letter
        else:  # "down"
            for i, letter in enumerate(word):
                grid[row + i][col] = letter

    def _find_overlaps(self, word, grid, placed_words, directions):
        """
        Find all possible valid overlaps for the word with already placed words.
        
        Args:
            word (str): The word to be placed.
            grid (List[List[str]]): The crossword grid.
            placed_words (Dict[str, Tuple[int, int, str]]): A dictionary of placed words and their positions.
            directions (Dict[str, str]): A dictionary of words and their directions.

        Returns:
            List[Tuple[int, int, str]]: A list of possible overlaps in the format (row, col, direction
        """
        overlaps = []
        for placed_word, (p_row, p_col, p_direction) in placed_words.items():
            for i, letter in enumerate(word):
                for j, placed_letter in enumerate(placed_word):
                    if letter == placed_letter:
                        # Determine the possible position based on the direction of the placed word
                        if p_direction == 'across':
                            row = p_row - i
                            col = p_col + j
                            if directions[word] == 'down' and 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                                if self._can_place_word(grid, word, 'down', row, col):
                                    overlaps.append((row, col, 'down'))
                        elif p_direction == 'down':
                            row = p_row + j
                            col = p_col - i
                            if directions[word] == 'across' and 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                                if self._can_place_word(grid, word, 'across', row, col):
                                    overlaps.append((row, col, 'across'))
        return overlaps

    
    def _render_board(self, grid, show_letters=False):
        """ Print the grid for text display """
        ## should be C01, C03, ... C10, C11, ...
        header = "   " + " ".join(f"C{i:02}" for i in range(len(grid)))
        lines = [header]
        for i, row in enumerate(grid):
            ## should be R01, R02, ... R10, R11, ...
            row_str = f"R{i:02} "
            for j, val in enumerate(row):
                if show_letters:
                    row_str += f" {val}  "
                else:
                    row_str += f" _  " if val != "." else " .  "
            lines.append(row_str)

        return "\n".join(lines)

    def _hide_letters(self, grid):
        """ Hide the letters in the grid """
        return [['_' if cell != "." else cell for cell in row] for row in grid]
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action) 

        ## validate the actions; note that the response can have multiple guesses at one go.
        matches = set(self.action_space[self.state.current_player_id].findall(action)) # [row column letter]

        if not matches:
            self.state.set_invalid_move(reason="The Player did not respond with valid 'row column letter'.")
        else:
            for match in matches:
                row, col, letter = match
                row, col, letter = int(row), int(col), str(letter)
                if row < 0 or row >= len(self.state.game_state["board"]) or col < 0 or col >= len(self.state.game_state["board"][0]):
                    self.state.set_invalid_move(reason="The specified coordinate is out of bounds."); break
                elif self.state.game_state["board"][row][col] == ".":
                    self.state.set_invalid_move(reason=f"The specified coordinate is a black cell."); break
                elif not self.game_board[row][col].upper() == letter.upper():
                    self.state.set_invalid_move(reason=f"Invalid move. The specified letter is incorrect."); break
                else:
                    self.state.game_state["board"][row][col] = letter.upper()
                    self.state.add_observation(message=f"Board state: \n{self._render_board(self.state.game_state['board'], show_letters=True)}")

            if self._is_game_over():  ## check if the game is over
                self.state.set_outcome(reward=1, reason=f"Congratulations! You completed the Crosswords puzzle.")
            elif self.state.check_turn_limit():
                pct_complete=self._get_percentage_completion()
                self.state.set_singleplayer_game_outcome(reward=pct_complete, reason=f"The turn limit has been reached. You completed {pct_complete*100} percent of the Crossword puzzle.")
        return self.state.step()

        def _get_percentage_completion(self) -> float:
            """ Compute the percentage of the crossword that has been solved so far """
            total_letter_cells = 0 # Count every cell that should eventually contain a letter
            filled_letter_cells = 0
            for row in self.state.game_state["board"]:
                for cell in row:
                    if cell != ".": # not a black square
                        total_letter_cells += 1
                        if cell != "_" and cell.isalpha(): # already revealed / guessed
                            filled_letter_cells += 1
            if total_letter_cells == 0: # safety guard
                return 0.0
            return filled_letter_cells / total_letter_cells

    def _is_game_over(self) -> bool:
        """ Check if the game is over; Returns: (bool) True if the game is over, False otherwise """
        return all("_" not in row for row in self.state.game_state["board"])

    def _clue_generator(self, string_format=True):
        """ Generate a clue for a word; Returns: (str) The clue for the word. """
        res = []
        for i, set in enumerate(zip(self.state.game_state["placed_words"].values(), self.state.game_state["clues"].values())):
            res.append(f"{i+1}. {set[1]}: {set[0]}")
        return "\n".join(res) if string_format else res
