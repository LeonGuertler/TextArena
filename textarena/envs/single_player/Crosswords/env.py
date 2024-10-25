from typing import Any, Dict, Optional, Tuple, Union
import copy
import random
import textarena as ta
import re
import json

class CrosswordsEnv(ta.Env):
    """
    Crosswords environment.
    """

    def __init__(
        self, 
        hardcore: Optional[bool] = False, 
        max_turns: Optional[int] = 100, 
        num_words: Optional[int] = 5
    ):
        """
        TODO
        """

        super().__init__()
        self.environment_name = "Crosswords"
        self.hardcore = hardcore
        self.max_turns = max_turns
        self.num_words = num_words

        ## initialise the game_state
        self.state = ta.State(
            num_players=1,
            max_turns=max_turns,
            render_keys=["rendered_board"], ## ensure that the rendered board is in the game state
        )

        ## load the word list
        with open("textarena/envs/single_player/Crosswords/words_clues.jsonl", "r") as f:
            word_data = f.readlines()
        self.word_data = [json.loads(x) for x in word_data if json.loads(x)["hardcore"]==hardcore]

    
    def reset(
        self, 
        seed: Optional[int] = None
    ) -> Optional[ta.Observations]:
        """
        Reset the game to its initial state.

        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.

        Returns:
            Tuple[str, str, Dict[str, str]]: Initial observations for both players and their secret words.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        ## load the game board
        self.game_board, self.placed_words, self.clues = self._generate_board() ## generate the game board and the placed words for the clues
        self.game_board_hidden = self._hide_letters(self.game_board) ## hide the letters in the game board

        # reset the state
        return self.state.reset(
            game_state={
                "board": copy.deepcopy(self.game_board_hidden),
                "rendered_board": self._render_board(self.game_board_hidden, show_letters=True),
            },
            player_prompt_function=self._generate_player_prompt
        )


    def _generate_player_prompt(self, player_id: int) -> str:
        """
        Generate the prompt for the player based on the current state of the game.
        """
        prompt = (
            f"You are Player {player_id}. You are playing Crosswords ({'Hardcore' if self.hardcore else 'Basic'}).\n"
            "Here is the current state of the Crosswords grid. Each row is numbered, and each column is also numbered.\n"
            "The cells that need to be populated with letters are represented by '_', and those that do not need words are represented by '.'.\n\n"
            "Current Crosswords Grid:\n"
        )

        grid_str = self._render_board(self.game_board_hidden, show_letters=False)
        prompt += grid_str

        prompt += "\n\nHere are the clues for the words you need to find:\n"
        prompt += self._clue_generator()
        prompt += ("\n\nYou may provide your response in any manner. However, note that any wrong guesses will result in you losing. Hence, plan your approach and risk appetite. Only guesses in the format of [row column letter] will be fetched from your response.\n"
                   "As you play, the history of your choices will be appended below. Use the information to complete the game.\n")
    
        return (-1, prompt)


    def _generate_board(self):
        """
        Generate a crossword grid with the given words and their directions.
        """
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

            if not placed:
                print(f"Could not place the word: {word}")

        return grid, placed_words, clues

    def _determine_initial_grid_size(self, words):
        """
        Determine the initial size of the grid based on the length of the longest word.
        """
        max_length = max(len(word) for word in words)
        return round(max_length * 1.5)  # Ensures that the grid size is larger than the longest word to allow placement

    def _create_empty_grid(self, size):
        """
        Create an empty grid of the specified size.
        """
        return [["." for _ in range(size)] for _ in range(size)]

    def _can_place_word(self, grid, word, direction, row, col):
        """
        Check if a word can be placed on the grid at the specified position.
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
        """
        if direction == "across":
            for i, letter in enumerate(word):
                grid[row][col + i] = letter
        else:  # "down"
            for i, letter in enumerate(word):
                grid[row + i][col] = letter

    def _find_overlaps(self, word, grid, placed_words, directions):
        """Find all possible valid overlaps for the word with already placed words."""
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
        """Print the grid for text display."""
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
        """Hide the letters in the grid."""
        return [['_' if cell != "." else cell for cell in row] for row in grid]
    
    def step(
        self,
        player_id: int,
        action: str
    ) -> Tuple[
        Optional[ta.Observations], # observations
        Optional[ta.Rewards], # reward
        bool, # truncated
        bool, # terminated
        ta.Info # info
    ]:
        """
        TODO
        """

        ## update the observations
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=False
        )

        ## validate the actions
        ## note that the response can have multiple guesses at one go.
        action_search_pattern = re.compile(r"\[(\d+)\s(\d+)\s([a-zA-Z])\]")
        # print("Actions", action)
        matches = action_search_pattern.findall(action)
        matches = set(matches) ## remove duplicates
        # print("Matches", matches)

        if not matches:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid move format. Player {player_id} did not respond with valid 'row column letter'."]
            )
        else:
            for match in matches:
                row, col, letter = match
                row, col, letter = int(row), int(col), str(letter)
                if row < 0 or row >= len(self.state.game_state["board"]) or col < 0 or col >= len(self.state.game_state["board"][0]):
                    self.state.set_invalid_move(
                        player_ids=[player_id],
                        reasons=[f"Invalid move. The specified row or column is out of bounds."]
                    )
                    break
                elif self.state.game_state["board"][row][col] != "_":
                    self.state.set_invalid_move(
                        player_ids=[player_id],
                        reasons=[f"Invalid move. The specified cell is already filled."]
                    )
                    break
                elif self._is_move_correct(row, col, letter):
                    self.state.game_state["board"][row][col] = letter.upper()
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=f"Board state: {self._render_board(self.state.game_state['board'], show_letters=True)}",
                        for_logging=False
                    )
                else:
                    self.state.set_invalid_move(
                        player_ids=[player_id],
                        reasons=[f"Invalid move. The specified letter is incorrect."]
                    )
                    break

            ## check if the game is over
            if self._is_game_over(): 
                self.state.set_winners(
                        player_ids=[player_id],
                        reason=f"Congratulations! Player {player_id} completed the Crosswords puzzle."
                    )
                
            ## update the game board
            self.state.game_state["rendered_board"] = self._render_board(self.state.game_state["board"], show_letters=True)

        return self.state.step()

    def _is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        return all("_" not in row for row in self.state.game_state["board"])
    
    def render(self):
        """
        Render the current state of the game.
        """
        print(self.state.game_state["rendered_board"])

    def _clue_generator(self):
        """
        Generate a clue for a word.
        """
        res = []
        for i, set in enumerate(zip(self.placed_words.values(), self.clues.values())):
            res.append(f"{i+1}. {set[1]}: {set[0]}")

        return "\n".join(res)
    
    def _is_move_correct(self, row, col, letter):
        """
        Check if the move is correct.
        """
        return self.game_board[row][col].upper() == letter.upper()