import re, os, json, random 
import importlib.resources
from typing import Optional, Tuple, Dict, List, Any, Union

import textarena as ta


class TabooEnv(ta.Env):
    """ Environment for Taboo Game. """
    def __init__(self, categories: Union[str, List[str]], max_turns: Optional[int], data_path: Optional[str]=None):
        """
        Initialize the Taboo game environment.

        Roles:
            - Player 0 is the Clue Giver
            - Player 1 is the Guesser

        Args:
            categories (Union[str, List[str]]): Either a single category or a list of categories to include in the game.
                                               If a list is provided, one category will be randomly selected.
            max_turns (int): Maximum number of conversation turns.
            data_path (str, optional): Path to the JSON file containing the taboo words.
        """
        # Handle random selection if categories is a list
        if isinstance(categories, list):
            if not categories:
                raise ValueError("Empty list of categories provided.")
            self.categories = [random.choice(categories)]
        else:
            # Convert single string to list for consistent handling
            self.categories = [categories]
            
        self.max_turns = max_turns
        self.action_space = lambda player_id: re.compile(r".*", re.DOTALL)

        # Load the data 
        self.data = self._load_data(data_path)

    @property
    def terminal_render_keys(self):
        return ["word_to_guess", "taboo_words"]

    def _load_data(self, data_path: Optional[str] = None):
        """
        Load the word list based on the specified categories from the JSON file.
        
        The JSON structure is expected to be:
        {
            "category1": {
                "word1": ["taboo1", "taboo2", ...],
                "word2": ["taboo1", "taboo2", ...],
                ...
            },
            "category2": {
                ...
            },
            ...
        }
        
        Args:
            data_path (str, optional): Path to the JSON file containing taboo words data.
            
        Returns:
            dict: A dictionary containing words and their associated taboo words for the selected categories.
            
        Raises:
            FileNotFoundError: If the `data_path` does not exist.
            ValueError: If the JSON file has an invalid format or selected categories are not found.
        """
        try:
            if data_path is not None:
                # Use provided path
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Taboo words data file not found at: {data_path}")
                with open(data_path, "r", encoding="utf-8") as file:
                    full_data = json.load(file)
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.Taboo').joinpath('words.json').open('r') as file:
                    full_data = json.load(file)
            
            # Validate that all specified categories exist in the data
            missing_categories = [cat for cat in self.categories if cat not in full_data]
            if missing_categories:
                raise ValueError(f"Categories not found in data file: {', '.join(missing_categories)}")
                
            # Subsample to the selected categories
            data = {}
            for category in self.categories:
                data.update(full_data[category])
                
            if not data:
                raise ValueError(f"No words found for selected categories: {', '.join(self.categories)}")
                
            return data
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load taboo words data: {str(e)}")

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the Taboo game to its initial state """
        # Initialize game state 
        self.state = ta.State(
            num_players=2, min_players=2, max_players=2, max_turns=self.max_turns, 
            role_mapping={0: "Clue Giver", 1: "Guesser"}, seed=seed
        )



        word_to_guess, taboo_words = random.choice(list(self.data.items()))

        game_state = {"word_to_guess": word_to_guess, "taboo_words": taboo_words}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for a player based on their role """
        if self.state.role_mapping[player_id] == "Clue Giver":
            prompt = (
                f"You are Player {player_id}, the Clue Giver in the Taboo game.\n"
                f"The word to guess is '{game_state['word_to_guess']}'.\n"
                f"Taboo words: {', '.join(game_state['taboo_words'])}.\n"
                "Your goal is to provide clues to help the Guesser guess the word without using the taboo words or the word to guess.\n"
                f"You have {self.state.max_turns} turns to assist the Guesser.\n"
                "On your turn, simply type your clue.\n"
            )

        elif self.state.role_mapping[player_id] == "Guesser":
            prompt = (
                f"You are Player {player_id}, the Guesser in the Taboo game.\n"
                "Your goal is to guess the secret word based on the clues provided by the Clue Giver.\n"
                f"You have {self.state.max_turns} turns to guess the word.\n"
                "On your turn, simply type your guess in squared brackets. For example: '[elephant'].\n"
            )

        else:
            # unexpected
            raise ValueError(f"Unexpected role mapping: {self.state.role_mapping[player_id]}. Expected 'Clue Giver' or 'Guesser'.")

        return prompt 

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action """
        player_id = self.state.current_player_id

        # update the observations and log the action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)


        # Clue Giver's turn
        if self.state.role_mapping[player_id] == "Clue Giver":
            # Check for taboo words or the word to guess in the clue 
            forbidden_words = self.state.game_state["taboo_words"] + [
                self.state.game_state["word_to_guess"]
            ]
            pattern = re.compile(
                r"\b(" + "|".join(map(re.escape, forbidden_words)) + r")\b",
                re.IGNORECASE,
            )
            if pattern.search(action):
                # Clue Giver used a forbidden word.
                reason=f"The Clue Giver (Player {player_id}) mentioned a taboo word, or the target word."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            
        # Guesser's turn
        elif self.state.role_mapping[player_id] == "Guesser":
            # Guesser must provide a guess within squared brackets
            guess_pattern = re.compile(r"\[(.*?)\]")
            match = guess_pattern.search(action)
            if not match:
                # Invalid guess format
                reason="Invalid guess format. Please provide your guess within squared brackets, e.g., '[apple]'."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
                return self.state.step()

            guess = match.group(1).strip().lower()
            correct_word = self.state.game_state["word_to_guess"].lower()

            if guess == correct_word:
                # Guesser guessed correctly
                reason=f"Player {player_id} (Guesser) correctly guessed the word. Both players win!"
                self.state.set_winners(player_ids=[0, 1], reason=reason)
                return self.state.step()


        else:
            # unexpected
            raise ValueError(f"Unexpected role mapping: {self.state.role_mapping[player_id]}. Expected 'Clue Giver' or 'Guesser'.")

        return self.state.step()

