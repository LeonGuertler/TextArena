import re, random
from typing import Optional, Tuple, Dict, Any
import numpy as np

import textarena as ta
from textarena.envs.utils.word_lists import EnglishDictionary


class SpellingBeeEnv(ta.Env):
    """
    Environment for the Spelling Bee Game with increasing word length.
    """

    def __init__(self, num_letters: int):
        """
        Initialize the Spelling Bee Game environment.

        Args:
            num_letters (int): Number of unique allowed letters.
        """
        self.environment_name = "Spelling Bee Game"

        self.num_letters = num_letters

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=None,
        )

        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)

    @property
    def offline_renderer(self):
        raise NotImplementedError

    @property
    def terminal_render_keys(self):
        return ["allowed_letters"]

    def reset(self, seed: Optional[int] = None):
        """
        Reset the Spelling Bee game to its initial state.

        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.

        Returns:
            Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)

        self.state.reset(
            game_state={
                "allowed_letters": self._generate_allowed_letters(),
                "word_history": [],
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def _generate_allowed_letters(self) -> set:
        """
        Generate a random set of unique lowercase letters with a frequency-weighted distribution.
        """
        if self.num_letters > 26:
            raise ValueError("num_letters cannot exceed 26.")

        # Frequency of letters in the English language (rough estimates)
        letter_frequencies = {
            "a": 8.17,
            "b": 1.49,
            "c": 2.78,
            "d": 4.25,
            "e": 12.70,
            "f": 2.23,
            "g": 2.02,
            "h": 6.09,
            "i": 7.00,
            "j": 0.15,
            "k": 0.77,
            "l": 4.03,
            "m": 2.41,
            "n": 6.75,
            "o": 7.51,
            "p": 1.93,
            "q": 0.10,
            "r": 5.99,
            "s": 6.33,
            "t": 9.06,
            "u": 2.76,
            "v": 0.98,
            "w": 2.36,
            "x": 0.15,
            "y": 1.97,
            "z": 0.07,
        }

        letters = list(letter_frequencies.keys())
        weights = list(letter_frequencies.values())

        # Convert weights to probabilities that sum to 1.
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        # Use numpy.random.choice to sample without replacement
        allowed = np.random.choice(
            letters, size=self.num_letters, replace=False, p=probs
        )
        return set(allowed)

    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[int, Any]
    ) -> str:
        """
        Generate the initial prompt for a player.

        Args:
            player_id (int): The player's ID.

        Returns:
            str: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id} in the Spelling Bee Game.\n"
            f"Allowed Letters: {''.join(sorted(game_state['allowed_letters']))}\n"
            "Each word must be at least as long as the previous word.\n"
            "Repeated words are not allowed.\n"
            "Wrap your word in square brackets, e.g., '[example]'.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process the player's action.

        Args:
            action (str): The player's submitted word.

        Returns:
            tuple: (observations, rewards, truncated, terminated, info)
        """
        # update the observations and log the action
        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=-1,  # Broadcast to all
            message=action,
            for_logging=True,
        )

        # extract provided word
        word = action.strip().lower()
        match = re.search(r"\[(\w+)\]", word)

        if match:
            word = match.group(1)

            # check if the word is longer/equal than the last word, and not a repeated word
            if len(self.state.game_state["word_history"]) == 0:
                # check if the letters are correct
                if not set(word).issubset(self.state.game_state["allowed_letters"]):
                    self.state.set_invalid_move(
                        player_id=self.state.current_player_id,
                        reason=f"Player {self.state.current_player_id} tried submitting a word containing illegal characters.",
                    )
                else:
                    self.state.game_state["word_history"].append(word)

            else:
                if len(word) < len(self.state.game_state["word_history"][-1]):
                    self.state.set_invalid_move(
                        player_id=self.state.current_player_id,
                        reason=f"Player {self.state.current_player_id} tried submitting a shorter word than the previous word.",
                    )
                elif word in self.state.game_state["word_history"]:
                    self.state.set_invalid_move(
                        player_id=self.state.current_player_id,
                        reason=f"Player {self.state.current_player_id} tried submitting a repeated word.",
                    )

                elif not (self.dictionary.is_english_word(word)):
                    self.state.set_invalid_move(
                        player_id=self.state.current_player_id,
                        reason=f"Player {self.state.current_player_id} tried submitting a non-english word.",
                    )

                else:
                    # correct length

                    # check if the letters are correct
                    if not set(word).issubset(self.state.game_state["allowed_letters"]):
                        self.state.set_invalid_move(
                            player_id=self.state.current_player_id,
                            reason=f"Player {self.state.current_player_id} tried submitting a word containing illegal characters.",
                        )
                    else:
                        self.state.game_state["word_history"].append(word)

        else:
            self.state.set_invalid_move(
                player_id=self.state.current_player_id,
                reason=f"Player {self.state.current_player_id} tried submitting a word in the wrong format. Please make sure to use squared brackets.",
            )

        return self.state.step()
