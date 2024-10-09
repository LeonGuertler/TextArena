"""
Word Chains Game

In this game, two players take turns to say words that start with the last letter of the previous word.

**Gameplay:**

- The game starts with a random English word.
- Players take turns to provide a word that starts with the last letter of the previous word.
- Words cannot be repeated.
- If a player provides an invalid word, repeats a word, or fails to follow the sequence, they lose.
- The game ends when a player loses or the maximum number of turns is reached.

**Key Rules:**

- The word must start with the last letter of the previous word.
- The word must be a valid English word.
- Words cannot be repeated.
- Players must provide their word enclosed in square brackets, e.g., `[apple]`.

**Parameters:**

- `max_turns`: The maximum number of turns before the game ends in a draw.

**Game Outcomes:**

- A player loses if they provide an invalid word, repeat a word, or fail to follow the sequence.
- The game is a draw if the maximum number of turns is reached without any player losing.
"""

from typing import Optional, Tuple
import random
import re
import textarena as ta
import nltk
from nltk.corpus import words

nltk.download("words")
nltk_words = set(word.lower() for word in words.words())


class WordChainsEnv(ta.Env):
    """Environment for Word Chains Game"""

    def __init__(
        self,
        max_turns: Optional[int] = None,
    ):
        """
        Initialize the Word Chains Game.

        Args:
            max_turns (int): Maximum number of turns before the game ends in a draw.
        """
        self.environment_name = "Word Chains Game"

        # Ensure NLTK words are loaded
        self.word_list = list(nltk_words)

        # Initialize game state
        self.game_state = ta.State(
            {
                "turn": 0,
                "max_turns": max_turns,
                "starting_word": None,
                "required_start_letter": None,
                "used_words": set(),
                "logs": [],
                "render": [
                    "turn",
                    "max_turns",
                    "starting_word",
                    "required_start_letter",
                ],
            }
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[ta.Observation, ta.Info]:
        """
        Reset the game to its initial state.

        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.

        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        self.game_state["turn"] = 0
        self.game_state["used_words"] = set()
        self.game_state.logs = []

        # Select a random starting word
        self.game_state["starting_word"] = random.choice(self.word_list)
        self.game_state["used_words"].add(self.game_state["starting_word"].lower())

        # Set the required starting letter for the next word
        self.game_state["required_start_letter"] = self.game_state["starting_word"][
            -1
        ].lower()

        self.game_state.logs.append(
            f"[GAME] Starting word is '{self.game_state['starting_word']}'."
        )

        # Generate initial prompts for both players
        observations = {
            0: self._generate_player_prompt(player_id=0),
            1: self._generate_player_prompt(player_id=1),
        }

        info = {
            "starting_word": self.game_state["starting_word"],
            "required_start_letter": self.game_state["required_start_letter"],
        }

        return observations, info

    def _generate_player_prompt(self, player_id: int) -> ta.Message:
        """
        Generate the initial prompt for a player.

        Args:
            player_id (int): The player's ID.

        Returns:
            str: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id} in the Word Chains Game.\n"
            "Players take turns to provide valid English words that start with the last letter of the previous word.\n"
            "Repetition of words is not allowed.\n"
            "If you provide an invalid word, repeat a word, or fail to follow the sequence, you lose.\n"
            "Please wrap your word in square brackets, e.g., '[apple]', '[monkey'], etc. .\n"
            f"The starting word is [{self.game_state['starting_word']}]. Please provide the next word.\n"
            "On your turn, simply type your word.\n"
        )
        if self.game_state["max_turns"]:
            prompt += f"The game will end after {self.game_state['max_turns']} turns if no player loses.\n"
        return -1, prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observation],  # observations
        Optional[ta.Reward],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's word.

        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        other_player_id = 1 - player_id
        terminated = False
        truncated = False
        reward = None
        info = {}
        message = [(player_id, action)]

        self.game_state["turn"] += 1
        # Log the player's action
        self.game_state.logs += message

        # Extract the word from the action
        word_match = re.search(r"\[(\w+)\]", action)
        if word_match:
            word = word_match.group(1).lower()
        else:
            # Invalid format
            terminated = True
            reward = {player_id: -1, other_player_id: 1}
            info["reason"] = "Invalid format. Word must be enclosed in square brackets."

        # Check if the word starts with the required letter
        if not word.startswith(self.game_state["required_start_letter"]):
            terminated = True
            reward = {player_id: -1, other_player_id: 1}
            info["reason"] = (
                f"Word does not start with the required letter '{self.game_state['required_start_letter']}'."
            )

        # Check if the word is a valid English word
        if word not in self.word_list:
            terminated = True
            reward = {player_id: -1, other_player_id: 1}
            info["reason"] = f"'{word}' is not a valid English word."

        # Check if the word has been used before
        if word in self.game_state["used_words"]:
            terminated = True
            reward = {player_id: -1, other_player_id: 1}
            info["reason"] = f"Word '{word}' has already been used."
            self.game_state["logs"].append(ta.GAME_ID, {info["reason"]})

            # Check if max turns have been reached
        if (
            self.game_state["max_turns"]
            and self.game_state["turn"] >= self.game_state["max_turns"]
        ):
            truncated = True
            reward = {0: 0, 1: 0}
            info["reason"] = "Maximum number of turns reached. The game is a draw."

        if terminated or truncated:
            self.game_state.logs.append((ta.GAME_ID, {info["reason"]}))
            observations = {player_id: message, other_player_id: message}
            return observations, reward, truncated, terminated, info

        # Add the word to used words and update required start letter
        self.game_state["used_words"].add(word)
        self.game_state["required_start_letter"] = word[-1]

        # Valid turn, continue the game
        observations = {player_id: action, other_player_id: action}
        return observations, reward, truncated, terminated, info

    def render(self):
        """
        Render the current game state.
        """
        print(
            f"Turn: {self.game_state['turn']}/{self.game_state['max_turns'] if self.game_state['max_turns'] else '∞'}"
        )
        print("Game Logs:")
        for id_, log in self.game_state.logs:
            if id_ >= 0:
                print(f"Player {id_}: {log}")
            else:
                print(f"Game: {log}")
        print("\n")
