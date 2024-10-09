"""
Spelling Bee Game

In this game, each player tries to create the longest possible English word using a given set of allowed letters.

**Gameplay:**

- A set of six unique lowercase letters is randomly generated and provided to both players.
- Players must create the longest valid English word using only the allowed letters.
- Each letter can be used multiple times in a word.
- Players submit their words simultaneously.
- The player with the longer valid word wins.
- If both words are of equal length, the game is a draw.
- If a player submits an invalid word, they lose.

**Key Rules:**

- Words must be composed only of the allowed letters.
- Each letter can be used multiple times.
- Words must be valid English words.
- Players must wrap their word in square brackets, e.g., `[example]`.

**Parameters:**

- No additional parameters are required for this game.

**Game Outcomes:**

- A player wins by submitting a longer valid word than the opponent.
- A player loses if they submit an invalid word.
- The game is a draw if both players submit words of equal length.
"""

import random
import re
import string
from typing import Any, Dict, Optional, Tuple

import enchant

import textarena as ta


class SpellingBeeEnv(ta.Env):
    def __init__(self):
        """
        Initialize the Spelling Bee Game environment.
        """
        self.environment_name = "Spelling Bee Game"

        # Initialize game state variables
        self.game_state = ta.State(
            {
                "allowed_letters": set(),
                "player_words": {0: None, 1: None},
                "logs": [],
                "render": ["allowed_letters", "player_words"],
            }
        )
        self.observation = None

        # Initialize Enchant dictionaries for US and UK English
        try:
            self.word_checker_us = enchant.Dict("en_US")
            self.word_checker_uk = enchant.Dict("en_GB")
        except enchant.errors.DictNotFoundError as e:
            raise ValueError(
                f"Enchant dictionary not found: {e}. Ensure that the en_US and en_GB dictionaries are installed."
            )

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[ta.Observation], ta.Info]:
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

        self.game_state["allowed_letters"] = self._generate_allowed_letters()
        self.game_state["player_words"] = {0: None, 1: None}
        self.game_state["logs"] = []

        # Generate initial prompts for both players
        self.observation = {
            0: [self._generate_player_prompt(player_id=0)],
            1: [self._generate_player_prompt(player_id=1)],
        }

        info = {
            "allowed_letters": "".join(sorted(self.game_state["allowed_letters"])),
        }

        self.game_state["logs"].append("[GAME] New game started.")
        self.game_state["logs"].append(
            f"[GAME] Allowed letters are: {''.join(sorted(self.game_state['allowed_letters']))}"
        )

        return self.observation, info

    def _generate_allowed_letters(self) -> set:
        """
        Generate a random set of six unique lowercase letters.

        Returns:
            set: A set of allowed letters.
        """
        return set(random.sample(string.ascii_lowercase, 6))

    def _generate_player_prompt(self, player_id: int) -> ta.Message:
        """
        Generate the initial prompt for a player.

        Args:
            player_id (int): The player's ID.

        Returns:
            str: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id} in the Spelling Bee Game.\n"
            f"Allowed Letters: {''.join(sorted(self.game_state['allowed_letters']))}\n"
            "Create the longest possible English word using only the allowed letters. You may use each letter multiple times.\n"
            "Please wrap your word in square brackets, e.g., '[example]'.\n"
            "On your turn, simply type your word.\n"
        )
        return ta.GAME_ID, prompt

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
        terminated = False
        truncated = False
        reward = None
        info = {}
        other_player_id = 1 - player_id

        # Log the player's action
        self.game_state["logs"].append((player_id, action))

        assert (
            self.game_state["player_words"][player_id] is None
        ), f"Player {player_id} has already provided a word. Please reset the environment."

        # find the word
        word = action.strip().lower()
        match = re.search(r"\[(\w+)\]", word)
        if match:
            word = match.group(1)
            self.game_state["player_words"][player_id] = word
        else:
            # no word was provided in the correct format
            terminated = True
            reward = {player_id: -1, other_player_id: 0}
            info["reason"] = (
                f"Player {player_id} did not submit a word in the correct format."
            )

        if (
            self.game_state["player_words"][player_id] is None
            or self.game_state["player_words"][other_player_id] is None
        ):
            return (
                self.observation if not terminated else None,
                reward,
                truncated,
                terminated,
                info,
            )

        # check the words for validity and length
        terminated = True

        # 1. check if player 0 word is valid
        w1_is_valid, w1_reason = self._check_word_validity(
            word=self.game_state["player_words"][0]
        )
        w2_is_valid, w2_reason = self._check_word_validity(
            word=self.game_state["player_words"][1]
        )

        # Neither word is valid
        if (not w1_is_valid) and (not w2_is_valid):
            reward = {0: -1, 1: -1}
            info["reason"] = (
                f"Neither word is valid: Player 0: {w1_reason}. Player 1: {w2_reason}"
            )

        # w1 is valid but w2 isn't
        elif (w1_is_valid) and (not w2_is_valid):
            reward = {0: 1, 1: -1}
            info["reason"] = f"Player 1 provided an invalid word ({w2_reason})."

        # w2 is valid but w1 isn't
        elif (not w1_is_valid) and (w2_is_valid):
            reward = {0: -1, 1: 1}
            info["reason"] = f"Player 0 provided an invalid word ({w1_reason})."

        # both words are valid
        elif w1_is_valid and w2_is_valid:
            # check which one is longer
            len_p0_word = len(self.game_state["player_words"][0])
            len_p1_word = len(self.game_state["player_words"][1])

            if len_p0_word > len_p1_word:
                # player 0 wins
                reward = {0: 1, 1: -1}
                info["reason"] = f"Player 0 provided a longer word."

            elif len_p1_word > len_p0_word:
                # player 1 wins
                reward = {0: -1, 1: 1}
                info["reason"] = f"Player 1 provided a longer word."

            else:
                # same length. It's a draw
                reward = {0: 0, 1: 0}
                info["reason"] = f"Draw. Both words are equal length."

            self.game_state["logs"].append((ta.GAME_ID, info["reason"]))
            return None, reward, truncated, terminated, info

    def _check_word_validity(self, word) -> Tuple[bool, Optional[str]]:
        """
        Check if the submitted word is valid.

        Args:
            word (str): The submitted word.

        Returns:
            Tuple[bool, Optional[str], str]: (is_valid, reason)
        """
        # 1.st check it only uses the allowed letters
        word_letter_set = set(word)

        # check if the set of letters are a subset of the allowed letter set
        if not word_letter_set.issubset(self.game_state["allowed_letters"]):
            return False, "The word used Illegal characters."

        # check if the word is a valid English word using Enchant
        is_valid = self.word_checker_us.check(word) or self.word_checker_uk.check(word)
        if not is_valid:
            return False, "The word is not a valid english word."

        # the word is valid
        return True, None

    def render(self):
        """
        Render the current game state.
        """
        print("Allowed Letters:")
        print(" ".join(sorted(self.game_state["allowed_letters"])))
        print("\nGame Logs:")
        for player_id, message in self.game_state["logs"]:
            if player_id == ta.GAME_ID:
                print(f"[GAME] {message}")
            else:
                print(f"[Player {player_id}] {message}")
