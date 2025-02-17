# import re, random 
# from typing import Any, Dict, Optional, Tuple 

# import nltk 
# import enchant 
# from nltk.corpus import words 
# nltk.download("words")

# import textarena as ta 


# class WordChainsEnv(ta.Env):
#     """ Environment for playing the Word Chains game """
    
#     def __init__(self, max_turns: Optional[int]=None):
#         """ 
#         Initialize the Word Chains game environment

#         Args:
#             max_turns (int): Maximum number of turns before the game ends in a draw.
#         """

#         # Ensure NLTK words are loaded
#         self.word_list = list((set(word.lower() for word in words.words())))

#         # Initialize Enchant dictionaries for US and UK English
#         try:
#             self.word_checker_us = enchant.Dict("en_US")
#             self.word_checker_uk = enchant.Dict("en_GB")
#         except enchant.errors.DictNotFoundError as e:
#             raise ValueError(f"Enchant dictionary not found: {e}. Ensure that the en_US and en_GB dictionaries are installed.")


#         # Initialize game state variables
#         self.state = ta.State(
#             num_players=2,
#             max_turns=max_turns,
#         )

#     @property 
#     def offline_renderer(self):
#         raise NotImplementedError

#     @property 
#     def terminal_render_keys(self):
#         return ["current_word", "required_start_letter"]


#     def reset(self, seed: Optional[int]=None):
#         """
#         Reset the game to its initial state.
#         Args:
#             seed (Optional[int]): Seed for random number generator to ensure reproducibility.
#         """
#         if seed is not None:
#             random.seed(seed)

#         # pick a starting word
#         starting_word = random.choice(self.word_list)

#         # reset state
#         self.state.reset(
#             game_state={
#                 "current_word": starting_word,
#                 "used_words": set(starting_word),
#                 "required_start_letter": starting_word[-1].lower()
#             },
#             player_prompt_function=self._generate_player_prompt
#         )

#     def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
#         """
#         Generate the initial prompt for a player.
#         Args:
#             player_id (int): ID of the player
#             game_state (Dict): game specific state
#         Returns:
#             str: The initial prompt for the player.
#         """
#         prompt = (
#             f"You are Player {player_id} in the Word Chains Game.\n"
#             "Players take turns to provide valid English words that start with the last letter of the previous word.\n"
#             "Repetition of words is not allowed.\n"
#             "If you provide an invalid word, repeat a word, or fail to follow the sequence, you lose.\n"
#             "Please wrap your word in square brackets, e.g., '[apple]', '[monkey'], etc. .\n"
#             f"The starting word is [{game_state['current_word']}]. Please provide the next word.\n"
#             "On your turn, simply type your word.\n"
#         )
#         if self.state.max_turns:
#             prompt += f"The game will end in a draw after {self.state.max_turns} turns if no player loses.\n"
#         return prompt

#     def step(self, action: str) -> Tuple[bool, ta.Info]:
#         """
#         Process the player's move.
#         Args:
#             action (str): The next word (wrapped in squared brackets)
#         Returns:
#             tuple: (done, info)
#         """
#         # add action to log and observation
#         self.state.add_observation(
#             from_id=self.state.current_player_id,
#             to_id=-1, # Broadcast to all 
#             message=action,
#             for_logging=True
#         )

#         # Extract the word from the action
#         word_match = re.search(r"\[(\w+)\]", action)
#         if not word_match:
#             # Invalid action format
#             self.state.set_invalid_move(
#                 player_id=self.state.current_player_id,
#                 reason=f"The Player {self.state.current_player_id} did not provide a word in the valid format."
#             )

#         else:
#             word = word_match.group(1).lower()
#             # Check if the word starts with the required letter
#             if not word.startswith(self.state.game_state["required_start_letter"]):
#                 # Invalid format
#                 self.state.set_invalid_move(
#                     player_id=self.state.current_player_id,
#                     reason=f"The word provided by Player {self.state.current_player_id} did start with the correct letter."
#                 )

#             # check if the word is a valid english word
#             elif not (self.word_checker_us.check(word) or self.word_checker_uk.check(word)):
#                 # Invalid word
#                 self.state.set_invalid_move(
#                     player_id=self.state.current_player_id,
#                     reason=f"The word provided by Player {self.state.current_player_id} is not a valid english word."
#                 )

#             # Check if the word has already been used
#             elif word in self.state.game_state["used_words"]:
#                 self.state.set_invalid_move(
#                     player_id=self.state.current_player_id,
#                     reason=f"Player {self.state.current_player_id} repeated the word '{word}', which has already been used."
#                 )


#             # The move is valid: update the game state
#             else:
#                 self.state.game_state["used_words"].add(word)
#                 self.state.game_state["current_word"] = word
#                 self.state.game_state["required_start_letter"] = word[-1].lower()

#                 # Broadcast a message about the valid move
#                 self.state.add_observation(
#                     from_id=ta.GAME_ID,
#                     to_id=-1,  # Broadcast to all players
#                     message=f"Player {self.state.current_player_id} played the word: [{word}]. Next word must start with '{word[-1].lower()}'."
#                 )

#         return self.state.step()

import re, random 
from typing import Any, Dict, Optional, Tuple 

import nltk 
import enchant 
from nltk.corpus import words 
nltk.download("words")

import textarena as ta 

class WordChainsEnv(ta.Env):
    """ Environment for playing the Word Chains game with increasing word length requirement """
    
    def __init__(self):
        """ 
        Initialize the Word Chains game environment

        Args:
            max_turns (int): Maximum number of turns before the game ends in a draw.
        """
        # Ensure NLTK words are loaded
        self.word_list = list((set(word.lower() for word in words.words())))
        
        # only conserd words shorter then 6 characters
        self.word_list = [word for word in self.word_list if len(word) <= 5]
        

        # Initialize Enchant dictionaries for US and UK English
        try:
            self.word_checker_us = enchant.Dict("en_US")
            self.word_checker_uk = enchant.Dict("en_GB")
        except enchant.errors.DictNotFoundError as e:
            raise ValueError(f"Enchant dictionary not found: {e}. Ensure that the en_US and en_GB dictionaries are installed.")

        # Initialize game state variables
        self.state = ta.State(
            num_players=2,
            max_turns=None,
        )

    @property 
    def offline_renderer(self):
        raise NotImplementedError

    @property 
    def terminal_render_keys(self):
        return ["current_word", "required_start_letter", "required_length"]

    def reset(self, seed: Optional[int]=None):
        """
        Reset the game to its initial state.
        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        # Pick a starting word of minimum length
        starting_word = random.choice(self.word_list) 
        # Reset state
        self.state.reset(
            game_state={
                "current_word": starting_word,
                "used_words": set([starting_word]),
                "required_start_letter": starting_word[-1].lower(),
                "required_length": len(starting_word) + 1  # Next word must be one letter longer
            },
            player_prompt_function=self._generate_player_prompt
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the initial prompt for a player.
        Args:
            player_id (int): ID of the player
            game_state (Dict): game specific state
        Returns:
            str: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id} in the Word Chains Game.\n"
            "Players take turns to provide valid English words that:\n"
            "1. Start with the last letter of the previous word\n"
            "2. Must be longer than the previous word\n"
            "3. Cannot be a word that was previously used\n\n"
            "If you provide an invalid word, repeat a word, or fail to follow the rules, you lose.\n"
            "Please wrap your word in square brackets, e.g., '[apple]', '[monkey]', etc.\n"
            f"The starting word is [{game_state['current_word']}].\n"
            f"Your word must start with '{game_state['required_start_letter']}' and "
            f"be at exactly {game_state['required_length']} letters long.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process the player's move.
        Args:
            action (str): The next word (wrapped in squared brackets)
        Returns:
            tuple: (done, info)
        """
        # Add action to log and observation
        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=-1,  # Broadcast to all 
            message=action,
            for_logging=True
        )

        # Extract the word from the action
        word_match = re.search(r"\[(\w+)\]", action)
        if not word_match:
            # Invalid action format
            self.state.set_invalid_move(
                player_id=self.state.current_player_id,
                reason=f"Player {self.state.current_player_id} did not provide a word in the valid format."
            )

        else:
            word = word_match.group(1).lower()
            required_length = self.state.game_state["required_length"]
            
            # Check if the word has the correct length
            if len(word) != required_length:
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id,
                    reason=f"The word must be exactly {required_length} letters long. '{word}' has {len(word)} characters."
                )
            
            # Check if the word starts with the required letter
            elif not word.startswith(self.state.game_state["required_start_letter"]):
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id,
                    reason=f"The word must start with '{self.state.game_state['required_start_letter']}'."
                )

            # Check if the word is a valid English word
            elif not (self.word_checker_us.check(word) or self.word_checker_uk.check(word)):
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id,
                    reason=f"'{word}' is not a valid English word."
                )

            # Check if the word has already been used
            elif word in self.state.game_state["used_words"]:
                self.state.set_invalid_move(
                    player_id=self.state.current_player_id,
                    reason=f"The word '{word}' has already been used."
                )

            # The move is valid: update the game state
            else:
                self.state.game_state["used_words"].add(word)
                self.state.game_state["current_word"] = word
                self.state.game_state["required_start_letter"] = word[-1].lower()
                self.state.game_state["required_length"] = len(word) + 1

                # Broadcast a message about the valid move
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,  # Broadcast to all players
                    message=(
                        f"Player {self.state.current_player_id} played: [{word}]\n"
                        f"Next word must:\n"
                        f"1. Start with '{word[-1].lower()}'\n"
                        f"2. Be exactly {len(word) + 1} letters long"
                    )
                )

        return self.state.step()