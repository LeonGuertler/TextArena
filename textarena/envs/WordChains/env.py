import re, random 
from typing import Any, Dict, Optional, Tuple 

import nltk 
from nltk.corpus import words 
nltk.download("words")

import textarena as ta 
from textarena.envs.WordChains.renderer import create_board_str
from textarena.utils.word_lists import EnglishDictionary


class WordChainsEnv(ta.Env):
    """ Environment for playing the Word Chains game with increasing word length requirement """
    
    def __init__(self):
        """  Initialize the Word Chains game environment """
        # Ensure NLTK words are loaded
        self.word_list = list((set(word.lower() for word in words.words())))
        
        # only conserd words shorter then 6 characters
        self.word_list = [word for word in self.word_list if len(word) <= 5]
        
        # Initialize dictionaries for US and UK English
        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the game to its initial state """
        self.state = ta.State(num_players=2, min_players=2, max_players=2)

        # Pick a starting word of minimum length
        starting_word = random.choice(self.word_list) 
        
        # Reset state
        game_state={
            "current_word": starting_word,
            "used_words": set([starting_word]),
            "required_start_letter": starting_word[-1].lower(),
            "required_length": len(starting_word) + 1  # Next word must be one letter longer
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for a player """
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
        """ Process the player's move """
        # Add action to log and observation
        self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action, for_logging=True)

        # Extract the word from the action
        word_match = re.search(r"\[(\w+)\]", action)
        is_invalid = False
        if not word_match:
            # Invalid action format
            is_invalid=True
            reason=f"Player {self.state.current_player_id} did not provide a word in the valid format."

        else:
            word = word_match.group(1).lower()
            required_length = self.state.game_state["required_length"]
            
            # Check if the word has the correct length
            if len(word) != required_length:
                is_invalid=True
                reason=f"The word must be exactly {required_length} letters long. '{word}' has {len(word)} characters."

            
            # Check if the word starts with the required letter
            elif not word.startswith(self.state.game_state["required_start_letter"]):
                is_invalid=True
                reason=f"The word must start with '{self.state.game_state['required_start_letter']}'."


            # Check if the word is a valid English word
            elif not self.dictionary.is_english_word(word):
                is_invalid=True
                reason=f"'{word}' is not a valid English word."


            # Check if the word has already been used
            elif word in self.state.game_state["used_words"]:
                is_invalid=True
                reason=f"The word '{word}' has already been used."


            # The move is valid: update the game state
            else:
                self.state.game_state["used_words"].add(word)
                self.state.game_state["current_word"] = word
                self.state.game_state["required_start_letter"] = word[-1].lower()
                self.state.game_state["required_length"] = len(word) + 1

                # Broadcast a message about the valid move
                message=(
                    f"Player {self.state.current_player_id} played: [{word}]\n"
                    f"Next word must:\n"
                    f"1. Start with '{word[-1].lower()}'\n"
                    f"2. Be exactly {len(word) + 1} letters long"
                )
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

        if is_invalid:
            self.state.set_invalid_move(player_id=self.state.current_player_id, reason=reason)

        return self.state.step()