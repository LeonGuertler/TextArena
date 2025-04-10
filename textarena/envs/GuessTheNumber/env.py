import re, random, copy
from typing import Any, Dict, Optional, Tuple, Union

import textarena as ta
from textarena.envs.GuessTheNumber.renderer import create_board_str

class GuessTheNumberEnv(ta.Env):
    """ Guess the number game environment """

    def __init__(self, min_number: int = 1, max_number: int = 20, max_turns: int = 20):
        """
        Initialize the environment.

        Args:
           min_number: The lower bound
           max_number: The upper bound
           max_turns: The number of guesses
        """
        super().__init__()
        self.min_number = min_number
        self.max_number = max_number 
        self.max_turns = max_turns

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """
        ## intitialise the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## load the game number
        self.game_number = random.randint(self.min_number, self.max_number)
        self.guessed_numbers = set()

        ## reset the game state
        game_state = {
            "game_number": self.game_number,
            "rendered_text": "Guess the number between {} and {}.".format(self.min_number, self.max_number),
            "guess_history": []
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the player prompt """
        prompt = (
            f"You are Player {player_id}. You are playing Guess The Number.\n"
            f"You have to guess the number between {self.min_number} and {self.max_number} within {self.max_turns} turns.\n"
            "As you enter your guess, the game will provide you with hints such as 'higher' or 'lower'.\n"
            "You may provide your response in any manner. Only the number that is wrapped in square brackets will be considered as your guess. For example, [5].\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
            "Enter your guess."
        )
        return prompt
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Take a step in the environment """
        player_id = self.state.current_player_id
        
        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, for_logging=True)

        ## validate the action
        action_search_pattern = re.compile(r"\[(\d+)\]") # e.g. [5]
        match = action_search_pattern.search(action)

        if not match:
            reason=f"Invalid move format. Player {player_id} did not respond with valid '[number]'."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            player_guess = int(match.group(1))
            if player_guess < self.min_number or player_guess > self.max_number:
                reason=f"Invalid move. Player {player_id} guessed a number outside the range specified."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            elif player_guess in self.guessed_numbers:
                reason=f"Invalid move. Player {player_id} has already guessed the number."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                self.guessed_numbers.add(player_guess)
                if player_guess == self.game_number:
                    reason=f"Congratulations! Player {player_id} guessed the correct number."
                    self.state.set_winners(player_ids=[player_id], reason=reason)
                else:
                    if player_guess > self.game_number:
                        hint = "lower"
                    else:
                        hint = "higher"
                    message=f"The target number is {hint}."
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message, for_logging=False)
                    self.state.game_state["guess_history"].append((player_guess, hint))

            self.state.game_state["rendered_text"] = f"Player {player_id} guessed {player_guess}."

        return self.state.step()
