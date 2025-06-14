import re, random
from typing import Optional, Tuple, List, Dict, Any

import textarena as ta
from textarena.envs.games.Mastermind.renderer import create_board_str

class MastermindEnv(ta.Env):
    def __init__(self, code_length: Optional[int] = 4, num_numbers: Optional[int] = 6, max_turns: Optional[int] = 20, duplicate_numbers: Optional[bool] = False):
        """
        Args:
            code_length (int): the number of options to get right
            max_turns (int): the number of turns until draw
            duplicate_numbers (bool): whether numbers can be duplicates
        """
        super().__init__()
        self.max_turns = max_turns
        self.code_length = code_length 
        self.num_numbers = num_numbers
        self.duplicate_numbers = duplicate_numbers
        self.action_space = lambda player_id: re.compile(r"\[(\d+(?:\s+\d+)*)\]")
    
    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed, max_turns=self.max_turns) # Initialize game state variables
        sample_fn = random.choices if self.duplicate_numbers else random.sample # generate secret code 
        code = sample_fn(list(range(1, self.num_numbers + 1)), k=self.code_length)
        game_state={"secret_code":code, "guess": [], "code_length": self.code_length, "num_numbers": self.num_numbers, "duplicate_numbers": self.duplicate_numbers, "history": []}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        prompt = (
            f"You are playing Mastermind.\n"
            f"You need to find the code that is {game_state['code_length']} digits long, each digit from 1 to {game_state['num_numbers']}, "
            f"{'with possible repeats' if game_state['duplicate_numbers'] else 'with no duplicates'}.\n"
            "In your response, you can submit your guess in the following format: '[2 1 4 5]'.\n"
            "After each guess, you will receive feedback in the form of black and white pegs.\n"
            "A black peg indicates a correct digit in the correct position, while a white peg indicates a correct digit in the wrong position.\n"
            f"You have {self.state.max_turns:.0f} turns to guess the code.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION) # Update the observation with the player's action
        match = self.action_space(self.state.current_player_id).search(action) # e.g., [1 2 3 4]

        if match is None:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"You did not respond with a space-separated list of numbers wrapped in square brackets.")
            return self.state.step()

        # Extract and validate the numbers from the action
        try:
            player_guess = list(map(int, match.group(1).split()))
        except ValueError:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"All entries must be valid integers.")
            return self.state.step()

        # Validate guess length
        if len(player_guess) != self.state.game_state["code_length"]:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"The guess should contain exactly {self.state.game_state['code_length']} numbers.")
            return self.state.step()

        # Validate number range
        if any(num < 1 or num > self.state.game_state["num_numbers"] for num in player_guess):
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"All numbers must be between 1 and {self.state.game_state['num_numbers']}.")
            return self.state.step()

        # Validate no duplicates if not allowed
        if not self.state.game_state["duplicate_numbers"] and len(set(player_guess)) != len(player_guess):
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Duplicate numbers are not allowed.")
            return self.state.step()

        # Check if the guess has been made before
        previous_guesses = [entry["guess"] for entry in self.state.game_state["history"]]
        if player_guess in previous_guesses:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"You have already guessed {player_guess}. Please try a different guess.")
            return self.state.step()

        black_pegs, white_pegs = self._evaluate_guess(player_guess) # Evaluate the guess
        self.state.game_state["history"].append({"guess": player_guess, "black": black_pegs, "white": white_pegs})
        
        if black_pegs == self.state.game_state["code_length"]: # Check for win condition
            self.state.set_outcome(reward=1, reason=f"You have cracked the code, solving {black_pegs} out of {self.state.game_state['code_length']} pegs correctly")
        else: # Add feedback message to observations
            self.state.add_observation(from_id=ta.GAME_ID, message=f"Submitted [{match.group(1)}]. Feedback: {black_pegs} black peg(s), {white_pegs} white peg(s).", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

        # check turn count
        if self.state.check_turn_limit():
            pct_completion = self._get_percentage_completion()
            self.state.set_outcome(reward=pct_completion, reason=f"Turn limit reached. You guessed {pct_completion*100:.2f} percent of the numbers correctly.")
        return self.state.step()
    
    def _evaluate_guess(self, player_guess: List[int]) -> Tuple[int, int]:
        black_pegs, white_pegs = 0, 0
        secret_copy = self.state.game_state["secret_code"].copy()
        guess_copy = player_guess.copy()
        # First pass: count black pegs and mark them as None
        for i in range(self.state.game_state["code_length"]):
            if guess_copy[i] == secret_copy[i]:
                black_pegs += 1
                secret_copy[i] = None
                guess_copy[i] = None
        # Second pass: count white pegs using the remaining numbers
        for i in range(self.state.game_state["code_length"]):
            if guess_copy[i] is not None and guess_copy[i] in secret_copy:
                white_pegs += 1
                secret_copy[secret_copy.index(guess_copy[i])] = None # Remove the first occurrence to prevent over-counting
        return black_pegs, white_pegs

    def _get_percentage_completion(self) -> float:
        """
        Calculate a percentage completion score based on the player's latest performance.

        """
        if not self.state.game_state["history"]:
            return 0.0
        
        code_length = self.state.game_state["code_length"]
        
        # calculate the latest guess quality
        latest_entry = self.state.game_state["history"][-1]
        black_pegs = latest_entry["black"] * 1.0
        white_pegs = latest_entry["white"] * 0.5

        latest_quality = (black_pegs + white_pegs) / code_length

        return latest_quality