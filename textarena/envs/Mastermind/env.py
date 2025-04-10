import re, random
from typing import Optional, Tuple, List, Dict, Any

import textarena as ta
from textarena.envs.Mastermind.renderer import create_board_str


class MastermindEnv(ta.Env):
    """ Environment for Mastermind game """
    def __init__(
        self,
        code_length: Optional[int] = 4,
        num_numbers: Optional[int] = 6,
        max_turns: Optional[int] = 20,
        duplicate_numbers: Optional[bool] = False
    ):
        """
        Initializes the Mastermind environment.
        
        Parameters:
            code_length (int): the number of options to get right
            max_turns (int): the number of turns until draw
            duplicate_numbers (bool): whether numbers can be duplicates
        """
        super().__init__()
        self.max_turns = max_turns
        self.code_length = code_length 
        self.num_numbers = num_numbers
        self.duplicate_numbers = duplicate_numbers
    
    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Resets the environment to its initial state. """
        # Initialize game state variables
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        # generate secret code 
        available_numbers = list(range(1, self.num_numbers + 1))
        sample_fn = random.choices if self.duplicate_numbers else random.sample
        code = sample_fn(available_numbers, k=self.code_length)
        # if self.duplicate_numbers:
        #     code = random.choices(available_numbers, k=self.code_length)
        # else:
        #     code = random.sample(available_numbers, k=self.code_length)

        ## return the initial observations
        game_state={"secret_code": code, "guess": []}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generates the initial prompt for a player """
        prompt = (
            f"You are Player {player_id}. You are playing Mastermind.\n"
            f"... that is {self.code_length} digits long, each digit from 1 to {self.num_numbers}, "
            f"{'with possible repeats' if self.duplicate_numbers else 'with no duplicates'}.\n"
            "In your response, you can mention any code or previously submitted code in the format of 1 2 3 4. Only when you have decided to make your guess, then you must strictly enter the code in square brackets like [2 1 4 5]. This is to avoid submitting a wrong code to the game environment.\n"
            "Hence, if you are quoting a recent guess, you must mention the numbers without the square brackets.\n"
            "After each guess, you will receive feedback in the form of black and white pegs.\n"
            "A black peg indicates a correct digit in the correct position, while a white peg indicates a correct digit in the wrong position.\n"
            f"You have only {self.state.max_turns:.0f} turns to guess the code.\n"
        )
        return prompt
    

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action and update the game state """
        player_id = self.state.current_player_id

        ## update the observations
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, for_logging=True)

        ## search for the action pattern
        action_search_pattern = re.compile(r"\[(\d+(?:\s+\d+)*)\]") ## e.g., [1 2 3 4]
        match = action_search_pattern.search(action)

        if match is None:
            reason=f"Invalid move format. Player {player_id}, did not respond with a space-separated list of numbers wrapped in square brackets."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return self.state.step()

        # Extract and validate the numbers from the action
        try:
            player_guess = list(map(int, match.group(1).split()))
        except ValueError:
            reason = f"Invalid move format. Player {player_id}, all entries must be valid integers."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return self.state.step()

        # Validate guess length
        if len(player_guess) != self.code_length:
            reason = f"Invalid move format. Player {player_id}, the guess should contain exactly {self.code_length} numbers."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return self.state.step()

        # Validate number range
        if any(num < 1 or num > self.num_numbers for num in player_guess):
            reason = f"Invalid move format. Player {player_id}, all numbers must be between 1 and {self.num_numbers}."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return self.state.step()

        # Validate no duplicates if not allowed
        if not self.duplicate_numbers and len(set(player_guess)) != len(player_guess):
            reason = f"Invalid move format. Player {player_id}, duplicate numbers are not allowed."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            return self.state.step()

        # Evaluate the guess
        black_pegs, white_pegs = self._evaluate_guess(player_id, player_guess)

        if "history" not in self.state.game_state:
            self.state.game_state["history"] = []

        self.state.game_state["history"].append({
            "guess": player_guess,
            "black": black_pegs,
            "white": white_pegs
        })


        # Check for win condition
        if black_pegs == self.code_length:
            self.state.set_winners(
                player_ids=[player_id], 
                reason=f"Player {player_id} has cracked the code!"
            )
        else:
            # Add feedback message to observations
            message = (
                f"Player {player_id} submitted [{match.group(1)}]. "
                f"Feedback: {black_pegs} black peg(s), {white_pegs} white peg(s)."
            )
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message,
                for_logging=False
            )

        self.state.game_state["guess"] = player_guess
        return self.state.step()

    
    def _evaluate_guess(self, player_id: int, player_guess: List[int]) -> Tuple[int, int]:
        """
        Evaluates the player's guess and returns the number of black and white pegs.
        
        Black peg: correct digit in the correct position.
        White peg: correct digit in the wrong position.
        
        Args:
            player_id (int): ID of the player making the guess.
            player_guess (List[int]): The player's guess.
        
        Returns:
            Tuple[int, int]: Number of black and white pegs.
        """
        secret_code = self.state.game_state["secret_code"]
        black_pegs = 0
        white_pegs = 0

        # Create copies to mark matched positions
        secret_copy = secret_code.copy()
        guess_copy = player_guess.copy()

        # First pass: count black pegs and mark them as None
        for i in range(self.code_length):
            if guess_copy[i] == secret_copy[i]:
                black_pegs += 1
                secret_copy[i] = None
                guess_copy[i] = None

        # Second pass: count white pegs using the remaining numbers
        for i in range(self.code_length):
            if guess_copy[i] is not None and guess_copy[i] in secret_copy:
                white_pegs += 1
                # Remove the first occurrence to prevent over-counting
                secret_copy[secret_copy.index(guess_copy[i])] = None

        return black_pegs, white_pegs