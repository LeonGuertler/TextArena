import random
from typing import Any, Dict, List, Optional, Set, Tuple

import textarena as ta


class ASCIIArtGameEnv(ta.Env):
    """Environment for Collaborative ASCII Art Game"""
    
    def __init__(
        self,
        max_guesses_per_round: Optional[int] = 3,
        rounds_per_game: Optional[int] = 3,
        term_difficulty_levels: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the Collaborative ASCII Art Game environment.

        Args:
            max_guesses_per_round (int): Maximum number of guess attempts per drawing.
            rounds_per_game (int): Number of rounds to play (default is 3).
            term_difficulty_levels (Dict[str, int]): Dictionary mapping terms to their difficulty ratings.
        """
        # Game configuration
        self.max_guesses_per_round = max_guesses_per_round
        self.rounds_per_game = rounds_per_game
        
        # Load term database with difficulty levels
        self._load_term_database(term_difficulty_levels)
        
        # ASCII symbol sets of varying complexity
        self.symbol_sets = [
            # Basic set (easier)
            {'-', '|', '/', '\\', '+', 'o', '.', '*'},
            # Medium set
            {'-', '|', '/', '\\', '+', 'o', '.', '*', '^', 'v', '<', '>', '(', ')', '[', ']'},
            # Complex set (harder)
            {'-', '|', '/', '\\', '+', 'o', '.', '*', '^', 'v', '<', '>', '(', ')', '[', ']', 
             '#', '=', '_', '~', ':', ';', '"', '`', '\'', ','}
        ]
        
        # Initialize game state
        self.state = ta.State(
            num_players=2,
            max_turns=rounds_per_game * 2,  # 2 turns per round (one for each player)
        )

    @property
    def terminal_render_keys(self):
        return ["terms", "current_symbols", "shared_score", "current_round", "current_artist", "current_guesser"]

    def _load_term_database(self, custom_difficulty_levels: Optional[Dict[str, int]] = None) -> None:
        """
        Load the database of terms to draw with their difficulty ratings.

        Args:
            custom_difficulty_levels (Dict[str, int]): Optional custom difficulty ratings.
        """
        # Default term database with difficulty levels (1-5, where 5 is hardest)
        default_terms = {
            # Easy terms (difficulty 1-2)
            "cat": 1, "dog": 1, "house": 1, "tree": 1, "car": 2, "boat": 2, "sun": 1, "moon": 1,
            # Medium terms (difficulty 3)
            "bicycle": 3, "airplane": 3, "robot": 3, "castle": 3, "train": 3,
            # Hard terms (difficulty 4-5)
            "elephant": 4, "giraffe": 4, "skyscraper": 4, "helicopter": 4,
            "submarine": 5, "dinosaur": 5, "spacecraft": 5
        }
        
        # Use custom difficulty levels if provided, otherwise use defaults
        self.term_database = custom_difficulty_levels if custom_difficulty_levels else default_terms

    def reset(self, seed: Optional[int] = None):
        """
        Reset the ASCII Art Game to its initial state.

        Args:
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.

        Returns:
            Optional[ta.Observations]: Initial observations for both players.
        """
        if seed is not None:
            random.seed(seed)
        
        # Select terms for each round
        terms = random.sample(list(self.term_database.keys()), k=self.rounds_per_game * 2)
        
        # Assign terms to players for each round
        assigned_terms = {}
        for round_num in range(self.rounds_per_game):
            assigned_terms[round_num] = {
                0: terms[round_num * 2],     # Player 0's term for this round
                1: terms[round_num * 2 + 1]  # Player 1's term for this round
            }
            
        # Initialize game state
        self.state.reset(
            game_state={
                "terms": assigned_terms,
                "current_round": 0,
                "current_artist": 0,  # Player 0 starts as artist in round 0
                "current_guesser": 1, # Player 1 starts as guesser in round 0
                "shared_score": 0,
                "current_symbols": random.choice(self.symbol_sets),
                "guesses_remaining": self.max_guesses_per_round,
                "current_drawing": None,
                "round_results": []
            },
            player_prompt_function=self._generate_player_prompt
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the initial prompt for a player with their role and instructions.

        Args:
            player_id (int): ID of the player (0 or 1).
            game_state (Dict[str, Any]): Current game state.

        Returns:
            str: Initial prompt for the player.
        """
        round_num = game_state["current_round"]
        is_artist = (player_id == game_state["current_artist"])
        
        if is_artist:
            term_to_draw = game_state["terms"][round_num][player_id]
            allowed_symbols = ", ".join(sorted(game_state["current_symbols"]))
            
            prompt = (
                f"Welcome to the ASCII Art Game! You are Player {player_id}.\n"
                f"ROUND {round_num + 1}/{self.rounds_per_game}: You are the ARTIST.\n"
                f"Your term to draw is: '{term_to_draw}'\n"
                f"Allowed symbols: {allowed_symbols}\n"
                f"RULES:\n"
                "- Use ONLY the allowed symbols in your drawing\n"
                "- DO NOT use words or emojis\n"
                "- Create a pure visual representation\n"
                "- The other player will guess what you've drawn\n"
                f"Current shared score: {game_state['shared_score']}\n"
                "Create your ASCII art drawing now:"
            )
        else:
            prompt = (
                f"Welcome to the ASCII Art Game! You are Player {player_id}.\n"
                f"ROUND {round_num + 1}/{self.rounds_per_game}: You are the GUESSER.\n"
                f"The other player is creating an ASCII art drawing.\n"
                f"You will have {self.max_guesses_per_round} guesses.\n"
                f"Current shared score: {game_state['shared_score']}\n"
                "Wait for the artist to submit their drawing."
            )
        
        return prompt

    def _validate_drawing(self, drawing: str, allowed_symbols: Set[str]) -> bool:
        """
        Validate that the drawing uses only allowed symbols.

        Args:
            drawing (str): The ASCII art drawing to validate.
            allowed_symbols (Set[str]): Set of allowed symbols.

        Returns:
            bool: True if the drawing is valid, False otherwise.
        """
        # Allow whitespace in addition to the specified symbols
        allowed_symbols_with_whitespace = allowed_symbols.union({' ', '\n', '\t'})
        
        for char in drawing:
            if char not in allowed_symbols_with_whitespace:
                return False
                
        return True

    def _calculate_score(self, term: str, guesses_used: int, drawing: str) -> int:
        """
        Calculate the score for the current round.

        Args:
            term (str): The term that was drawn/guessed.
            guesses_used (int): Number of guesses used.
            drawing (str): The ASCII art drawing.

        Returns:
            int: Points earned for this round.
        """
        # Base points based on term difficulty
        difficulty = self.term_database.get(term, 3)  # Default to medium if not found
        base_points = difficulty * 10
        
        # Bonus for solving on first guess
        guess_multiplier = 1.0
        if guesses_used == 1:
            guess_multiplier = 1.5  # 50% bonus for first-try
        elif guesses_used == 2:
            guess_multiplier = 1.2  # 20% bonus for second-try
            
        # Drawing efficiency bonus (reward creative use of minimal symbols)
        unique_symbols_used = len(set(drawing) - {' ', '\n', '\t'})
        efficiency_bonus = 0
        if unique_symbols_used <= 5:
            efficiency_bonus = 5  # Bonus for using 5 or fewer unique symbols
            
        # Calculate final score
        final_score = int(base_points * guess_multiplier) + efficiency_bonus
        
        return final_score

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a player's action (either a drawing submission or a guess).

        Args:
            action (str): The player's action (drawing or guess).

        Returns:
            tuple: (observations, rewards, truncated, terminated, info)
        """
        current_player = self.state.current_player_id
        
        # Log the action
        self.state.add_observation(
            from_id=current_player,
            to_id=-1,  # Broadcast to all
            message=action,
            for_logging=True
        )
        
        # Process based on player role (artist or guesser)
        if current_player == self.state.game_state["current_artist"]:
            # Artist's turn - submitting a drawing
            if not self._validate_drawing(action, self.state.game_state["current_symbols"]):
                # Invalid drawing using disallowed symbols
                self.state.add_observation(
                    from_id=-1,  # System message
                    to_id=current_player,
                    message="Your drawing contains disallowed symbols! Please try again using only the allowed symbols."
                )
                # Don't advance turn, let them try again
                return self.state.step(advance_turn=False)
            
            # Valid drawing - store it and wait for guesser
            self.state.game_state["current_drawing"] = action
            
            # Notify the guesser that a drawing is ready
            self.state.add_observation(
                from_id=-1,  # System message
                to_id=self.state.game_state["current_guesser"],
                message=f"The artist has submitted their drawing:\n\n{action}\n\nMake your guess:"
            )
            
            return self.state.step()
            
        else:
            # Guesser's turn - making a guess
            current_round = self.state.game_state["current_round"]
            current_artist = self.state.game_state["current_artist"]
            target_term = self.state.game_state["terms"][current_round][current_artist]
            
            if action.lower() == target_term.lower():
                # Correct guess!
                guesses_used = self.max_guesses_per_round - self.state.game_state["guesses_remaining"] + 1
                drawing = self.state.game_state["current_drawing"]
                
                # Calculate score
                round_score = self._calculate_score(target_term, guesses_used, drawing)
                self.state.game_state["shared_score"] += round_score
                
                # Record round result
                self.state.game_state["round_results"].append({
                    "round": current_round,
                    "artist": current_artist,
                    "term": target_term,
                    "guesses_used": guesses_used,
                    "score": round_score
                })
                
                # Notify players of success
                self.state.add_observation(
                    from_id=-1,  # System message
                    to_id=-1,  # Broadcast to all
                    message=f"Correct! The term was '{target_term}'. You earned {round_score} points!\n"
                            f"Total shared score: {self.state.game_state['shared_score']}"
                )
                
                # Advance to next round or end game
                self._advance_round()
                
            else:
                # Incorrect guess
                self.state.game_state["guesses_remaining"] -= 1
                
                if self.state.game_state["guesses_remaining"] <= 0:
                    # Out of guesses for this round
                    self.state.add_observation(
                        from_id=-1,  # System message
                        to_id=-1,  # Broadcast to all
                        message=f"Out of guesses! The term was '{target_term}'.\n"
                                f"No points awarded for this round."
                    )
                    
                    # Record round result with 0 score
                    self.state.game_state["round_results"].append({
                        "round": current_round,
                        "artist": current_artist,
                        "term": target_term,
                        "guesses_used": self.max_guesses_per_round,
                        "score": 0
                    })
                    
                    # Advance to next round or end game
                    self._advance_round()
                    
                else:
                    # Still has guesses remaining
                    self.state.add_observation(
                        from_id=-1,  # System message
                        to_id=current_player,
                        message=f"Incorrect guess. You have {self.state.game_state['guesses_remaining']} "
                                f"guesses remaining. Try again:"
                    )
                    
                    # Don't advance turn, let them keep guessing
                    return self.state.step(advance_turn=False)
            
            return self.state.step()

    def _advance_round(self):
        """
        Advance to the next round or end the game if all rounds are complete.
        """
        current_round = self.state.game_state["current_round"]
        
        # Check if we've completed all rounds
        if current_round + 1 >= self.rounds_per_game:
            # Game complete
            self._end_game()
            return
            
        # Advance to next round
        self.state.game_state["current_round"] += 1
        
        # Swap roles (artist and guesser)
        self.state.game_state["current_artist"] = 1 - self.state.game_state["current_artist"]
        self.state.game_state["current_guesser"] = 1 - self.state.game_state["current_guesser"]
        
        # Reset guesses and select new symbol set
        self.state.game_state["guesses_remaining"] = self.max_guesses_per_round
        self.state.game_state["current_symbols"] = random.choice(self.symbol_sets)
        self.state.game_state["current_drawing"] = None
        
        # Provide new instructions to both players
        new_round = self.state.game_state["current_round"]
        artist_id = self.state.game_state["current_artist"]
        guesser_id = self.state.game_state["current_guesser"]
        
        # Artist's instructions
        term_to_draw = self.state.game_state["terms"][new_round][artist_id]
        allowed_symbols = ", ".join(sorted(self.state.game_state["current_symbols"]))
        
        self.state.add_observation(
            from_id=-1,  # System message
            to_id=artist_id,
            message=(
                f"ROUND {new_round + 1}/{self.rounds_per_game}: Now you are the ARTIST.\n"
                f"Your term to draw is: '{term_to_draw}'\n"
                f"Allowed symbols: {allowed_symbols}\n"
                f"Current shared score: {self.state.game_state['shared_score']}\n"
                "Create your ASCII art drawing now:"
            )
        )
        
        # Guesser's instructions
        self.state.add_observation(
            from_id=-1,  # System message
            to_id=guesser_id,
            message=(
                f"ROUND {new_round + 1}/{self.rounds_per_game}: Now you are the GUESSER.\n"
                f"The other player is creating an ASCII art drawing.\n"
                f"You will have {self.max_guesses_per_round} guesses.\n"
                f"Current shared score: {self.state.game_state['shared_score']}\n"
                "Wait for the artist to submit their drawing."
            )
        )
    
    def _end_game(self):
        """
        End the game and calculate final scores and achievements.
        """
        # Calculate total score and other metrics
        total_score = self.state.game_state["shared_score"]
        
        # Count successful rounds (where points were earned)
        successful_rounds = sum(1 for result in self.state.game_state["round_results"] if result["score"] > 0)
        
        # Calculate bonus for consistency if all rounds were successful
        consistency_bonus = 0
        if successful_rounds == self.rounds_per_game:
            consistency_bonus = 20
            total_score += consistency_bonus
        
        # Generate final summary
        summary = (
            f"Game complete! Final shared score: {total_score} points\n"
            f"Successfully completed {successful_rounds}/{self.rounds_per_game} rounds\n"
        )
        
        if consistency_bonus > 0:
            summary += f"Earned consistency bonus: +{consistency_bonus} points\n"
        
        # Add round-by-round breakdown
        summary += "\nRound-by-round summary:\n"
        for result in self.state.game_state["round_results"]:
            round_num = result["round"] + 1
            artist = f"Player {result['artist']}"
            term = result["term"]
            guesses = result["guesses_used"]
            score = result["score"]
            
            summary += f"Round {round_num}: {artist} drew '{term}' - "
            if score > 0:
                summary += f"Guessed in {guesses} attempts, scored {score} points\n"
            else:
                summary += "Not guessed correctly\n"
        
        # Send final summary to both players
        self.state.add_observation(
            from_id=-1,  # System message
            to_id=-1,  # Broadcast to all
            message=summary
        )
        
        # Set winners (both players since it's collaborative)
        self.state.set_winners(
            player_ids=[0, 1],  # Both players win or lose together
            reason=f"Game complete with shared score of {total_score} points"
        )
