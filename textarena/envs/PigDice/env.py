import re, random
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.PigDice.renderer import create_board_str

class PigDiceEnv(ta.Env):
    """
    A simple two-player Pig dice game environment.

    Rules:
      - Two players alternate turns.
      - On each turn, the player can either 'roll' or 'hold' ('r' and 'h' are also acceptable moves).
      - If they 'roll' and get a 2..6, that amount is added to the turn total.
      - If they 'roll' and get a 1, the player loses the turn total, and the turn ends.
      - If they 'hold', the turn total is added to their overall score, and the turn ends.
      - The first player to reach 100 or more points wins.
    """

    def __init__(self, winning_score: int = 100, max_turns: int = 500):
        """
        Initialize the Pig environment.

        Args:
            winning_score (int): The score needed to win.
            max_turns (int): Maximum number of turns before the game ends.
        """
        super().__init__()
        self.winning_score = winning_score
        self.max_turns = max_turns

        # Action pattern for parsing player input
        self.action_pattern = re.compile(r"\[(roll|hold|r|h)\]", re.IGNORECASE)

        self.roll_value = None

    def get_board_str(self):
        return create_board_str(
            scores=self.state.game_state["scores"],
            turn_total=self.state.game_state["turn_total"],
            current_player=self.state.current_player_id,
            current_roll=self.roll_value  # This is only available inside `_perform_roll`
        )

    def reset(self, num_players: int, seed: Optional[int] = None) -> None:
        # Create a new State
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=2,
            max_turns=self.max_turns,
            check_truncated=False
        )

        # Initialize game_state
        game_state = {"scores": [0] * num_players, "turn_total": 0, "turn_count": 0}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._player_prompt)
        

    def _player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        prompt = (
            f"[GAME] Welcome to Pig Dice Game!\n\n"
            f"You are Player {player_id}.\n\n"
            f"Rules:\n"
            f"- On your turn, you can either '[roll]' or '[hold]'\n"
            f"- Roll a 2-6: Add to your turn total\n"
            f"- Roll a 1: Lose turn total and end turn\n"
            f"- Hold: Add turn total to your score and end turn\n"
            f"- First to {self.winning_score} points wins\n\n"
            f"When it's your turn, you'll see the current scores and turn total.\n"
            f"Respond with '[roll]' to roll the die or '[hold]' to bank your points.\n"
        )
        if player_id == 0:
            prompt+=f"\n\nCurrent turn total: {self.state.game_state['turn_total']}.\nAvailable actions: '[roll]', '[hold]'."

        return prompt


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process a single move in the game """
        player_id = self.state.current_player_id
        
        # Log the player's raw action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)
        
        # Parse the action using regex
        match = self.action_pattern.search(action.strip())
        if not match:
            self.state.set_invalid_move(player_id, f"Invalid action format. Use '[roll]' or '[hold]' or one letter shorthands.")
            return self.state.step(rotate_player=False) 
            
        # Extract the actual action
        action = match.group(1).lower()
        
        # Execute the action
        if action in ["roll", "r"]:
            self._perform_roll(player_id)
        elif action in ["hold", "h"]:
            self._perform_hold(player_id)

        # add available actions
        message="Available actions: '[roll]' or '[hold]'"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=self.state.current_player_id, message=message)
        return self.state.step(rotate_player=False)
        
        
    def _determine_winner(self, scores):
        if scores[0] > scores[1]:
            return 0
        elif scores[0] < scores[1]:
            return 1
        return None

    def _rotate_to_next_player(self):
        game_state = self.state.game_state
        scores = game_state['scores']
        turn_count = game_state["turn_count"]
        max_turns = self.state.max_turns

        # End game if the turn limit is reached
        if turn_count + 1 >= max_turns:
            winner_id = self._determine_winner(scores)
            if winner_id is None:
                reason = f"The turn limit has been reached and all players have the same score: {scores}"
                self.state.set_draw(reason=reason)
            else:
                reason = f"Player {winner_id} won by having a higher score at the turn limit ({scores})"
                self.state.set_winners(player_ids=[winner_id], reason=reason)
            return

        # End game if the winning score is reached
        if any(score >= self.winning_score for score in scores):
            winner_id = 0 if scores[0] > scores[1] else 1
            reason = f"Player {winner_id} won by reaching the target score of {self.winning_score}!"
            self.state.set_winners(player_ids=[winner_id], reason=reason)
            return

        # Otherwise, continue the game
        game_state["turn_count"] += 1
        game_state["turn_total"] = 0
        next_player_id = (self.state.current_player_id + 1) % self.state.num_players

        self.state.manually_update_current_player(new_player_id=next_player_id)

        scores_str = "\n".join(f"Player {i} score: {score}" for i, score in enumerate(scores))
        message = f"Player {next_player_id}'s turn\n\n{scores_str}\n\nCurrent turn total: {game_state['turn_total']}\n"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=next_player_id, message=message)


    def _perform_roll(self, player_id: int) -> None:
        """ Perform the dice roll logic """
        roll_value = random.randint(1, 6)
        self.roll_value = roll_value
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Player {player_id} rolls a {roll_value}.")

        if roll_value == 1:
            # Bust! Lose the turn total, end the turn
            message = f"Player {player_id} busted! Lost all points for this turn.\n\n"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
            self._rotate_to_next_player()

        else:
            # Accumulate turn total
            self.state.game_state["turn_total"] += roll_value
            message = f"Turn total is now {self.state.game_state['turn_total']}."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)


    def _perform_hold(self, player_id: int) -> None:
        """ The player holds, adding turn_total to their overall score and ending the turn """
        # Add turn total to player's score
        self.state.game_state['scores'][player_id] += self.state.game_state['turn_total']
        message = (
            f"Player {player_id} holds and banks {self.state.game_state['turn_total']} points.\n\n"
            f"Total score: {self.state.game_state['scores'][player_id]}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
        self._rotate_to_next_player()

