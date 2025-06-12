import re, random
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.games.PigDice.renderer import create_board_str

class PigDiceEnv(ta.Env):
    def __init__(self, winning_score: int = 100, max_turns: int = 500):
        """
        Args:
            winning_score (int): The score needed to win.
            max_turns (int): Maximum number of turns before the game ends.
        """
        super().__init__()
        self.winning_score = winning_score
        self.max_turns = max_turns
        self.roll_value = None
        self.action_space = {i: re.compile(r"\[(roll|hold)\]", re.IGNORECASE) for i in range(2)}

    def get_board_str(self):
        return create_board_str(scores=self.state.game_state["scores"], turn_total=self.state.game_state["turn_total"], current_player=self.state.current_player_id, current_roll=self.roll_value)

    def reset(self, num_players: int, seed: Optional[int] = None) -> None:
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        self.state.reset(game_state={"scores": [0]*num_players, "turn_total": 0, "turn_rolls": []}, player_prompt_function=self._prompt)
        self._add_current_board_observation()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} playing a game of Pig Dice.\n"
            f"Rules:\n- On your turn, you can either '[roll]' or '[hold]'\n- Roll a 2-6: Add to your turn total\n"
            f"- Roll a 1: Lose turn total and end turn\n- Hold: Add turn total to your score and end turn\n"
            f"- First to {self.winning_score} points wins\n\nWhen it's your turn, you'll see the current scores and turn total.\n"
            f"Respond with '[roll]' to roll the die or '[hold]' to bank your points."
        )
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match = self.action_space[self.state.current_player_id].search(action.strip()) # Parse the action using regex
        if not match:
            self.state.set_invalid_move(reason=f"Invalid action format. Use '[roll]' or '[hold]'.")
            return self.state.step(rotate_player=False) 
        action = match.group(1).lower() # Extract the actual action
        # Execute the action
        if action == "roll": self._perform_roll(self.state.current_player_id)
        elif action == "hold": self._perform_hold(self.state.current_player_id)
        self._add_current_board_observation()
        return self.state.step(rotate_player=False)
    
    def _add_current_board_observation(self):
        observation = f"You current turn total is {self.state.game_state['turn_total']}. "
        observation += f"\nYour roll history for this turn: {', '.join(self.state.game_state['turn_rolls'])}" if self.state.game_state['turn_rolls'] else "\nThis is the first roll of your turn."
        observation += "\nAvailable actions: '[roll]' or '[hold]'"
        self.state.add_observation(message=observation, observation_type=ta.ObservationType.GAME_BOARD)
        
    def _determine_winner(self, scores):
        if scores[0] > scores[1]: return 0
        elif scores[0] < scores[1]: return 1
        return None

    def _rotate_to_next_player(self):
        scores = self.state.game_state['scores']
        # End game if the turn limit is reached
        if self.state.check_turn_limit():
            winner_id = self._determine_winner(scores)
            if winner_id is None: self.state.set_draw(reason=f"The turn limit has been reached and all players have the same score: {scores}")
            else: self.state.set_winner(player_id=winner_id, reason=f"Player {winner_id} won by having a higher score at the turn limit ({scores})")
            return
        # End game if the winning score is reached
        if any(score >= self.winning_score for score in scores):
            winner_id = 0 if scores[0] > scores[1] else 1
            self.state.set_winner(player_id=winner_id, reason=f"Player {winner_id} won by reaching the target score of {self.winning_score}!")
            return
        # Otherwise, continue the game
        self.state.game_state["turn_total"] = 0
        self.state.game_state["turn_rolls"] = []
        next_player_id = (self.state.current_player_id + 1) % self.state.num_players
        self.state.manually_set_current_player_id(new_player_id=next_player_id)

        # add current scores to observation
        self.state.add_observation(message="Current Scores: "+"; ".join(f"Player {i}: '{score}'" for i, score in enumerate(self.state.game_state['scores'])), observation_type=ta.ObservationType.GAME_MESSAGE)


    def _perform_roll(self, player_id: int) -> None:
        """ Perform the dice roll logic """
        roll_value = random.randint(1, 6)
        self.roll_value = roll_value
        if roll_value == 1: # Bust! Lose the turn total, end the turn
            self.state.add_observation(message=f"Player {player_id} busted, losing {self.state.game_state['turn_total']} points!", observation_type=ta.ObservationType.GAME_MESSAGE)
            self._rotate_to_next_player()
        else: # Accumulate turn total
            self.state.game_state["turn_total"] += roll_value
            self.state.game_state["turn_rolls"].append(f"'{roll_value}'")

    def _perform_hold(self, player_id: int) -> None:
        """ The player holds, adding turn_total to their overall score and ending the turn """
        # Add turn total to player's score
        self.state.game_state['scores'][player_id] += self.state.game_state['turn_total']
        self.state.add_observation(message=f"Player {player_id} holds and banks {self.state.game_state['turn_total']} points.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        self._rotate_to_next_player()

