import re, random
from typing import Optional, Tuple, Dict, Any

import textarena as ta
from textarena.envs.CharacterConclave.renderer import create_board_str

class CharacterConclaveEnv(ta.Env):
    """
    A multi-player text-based game where each player has a fixed character budget for discussion
    and then votes for another player. The environment is strictly turn-based.
    """

    def __init__(self, character_budget: int = 1_000):
        """
        Initialize the Character Conclave environment.

        Args:
            character_budget (int): Maximum number of characters each player can use during discussion.
        """
        self.character_budget = character_budget

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment to its initial state """
        # Create the underlying State object
        self.state = ta.State(num_players=num_players, min_players=3, max_players=15)

        # Initialize the shared game_state for all players
        game_state = {
            "phase": "discussion",
            "budget_remaining": {p: self.character_budget for p in range(self.state.num_players)},
            "votes": {},
        }
        # Reset the State object
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._player_prompt)

    def _player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Initial prompt each player sees when the game starts """
        prompt = (
            f"You are Player {player_id} in a {self.state.num_players} player game of Character Conclave.\n"
            f"Each of you has a limited character budget of {self.character_budget} characters.\n"
            f"Use them up across multiple turns by sending messages.\n\n"
            f"Once all players have used their budgets, each will vote exactly once "
            f"(in square brackets) for the player they found most impressive.\n"
            f"You cannot vote for yourself.\n"
            f"The player with the most votes wins.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ TODO """
        if self.state.game_state["phase"] == "discussion":
            # Check players budget 
            remaining_char_budget = self.state.game_state["budget_remaining"][self.state.current_player_id]
            if len(action) > remaining_char_budget:
                # truncate action
                action = action[:remaining_char_budget]

            # broadcast
            self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action)

            # update player budget
            self.state.game_state["budget_remaining"][self.state.current_player_id] -= len(action)

            # try rotating players
            self._attempt_player_rotation()
            return self.state.step(rotate_player=False)

        elif self.state.game_state["phase"] == "voting":
            # collect current votes until everybody has voted
            vote, reason = self._validate_player_vote(action=action)
            if vote is None:
                self.state.set_invalid_move(player_id=self.state.current_player_id, reason=reason)
                return self.state.step()
            
            else:
                self.state.game_state["votes"][self.state.current_player_id] = vote

                # confirm vote in private
                message = f"You have successfully voted for Player {vote}."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=self.state.current_player_id, message=message)

                # check if everybody has voted
                self._check_and_evaluate_outcome()
                return self.state.step()


    def _attempt_player_rotation(self):
        current_player_id = self.state.current_player_id
        # try rotating through all players and find the first one with left over budget
        next_player_id = (current_player_id + 1) % self.state.num_players
        while next_player_id != current_player_id:
            # check character budget
            if self.state.game_state["budget_remaining"][next_player_id] > 0:
                self.state.manually_update_current_player(new_player_id=next_player_id)
                break
            next_player_id = (next_player_id + 1) % self.state.num_players
        else:
            # no players remaining. Exit and change to voting phase
            self.state.game_state["phase"] = "voting"

            # add appropriate observation for everybody
            message = f"The discussion phase has concluded. Please now vote for a player. Votes have to be submitted as '[player x]' or '[x]'."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)


    def _validate_player_vote(self, action: str):
        # More permissive pattern that allows text before and after the vote
        pattern = r"\[\s*(?:player\s+)?(\d+)\s*\]"
        match = re.search(pattern, action.strip(), re.IGNORECASE)
        if not match:
            return None, "Invalid voting format. Please include your vote as '[x]' or '[player x]'."

        # Extract the first vote if multiple are present
        target_str = match.group(1)
        try:
            target_pid = int(target_str)
        except ValueError:
            return None, f"Could not parse the player ID from your brackets."

        # Validate the target is a real, other player
        if target_pid < 0 or target_pid >= self.state.num_players:
            return None, f"Invalid vote. Player {target_pid} does not exist."

        if target_pid == self.state.current_player_id:
            return None, "You cannot vote for yourself!"

        # Check if there are multiple votes in the text
        all_votes = re.findall(pattern, action.strip(), re.IGNORECASE)
        if len(all_votes) > 1:
            return None, "Please submit only one vote."

        # vote is valid, return accordingly
        return target_pid, None


    def _check_and_evaluate_outcome(self):
        if len(self.state.game_state["votes"]) == self.state.num_players:
            # conclude game by counting votes
            received_votes = {}
            for voting_pid, target_pid in self.state.game_state["votes"].items():
                received_votes[target_pid] = received_votes.get(target_pid, 0) + 1

            # check for most votes
            max_votes = max(received_votes.values())

            winner_ids = [k for k,v in received_votes.items() if v==max_votes]
            print(winner_ids, max_votes)
            reason=f"Player(s) {','.join(map(str, winner_ids))} received the most votes ({max_votes})."
            self.state.set_winners(player_ids=winner_ids, reason=reason)

