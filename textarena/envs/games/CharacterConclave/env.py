import re
from typing import Optional, Tuple, Dict, Any, Callable

import textarena as ta
from textarena.envs.games.CharacterConclave.renderer import create_board_str

class CharacterConclaveEnv(ta.Env):
    def __init__(self, character_budget: int = 1_000):
        """
        Args:
            character_budget (int): Maximum number of characters each player can use during discussion.
        """
        self.character_budget = character_budget
        self.action_space = {i: re.compile(r"\[\s*(?:player\s+)?(\d+)\s*\]") for i in range(15)}

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert 3<=num_players<=15, f"The number of players has to be 3<=x<=15, received {num_players}"
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        game_state = {"phase": "discussion", "budget_remaining": {p: self.character_budget for p in range(self.state.num_players)}, "votes": {}}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.state.num_players} player game of Character Conclave.\nEach of you has a limited character budget of {self.character_budget} characters.\n"
            f"Use them up across multiple turns by sending messages.\n\nOnce all players have used their budgets, each will vote exactly once "
            f"(in square brackets) for the player they found most impressive.\nYou cannot vote for yourself.\nThe player with the most votes wins.\n"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.state.game_state["phase"] == "discussion": # discussion phase
            if len(action) > self.state.game_state["budget_remaining"][self.state.current_player_id]: # Check players budget 
                action = action[:self.state.game_state["budget_remaining"][self.state.current_player_id]] # truncate action
            self.state.add_observation(from_id=self.state.current_player_id, message=action) # broadcast
            self.state.game_state["budget_remaining"][self.state.current_player_id] -= len(action) # update player budget
            self._attempt_player_rotation_discussion() # try rotating players
            return self.state.step(rotate_player=False)

        else: # voting phase
            vote, reason = self._validate_player_vote(action=action) # collect current votes until everybody has voted
            if vote is None:
                was_eliminated = self.state.set_invalid_move(reason=reason) # skip this vote & rotate players
                if was_eliminated:
                    self.state.game_state["votes"][self.state.current_player_id] = -1
                    self.state.add_observation(to_id=self.state.current_player_id, message=f"You have submitted and invalid vote: {vote}. It won't be counted.")
                else:
                    return self.state.step(rotate_player=False)
            else:
                self.state.game_state["votes"][self.state.current_player_id] = vote
                self.state.add_observation(to_id=self.state.current_player_id, message=f"You have successfully voted for Player {vote}.") # confirm vote in private

            self._attempt_player_rotation_voting()
            return self.state.step(rotate_player=False)            

    # def _attempt_player_rotation_discussion(self):
    #     # try rotating through all players and find the first one with left over budget
    #     next_player_id = (self.state.current_player_id + 1) % self.state.num_players
    #     while next_player_id != self.state.current_player_id:
    #         # check character budget
    #         if self.state.game_state["budget_remaining"][next_player_id] > 0:
    #             self.state.manually_set_current_player_id(new_player_id=next_player_id); break
    #         next_player_id = (next_player_id + 1) % self.state.num_players
    #     else:
    #         if self.state.game_state["budget_remaining"][next_player_id] > 0:
    #             self.state.manually_set_current_player_id(new_player_id=next_player_id)
    #         else:
    #             self.state.game_state["phase"] = "voting" # no players remaining. Exit and change to voting phase
    #             self.state.add_observation(message=f"The discussion phase has concluded. Please now vote for a player. Votes have to be submitted as '[player x]' or '[x]'.") # add appropriate observation for everybody

    # def _attempt_player_rotation_voting(self):
    #     # try rotating through all players and find the first one with left over budget
    #     next_player_id = (self.state.current_player_id + 1) % self.state.num_players
    #     while next_player_id != self.state.current_player_id:
    #         if next_player_id not in self.state.game_state["votes"]: # check player has voted
    #             self.state.manually_set_current_player_id(new_player_id=next_player_id); break
    #         next_player_id = (next_player_id + 1) % self.state.num_players
    #     else:
    #         self._check_and_evaluate_outcome() # everybody has voted, evaluate
    def _rotate_player(self, can_take_turn: Callable[[int], bool], on_exhausted: Optional[Callable[[], None]] = None) -> None:
        """Move `current_player_id` to the next player who satisfies `can_take_turn`. If nobody does, call `on_exhausted`."""
        next_pid = (self.state.current_player_id + 1) % self.state.num_players
        while next_pid != self.state.current_player_id:
            if can_take_turn(next_pid): self.state.manually_set_current_player_id(new_player_id=next_pid); return
            next_pid = (next_pid + 1) % self.state.num_players
        # We’ve looped back to the starting player → check them once more
        if can_take_turn(next_pid):
            self.state.manually_set_current_player_id(new_player_id=next_pid)
        elif on_exhausted is not None:
            on_exhausted()
    def _attempt_player_rotation_discussion(self) -> None:
        self._rotate_player(can_take_turn=lambda pid: self.state.game_state["budget_remaining"][pid] > 0, on_exhausted=self._end_discussion_phase)

    def _attempt_player_rotation_voting(self) -> None:
        self._rotate_player(can_take_turn=lambda pid: pid not in self.state.game_state["votes"], on_exhausted=self._check_and_evaluate_outcome)

    def _end_discussion_phase(self) -> None:
        self.state.game_state["phase"] = "voting"
        self.state.add_observation(message="The discussion phase has concluded. Please now vote for a player. Votes must be submitted as '[player x]' or '[x]'.")


    def _validate_player_vote(self, action: str):
        match = self.action_space[self.state.current_player_id].search(action.strip()) # More permissive pattern that allows text before and after the vote
        if not match: return None, "Invalid voting format. Please include your vote as '[x]' or '[player x]'."
        # Extract the first vote if multiple are present
        try: target_pid = int(match.group(1))
        except ValueError: return None, f"Could not parse the player ID from your brackets."
        if target_pid < 0 or target_pid >= self.state.num_players: return None, f"Invalid vote. Player {target_pid} does not exist." # Validate the target is a real, other player
        if target_pid == self.state.current_player_id: return None, "You cannot vote for yourself!"
        if len(re.findall(r"\[\s*(?:player\s+)?(\d+)\s*\]", action.strip(), re.IGNORECASE)) > 1: return None, "Please submit only one vote." # Check if there are multiple votes in the text
        return target_pid, None # vote is valid, return accordingly

    def _check_and_evaluate_outcome(self):
        if len(self.state.game_state["votes"]) == self.state.num_players:
            vote_counts = {} # conclude game by counting votes
            for voting_pid, target_pid in self.state.game_state["votes"].items():
                if target_pid not in self.state.elimination_order and target_pid !=- 1: # check if player made invalid move (can't be voted for in that case)
                    vote_counts[target_pid] = vote_counts.get(target_pid, 0) + 1
            valid_players = [pid for pid in range(self.state.num_players) if pid not in self.state.elimination_order] # Build a list of players not eliminated and not voted invalidly
            ranked_players = sorted(valid_players, key=lambda pid: vote_counts.get(pid, 0)) # Sort players by vote count (ascending)
            n = len(ranked_players) # Get number of distinct players to rank
            reward_dict = {}
            for i, pid in enumerate(ranked_players):
                if vote_counts.get(pid, 0) == vote_counts.get(ranked_players[0], 0): reward = -1.0  # Lowest score (tie allowed)
                elif vote_counts.get(pid, 0) == vote_counts.get(ranked_players[-1], 0): reward = 1.0  # Highest score (tie allowed)
                else: reward = -1.0 + 2.0 * (i / (n - 1))
                reward_dict[pid] = reward
            for pid in self.state.elimination_order: reward_dict[pid] = -1.0 # Set eliminated players' rewards to -1
            self.state.set_game_outcome(reward_dict=reward_dict, reason=f"Player(s) {[pid for pid, r in reward_dict.items() if r == 1.0]} win(s) with the most votes.")

