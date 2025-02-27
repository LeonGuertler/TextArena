import re, random, copy
from typing import Any, Dict, Optional, Tuple, Union

import textarena as ta

class TowerOfHanoiEnv(ta.Env):
    """ Tower of Hanoi game environment """

    def __init__(self, num_disks: int=3, max_turns: int=100):
        """ Initialize the envrionment.
        
        Args:
            num_disks (int): The number of disks
            max_turns (int): The max number of turns
        """
        super().__init__()
        self.num_disks = num_disks
        self.max_turns = max_turns

    @property
    def terminal_render_keys(self):
        return ["rendered_board"]

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """
        ## intitialise the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## load the game state
        self.towers = self._generate_board()

        ## reset the game state
        game_state={
            "towers": copy.deepcopy(self.towers),
            "rendered_board": self._render_board()
        }
        return self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate a player prompt """
        prompt = (
            f"You are Player {player_id}. You are playing Tower of Hanoi with {self.num_disks} disks.\n"
            f"You have to move the disks from tower A to tower C.\n"
            "To move a disk, type the source tower and the target tower (e.g., '[A C]').\n"
            "Note that you can only move the top disk of a tower, and that a bigger disk cannot be placed on a smaller disk.\n"
            "As you play, the history of your moves will be displayed.\n"
            "Here is the current state of the towers:\n"
        )
        prompt += self._render_board()
        return prompt

    def _generate_board(self):
        """ Generate the board """
        towers = {"A": list(range(self.num_disks, 0, -1)), "B": [], "C": []}
        return towers
    
    def _render_board(self):
        """ Render the board """
        rendered_board = ""
        for tower, disks in self.towers.items():
            rendered_board += f"{tower}: {disks}\n"
        return rendered_board
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Take a step in the environment """
        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## validate the action
        action_search_pattern = re.compile(r"\[([ABCabc])\s*,?\s*([ABCabc])\]") # e.g. [A, C], [A C], [a c], [a, c]
        matches = action_search_pattern.findall(action)

        if not matches:
            reason=f"Invalid move format. Player {player_id} did not respond with valid '[source] [target]'."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            for match in matches:
                source, target = match
                source = source.upper()
                target = target.upper()
                if source not in self.towers or target not in self.towers:
                    reason=f"Invalid move. Player {player_id} specified an invalid source or target tower."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                    break
                elif not self.towers[source]:
                    reason=f"Invalid move. Player {player_id} tried to move a disk from an empty tower."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                elif self.towers[target] and self.towers[target][-1] < self.towers[source][-1]:
                    reason=f"Invalid move. Player {player_id} tried to place a larger disk on a smaller disk."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                else:
                    disk = self.towers[source].pop()
                    self.towers[target].append(disk)
                    message=(f"Player {player_id} moved disk from {source} to {target}. Here is the current state of the towers:\n{self._render_board()}")
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id,message=message, for_logging=False)

            ## check if the game is over
            if self.towers["C"] == list(range(self.num_disks, 0, -1)):
                reason=f"Congratulations! Player {player_id} solved the Tower of Hanoi puzzle."
                self.state.set_winners(player_ids=[player_id], reason=reason)
            
            self.state.game_state["rendered_board"] = self._render_board()
        return self.state.step()
    