import re, random
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.envs.games.MemoryGame.renderer import create_board_str

# TODO introduce turn limit
# TODO check. On match I think we shouldn't rotate players

class MemoryGameEnv(ta.Env):
    """ Environment for Memory Game """
    def __init__(self, grid_size: Optional[int] = 4):
        """
        Args:
            grid_size (int): The grid size used
        """
        self.grid_size = grid_size
        self.action_space = lambda player_id: re.compile(r"\[([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)\]")

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {"board": self._generate_board(), "matched_positions": set(), "score": {0: 0, 1: 0}, "scores": {0: {"Score": 0}, 1: {"Score": 0}}}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self.state.add_observation(message=f"Current board:\n{self._render_board()}", observation_type=ta.ObservationType.GAME_BOARD)
    
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id}. You are playing the Memory Game.\n"
            "Your goal is to match more pairs of cards on the board, than your opponent.\n"
            "On your turn, select two cards to flip by entering the row and column numbers of the first and second card respectively like '[0 1 1 0]', where the first card is in row 0 and column 1, and the second card is in row 1 and column 0.\n"
            "If the two cards match, you get a point and the cards remain face up. If they do not match, the cards are flipped back face down, e.g. '.'.\n"
            "The game ends when all pairs have been matched."
        )
    
    def _generate_board(self) -> List[List[str]]:
        symbols = [chr(65 + i) for i in range((self.grid_size ** 2) // 2)] * 2
        random.shuffle(symbols)
        return [symbols[i * self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]
    
    def _render_board(self) -> str:
        rendered_board = "  " + " ".join(str(c) for c in range(self.grid_size)) + "\n"
        for r in range(self.grid_size):
            row = f"{r} "
            for c in range(self.grid_size):
                if (r, c) in self.state.game_state["matched_positions"]: row += f"{self.state.game_state['board'][r][c]} "
                else: row += ". "
            rendered_board += row.strip() + "\n"
        return rendered_board
    
    def step(self, action: List[int]) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match = self.action_space(player_id).search(action) # e.g. [0 1 1 0]

        if match is None:
            self.state.set_invalid_move(reason=f"Invalid move format. Player {player_id} did not respond with a valid direction in square brackets.")
        else:
            r1, c1, r2, c2 = map(int, match.groups())
            if r1 < 0 or r1 >= self.grid_size or c1 < 0 or c1 >= self.grid_size or r2 < 0 or r2 >= self.grid_size or c2 < 0 or c2 >= self.grid_size: self.state.set_invalid_move(reason=f"Invalid move. Player {player_id} selected an out-of-bounds position.")
            elif (r1, c1) == (r2, c2): self.state.set_invalid_move(reason=f"Invalid move. Player {player_id} selected the same card twice.")
            elif (r1, c1) in self.state.game_state["matched_positions"] or (r2, c2) in self.state.game_state["matched_positions"]: self.state.set_invalid_move(reason=f"Invalid move. Player {player_id} selected one or both cards that have already been matched.")
            else:
                if self.state.game_state['board'][r1][c1] == self.state.game_state['board'][r2][c2]:
                    self.state.game_state["score"][player_id] += 1 # update the score
                    self.state.game_state["matched_positions"].update([(r1, c1), (r2, c2)]) # update the matched positions
                    if len(self.state.game_state["matched_positions"]) == self.grid_size ** 2: # check if the game is over
                        if self.state.game_state["score"][0] == self.state.game_state["score"][1]: # check if there is a tie
                            self.state.set_draw(reason="Both players matched the same number of pairs of cards.")
                        else: # set the winner
                            winner_id = max(self.state.game_state["score"], key=self.state.game_state["score"].get)
                            self.state.set_winner(player_id=winner_id, reason=f"Player {winner_id} has won!")

                    ## log the action
                    self.state.add_observation(message=f"The cards selected by Player {player_id} at positions [{r1} {c1}] and [{r2} {c2}] match!", observation_type=ta.ObservationType.GAME_MESSAGE)
                    self.state.add_observation(message=f"Current Game Board:\n{self._render_board()}", observation_type=ta.ObservationType.GAME_BOARD)
                else:
                    pos1 = self.state.game_state['board'][r1][c1]; pos2 = self.state.game_state['board'][r2][c2]
                    self.state.add_observation(message=f"The cards selected by Player {player_id} do not match. Cards at positions [{r1} {c1}] and [{r2} {c2}] are {pos1} and {pos2} respectively.", observation_type=ta.ObservationType.GAME_MESSAGE)
        self.state.game_state["scores"] = {0: {"Score": self.state.game_state["score"][0]}, 1: {"Score": self.state.game_state["score"][1]}}
        return self.state.step()


    