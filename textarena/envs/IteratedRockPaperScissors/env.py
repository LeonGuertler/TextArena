import re, random
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.IteratedRockPaperScissors.renderer import create_board_str

class IteratedRockPaperScissorsEnv(ta.Env):
    """ Environment for a two-player iterated Rock-Paper-Scissors game """
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment to the initial state """
        random.seed(seed)
        self.state = ta.State(num_players=2, min_players=2, max_players=2)
        game_state = {"round": 1, "points": {0:0, 1:0}, "moves": {0:None, 1:None}, "history": []}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        prompt = (
            f"You are Player {player_id} in a {self.num_rounds}-round Rock-Paper-Scissors game.\n"
            "Your goal is to win as many rounds as possible.\n"
            "In each round, respond with one of: [rock], [paper], or [scissors].\n"
            "You may also use [r], [p], or [s] as shorthand.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=player_id, message=action)

        move = self._parse_action(action)
        if move not in {"rock", "paper", "scissors"}:
            self.state.set_invalid_move(player_id, reason=f"Invalid move '{action}'. Use [rock], [paper], or [scissors].")
        else:
            # add to game moves
            self.state.game_state["moves"][player_id] = move

            if self.state.game_state["moves"][1-player_id] != None:
                # Resolve the round
                p0_move = self.state.game_state["moves"][0]
                p1_move = self.state.game_state["moves"][1]
                result = self._resolve_round(p0_move, p1_move)
                self.state.game_state["history"].append({0:p0_move,1:p1_move})
                self.state.game_state["round"] += 1
                self.state.game_state["moves"] = {0:None, 1:None}

                if result == 0:
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message="Round result: Draw")
                else:
                    winner = result - 1  # Convert to player_id: 0 or 1
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=f"Round result: Player {winner} wins!")
                    self.state.game_state["points"][winner] += 1

                # Check end condition
                if self.state.game_state["round"] > self.num_rounds:
                    wins = self.state.game_state.get("points", {0: 0, 1: 0})
                    if wins[0] > wins[1]:
                        self.state.set_winners([0], reason="Player 0 won the most rounds!")
                    elif wins[1] > wins[0]:
                        self.state.set_winners([1], reason="Player 1 won the most rounds!")
                    else:
                        self.state.set_draw("The match is a draw!")
        return self.state.step()

    def _parse_action(self, action: str) -> str:
        match = re.search(r"\[(rock|r|paper|p|scissors|s)\]", action.strip().lower())
        if not match:
            return ""
        token = match.group(1)
        return {"r": "rock", "p": "paper", "s": "scissors"}.get(token, token)

    def _resolve_round(self, p0: str, p1: str) -> int:
        if p0 == p1:
            return 0
        wins = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper",
        }
        return 1 if wins[p0] == p1 else 2
