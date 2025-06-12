import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.games.IteratedRockPaperScissors.renderer import create_board_str


class IteratedRockPaperScissorsEnv(ta.Env):
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds
        self.action_space = {i: re.compile(r"\[(rock|r|paper|p|scissors|s)\]") for i in range(2)}

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {"round": 1, "points": {0:0,1:0}, "moves": {0:None,1:None}, "history": []}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.num_rounds}-round Rock-Paper-Scissors game.\nYour goal is to win as many rounds as possible.\n"
            "In each round, respond with one of: '[rock]', '[paper]', or '[scissors]'.\nYou may also use '[r]', '[p]', or '[s]' as shorthand.\n"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        move = self._parse_action(action)
        if move not in {"rock", "paper", "scissors"}:
            self.state.set_invalid_move(reason=f"Move not recognized. Use [rock], [paper], or [scissors].")
        else:
            self.state.game_state["moves"][player_id] = move
            self.state.add_observation(from_id=player_id, to_id=player_id, message=f"Player {player_id} selects move {move}.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
            
            if self.state.game_state["moves"][1-player_id] != None: # Resolve the round
                p0_move = self.state.game_state["moves"][0]
                p1_move = self.state.game_state["moves"][1]
                result = self._resolve_round(p0_move, p1_move)
                self.state.game_state["history"].append({0:p0_move,1:p1_move})
                self.state.game_state["round"] += 1
                self.state.game_state["moves"] = {0:None, 1:None}

                if result == 0:
                    self.state.add_observation(message="Round result: Draw", observation_type=ta.ObservationType.GAME_MESSAGE)
                else:
                    self.state.add_observation(message=f"Round result: Player {result-1} wins!", observation_type=ta.ObservationType.GAME_MESSAGE)
                    self.state.game_state["points"][result-1] += 1

                if self.state.game_state["round"] > self.num_rounds: # Check end condition
                    wins = self.state.game_state.get("points", {0: 0, 1: 0})
                    if wins[0] > wins[1]:   self.state.set_winner(player_id=0, reason="Player 0 won the most rounds!")
                    elif wins[1] > wins[0]: self.state.set_winner(player_id=1, reason="Player 1 won the most rounds!")
                    else:                   self.state.set_draw("The match is a draw!")
        
        return self.state.step()

    def _parse_action(self, action: str) -> str:
        match = self.action_space[self.state.current_player_id].search(action.strip().lower())
        if not match: return ""
        return {"r": "rock", "p": "paper", "s": "scissors"}.get(match.group(1), match.group(1))

    def _resolve_round(self, p0: str, p1: str) -> int:
        if p0 == p1: return 0
        return 1 if {"rock": "scissors", "paper": "rock", "scissors": "paper",}[p0] == p1 else 2
