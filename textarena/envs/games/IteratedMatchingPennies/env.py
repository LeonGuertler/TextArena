import re
from typing import Optional, Dict, Any, Tuple

import textarena as ta


class IteratedMatchingPenniesEnv(ta.Env):
    def __init__(self, num_rounds: int = 5):
        """
        Iterated Matching Pennies.

        Players: 2
        Rounds: num_rounds

        Each round both players simultaneously pick Heads or Tails.
        - If their choices match, Player 0 (“Matcher”) wins the round.
        - If they differ, Player 1 (“Mismatcher”) wins the round.
        After num_rounds, whoever has more round-wins is the overall winner.
        """
        self.num_rounds = num_rounds
        self.action_space = {i: re.compile(r"\[\s*(heads|h|tails|t)\s*\]", re.IGNORECASE) for i in range(2)} # regex to parse [heads], [tails], or shorthand [h], [t]

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {"round": 1, "points": {0:0, 1:0}, "moves": {}, "history": []}
        self.state.reset(game_state=game_state, player_prompt_function=self._make_prompt)

    def _make_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        role = "Matcher" if player_id == 0 else "Mismatcher"
        return (
            f"You are Player {player_id} ({role}) in a {self.num_rounds}-round Matching Pennies game.\n- Each round, submit [heads] or [tails] (or [h], [t]).\n"
            "- If your choice matches your opponent’s, Player 0 wins the round; otherwise Player 1 wins.\nReply with your choice in the format '[heads]' or '[tails]'.\n"
        )

    def get_board_str(self) -> str:
        gs = self.state.game_state
        s = f"Round {gs['round']}/{self.num_rounds}\n"
        if gs["history"]:
            s += "History:\n"
            for i, past in enumerate(gs["history"], start=1):
                s += (f"  Round {i}: " + ", ".join(f"P{pid}→{choice}" for pid, choice in past.items()) + "\n")
        return s

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        pid = self.state.current_player_id

        # parse choice
        m = self.action_space[pid].search(action)
        if not m:
            self.state.set_invalid_move(reason="Invalid format; please submit '[heads]' or '[tails]'.")
        else:
            token = m.group(1).lower()
            choice = "heads" if token in ("heads", "h") else "tails"
            self.state.game_state["moves"][pid] = choice

            # if both have moved, resolve the round
            if len(self.state.game_state["moves"]) == 2:
                moves = self.state.game_state["moves"]
                same = (moves[0] == moves[1])
                winner = 0 if same else 1

                # log result
                self.state.game_state["history"].append(moves.copy())
                self.state.add_observation(message=f"Player 0 picked {self.state.game_state['moves'][0]}; Player 1 picked {self.state.game_state['moves'][1]}. {'Match -> Player 0 wins' if same else 'Mismatch -> Player 1 wins.'}", observation_type=ta.ObservationType.GAME_MESSAGE)
                self.state.game_state["points"][winner] += 1

                # advance
                self.state.game_state["round"] += 1
                self.state.game_state["moves"].clear()

                # end-of-game check
                if self.state.game_state["round"] > self.num_rounds:
                    p0, p1 = self.state.game_state["points"][0], self.state.game_state["points"][1]
                    if p0 > p1:     self.state.set_winner(0, reason="Player 0 won more rounds.")
                    elif p1 > p0:   self.state.set_winner(1, reason="Player 1 won more rounds.")
                    else:           self.state.set_draw(reason="Overall game is a draw.")

        return self.state.step()
