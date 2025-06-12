import re
from typing import Optional, Dict, Any, Tuple

import textarena as ta


class ChopsticksEnv(ta.Env):
    def __init__(self, max_turns: int = 40):
        """
        Chopsticks (Finger Game).

        Players: 2
        Each player has two hands, each with 0–4 fingers.
        - On your turn you may either:
            • Attack: “[attack M O]” where M is your hand index (0 or 1) and O is opponent’s hand (0 or 1).
              Opponent’s hand count ← opponent’s hand + your hand; if ≥5 it becomes 0 (“dead”).
            • Split: “[split L R]” to redistribute your total fingers into your two hands L and R,
              where L+R equals your current total.
        First to kill both opponent hands (both zero) wins. If both die on same move, attacker wins.
        """
        self.max_turns = max_turns
        self.action_space = {i: re.compile(r"\[\s*attack\s+([01])\s+([01])\s*\]|\[\s*split\s+(\d+)\s+(\d+)\s*\]", re.IGNORECASE) for i in range(2)}

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        self.state.reset(game_state={"hands": {0: [1, 1], 1: [1, 1]}, "history": []}, player_prompt_function=self._prompt)
        self.state.add_observation(message=f"Current Board:\nPlayer 0: {self.state.game_state['hands'][0]}\nPlayer 1: {self.state.game_state['hands'][1]}", observation_type=ta.ObservationType.GAME_BOARD)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Chopsticks. On your turn, choose one of:\n"
            "  + Attack:  “[attack M O]”  where M=your hand (0 or 1), O=opponent hand (0 or 1).\n"
            "    - Opponent’s hand count increases by your hand; if >=5, it becomes 0.\n"
            "  + Split:   “[split L R]”  to redistribute your total fingers into L and R (L+R = your total).\n"
            "Reply with exactly one of those commands."
        )
    def get_board_str(self) -> str:
        gs = self.state.game_state
        h0, h1 = gs["hands"][0], gs["hands"][1]
        s = f"Hands:\n  Player 0: [{h0[0]}, {h0[1]}]\n  Player 1: [{h1[0]}, {h1[1]}]\n"
        if gs["history"]: s += "History:\n" + "\n".join(f"  {entry}" for entry in gs["history"]) + "\n"
        return s

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        pid = self.state.current_player_id
        gs = self.state.game_state
        self.state.add_observation(from_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        # try attack
        m_atk = re.compile(r"\[\s*attack\s+([01])\s+([01])\s*\]", re.IGNORECASE).search(action)
        if m_atk:
            my_idx, opp_idx = map(int, m_atk.groups())
            my_val = gs["hands"][pid][my_idx]
            opp_val = gs["hands"][1 - pid][opp_idx]

            # validation
            if my_val == 0:     self.state.set_invalid_move(reason=f"Your hand {my_idx} is dead.")
            elif opp_val == 0:  self.state.set_invalid_move(reason=f"Opponent hand {opp_idx} is already dead.")
            else:
                new_val = my_val + opp_val
                gs["hands"][1-pid][opp_idx] = 0 if new_val >= 5 else new_val
                desc = f"P{pid} attacks P{1-pid}’s hand {opp_idx}: it goes from {opp_val} to {gs['hands'][1 - pid][opp_idx]}."
                self.state.add_observation(message=desc, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
                gs["history"].append(f"P{pid} {desc}")
                # check for win
                if gs["hands"][1 - pid] == [0, 0]: self.state.set_winner(player_id=pid, reason="Both opponent hands dead.")
        else:
            # try split
            m_sp = re.compile(r"\[\s*split\s+(\d+)\s+(\d+)\s*\]", re.IGNORECASE).search(action)
            if m_sp:
                L, R = map(int, m_sp.groups())
                cur_L, cur_R = gs["hands"][pid]
                total = cur_L + cur_R
                if L + R != total:              self.state.set_invalid_move(reason=f"Split must sum to {total}.")
                elif (L, R) == (cur_L, cur_R):  self.state.set_invalid_move(reason="Split must change your hand distribution.")
                else:
                    gs["hands"][pid] = [L, R]
                    desc = f"P{pid} splits into [{L}, {R}]."
                    self.state.add_observation(message=desc, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
                    gs["history"].append(f"P{pid} {desc}")
            else:
                self.state.set_invalid_move(reason="Invalid move. Use “[attack M O]” or “[split L R]”.") # invalid command

        # add current board
        self.state.add_observation(message=f"Current Board:\nPlayer 0: {self.state.game_state['hands'][0]}\nPlayer 1: {self.state.game_state['hands'][1]}", observation_type=ta.ObservationType.GAME_BOARD)
        # check turn limit
        if self.state.check_turn_limit(): self.state.set_draw(reason="The turn limit has been reached.")

        return self.state.step()
