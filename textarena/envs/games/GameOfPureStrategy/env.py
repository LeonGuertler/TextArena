import random, re
from typing import Dict, Any, Optional, Tuple

import textarena as ta
# from textarena.envs.games.GameOfPureStrategy.renderer import create_board_str  # TODO


class GameOfPureStrategyEnv(ta.Env):
    """
    GOPS / One-Card War  – 2-player bidding game.

    * Deck of 13 prize cards (A=1 … K=13) shuffled face-down.
    * Every round reveal top prize; both players secretly bid a card from their own A-K hand.
    * Higher bid wins the entire prize **plus** any carry-over pot.
    * Tie -> prize value rolls into pot, next round worth more.
    * Bids are single-use; after 13 rounds high score wins.
    """

    def __init__(self):
        super().__init__()
        self.full_hand = list(range(1, 14))
        self.action_space = {i: re.compile(r"\[(a|k|q|j|10|[2-9])]", re.I) for i in range(2)}

    @staticmethod
    def _face_to_val(face: str) -> int:
        face = face.strip().lower()
        faces = {"a": 1, "j": 11, "q": 12, "k": 13}
        if face.isdigit(): return int(face)
        return faces.get(face)
    
    @staticmethod
    def _val_to_face(v: int) -> str: return {1: "A", 11: "J", 12: "Q", 13: "K"}.get(v, str(v))
    # def get_board_str(self): return create_board_str(self.state.game_state) # TODO

    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state={
            "round": 0, "prize_deck": random.sample(self.full_hand, k=13), "carry_pot": 0, "current_prize": None, 
            "player_hands": {0: self.full_hand.copy(), 1: self.full_hand.copy()}, "pending_bids": {}, "player_scores": {0: 0, 1: 0}, "starting_player": 0,
            }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._start_round()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in GOPS.\n- You hold the 13 cards A-K; each can be used once.\n- Each round a prize card is revealed. Play exactly ONE card "
            f"by writing something that contains a bracketed token like '[Q]', '[10]', '[2]' ...\n- Higher card wins the prize (plus carry-over). Ties roll prize "
            f"into the pot for next round.\n- Highest total after 13 rounds wins."
        )

    def _start_round(self):
        gs = self.state.game_state
        gs["round"] += 1

        if gs["round"] > 13:
            s0, s1 = gs["player_scores"].values()
            if s0 > s1:     self.state.set_winner(0, f"P0 {s0} vs P1 {s1}")
            elif s1 > s0:   self.state.set_winner(1, f"P1 {s1} vs P0 {s0}")
            else:           self.state.set_draw(f"Both scored {s0}")
            return

        gs["current_prize"] = gs["prize_deck"][gs["round"] - 1]
        gs["pending_bids"] = {}
        gs["starting_player"] = 1 - gs["starting_player"]
        self.state.manually_set_current_player_id(gs["starting_player"])

        for pid in (0, 1):
            hand_str = " ".join(f"'[{self._val_to_face(c)}]'" for c in gs["player_hands"][pid])
            self.state.add_observation(to_id=pid, message=f"### Round {gs['round']}/13 - Prize: {self._val_to_face(gs['current_prize'])}  (worth {gs['current_prize'] + gs['carry_pot']})", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.add_observation(to_id=pid, message=f"\nYour remaining hand: {hand_str}", observation_type=ta.ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        pid = self.state.current_player_id
        gs = self.state.game_state
        self.state.add_observation(from_id=pid, to_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        tokens = self.action_space[pid].findall(action.lower())
        print(tokens)
        if len(tokens) != 1 or len(gs["pending_bids"]) >= 2: 
            self.state.set_invalid_move(reason="Action must contain exactly ONE bracketed card token.")
            return self.state.step()

        face = tokens[0]
        bid_val = self._face_to_val(face)
        if bid_val not in gs["player_hands"][pid]:
            self.state.set_invalid_move(reason="You no longer have that card.")
            return self.state.step()

        # record bid secretly
        gs["pending_bids"][pid] = bid_val
        gs["player_hands"][pid].remove(bid_val)

        # waiting for opponent?
        if len(gs["pending_bids"]) == 1: return self.state.step(rotate_player=True)

        bid0, bid1 = gs["pending_bids"][0], gs["pending_bids"][1]
        pot_value  = gs["current_prize"] + gs["carry_pot"]
        gs["carry_pot"] = 0

        reveal = (f"Bids: P0 {self._val_to_face(bid0)} vs P1 {self._val_to_face(bid1)} - ")
        if bid0 > bid1:     gs["player_scores"][0] += pot_value;    reveal += f"Player 0 wins {pot_value}."
        elif bid1 > bid0:   gs["player_scores"][1] += pot_value;    reveal += f"Player 1 wins {pot_value}."
        else:               gs["carry_pot"] += pot_value;           reveal += f"Tie -> pot now {gs['carry_pot']}."
        self.state.add_observation(message=reveal, observation_type=ta.ObservationType.GAME_MESSAGE)
        self.state.add_observation(message=f"Scores -> P0:{gs['player_scores'][0]}  P1:{gs['player_scores'][1]}", observation_type=ta.ObservationType.GAME_MESSAGE)

        self._start_round()
        return self.state.step(rotate_player=False)