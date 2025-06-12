import random, re
from typing import Dict, Any, Optional, Tuple
import textarena as ta


class HighSocietyEnv(ta.Env):
    """
    2-player Knizia 'High Society' (simplified: only 10 positive prestige cards).
    • Each player has one set of money cards: 1-11 (single use).
    • 10 prestige cards 1-10 are auctioned in order.
    • Each round players simultaneously choose one money card to bid.
      - Higher bid wins the prestige card *but loses the money card*.
      - Losing bid card is returned to hand.
    • At the end: player with LESS cash remaining is *eliminated*.
      Other player wins if they weren't eliminated, ties => draw.
    """

    def __init__(self):
        super().__init__()
        self.money_cards = list(range(1, 12))   # 1-11
        self.action_space = {i: re.compile(r"\[(11|10|[1-9])]", re.I) for i in range(2)}

    @staticmethod
    def _intlist_to_str(lst): return " ".join(f"'[{str(x)}]'" for x in sorted(lst))

    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        deck = list(range(1, 11))
        random.shuffle(deck)
        game_state = {"round": 0, "prestige_deck": deck, "player_money": {0: self.money_cards.copy(), 1: self.money_cards.copy()}, "player_prestige": {0: 0, 1: 0}, "pending_bids": {}, "starting_player": 0}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._next_auction()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a game of HighSociety (2-player version).\n"
            f"Game flow:  Ten prestige cards are auctioned one after another.\n"
            f"Bidding:    Each auction, secretly choose a money card 1-11 and reveal.\n"
            f"            • Higher bid wins the prestige card and discards that money card.\n"
            f"            • Lower bid keeps their money card.\n"
            f"            • Tie -> both bids are returned and the same prestige card is re-auctioned.\n"
            f"Scoring:    After all ten auctions, add **remaining cash + prestige points**.\n"
            f"            Higher *net-worth* wins (exact tie -> draw).\n\n"
            f"**Action syntax**  →  bid a single card like '[7]' or '[11]'."
        )

    def _next_auction(self):
        gs = self.state.game_state
        gs["round"] += 1
        if not gs["prestige_deck"]: self._end_match(); return

        prize = gs["prestige_deck"].pop()
        gs["current_prize"] = prize
        gs["pending_bids"] = {}

        for pid in (0, 1):
            self.state.add_observation(to_id=pid, message=f"\n### Auction {gs['round']}/10  |  Prestige card: {prize}", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.add_observation(to_id=pid, message=f"Your remaining money cards: {self._intlist_to_str(gs['player_money'][pid])}", observation_type=ta.ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        pid = self.state.current_player_id
        gs  = self.state.game_state
        self.state.add_observation(from_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        t = self.action_space[pid].findall(action)
        if len(t) != 1:
            self.state.set_invalid_move("Must include exactly one [X] money card.")
            return self.state.step()

        bid = int(t[0])
        if bid not in gs["player_money"][pid]:
            self.state.set_invalid_move("You no longer have that money card.")
            return self.state.step()

        gs["pending_bids"][pid] = bid

        if len(gs["pending_bids"]) == 1:
            return self.state.step(rotate_player=True)  # wait for opponent

        # both bids in
        bid0, bid1 = gs["pending_bids"][0], gs["pending_bids"][1]
        prize = gs["current_prize"]

        if bid0 > bid1:     winner, loser = 0, 1
        elif bid1 > bid0:   winner, loser = 1, 0
        else:  # tie -> redraw bids
            self.state.add_observation(message="Tie - bids returned. Rebid!", observation_type=ta.ObservationType.GAME_MESSAGE)
            gs["pending_bids"] = {}
            return self.state.step(rotate_player=True)

        gs["player_prestige"][winner] += prize
        gs["player_money"][winner].remove(gs["pending_bids"][winner])  # pay cost
        self.state.add_observation(message=f"P0 bid {bid0}, P1 bid {bid1}. Player {winner} wins prestige {prize} (total {gs['player_prestige'][winner]}).", observation_type=ta.ObservationType.GAME_MESSAGE)
        self._next_auction()
        return self.state.step(rotate_player=False)

    def _end_match(self):
        def networth(pid: int) -> int: return sum(self.state.game_state["player_money"][pid]) + self.state.game_state["player_prestige"][pid]
        nw0, nw1 = networth(0), networth(1)
        if nw0 > nw1:   self.state.set_winner(0, f"Net-worth {nw0} > {nw1}")
        elif nw1 > nw0: self.state.set_winner(1, f"Net-worth {nw1} > {nw0}")
        else:           self.state.set_draw(f"Both net-worth {nw0} – exact tie.")
