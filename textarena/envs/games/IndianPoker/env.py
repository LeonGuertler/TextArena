import re, random
from typing import Dict, Any, Optional, Tuple

import textarena as ta
# from textarena.envs.games.IndianPoker.renderer import create_board_str # TODO


class IndianPokerEnv(ta.Env):
    """
    Two-player Indian (Blind-Man’s-Bluff) Poker.
    - 52-card deck, ranks 2 < … < A; suits ignored for showdown.
    - Every round: ante → each player receives ONE face-down card (visible only to the opponent) → single betting street with unlimited raises → showdown / fold.
    - Actions: [check] | [bet X] | [call] | [raise X] | [fold] where X is a positive integer.
    """

    def __init__(self, ante: int=1, max_rounds: int=1, starting_chips: int=100):
        super().__init__()
        self.ante = ante
        self.max_rounds = max_rounds
        self.starting_bank = starting_chips
        self.full_deck = list(range(52))
        self.action_space = lambda player_id: re.compile(r"\[check\]|\[fold\]|\[call\]|\[bet (\d+)\]|\[raise (\d+)\]", re.IGNORECASE)

    @staticmethod
    def _rank(card: int) -> int: return (card % 13) + 2 # 0-51 → 2-14
    @staticmethod
    def _rank_to_str(card: int) -> str: return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"][(card % 13)]

    # def get_board_str(self):
    #     return create_board_str(self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"player_chips": {0: self.starting_bank, 1: self.starting_bank}, "current_round": 0, "starting_player": 0}, player_prompt_function=self._prompt)
        self._init_round()

    def _init_round(self):
        gs = self.state.game_state
        gs["current_round"] += 1

        # check if match finished
        if gs["current_round"] > self.max_rounds:
            bank0, bank1 = gs["player_chips"].values()
            if bank0 > bank1:   self.state.set_winner(0, f"Player 0 wins ({bank0} > {bank1})")
            elif bank1 > bank0: self.state.set_winner(1, f"Player 1 wins ({bank1} > {bank0})")
            else:               self.state.set_draw("Equal chips after all rounds")
            return

        deck = self.full_deck.copy()
        random.shuffle(deck)
        gs["player_cards"] = {0: deck[0], 1: deck[1]}

        gs["pot"] = self.ante * 2
        for pid in (0, 1): gs["player_chips"][pid] -= self.ante

        # per-round betting bookkeeping
        gs["current_bets"] = {0: 0, 1: 0} # chips committed this round
        gs["highest_bet"] = 0 # current bet to match
        gs["prev_action"] = None # track check-check

        # rotate first player
        gs["starting_player"] = 1-gs["starting_player"]
        self.state.manually_set_current_player_id(gs["starting_player"])
        for pid in (0, 1):
            self.state.add_observation(to_id=pid, message=f"### Round {gs['current_round']}/{self.max_rounds}\nYour opponent's card is: {self._rank_to_str(gs['player_cards'][1-pid])}", observation_type=ta.ObservationType.GAME_MESSAGE)
        self._announce_actions(self.state.current_player_id)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Indian Poker.\n- 52-card deck; you see only the opponent's card.\n- Ante {self.ante} chip(s) each round, {self.max_rounds} round(s) total.\n"
            f"- Valid moves: '[check]'  |  '[bet X]'  |  '[call]'  |  '[raise X]'  |  '[fold]'  (X is a positive integer <= your chip count.)\n- Highest hidden card wins the pot at showdown.\n"
        )
    
    def _find_token(self, msg: str):
        patterns = [("check", re.compile(r"\[check\]", re.I)), ("fold", re.compile(r"\[fold\]", re.I)), ("call", re.compile(r"\[call\]", re.I)), ("bet", re.compile(r"\[bet (\d+)\]", re.I)), ("raise", re.compile(r"\[raise (\d+)\]", re.I))]
        found = [(name, m) for name, rx in patterns if (m := rx.search(msg))]
        if len(found) != 1: return None, None # none or ambiguous
        name, match = found[0]
        amt  = int(match.group(1)) if name in ("bet", "raise") else None
        return name, amt

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        pid = self.state.current_player_id
        gs = self.state.game_state
        self.state.add_observation(from_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        def chips_to_call() -> int: return gs["highest_bet"] - gs["current_bets"][pid]

        move, amount = self._find_token(action)
        if move is None:
            self.state.set_invalid_move("Supply exactly ONE bracketed action, e.g. '[check]', '[bet 2]', '[call]', '[raise 3]', or '[fold]'.")
            return self.state.step()

        to_call = chips_to_call()

        # legality checks
        if move == "check" and to_call != 0:            self.state.set_invalid_move("Cannot [check] - you are facing a bet.");          return self.state.step()
        if move in ("bet", "raise") and amount <= 0:    self.state.set_invalid_move("Bet / Raise amount must be ≥ 1.");                 return self.state.step()
        if move == "bet" and to_call != 0:              self.state.set_invalid_move("Cannot [bet] - must [call] / [raise] / [fold].");  return self.state.step()
        if move == "call" and to_call == 0:             self.state.set_invalid_move("Nothing to call; you may [check] instead.");       return self.state.step()
        if move == "raise" and to_call == 0:            self.state.set_invalid_move("Use [bet X] to open; there is no bet to raise.");  return self.state.step()
        
        # bankroll check
        cost = 0
        if move == "bet":       cost = amount
        elif move == "call":    cost = to_call
        elif move == "raise":   cost = to_call + amount

        if cost > gs["player_chips"][pid]:              self.state.set_invalid_move("Insufficient chips for that action.");             return self.state.step()

        rotate = True
        self.state.add_observation(message=f"Player {pid} -> [{move}{' ' + str(amount) if amount else ''}]", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

        if move == "check":
            gs["prev_action"], rotate = ("check", True)
            if gs["prev_action"] == "check" and gs.get("second_check"): # second consecutive check -> showdown
                self._showdown()
                rotate = False
            else:
                gs["second_check"] = True

        elif move == "bet":
            gs["second_check"] = False
            gs["highest_bet"] = amount
            gs["current_bets"][pid] += amount
            gs["player_chips"][pid] -= amount
            gs["pot"] += amount
            gs["prev_action"] = "bet"

        elif move == "call":
            gs["second_check"] = False
            gs["current_bets"][pid] += to_call
            gs["player_chips"][pid] -= to_call
            gs["pot"] += to_call
            self._showdown()
            rotate = False

        elif move == "raise":
            gs["second_check"] = False
            new_bet = gs["highest_bet"] + amount
            chips_needed = new_bet - gs["current_bets"][pid]
            gs["highest_bet"] = new_bet
            gs["current_bets"][pid] += chips_needed
            gs["player_chips"][pid] -= chips_needed
            gs["pot"] += chips_needed
            gs["prev_action"] = "raise"

        elif move == "fold":
            self._end_round(1-pid, f"Player {pid} folded.")
            rotate = False

        # prompt next player if round continues
        if rotate: self._announce_actions(1-pid)
        return self.state.step(rotate_player=rotate)

    def _announce_actions(self, to_pid: int):
        gs = self.state.game_state
        to_call = gs["highest_bet"] - gs["current_bets"][to_pid]
        if to_call == 0:    legal = "'[check]', '[bet X]'"
        else:               legal = f"'[call]' (cost {to_call}), '[raise X]', '[fold]'"
        self.state.add_observation(to_id=to_pid, message=f"Your possible actions: {legal}", observation_type=ta.ObservationType.GAME_BOARD)

    def _end_round(self, winner: int, reason: str):
        gs = self.state.game_state
        gs["player_chips"][winner] += gs["pot"]
        self.state.add_observation(message=f"{reason}  Pot {gs['pot']} → Player {winner}. (Bankrolls P0:{gs['player_chips'][0]}, P1:{gs['player_chips'][1]})", observation_type=ta.ObservationType.GAME_MESSAGE)
        self._init_round()

    def _showdown(self):
        gs = self.state.game_state
        c0, c1 = gs["player_cards"][0], gs["player_cards"][1]
        r0, r1 = self._rank(c0), self._rank(c1)

        if r0 > r1:     self._end_round(0, f"Showdown: {self._rank_to_str(c0)} beats {self._rank_to_str(c1)}.")
        elif r1 > r0:   self._end_round(1, f"Showdown: {self._rank_to_str(c1)} beats {self._rank_to_str(c0)}.")
        else:  # tie – split pot
            split = gs["pot"] // 2
            gs["player_chips"][0] += split
            gs["player_chips"][1] += gs["pot"] - split
            self.state.add_observation(message=f"Showdown tie: both {self._rank_to_str(c0)}. Pot split – each receives {split}.", observation_type=ta.ObservationType.GAME_MESSAGE)
            self._init_round()
