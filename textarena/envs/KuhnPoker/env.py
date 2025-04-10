import re, random
from typing import Tuple, Dict, Any, Optional

import textarena as ta
from textarena.envs.KuhnPoker.renderer import create_board_str

# For reference:
#  - J (Jack): numeric rank 0
#  - Q (Queen): numeric rank 1
#  - K (King): numeric rank 2

class KuhnPokerEnv(ta.Env):
    """
    A simplified two-player Kuhn Poker environment.

    Deck: 3 cards - J, Q, K.
    Each player antes 1 chip => starting pot = 2 chips.
    Betting actions: [Check], [Bet], [Call], [Fold]
    Single betting round: 
        - Player0 acts first (check or bet).
        - If check => Player1 can check or bet.
        - If bet => next player can fold or call.
    Showdown if both players check or if a call occurs.
    Winner takes the pot.
    Game continues for max_rounds number of rounds.
    Final winner is determined by chip count.
    """
    def __init__(self, ante: int = 1, max_rounds: int = 1):
        super().__init__()
        self.ante = ante
        self.max_rounds = max_rounds
        self.deck = [0, 1, 2]  # 0=J, 1=Q, 2=K

        self.legal_action_tree = {
            "check": {
                "check": "showdown",
                "bet": {
                    "fold": "loser",
                    "call": "showdown"
                }
            },
            "bet": {
                "fold": "loser",
                "call": "showdown"
            }
        }

        # Regular expression to capture valid actions: e.g. [Check], [Bet], [Fold], [Call]
        self.action_pattern = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(self.state.game_state)


    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=2,
            max_turns=self.max_rounds,
            check_truncated=False
        )

        game_state = {"pot": None, "player_chips": {0: 0, 1: 0}, "current_round": 1, "starting_player": 0}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt, seed=seed)

        # Initialize the first round
        self._init_round()

    def _init_round(self):
        # check if game is complete
        if self.state.game_state["current_round"]+1 >= self.state.max_turns:
            # determine winner 
            if self.state.game_state["player_chips"][0] > self.state.game_state["player_chips"][0]:
                winner_id = 0
            elif self.state.game_state["player_chips"][0] > self.state.game_state["player_chips"][0]:
                winner_id = 0
            else:
                winner_id = None
                self.state.set_draw(reason=f"At the end of {self.state.max_turns} rounds, both players had the same number of chips.")
                
            if winner_id is not None:
                self.state.set_winners(player_ids=[winner_id], reason=f"Player {winner_id} won by having more chips at the end of all {self.state.max_turns} rounds.")

        # shuffle the deck 
        random.shuffle(self.deck)

        # assign player cards
        self.state.game_state["player_cards"] = {0: self.deck[0], 1: self.deck[1]}

        # reset pot
        self.state.game_state["pot"] = self.ante * 2
        self.state.game_state["player_chips"][0] -= self.ante
        self.state.game_state["player_chips"][1] -= self.ante


        # increment round counter
        self.state.game_state["current_round"] += 1
        self.state.game_state["current_legal_action_tree"] = self.legal_action_tree.copy()

        # set starting player
        starting_player = 1 - self.state.game_state["starting_player"]
        self.state.game_state["starting_player"] = starting_player 
        self.state.manually_update_current_player(new_player_id=starting_player)

        for player_id in range(2):
            message = (
                f"Starting round {self.state.game_state['current_round']} out of {self.state.max_turns} rounds.\n"
                f"Your card is: {self._rank_to_str(self.state.game_state['player_cards'][player_id])}\n"
            )
            
            if player_id == starting_player:
                legal_actions = ', '.join(f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys())
                message += f"Your available actions are: {legal_actions}"

            self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)




    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Provide detailed instructions to the players, explaining the rules of Kuhn Poker
        and available actions without revealing each other's cards.
        """
        # Basic game information
        prompt = (
            f"You are Player {player_id} in a {self.state.max_turns} round game of Kuhn Poker.\n"
            f"Game Rules:\n"
            f"- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            f"- Each player antes {self.ante} chip and receives 1 card each round\n"
            f"- Game continues for {self.state.max_turns} rounds\n"
            f"- The player with the most chips after all rounds wins\n\n"
            f"Action Rules:\n"
            f"- '[check]': Pass without betting (only if no bet is on the table)\n"
            f"- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            f"- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            f"- '[fold]': Surrender your hand and let your opponent win the pot\n\n"
            # f"Game Flow:\n"
            # f"- Player 0 can '[check]' or '[bet]'\n"
            # f"- If Player 0 Checks, Player 1 can '[check]' or '[bet]'\n"
            # f"  - If Player 1 Checks, showdown occurs (higher card wins)\n"
            # f"  - If Player 1 Bets, Player 0 must '[call]' or '[fold]'\n"
            # f"- If Player 0 Bets, Player 1 must '[call]' or '[fold]'\n"
            # f"- Showdown occurs if both players Check or if a bet is Called\n\n"
        )
        return prompt


    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """ Process the player's move """
        player_id = self.state.current_player_id

        # Log the raw action to both players
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # Parse the action
        match = self.action_pattern.search(action.strip())
        if not match:
            # Invalid action
            self.state.set_invalid_move(player_id=player_id, reason="Action must be [Check], [Bet], [Call], or [Fold].")
            return self.state.step()

        move = match.group(1).lower()  # 'check', 'bet', 'fold', 'call'

        if move not in self.state.game_state["current_legal_action_tree"].keys():
            legal_actions = ', '.join([f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys()])
            self.state.set_invalid_move(player_id=player_id, reason=f"Action must be {legal_actions}.")
            return self.state.step()

        # execute move
        self.state.game_state["current_legal_action_tree"] = self.state.game_state["current_legal_action_tree"][move]

        # check if round loser / showdown
        if self.state.game_state["current_legal_action_tree"] == "loser":
            reason=f"Player {self.state.current_player_id} has folded."
            self._set_round_winner(player_id=1-self.state.current_player_id, reason=reason)
        elif self.state.game_state["current_legal_action_tree"] == "showdown":
            self._handle_showdown()
        else:
            # show valid next actions
            legal_actions = ', '.join([f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys()])
            self.state.add_observation(from_id=ta.GAME_ID, to_id=1-player_id, message=f"Your available actions are: {legal_actions}")
        return self.state.step()



    def _set_round_winner(self, player_id: int, reason: str):
        self.state.game_state["player_chips"][player_id] += self.state.game_state["pot"]

        # initialize the next cound
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=reason)

        # start next round
        self._init_round()


    def _rank_to_str(self, rank: int) -> str:
        """Convert the numeric rank to a string 'J', 'Q', or 'K'."""
        return {0: 'J', 1: 'Q', 2: 'K'}.get(rank, '?')

    def _handle_showdown(self):
        player_cards = self.state.game_state["player_cards"]
        card_p0, card_p1 = player_cards[0], player_cards[1]

        # Show the cards
        cards_msg = f"Cards: Player 0 had {self._rank_to_str(card_p0)}, Player 1 had {self._rank_to_str(card_p1)}"
        self.state.add_observation(from_id=-1, to_id=-1, message=cards_msg)

        # Determine and announce the winner
        winner = 0 if card_p0 > card_p1 else 1
        winner_card, loser_card = (card_p0, card_p1) if winner == 0 else (card_p1, card_p0)
        reason = (
            f"Showdown: Player {winner}'s {self._rank_to_str(winner_card)} beats "
            f"Player {1 - winner}'s {self._rank_to_str(loser_card)}. "
            f"Player {winner} wins pot of {self.state.game_state['pot']} chips."
        )
        self._set_round_winner(player_id=winner, reason=reason)


