"""
Simplified Texas Hold'em Poker Game

In this simplified version of Texas Hold'em Poker, two players compete over a fixed number of rounds.

**Gameplay:**

- **Starting Chips**: Both players start with an equal number of chips.
- **Rounds**: The game is played over a fixed number of rounds.
- **Betting**:
  - Each round begins with both players placing an **ante** into the pot.
  - Players are dealt two private cards (hole cards).
  - A set of five community cards is dealt face-up.
  - Players can choose to **bet**, **call**, **raise**, **fold** or **check** during betting phases.
  - Betting continues until one player folds, or both **check**
- **Hand Ranking**: Standard Poker hand rankings are used to determine the winner.
- **Showdown**:
  - If both players bet or call, a showdown occurs where the best hand wins the pot.
  - If a player folds, the other player wins the pot without a showdown.

**Key Rules:**

- **Betting Actions**:
  - **[Bet] <amount>**: Place a bet of the specified amount.
  - **[Call]**: Match the opponent's bet.
  - **[Raise] <amount>**: Increase the bet by the specified amount.
  - **[Check]**: Check
  - **[Fold]**: Surrender the hand and forfeit any bets placed.
- **Hand Evaluation**:
  - Hands are evaluated using standard Poker hand rankings.
  - Each player's best possible hand is determined from their hole cards and the community cards.

**Parameters:**

- `starting_chips`: The number of chips each player starts with.
- `fixed_bet`: The fixed amount for bets and raises.
- `num_rounds`: The number of rounds to play.

**Game Outcomes:**

- The player with the most chips at the end of all rounds wins the game.
- A player can also win if the opponent runs out of chips before the rounds are completed.
"""

import random
from typing import Optional, Tuple

import textarena as ta


class SimplifiedPokerEnv(ta.Env):
    """Environment for the Simplified Texas Hold'em Poker game."""

    def __init__(
        self,
        num_rounds: int = 5,
        starting_chips: int = 100,
        fixed_bet: int = 10,
    ):
        """
        Initialize the Texas Hold'em Poker game.

        Args:
            num_rounds (int): Number of rounds to play.
            starting_chips (int): The number of chips each player starts with.
            fixed_bet (int): The fixed amount for bets and raises.
        """
        self.environment_name = "SimplifiedTexasHoldemPoker"

        self.num_rounds = num_rounds
        self.starting_chips = starting_chips
        self.fixed_bet = fixed_bet

        self.suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        self.ranks = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "Jack",
            "Queen",
            "King",
            "Ace",
        ]

        # Initialize game state
        self.game_state = ta.State(
            {
                "round": 0,
                "player_chips": {0: starting_chips, 1: starting_chips},
                "community_cards": [],
                "player_hands": {0: [], 1: []},
                "current_pot": 0,
                "current_bet": 0,
                "player_bets": {0: 0, 1: 0},
                "logs": [],
                "render": [
                    "round",
                    "player_chips",
                    "community_cards",
                    "current_pot",
                    "current_bet",
                ],
            }
        )

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[ta.Observation], ta.Info]:
        """
        Reset the game to its initial state.

        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.

        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)

        self.game_state["round"] = 0
        self.game_state["player_chips"] = {
            0: self.starting_chips,
            1: self.starting_chips,
        }
        self.game_state["logs"] = []

        # Start the first round
        self._start_new_round()

        # Generate initial prompts for both players
        observations = {
            0: [self._generate_player_prompt(player_id=0)],
            1: [self._generate_player_prompt(player_id=1)],
        }

        info = {
            "round": self.game_state["round"],
            "community_cards": self.game_state["community_cards"],
            "current_pot": self.game_state["current_pot"],
        }

        self.game_state["logs"].append((-1, "New game started."))

        return observations, info

    def _generate_player_prompt(self, player_id: int) -> ta.Message:
        """
        Generate the initial prompt for a player based on the current game state.

        Args:
            player_id (int): The player's ID (0 or 1).

        Returns:
            ta.Message: The initial prompt for the player.
        """
        prompt = (
            f"You are Player {player_id} in the Scenario-Style Texas Hold'em Poker game.\n"
            f"You have {self.game_state['player_chips'][player_id]} chips.\n"
            f"The current pot is {self.game_state['current_pot']} chips.\n"
            f"Your hole cards are: {self._format_cards(self.game_state['player_hands'][player_id])}\n"
            f"The community cards are: {self._format_cards(self.game_state['community_cards'])}\n"
            "Your goal is to make the best decision based on your hand and the community cards.\n"
            "On your turn, you can take one of the following actions:\n"
            "  - [Bet] <amount>: Place a bet of the specified amount.\n"
            "  - [Call]: Match the current bet.\n"
            "  - [Raise] <amount>: Increase the bet by the specified amount.\n"
            "  - [Check]: Pass the action to the next player without betting.\n"
            "  - [Fold]: Surrender your hand and forfeit the pot.\n"
        )
        return -1, prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observation],  # observations
        Optional[ta.Reward],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The action taken by the player.

        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        terminated = False
        truncated = False
        reward = None
        info = {}
        other_player_id = 1 - player_id

        action_type, amount = self._parse_action(action)

        if action_type is None:
            info["reason"] = "Invalid action."
            return None, reward, truncated, terminated, info

        self.game_state["logs"].append((player_id, action))

        if action_type == "fold":
            self._handle_fold(player_id)
            terminated = self._check_game_over()
        elif action_type == "check":
            self._handle_check(player_id)
        elif action_type == "call":
            self._handle_call(player_id)
        elif action_type == "bet":
            self._handle_bet(player_id, amount)
        elif action_type == "raise":
            self._handle_raise(player_id, amount)

        message = [(player_id, action)]
        observations = {player_id: message, other_player_id: message}

        if self._is_round_over():
            self._evaluate_hands()
            if self._check_game_over():
                terminated = True
                winner_id = self._determine_winner()
                reward = {winner_id: 1, 1 - winner_id: -1}
                info["reason"] = f"Player {winner_id} wins the game."
            else:
                self._start_new_round()

        return observations, reward, truncated, terminated, info

    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse the player's action string."""
        action = action.lower().strip()
        if action.startswith("[fold]"):
            return "fold", None
        elif action.startswith("[check]"):
            return "check", None
        elif action.startswith("[call]"):
            return "call", None
        elif action.startswith("[bet]"):
            try:
                amount = int(action.split()[-1])
                return "bet", amount
            except ValueError:
                return None, None
        elif action.startswith("[raise]"):
            try:
                amount = int(action.split()[-1])
                return "raise", amount
            except ValueError:
                return None, None
        return None, None

    def _handle_fold(self, player_id: int):
        """Handle a fold action."""
        other_player_id = 1 - player_id
        self.game_state["player_chips"][other_player_id] += self.game_state[
            "current_pot"
        ]
        self.game_state["current_pot"] = 0
        self.game_state["logs"].append(
            (-1, f"Player {player_id} folds. Player {other_player_id} wins the pot.")
        )

    def _handle_check(self, player_id: int):
        """Handle a check action."""
        self.game_state["logs"].append((-1, f"Player {player_id} checks."))

    def _handle_call(self, player_id: int):
        """Handle a call action."""
        amount_to_call = (
            self.game_state["current_bet"] - self.game_state["player_bets"][player_id]
        )
        self.game_state["player_chips"][player_id] -= amount_to_call
        self.game_state["player_bets"][player_id] += amount_to_call
        self.game_state["current_pot"] += amount_to_call
        self.game_state["logs"].append(
            (-1, f"Player {player_id} calls {amount_to_call} chips.")
        )

    def _handle_bet(self, player_id: int, amount: int):
        """Handle a bet action."""
        self.game_state["player_chips"][player_id] -= amount
        self.game_state["player_bets"][player_id] += amount
        self.game_state["current_pot"] += amount
        self.game_state["current_bet"] = amount
        self.game_state["logs"].append((-1, f"Player {player_id} bets {amount} chips."))

    def _handle_raise(self, player_id: int, amount: int):
        """Handle a raise action."""
        total_amount = self.game_state["current_bet"] + amount
        self.game_state["player_chips"][player_id] -= total_amount
        self.game_state["player_bets"][player_id] += total_amount
        self.game_state["current_pot"] += total_amount
        self.game_state["current_bet"] = total_amount
        self.game_state["logs"].append(
            (-1, f"Player {player_id} raises to {total_amount} chips.")
        )

    def _is_round_over(self) -> bool:
        """Check if the current round is over."""
        return self.game_state["player_bets"][0] == self.game_state["player_bets"][1]

    def _evaluate_hands(self):
        """Evaluate the players' hands and award the pot."""
        player0_hand = self._evaluate_hand(
            self.game_state["player_hands"][0] + self.game_state["community_cards"]
        )
        player1_hand = self._evaluate_hand(
            self.game_state["player_hands"][1] + self.game_state["community_cards"]
        )

        if player0_hand > player1_hand:
            winner_id = 0
        elif player1_hand > player0_hand:
            winner_id = 1
        else:
            # Tie, split the pot
            self.game_state["player_chips"][0] += self.game_state["current_pot"] // 2
            self.game_state["player_chips"][1] += self.game_state["current_pot"] // 2
            self.game_state["logs"].append((-1, "It's a tie. The pot is split."))
            self.game_state["current_pot"] = 0
            return

        self.game_state["player_chips"][winner_id] += self.game_state["current_pot"]
        self.game_state["logs"].append(
            (
                -1,
                f"Player {winner_id} wins the pot of {self.game_state['current_pot']} chips.",
            )
        )
        self.game_state["current_pot"] = 0

    def _evaluate_hand(self, hand):
        """Evaluate a poker hand and return a numeric value representing its strength."""
        # Simplified hand evaluation logic
        rank_values = {rank: index for index, rank in enumerate(self.ranks)}
        hand_value = sum(rank_values[card["rank"]] for card in hand)
        return hand_value

    def _start_new_round(self):
        """Start a new round of the game."""
        self.game_state["round"] += 1
        self.game_state["current_pot"] = 0
        self.game_state["current_bet"] = 0
        self.game_state["player_bets"] = {0: 0, 1: 0}

        # Deal new cards
        deck = self._create_deck()
        random.shuffle(deck)
        self.game_state["player_hands"] = {
            0: [deck.pop(), deck.pop()],
            1: [deck.pop(), deck.pop()],
        }
        self.game_state["community_cards"] = [deck.pop() for _ in range(5)]

        # Players ante up
        for player_id in [0, 1]:
            self.game_state["player_chips"][player_id] -= self.fixed_bet
            self.game_state["current_pot"] += self.fixed_bet
            self.game_state["player_bets"][player_id] = self.fixed_bet

        self.game_state["logs"].append(
            (-1, f"Starting Round {self.game_state['round']}")
        )

    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        if self.game_state["round"] >= self.num_rounds:
            self.game_state["logs"].append((-1, "Maximum number of rounds reached."))
            return True
        if any(chips <= 0 for chips in self.game_state["player_chips"].values()):
            self.game_state["logs"].append((-1, "A player has run out of chips."))
            return True
        return False

    def _determine_winner(self) -> int:
        """Determine the winner of the game."""
        if self.game_state["player_chips"][0] > self.game_state["player_chips"][1]:
            return 0
        elif self.game_state["player_chips"][1] > self.game_state["player_chips"][0]:
            return 1
        else:
            return -1  # Indicates a tie

    def _format_cards(self, cards):
        """Format a list of card dictionaries into a string."""
        return ", ".join([f"{card['rank']} of {card['suit']}" for card in cards])

    def _create_deck(self) -> list:
        """Create a standard 52-card deck."""
        return [
            {"rank": rank, "suit": suit} for suit in self.suits for rank in self.ranks
        ]

    def render(self):
        """Render the current game state."""
        print(f"Round: {self.game_state['round']}/{self.num_rounds}")
        print(f"Pot: {self.game_state['current_pot']} chips")
        print("Player Chips:")
        for player_id in [0, 1]:
            print(
                f"  Player {player_id}: {self.game_state['player_chips'][player_id]} chips"
            )
        print(
            f"Community Cards: {self._format_cards(self.game_state['community_cards'])}"
        )
        print("Game Logs:")
        for player_id, log in self.game_state["logs"][-5:]:  # Show last 5 logs
            if player_id == -1:
                print(f"[GAME] {log}")
            else:
                print(f"[Player {player_id}] {log}")
        print("\n")
