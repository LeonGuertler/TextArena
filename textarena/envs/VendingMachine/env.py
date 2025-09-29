from __future__ import annotations

"""
Vending Machine Environment (2 players):

Players
- Player 0: VendingMachine (VM)
- Player 1: Demand (consumer side)

Episode Rules
- Horizon: 20 days. Each day has two turns: VM first, then Demand.
- Start inventory: 0. Unlimited storage.
- Restock: VM can restock any non-negative integer quantity at the start of each day.
- Buy: Demand chooses a purchase quantity (max 20). If requested > inventory, sold = inventory.
- Visibility: Demand does NOT see current inventory; it sees fixed price and historical daily aggregates only.

Actions (strict, bracketed tokens)
- VM turn: "[Restock] qty=INTEGER"
- Demand turn: "[Buy] qty=INTEGER"

Observations
- Each turn, the acting player receives an English instruction prompt with the required action format.

Logging
- After Demand acts each day, the environment prints one log line:
  Day d: restock=R, requested=Q, sold=S, stock=I

Rewards
- At the end of day 20, we compute profit and store it in game_info, and set rewards as {0: profit, 1: 0}.
"""

import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.state import TwoPlayerState


RESTOCK_COST = 5
SALE_PRICE = 7
DEMAND_MAX_PER_DAY = 20
NUM_DAYS = 5


class VendingMachineEnv(ta.Env):
    """Two-player vending machine simulation with daily restock and demand purchase."""

    def __init__(self):
        # Core state variables (initialized in reset)
        self.state: TwoPlayerState
        self.inventory = 0
        self.current_day = 1
        self.pending_restock_qty = 0  # Restock chosen by VM for the current day
        self.last_requested_qty = 0
        self.last_sold_qty = 0
        # no internal random signal for demand (randomization will be prompted)

        # Aggregates
        self.total_restocked = 0
        self.total_sold = 0
        self.daily_logs = []  # list of dicts per day

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 2:
            raise ValueError("VendingMachineEnv requires exactly 2 players: VM (0) and Demand (1)")

        # Initialize TextArena two-player state: 40 turns (2 per day * 20 days)
        self.state = TwoPlayerState(num_players=2, max_turns=NUM_DAYS * 2, seed=seed)

        # Initialize environment state
        self.inventory = 0
        self.current_day = 1
        self.pending_restock_qty = 0
        self.last_requested_qty = 0
        self.last_sold_qty = 0
        self.total_restocked = 0
        self.total_sold = 0
        self.daily_logs = []

        # Compose game_state if needed by wrappers/agents
        game_state: Dict[str, Any] = {
            "day": self.current_day,
            "inventory": self.inventory,
            "price": SALE_PRICE,
            "cost": RESTOCK_COST,
            "demand_max": DEMAND_MAX_PER_DAY,
            "daily_logs": self.daily_logs,
        }

        # Provide initial prompts for both players
        self.state.reset(
            game_state=game_state,
            role_mapping={0: "VendingMachine", 1: "Demand"},
        )
    

    def get_observation(self) -> Tuple[int, Any]:
        """Provide basic game state observation - wrapper will handle context management."""
        # Follow the Env API: return (current_player_id, observation_list)
        pid = self.state.current_player_id
        obs_list = self.state.get_current_player_observation()

        # Add a simple game board message with current state
        if pid == 0:  # VM player
            board = (
                f"DAY {self.current_day} / {NUM_DAYS}\n"
                f"Price=${SALE_PRICE}, Cost=${RESTOCK_COST}\n"
                f"Inventory (visible to VM): {self.inventory}"
            )
        else:  # Demand player
            board = (
                f"DAY {self.current_day} / {NUM_DAYS}\n"
                f"Price=${SALE_PRICE}\n"
                f"Inventory is hidden."
            )

        obs_list.append((ta.GAME_ID, board, ta.ObservationType.GAME_BOARD))
        return pid, obs_list

    def _last_day_summary(self) -> str:
        if not self.daily_logs:
            return "N/A"
        last = self.daily_logs[-1]
        return (
            f"restock={last['restock']}, requested={last['requested']}, "
            f"sold={last['sold']}, stock_end={last['stock_end']}"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process a single action. VM acts on odd turns of the day (first turn), Demand on the second."""
        current_pid = self.state.current_player_id

        # Log raw action to the acting player
        self.state.add_observation(
            from_id=current_pid,
            to_id=current_pid,
            message=f"Your action: {action}",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )

        if current_pid == 0:
            # VM turn: expect [Restock] qty=N
            qty = self._parse_qty(action, token="Restock")
            if qty is None or qty < 0:
                self.state.set_invalid_move("Invalid VM action. Use '[Restock] qty=INTEGER' with INTEGER >= 0.")
                return self.state.step(rotate_player=False)

            self.pending_restock_qty = qty
            # Apply immediate restock
            self.inventory += self.pending_restock_qty
            self.total_restocked += self.pending_restock_qty

            # Announce to both players (without cost accounting details)
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=f"VM restocked qty={self.pending_restock_qty}.",
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )

            # Advance to Demand
            return self.state.step(rotate_player=True)

        else:
            # Demand turn: expect [Buy] qty=N (clamped to [0, DEMAND_MAX_PER_DAY])
            qty = self._parse_qty(action, token="Buy")
            if qty is None or qty < 0:
                self.state.set_invalid_move("Invalid Demand action. Use '[Buy] qty=INTEGER' with 0 <= INTEGER <= 20.")
                return self.state.step(rotate_player=False)

            requested = min(qty, DEMAND_MAX_PER_DAY)
            sold = min(requested, self.inventory)

            # Reduce inventory by sold amount
            self.inventory -= sold

            self.last_requested_qty = requested
            self.last_sold_qty = sold
            self.total_sold += sold

            # End-of-day log record
            day_log = {
                "day": self.current_day,
                "restock": self.pending_restock_qty,
                "requested": requested,
                "sold": sold,
                "stock_end": self.inventory,
            }
            self.daily_logs.append(day_log)


            # Print required daily line
            print(
                f"Day {self.current_day}: restock={self.pending_restock_qty}, "
                f"requested={requested}, sold={sold}, stock={self.inventory}"
            )

            # Announce to both players
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=(
                    f"Day {self.current_day} concluded: restock={self.pending_restock_qty}, "
                    f"requested={requested}, sold={sold}, stock_end={self.inventory}."
                ),
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )

            # Next day setup
            self.current_day += 1
            self.pending_restock_qty = 0

            # If reached day > NUM_DAYS, finish the game
            if self.current_day > NUM_DAYS:
                self._finalize_and_end()
                return self.state.step(rotate_player=False)

            # Else continue to next day -> VM's turn next
            return self.state.step(rotate_player=True)

    def _parse_qty(self, action: str, token: str) -> Optional[int]:
        """Extract an integer quantity from an action like: '[Token] qty=NUMBER'."""
        try:
            action = action.strip()
            # Look for the pattern [Token] qty=NUMBER anywhere in the action text
            pattern = rf"\[{token}\]\s*qty\s*=\s*(\d+)"
            m = re.search(pattern, action)
            if not m:
                return None
            return int(m.group(1))
        except Exception:
            return None

    def _finalize_and_end(self):
        """Compute end-of-episode profit (Method A) and set state.done."""
        total_profit = SALE_PRICE * self.total_sold - RESTOCK_COST * self.total_restocked

        # Store results in game_info for both players
        for pid in range(2):
            self.state.game_info[pid].update({
                "total_restocked": self.total_restocked,
                "total_requested": sum(d["requested"] for d in self.daily_logs),
                "total_sold": self.total_sold,
                "ending_inventory": self.inventory,
                "profit_method": "A",
                "unit_cost": RESTOCK_COST,
                "unit_price": SALE_PRICE,
                "total_profit": total_profit,
                "daily_logs": self.daily_logs,
            })

        # Set rewards: VM gets profit, Demand gets 0 (placeholder)
        self.state.rewards = {0: float(total_profit), 1: 0.0}
        self.state.done = True


