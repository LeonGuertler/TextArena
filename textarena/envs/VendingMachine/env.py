from __future__ import annotations

"""
Vending Machine Environment (2 players) - Multi-Item with Lead Time, Holding Cost, and News:

Players
- Player 0: VendingMachine (VM)
- Player 1: Demand (consumer side)

Episode Rules
- Horizon: NUM_DAYS days. Each day has two turns: VM first, then Demand.
- Start inventory: 10 units per item (on-hand).
- Multiple items: Each item has description, lead_time, profit, holding_cost.
- Order: VM orders items at start of each day. Orders arrive after lead_time days.
- Buy: Demand purchases from on-hand inventory only.
- Visibility: Demand does NOT see current inventory; sees prices and historical aggregates only.

News System
- News can be scheduled for specific days (e.g., holidays, promotions)
- Complete news schedule is visible to BOTH agents from the start
- News is purely informational - doesn't change game rules, but agents can adjust strategies
- Format: {day: news_text}

Actions (strict, bracketed tokens)
- VM turn: "[Order] item_1:qty=5, item_2:qty=10"
- Demand turn: "[Buy] item_1:qty=3, item_2:qty=5"

Inventory Pipeline
- I_t(j, 0): on-hand inventory (available for sale now)
- I_t(j, k): inventory arriving in k days

Observations
- Each turn, agents see: current day, news schedule, items info, inventory (VM only), history
- News schedule format: "Day X: [news text] <- TODAY" (marker on current day)

Rewards
- Daily reward: R_t = p_t · y_t - h_t · I_t
  where p_t·y_t is profit from sales (profit*sold) and h_t·I_t is holding cost
- Total reward: sum of all daily rewards R_t over the episode
- Final rewards: {0: total_reward, 1: 0}
"""

import re
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
from textarena.state import TwoPlayerState


# Global game parameters
NUM_DAYS = 10
INITIAL_INVENTORY_PER_ITEM = 5


class VendingMachineEnv(ta.Env):
    """Multi-item vending machine with lead time and inventory pipeline."""

    def __init__(self):
        # Core state variables (initialized in reset)
        self.state: TwoPlayerState
        self.current_day = 1
        
        # Item definitions: {item_id: {description, lead_time, profit, holding_cost}}
        self.items: Dict[str, Dict[str, Any]] = {}
        
        # Inventory pipeline: {item_id: [I(j,0), I(j,1), ..., I(j,L)]}
        # Position 0 = on-hand inventory (ready to sell)
        # Position k = arrives in k days
        self.inventory_pipeline: Dict[str, List[int]] = {}
        
        # News schedule: {day: news_text} revealed to both agents at start
        self.news_schedule: Dict[int, str] = {}
        
        # Tracking variables
        self.current_day_orders: Dict[str, int] = {}  # Orders placed this turn by VM
        
        # Aggregates
        self.total_ordered: Dict[str, int] = {}  # Total ordered per item
        self.total_sold: Dict[str, int] = {}     # Total sold per item
        self.daily_logs: List[Dict[str, Any]] = []  # Daily records

    def add_item(self, item_id: str, description: str, lead_time: int, profit: float, holding_cost: float):
        """
        Add an item to the vending machine.
        
        Args:
            item_id: Unique identifier for the item
            description: Human-readable description
            lead_time: Number of days for order to arrive (L_t(j))
            profit: Profit per unit sold (previously price - cost)
            holding_cost: Cost per unit per day for holding inventory (h_t(j))
        """
        if item_id in self.items:
            raise ValueError(f"Item {item_id} already exists")
        
        self.items[item_id] = {
            'description': description,
            'lead_time': lead_time,
            'profit': profit,
            'holding_cost': holding_cost
        }
        
        # Initialize pipeline: [on-hand, +1 day, +2 days, ..., +lead_time days]
        # Pipeline length = lead_time + 1 (positions 0 to lead_time)
        self.inventory_pipeline[item_id] = [0] * (lead_time + 1)
        self.total_ordered[item_id] = 0
        self.total_sold[item_id] = 0

    def add_news(self, day: int, news: str):
        """
        Add news for a specific day. News is revealed to both agents at game start.
        
        Args:
            day: The day number (1-indexed) when this news is relevant
            news: Textual information (e.g., "Holiday: Expect 50% higher demand")
        """
        if day < 1:
            raise ValueError(f"Day must be >= 1, got {day}")
        
        self.news_schedule[day] = news

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 2:
            raise ValueError("VendingMachineEnv requires exactly 2 players: VM (0) and Demand (1)")
        
        if not self.items:
            raise ValueError("No items added. Call add_item() before reset()")

        # Initialize TextArena two-player state
        self.state = TwoPlayerState(num_players=2, max_turns=NUM_DAYS * 2, seed=seed)

        # Initialize environment state
        self.current_day = 1
        self.current_day_orders = {}
        self.daily_logs = []
        
        # Initialize inventory pipeline: Set position 0 (on-hand) to initial inventory
        for item_id in self.items:
            lead_time = self.items[item_id]['lead_time']
            self.inventory_pipeline[item_id] = [INITIAL_INVENTORY_PER_ITEM] + [0] * lead_time
            self.total_ordered[item_id] = 0
            self.total_sold[item_id] = 0

        # Compose game_state for wrappers/agents
        game_state: Dict[str, Any] = {
            "day": self.current_day,
            "items": self.items,
            "inventory_pipeline": self.inventory_pipeline,
            "daily_logs": self.daily_logs,
        }

        # Provide initial prompts for both players
        self.state.reset(
            game_state=game_state,
            role_mapping={0: "VendingMachine", 1: "Demand"},
        )
    

    def get_observation(self) -> Tuple[int, Any]:
        """Provide basic game state observation - wrapper will handle context management."""
        pid = self.state.current_player_id
        obs_list = self.state.get_current_player_observation()

        # Build game board with multi-item information
        board_lines = [f"DAY {self.current_day} / {NUM_DAYS}"]
        
        # Add news schedule (complete schedule visible to both agents)
        if self.news_schedule:
            board_lines.append("\n=== NEWS SCHEDULE ===")
            for day in sorted(self.news_schedule.keys()):
                news_text = self.news_schedule[day]
                marker = " <- TODAY" if day == self.current_day else ""
                board_lines.append(f"Day {day}: {news_text}{marker}")
        
        board_lines.append("\n=== ITEMS ===")
        
        for item_id, item_info in self.items.items():
            desc = item_info['description']
            profit = item_info['profit']
            lead_time = item_info['lead_time']
            holding_cost = item_info['holding_cost']
            
            if pid == 0:  # VM player - show full info including inventory
                on_hand = self.inventory_pipeline[item_id][0]
                pipeline_str = ", ".join(str(self.inventory_pipeline[item_id][i]) 
                                        for i in range(1, len(self.inventory_pipeline[item_id])))
                board_lines.append(
                    f"{item_id} ({desc}): Profit=${profit}/unit, Holding=${holding_cost}/unit/day, Lead={lead_time}d"
                )
                board_lines.append(f"  On-hand: {on_hand}, Pipeline: [{pipeline_str}]")
            else:  # Demand player - hide profit and inventory
                board_lines.append(
                    f"{item_id} ({desc}): Lead={lead_time}d"
                )

        board = "\n".join(board_lines)
        obs_list.append((ta.GAME_ID, board, ta.ObservationType.GAME_BOARD))
        return pid, obs_list

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process a single action. VM orders, then Demand buys."""
        current_pid = self.state.current_player_id

        # Log raw action to the acting player
        self.state.add_observation(
            from_id=current_pid,
            to_id=current_pid,
            message=f"Your action: {action}",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )

        if current_pid == 0:
            # VM turn: expect JSON with action and optional rationale
            parsed = self._parse_json_action(action)
            if parsed is None:
                self.state.set_invalid_move('Invalid VM action. Use JSON format: {"action": {"item_id": qty, ...}, "rationale": "..."}')
                return self.state.step(rotate_player=False)
            
            orders, rationale = parsed

            # Validate all items exist
            for item_id in orders:
                if item_id not in self.items:
                    self.state.set_invalid_move(f"Unknown item: {item_id}")
                    return self.state.step(rotate_player=False)
                if orders[item_id] < 0:
                    self.state.set_invalid_move(f"Negative quantity for {item_id}")
                    return self.state.step(rotate_player=False)

            # Store orders for this turn
            self.current_day_orders = orders.copy()

            # Add orders to pipeline at position lead_time
            for item_id, qty in orders.items():
                if qty > 0:
                    lead_time = self.items[item_id]['lead_time']
                    self.inventory_pipeline[item_id][lead_time] += qty
                    self.total_ordered[item_id] += qty

            # Announce orders (without rationale in game history)
            order_str = ", ".join(f"{item_id}:{qty}" for item_id, qty in orders.items())
            message = f"VM ordered: {order_str}"
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=-1,
                message=message,
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )

            # Advance to Demand
            return self.state.step(rotate_player=True)

        else:
            # Demand turn: expect JSON with action
            parsed = self._parse_json_action(action)
            if parsed is None:
                self.state.set_invalid_move('Invalid Demand action. Use JSON format: {"action": {"item_id": qty, ...}}')
                return self.state.step(rotate_player=False)
            
            purchases, _ = parsed  # Demand doesn't need rationale

            # Validate all items exist
            for item_id in purchases:
                if item_id not in self.items:
                    self.state.set_invalid_move(f"Unknown item: {item_id}")
                    return self.state.step(rotate_player=False)
                if purchases[item_id] < 0:
                    self.state.set_invalid_move(f"Negative quantity for {item_id}")
                    return self.state.step(rotate_player=False)

            # Process sales: sell from on-hand inventory only (position 0)
            actual_sales = {}
            for item_id, requested_qty in purchases.items():
                on_hand = self.inventory_pipeline[item_id][0]
                sold = min(requested_qty, on_hand)
                actual_sales[item_id] = sold
                
                # Reduce on-hand inventory
                self.inventory_pipeline[item_id][0] -= sold
                self.total_sold[item_id] += sold

            # Calculate daily reward: R_t = p_t · y_t - h_t · I_t
            # p_t · y_t = sum of profit * sold for each item
            # h_t · I_t = sum of holding_cost * ending_inventory for each item
            daily_profit = 0.0
            daily_holding_cost = 0.0
            
            for item_id in self.items:
                profit = self.items[item_id]['profit']
                holding_cost = self.items[item_id]['holding_cost']
                
                sold = actual_sales.get(item_id, 0)
                ending_inventory = self.inventory_pipeline[item_id][0]
                
                # Profit from sales
                daily_profit += profit * sold
                
                # Holding cost on ending inventory
                daily_holding_cost += holding_cost * ending_inventory
            
            daily_reward = daily_profit - daily_holding_cost

            # Record daily log
            day_log = {
                "day": self.current_day,
                "news": self.news_schedule.get(self.current_day, None),  # News for this day if any
                "orders": self.current_day_orders.copy(),
                "requests": purchases.copy(),
                "sales": actual_sales.copy(),
                "ending_inventory": {item_id: self.inventory_pipeline[item_id][0] 
                                     for item_id in self.items},
                "daily_profit": daily_profit,
                "daily_holding_cost": daily_holding_cost,
                "daily_reward": daily_reward
            }
            self.daily_logs.append(day_log)

            # Print daily summary
            print(f"\n=== Day {self.current_day} Summary ===")
            if self.current_day in self.news_schedule:
                print(f"NEWS: {self.news_schedule[self.current_day]}")
            for item_id in self.items:
                ordered = self.current_day_orders.get(item_id, 0)
                requested = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                stock = self.inventory_pipeline[item_id][0]
                print(f"{item_id}: ordered={ordered}, requested={requested}, sold={sold}, stock={stock}")
            print(f"Daily Profit: ${daily_profit:.2f}, Daily Holding Cost: ${daily_holding_cost:.2f}")
            print(f"Daily Reward (R_t): ${daily_reward:.2f}")

            # Announce day conclusion with role-specific visibility
            # VM sees: ordered, requested, sold, stock
            vm_summary_lines = [f"Day {self.current_day} concluded:"]
            for item_id in self.items:
                ordered = self.current_day_orders.get(item_id, 0)
                requested = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                stock = self.inventory_pipeline[item_id][0]
                vm_summary_lines.append(f"  {item_id}: ordered={ordered}, requested={requested}, sold={sold}, stock={stock}")
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=0,  # VM only
                message="\n".join(vm_summary_lines),
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )
            
            # Demand sees: requested, sold (no ordered, no stock)
            demand_summary_lines = [f"Day {self.current_day} concluded:"]
            for item_id in self.items:
                requested = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                demand_summary_lines.append(f"  {item_id}: requested={requested}, sold={sold}")
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=1,  # Demand only
                message="\n".join(demand_summary_lines),
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )

            # Advance pipeline: shift everything left by 1 position
            self._advance_pipeline()

            # Next day setup
            self.current_day += 1
            self.current_day_orders = {}

            # If reached day > NUM_DAYS, finish the game
            if self.current_day > NUM_DAYS:
                self._finalize_and_end()
                return self.state.step(rotate_player=False)

            # Else continue to next day -> VM's turn next
            return self.state.step(rotate_player=True)

    def _parse_json_action(self, action: str) -> Optional[Tuple[Dict[str, int], Optional[str]]]:
        """
        Parse JSON action format: {"action": {"item_id": qty, ...}, "rationale": "..."}
        Returns (action_dict, rationale) or None if invalid.
        """
        try:
            import json
            action = action.strip()
            
            # Find JSON object in the string
            json_start = action.find('{')
            json_end = action.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = action[json_start:json_end]
            data = json.loads(json_str)
            
            if 'action' not in data:
                return None
            
            action_dict = data['action']
            if not isinstance(action_dict, dict):
                return None
            
            # Convert all values to integers
            result = {}
            for item_id, qty in action_dict.items():
                result[str(item_id)] = int(qty)
            
            # Extract rationale if present
            rationale = data.get('rationale', None)
            
            return (result, rationale)
        except Exception as e:
            print(f"Error parsing JSON action: {e}")
            return None
    
    def _parse_multi_item_action(self, action: str, token: str) -> Optional[Dict[str, int]]:
        """
        Parse multi-item action like: '[Order] item_1:qty=5, item_2:qty=10'
        Returns dict {item_id: quantity} or None if invalid.
        (Legacy format, kept for backward compatibility)
        """
        try:
            action = action.strip()
            # Look for pattern [Token] item_id:qty=N, item_id:qty=N, ...
            pattern = rf"\[{token}\]\s*(.+)"
            m = re.search(pattern, action, re.IGNORECASE)
            if not m:
                return None
            
            items_str = m.group(1).strip()
            result = {}
            
            # Parse each item:qty=N pair
            for item_pair in items_str.split(','):
                item_pair = item_pair.strip()
                if not item_pair:
                    continue
                    
                # Match item_id:qty=NUMBER
                item_match = re.match(r'(\w+)\s*:\s*qty\s*=\s*(\d+)', item_pair, re.IGNORECASE)
                if not item_match:
                    return None
                
                item_id = item_match.group(1)
                qty = int(item_match.group(2))
                result[item_id] = qty
            
            return result if result else {}
        except Exception:
            return None

    def _advance_pipeline(self):
        """
        Advance the inventory pipeline by 1 day.
        Shift all positions left: position k becomes position k-1.
        Position lead_time becomes 0 (new items added to pipeline in next step).
        """
        for item_id in self.items:
            pipeline = self.inventory_pipeline[item_id]
            # Shift left: [on-hand, +1, +2, ..., +L] -> [on-hand + old_+1, old_+2, ..., 0]
            for i in range(len(pipeline) - 1):
                if i == 0:
                    # On-hand inventory receives items from position 1
                    pipeline[0] += pipeline[1]
                else:
                    # Each position receives from next position
                    pipeline[i] = pipeline[i + 1]
            # Last position becomes 0 (no new orders yet)
            pipeline[-1] = 0

    def _finalize_and_end(self):
        """Sum up all daily rewards and finalize the episode."""
        # Sum all daily rewards R_t = sum(p_t · y_t - h_t · I_t) over all days
        total_reward = sum(day_log['daily_reward'] for day_log in self.daily_logs)
        
        # Calculate total profit and holding cost for reporting
        total_sales_profit = sum(day_log['daily_profit'] for day_log in self.daily_logs)
        total_holding_cost = sum(day_log['daily_holding_cost'] for day_log in self.daily_logs)
        
        # Store results in game_info for both players
        for pid in range(2):
            self.state.game_info[pid].update({
                "total_ordered": self.total_ordered,
                "total_sold": self.total_sold,
                "ending_inventory": {item_id: self.inventory_pipeline[item_id][0] 
                                     for item_id in self.items},
                "total_reward": total_reward,
                "total_sales_profit": total_sales_profit,
                "total_holding_cost": total_holding_cost,
                "daily_logs": self.daily_logs,
                "items": self.items,
            })

        # Set rewards: VM gets total reward, Demand gets 0 (placeholder)
        self.state.rewards = {0: float(total_reward), 1: 0.0}
        self.state.done = True


