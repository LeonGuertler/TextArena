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
INITIAL_INVENTORY_PER_ITEM = 0


class VendingMachineEnv(ta.Env):
    """Multi-item vending machine with lead time and inventory pipeline."""

    def __init__(self):
        # Core state variables (initialized in reset)
        self.state: TwoPlayerState
        self.current_day = 1
        
        # Item definitions: {item_id: {description, lead_time, profit, holding_cost}}
        self.items: Dict[str, Dict[str, Any]] = {}
        
        # On-hand inventory: {item_id: quantity}
        # This is inventory available for sale NOW
        self.on_hand_inventory: Dict[str, int] = {}
        
        # Pending orders: List of all orders that haven't arrived yet
        # Each order: {item_id, quantity, order_day, arrival_day, original_lead_time}
        # arrival_day = order_day + lead_time at time of ordering
        self.pending_orders: List[Dict[str, Any]] = []
        
        # News schedule: {day: news_text} revealed to both agents at start
        self.news_schedule: Dict[int, str] = {}
        
        # Tracking variables
        self.current_day_orders: Dict[str, int] = {}  # Orders placed this turn by VM
        self.current_day_arrivals: Dict[str, List[Tuple[int, int]]] = {}  # Arrivals: [(qty, order_day), ...]
        
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
        
        # Initialize on-hand inventory (will be set to INITIAL_INVENTORY_PER_ITEM in reset)
        self.on_hand_inventory[item_id] = 0
        
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
        self.current_day_arrivals = {}
        self.daily_logs = []
        self.pending_orders = []
        
        # Initialize on-hand inventory to INITIAL_INVENTORY_PER_ITEM for each item
        for item_id in self.items:
            self.on_hand_inventory[item_id] = INITIAL_INVENTORY_PER_ITEM
            self.total_ordered[item_id] = 0
            self.total_sold[item_id] = 0

        # Compose game_state for wrappers/agents
        game_state: Dict[str, Any] = {
            "day": self.current_day,
            "items": self.items,
            "on_hand_inventory": self.on_hand_inventory,
            "pending_orders": self.pending_orders,
            "daily_logs": self.daily_logs,
        }

        # Provide initial prompts for both players
        self.state.reset(
            game_state=game_state,
            role_mapping={0: "VendingMachine", 1: "Demand"},
        )
    

    def update_item_config(self, item_id: str, lead_time: int = None, profit: float = None, holding_cost: float = None, description: str = None):
        """
        Update item configuration dynamically (supports changing parameters per day).
        
        IMPORTANT: lead_time changes only affect NEW orders placed after the change.
        Existing orders in transit maintain their original arrival_day (as in real logistics).
        
        Args:
            item_id: Item identifier
            lead_time: New lead time for future orders (if None, unchanged)
            profit: New profit (if None, unchanged)
            holding_cost: New holding cost (if None, unchanged)
            description: New description (if None, unchanged)
        """
        if item_id not in self.items:
            raise ValueError(f"Unknown item: {item_id}")
        
        # Update lead_time for future orders only
        # Existing orders in self.pending_orders keep their original arrival_day
        if lead_time is not None:
            self.items[item_id]['lead_time'] = lead_time
        
        if profit is not None:
            self.items[item_id]['profit'] = profit
        
        if holding_cost is not None:
            self.items[item_id]['holding_cost'] = holding_cost
        
        if description is not None:
            self.items[item_id]['description'] = description
    
    def get_observation(self) -> Tuple[int, Any]:
        """Provide basic game state observation - wrapper will handle context management."""
        pid = self.state.current_player_id
        obs_list = self.state.get_current_player_observation()

        # Build game board with multi-item information
        board_lines = [f"DAY {self.current_day} / {NUM_DAYS}"]
        
        # Add news - only show today's news and past news (no future news visibility)
        if self.news_schedule:
            # Today's news (if any)
            today_news = self.news_schedule.get(self.current_day, None)
            if today_news:
                board_lines.append(f"\n=== TODAY'S NEWS (Day {self.current_day}) ===")
                board_lines.append(f"⚡ {today_news}")
            
            # Past news (for learning from history)
            past_news = {day: news for day, news in self.news_schedule.items() if day < self.current_day}
            if past_news:
                board_lines.append("\n=== Past News ===")
                for day in sorted(past_news.keys()):
                    board_lines.append(f"Day {day}: {past_news[day]}")
        
        board_lines.append("\n=== ITEMS ===")
        
        for item_id, item_info in self.items.items():
            desc = item_info['description']
            profit = item_info['profit']
            holding_cost = item_info['holding_cost']
            
            if pid == 0:  # VM player - show profit/holding_cost and inventory (but NOT lead_time)
                on_hand = self.on_hand_inventory[item_id]
                # Calculate total in-transit from pending_orders
                # Include ALL orders that haven't arrived yet (arrival_day >= current_day)
                in_transit = sum(order['quantity'] for order in self.pending_orders 
                                if order['item_id'] == item_id 
                                and self.current_day <= order['arrival_day'] < float('inf'))
                board_lines.append(
                    f"{item_id} ({desc}): Profit=${profit}/unit, Holding=${holding_cost}/unit/day"
                )
                board_lines.append(f"  On-hand: {on_hand}, In-transit: {in_transit} units")
            else:  # Demand player - only show description
                board_lines.append(
                    f"{item_id} ({desc})"
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

            # Store orders for this turn (will be modified if lead_time=inf)
            self.current_day_orders = {}
            
            # Add orders to pending_orders with calculated arrival_day
            # Force order=0 if lead_time=inf (cannot restock)
            for item_id, qty in orders.items():
                lead_time = self.items[item_id]['lead_time']
                
                # Check if lead_time is infinite - cannot restock
                if lead_time == float('inf'):
                    self.current_day_orders[item_id] = 0  # Force to 0
                    if qty > 0:
                        print(f"⚠️  Day {self.current_day}: Cannot restock {item_id} (lead_time=inf). Order forced to 0 (requested: {qty})")
                else:
                    self.current_day_orders[item_id] = qty
                    if qty > 0:
                        # Calculate arrival_day = order_day + lead_time
                        arrival_day = self.current_day + lead_time
                        
                        # Add to pending_orders
                        self.pending_orders.append({
                            'item_id': item_id,
                            'quantity': qty,
                            'order_day': self.current_day,
                            'arrival_day': arrival_day,
                            'original_lead_time': lead_time
                        })
                        
                        self.total_ordered[item_id] += qty

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

            # Find orders that arrive TODAY (arrival_day == current_day)
            # These orders will be moved from pending_orders to on_hand_inventory
            arrivals = {}
            for item_id in self.items:
                arrivals[item_id] = []
            
            # Process arrivals: find all orders with arrival_day == current_day
            arrived_orders = [order for order in self.pending_orders 
                             if order['arrival_day'] == self.current_day]
            
            # Group arrivals by item_id and add to on_hand_inventory
            for order in arrived_orders:
                item_id = order['item_id']
                qty = order['quantity']
                order_day = order['order_day']
                
                # Add to arrivals tracking
                arrivals[item_id].append((qty, order_day))
                
                # Move to on_hand_inventory
                self.on_hand_inventory[item_id] += qty
            
            # Remove arrived orders from pending_orders
            self.pending_orders = [order for order in self.pending_orders 
                                  if order['arrival_day'] != self.current_day]
            
            self.current_day_arrivals = arrivals
            
            # Record starting on-hand inventory (after arrivals, before sales)
            starting_inventory = {}
            for item_id in self.items:
                starting_inventory[item_id] = self.on_hand_inventory[item_id]
            
            # Process sales: sell from on-hand inventory only
            actual_sales = {}
            for item_id, requested_qty in purchases.items():
                on_hand = self.on_hand_inventory[item_id]
                sold = min(requested_qty, on_hand)
                actual_sales[item_id] = sold
                
                # Reduce on-hand inventory
                self.on_hand_inventory[item_id] -= sold
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
                ending_inventory = self.on_hand_inventory[item_id]
                
                # Profit from sales
                daily_profit += profit * sold
                
                # Holding cost on ending inventory
                daily_holding_cost += holding_cost * ending_inventory
            
            daily_reward = daily_profit - daily_holding_cost

            # Record daily log
            day_log = {
                "day": self.current_day,
                "news": self.news_schedule.get(self.current_day, None),
                "orders": self.current_day_orders.copy(),
                "arrivals": {item_id: arrivals[item_id].copy() for item_id in self.items},
                "starting_inventory": starting_inventory.copy(),
                "requests": purchases.copy(),
                "sales": actual_sales.copy(),
                "ending_inventory": {item_id: self.on_hand_inventory[item_id] 
                                     for item_id in self.items},
                "daily_profit": daily_profit,
                "daily_holding_cost": daily_holding_cost,
                "daily_reward": daily_reward
            }
            self.daily_logs.append(day_log)

            # Print daily summary (console output) - unified format with LLM observation
            print(f"\n=== Day {self.current_day} Summary ===")
            if self.current_day in self.news_schedule:
                print(f"NEWS: {self.news_schedule[self.current_day]}")
            for item_id in self.items:
                ordered = self.current_day_orders.get(item_id, 0)
                arrival_list = arrivals[item_id]
                start_inv = starting_inventory[item_id]
                demand = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                ending_inv = self.on_hand_inventory[item_id]
                
                # Format arrival information with lead_time calculation
                arrival_str = ""
                if arrival_list:
                    arrival_parts = []
                    for qty, order_day in arrival_list:
                        actual_lead_time = self.current_day - order_day
                        arrival_parts.append(f"{qty} units (ordered on Day {order_day}, lead_time was {actual_lead_time} days)")
                    arrival_str = f", arrived={', '.join(arrival_parts)}"
                else:
                    arrival_str = ", arrived=0"
                
                print(f"{item_id}: ordered={ordered}{arrival_str}, starting on-hand inventory={start_inv}, demand={demand}, sold={sold}, ending on-hand inventory={ending_inv}")
            print(f"Daily Profit: ${daily_profit:.2f}, Daily Holding Cost: ${daily_holding_cost:.2f}")
            print(f"Daily Reward (R_t): ${daily_reward:.2f}")

            # Announce day conclusion with role-specific visibility
            # VM sees: ordered, arrived (with lead_time calculation), starting inventory, demand, sold, ending inventory
            vm_summary_lines = [f"Day {self.current_day} concluded:"]
            for item_id in self.items:
                ordered = self.current_day_orders.get(item_id, 0)
                arrival_list = arrivals[item_id]
                start_inv = starting_inventory[item_id]
                demand = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                ending_inv = self.on_hand_inventory[item_id]
                
                item_line = f"  {item_id}: ordered={ordered}"
                
                # Add arrival information with lead_time calculation
                if arrival_list:
                    arrival_parts = []
                    for qty, order_day in arrival_list:
                        # Calculate lead_time: current_day - order_day
                        actual_lead_time = self.current_day - order_day
                        arrival_parts.append(f"{qty} units (ordered on Day {order_day}, lead_time was {actual_lead_time} days)")
                    arrival_str = ", ".join(arrival_parts)
                    item_line += f", arrived={arrival_str}"
                else:
                    item_line += f", arrived=0"
                
                item_line += f", starting on-hand inventory={start_inv}, demand={demand}, sold={sold}, ending on-hand inventory={ending_inv}"
                vm_summary_lines.append(item_line)
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=0,  # VM only
                message="\n".join(vm_summary_lines),
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )
            
            # Demand sees: demand, sold (no ordered, no stock)
            demand_summary_lines = [f"Day {self.current_day} concluded:"]
            for item_id in self.items:
                demand = purchases.get(item_id, 0)
                sold = actual_sales.get(item_id, 0)
                demand_summary_lines.append(f"  {item_id}: demand={demand}, sold={sold}")
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=1,  # Demand only
                message="\n".join(demand_summary_lines),
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
            )

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
                "ending_inventory": {item_id: self.on_hand_inventory[item_id] 
                                     for item_id in self.items},
                "pending_orders": self.pending_orders,
                "total_reward": total_reward,
                "total_sales_profit": total_sales_profit,
                "total_holding_cost": total_holding_cost,
                "daily_logs": self.daily_logs,
                "items": self.items,
            })

        # Set rewards: VM gets total reward, Demand gets 0 (placeholder)
        self.state.rewards = {0: float(total_reward), 1: 0.0}
        self.state.done = True


