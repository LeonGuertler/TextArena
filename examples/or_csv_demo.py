"""
OR Algorithm baseline demo with CSV-driven demand.

This demo uses:
- VM Agent: OR algorithm baseline (Operations Research heuristic)
- Demand: Fixed demand patterns loaded from CSV file

The OR algorithm uses a base-stock policy:
  base_stock = mu_hat + z*sigma_hat
  order = max(base_stock - current_inventory, 0)

where mu_hat and sigma_hat are estimated from historical demand data.

Usage:
  python or_csv_demo.py --demand-file path/to/demands.csv
"""

import os
import sys
import argparse
import json
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import norm
import textarena as ta


# Fix stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _sanitize_text(text: str) -> str:
    """Normalize to NFKC and escape remaining non-ASCII characters."""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.encode("ascii", "backslashreplace").decode("ascii")


def _safe_print(text: str) -> None:
    """Print text with encoding fallback for Windows."""
    print(_sanitize_text(str(text)))


class CSVDemandPlayer:
    """
    Simulates demand agent by reading from CSV file.
    Supports dynamic item configurations that can change per day.
    """
    def __init__(self, csv_path: str, initial_samples: dict = None):
        """
        Args:
            csv_path: Path to CSV file
            initial_samples: Optional dict of {item_id: [historical demand samples]}
                           If provided, will validate item_ids match CSV
        """
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        
        # Auto-detect item IDs from CSV columns (columns starting with 'demand_')
        self.item_ids = self._extract_item_ids()
        
        if not self.item_ids:
            raise ValueError("No item columns found in CSV. Expected columns like 'demand_<item_id>'")
        
        # Validate all required columns exist for each item
        self._validate_item_columns()
        
        # Validate initial_samples if provided
        if initial_samples is not None:
            self._validate_initial_samples(initial_samples)
        
        # Extract news if available
        self.has_news = 'news' in self.df.columns
        
        print(f"Loaded CSV with {len(self.df)} days of demand data")
        print(f"Detected {len(self.item_ids)} items: {self.item_ids}")
        if self.has_news:
            news_days = self.df[self.df['news'].notna()]['day'].tolist()
            print(f"News scheduled for days: {news_days}")
    
    def _extract_item_ids(self) -> list:
        """Extract item IDs from CSV columns that start with 'demand_'."""
        item_ids = []
        for col in self.df.columns:
            if col.startswith('demand_'):
                item_id = col[len('demand_'):]
                item_ids.append(item_id)
        return item_ids
    
    def _validate_item_columns(self):
        """Validate that CSV has all required columns for each item."""
        required_suffixes = ['demand', 'description', 'lead_time', 'profit', 'holding_cost']
        for item_id in self.item_ids:
            for suffix in required_suffixes:
                col_name = f'{suffix}_{item_id}'
                if col_name not in self.df.columns:
                    raise ValueError(f"CSV missing required column: {col_name}")
    
    def _validate_initial_samples(self, initial_samples: dict):
        """Validate that initial_samples item_ids match CSV."""
        sample_ids = set(initial_samples.keys())
        csv_ids = set(self.item_ids)
        
        if sample_ids != csv_ids:
            missing_in_csv = sample_ids - csv_ids
            missing_in_samples = csv_ids - sample_ids
            error_msg = "Initial samples item_ids do not match CSV items.\n"
            if missing_in_csv:
                error_msg += f"  Items in initial_samples but not in CSV: {missing_in_csv}\n"
            if missing_in_samples:
                error_msg += f"  Items in CSV but not in initial_samples: {missing_in_samples}\n"
            raise ValueError(error_msg)
    
    def get_item_ids(self) -> list:
        """Return list of item IDs detected from CSV."""
        return self.item_ids.copy()
    
    def get_initial_item_configs(self) -> list:
        """
        Get initial item configurations from first row of CSV.
        
        Returns:
            List of dicts with keys: item_id, description, lead_time, profit, holding_cost
        """
        if len(self.df) == 0:
            raise ValueError("CSV is empty")
        
        first_row = self.df.iloc[0]
        configs = []
        
        for item_id in self.item_ids:
            # Handle lead_time - could be int or "inf"
            lead_time_val = first_row[f'lead_time_{item_id}']
            if isinstance(lead_time_val, str) and lead_time_val.lower() == 'inf':
                lead_time = float('inf')
            elif isinstance(lead_time_val, float) and lead_time_val == float('inf'):
                # pandas reads "inf" as numpy.float64 inf
                lead_time = float('inf')
            else:
                lead_time = int(lead_time_val)
            
            config = {
                'item_id': item_id,
                'description': str(first_row[f'description_{item_id}']),
                'lead_time': lead_time,
                'profit': float(first_row[f'profit_{item_id}']),
                'holding_cost': float(first_row[f'holding_cost_{item_id}'])
            }
            configs.append(config)
        
        return configs
    
    def get_day_item_config(self, day: int, item_id: str) -> dict:
        """
        Get item configuration for a specific day (supports dynamic changes).
        
        Args:
            day: Day number (1-indexed)
            item_id: Item identifier
            
        Returns:
            Dict with keys: description, lead_time, profit, holding_cost
        """
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range (1-{len(self.df)})")
        
        if item_id not in self.item_ids:
            raise ValueError(f"Unknown item_id: {item_id}")
        
        row = self.df.iloc[day - 1]
        
        # Handle lead_time - could be int or "inf"
        lead_time_val = row[f'lead_time_{item_id}']
        if isinstance(lead_time_val, str) and lead_time_val.lower() == 'inf':
            lead_time = float('inf')
        elif isinstance(lead_time_val, float) and lead_time_val == float('inf'):
            # pandas reads "inf" as numpy.float64 inf
            lead_time = float('inf')
        else:
            lead_time = int(lead_time_val)
        
        return {
            'description': str(row[f'description_{item_id}']),
            'lead_time': lead_time,
            'profit': float(row[f'profit_{item_id}']),
            'holding_cost': float(row[f'holding_cost_{item_id}'])
        }
    
    def get_num_days(self) -> int:
        """Return number of days in CSV."""
        return len(self.df)
    
    def get_news_schedule(self) -> dict:
        """Extract news schedule from CSV."""
        if not self.has_news:
            return {}
        
        news_schedule = {}
        for _, row in self.df.iterrows():
            day = int(row['day'])
            if pd.notna(row['news']) and str(row['news']).strip():
                news_schedule[day] = str(row['news']).strip()
        
        return news_schedule
    
    def get_action(self, day: int) -> str:
        """
        Generate buy action for given day based on CSV data in JSON format.
        
        Args:
            day: Current day (1-indexed)
            
        Returns:
            JSON string like '{"action": {"chips(Regular)": 10, "chips(BBQ)": 5}}'
        """
        # Get row for this day (day is 1-indexed, df is 0-indexed)
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range (1-{len(self.df)})")
        
        row = self.df.iloc[day - 1]
        
        # Extract demand for each item
        action_dict = {}
        for item_id in self.item_ids:
            col_name = f'demand_{item_id}'
            qty = int(row[col_name])
            action_dict[item_id] = qty
        
        # Return JSON format
        result = {"action": action_dict}
        return json.dumps(result, indent=2)


class ORAgent:
    """
    OR algorithm baseline agent using base-stock policy.
    
    Supports two policies:
    1. Vanilla: order_quantity = max(base_stock - current_inventory, 0)
       where base_stock = μ̂ + z*σ̂
    2. Capped: order_quantity = max(min(base_stock - current_inventory, cap), 0)
       where cap = μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
    
    μ̂ = (1+L) × empirical_mean
    σ̂ = sqrt(1+L) × empirical_std
    z* = Φ^(-1)(q), where q = profit/(profit + holding_cost)
    """
    
    def __init__(self, items_config: dict, initial_samples: dict = None, policy: str = 'capped'):
        """
        Args:
            items_config: Dict of {item_id: {'lead_time': L, 'profit': p, 'holding_cost': h}}
            initial_samples: Optional dict of {item_id: [list of initial demand samples]}
                           If None or empty, will use only observed demands
            policy: 'vanilla' or 'capped' (default: 'capped')
        """
        self.items_config = items_config
        self.initial_samples = initial_samples if initial_samples else {}
        self.policy = policy
        
        # Store observed demands (will be updated each day)
        # Format: {item_id: [demand_day1, demand_day2, ...]}
        self.observed_demands = {item_id: [] for item_id in items_config}
        
        print(f"\n=== OR Agent Initialized (Policy: {policy.upper()}) ===")
        for item_id, config in items_config.items():
            L = config['lead_time']
            p = config['profit']
            h = config['holding_cost']
            q = p / (p + h)
            z_star = norm.ppf(q)
            
            samples = self.initial_samples.get(item_id, [])
            print(f"{item_id}:")
            print(f"  Lead time (L): {L}")
            print(f"  Profit (p): {p}, Holding cost (h): {h}")
            print(f"  Critical fractile (q): {q:.4f}")
            _safe_print(f"  z* = Φ^(-1)(q): {z_star:.4f}")
            print(f"  Initial samples: {samples if samples else 'None (will learn from observed demands)'}")
    
    def _parse_inventory_from_observation(self, observation: str, item_id: str) -> int:
        """
        Parse current total inventory (on-hand + in-transit) from observation.
        
        Observation format:
          chips(Regular) (...): Profit=$2/unit, Holding=$1/unit/day
            On-hand: 5, In-transit: 10 units
        
        Returns:
            Total inventory across all pipeline stages (on-hand + in-transit)
        """
        try:
            lines = observation.split('\n')
            for i, line in enumerate(lines):
                # Find the item header line
                if line.strip().startswith(f"{item_id}"):
                    # Next line should have the inventory info
                    if i + 1 < len(lines):
                        inventory_line = lines[i + 1]
                        
                        # Parse: "  On-hand: 5, In-transit: 10 units"
                        if "On-hand:" in inventory_line and "In-transit:" in inventory_line:
                            # Extract on-hand value
                            on_hand_start = inventory_line.find("On-hand:") + len("On-hand:")
                            on_hand_end = inventory_line.find(",", on_hand_start)
                            on_hand = int(inventory_line[on_hand_start:on_hand_end].strip())
                            
                            # Extract in-transit value: "In-transit: 10 units"
                            in_transit_start = inventory_line.find("In-transit:") + len("In-transit:")
                            # Find the end - look for "units" or end of line
                            in_transit_str = inventory_line[in_transit_start:].strip()
                            # Remove "units" if present
                            in_transit_str = in_transit_str.replace("units", "").strip()
                            in_transit = int(in_transit_str)
                            
                            total_inventory = on_hand + in_transit
                            return total_inventory
            
            # If not found, return 0 (shouldn't happen in normal operation)
            print(f"Warning: Could not parse inventory for {item_id}, assuming 0")
            return 0
        except Exception as e:
            print(f"Error parsing inventory for {item_id}: {e}")
            return 0
    
    def _calculate_order(self, item_id: str, current_inventory: int) -> dict:
        """
        Calculate order quantity using OR base-stock policy.
        
        Args:
            item_id: Item identifier
            current_inventory: Current total inventory (on-hand + in-transit)
            
        Returns:
            Dict with keys: order, empirical_mean, empirical_std, mu_hat, sigma_hat, 
                           L, z_star, q, base_stock, cap (if capped), order_uncapped (if capped)
        """
        config = self.items_config[item_id]
        L = config['lead_time']
        p = config['profit']
        h = config['holding_cost']
        
        # Collect all demand samples
        initial = self.initial_samples.get(item_id, [])
        all_samples = initial + self.observed_demands[item_id]
        
        # If no samples yet, use a conservative default (order 0)
        if not all_samples:
            return {
                'order': 0,
                'empirical_mean': 0,
                'empirical_std': 0,
                'mu_hat': 0,
                'sigma_hat': 0,
                'L': L,
                'z_star': 0,
                'q': p / (p + h),
                'base_stock': 0,
                'current_inventory': current_inventory
            }
        
        # Calculate empirical statistics
        empirical_mean = np.mean(all_samples)
        empirical_std = np.std(all_samples, ddof=1) if len(all_samples) > 1 else 0
        
        # Calculate μ̂ and σ̂ for lead time + review period
        mu_hat = (1 + L) * empirical_mean
        sigma_hat = np.sqrt(1 + L) * empirical_std
        
        # Calculate critical fractile and z*
        q = p / (p + h)
        z_star = norm.ppf(q)
        
        # Calculate base stock
        base_stock = mu_hat + z_star * sigma_hat
        
        result = {
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'mu_hat': mu_hat,
            'sigma_hat': sigma_hat,
            'L': L,
            'z_star': z_star,
            'q': q,
            'base_stock': base_stock,
            'current_inventory': current_inventory
        }
        
        # Calculate order quantity based on policy
        if self.policy == 'vanilla':
            order = max(int(np.ceil(base_stock - current_inventory)), 0)
            result['order'] = order
        else:  # capped policy
            # Vanilla order (for logging)
            order_uncapped = max(int(np.ceil(base_stock - current_inventory)), 0)
            
            # Cap calculation: μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
            cap_z = norm.ppf(0.95)
            cap = mu_hat / (1 + L) + cap_z * sigma_hat / np.sqrt(1 + L)
            
            # Capped order
            order = max(min(int(np.ceil(base_stock - current_inventory)), int(np.ceil(cap))), 0)
            
            result['order'] = order
            result['order_uncapped'] = order_uncapped
            result['cap'] = cap
        
        return result
    
    def update_demand_observation(self, item_id: str, observed_demand: int):
        """
        Update observed demand history for an item.
        
        Args:
            item_id: Item identifier
            observed_demand: The true demand observed on this day (requested quantity)
        """
        self.observed_demands[item_id].append(observed_demand)
    
    def get_action(self, observation: str) -> tuple[str, dict]:
        """
        Generate ordering decision in JSON format with detailed statistics.
        
        Args:
            observation: Current game observation
            
        Returns:
            Tuple of (action_json_string, statistics_dict)
            statistics_dict contains all calculation details for logging
        """
        action_dict = {}
        rationale_parts = []
        statistics = {}
        
        for item_id in self.items_config:
            # Parse current inventory from observation
            current_inventory = self._parse_inventory_from_observation(observation, item_id)
            
            # Calculate order quantity with full statistics
            calc_result = self._calculate_order(item_id, current_inventory)
            action_dict[item_id] = calc_result['order']
            statistics[item_id] = calc_result
            
            # Build rationale
            if calc_result['mu_hat'] == 0:
                rationale_parts.append(f"{item_id}: No samples yet, order=0")
            else:
                if self.policy == 'vanilla':
                    rationale_parts.append(
                        f"{item_id}: base_stock={calc_result['base_stock']:.1f} "
                        f"(μ̂={calc_result['mu_hat']:.1f}, σ̂={calc_result['sigma_hat']:.1f}, z*={calc_result['z_star']:.2f}), "
                        f"inv={current_inventory}, order={calc_result['order']}"
                    )
                else:  # capped
                    rationale_parts.append(
                        f"{item_id}: base_stock={calc_result['base_stock']:.1f}, cap={calc_result['cap']:.1f}, "
                        f"inv={current_inventory}, order={calc_result['order']} "
                        f"(uncapped would be {calc_result['order_uncapped']})"
                    )
        
        policy_name = "OR base-stock (capped)" if self.policy == 'capped' else "OR base-stock (vanilla)"
        rationale = f"{policy_name}: " + "; ".join(rationale_parts)
        
        result = {
            "action": action_dict,
            "rationale": rationale
        }
        
        return json.dumps(result, indent=2), statistics


def main():
    parser = argparse.ArgumentParser(description='Run OR algorithm baseline with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time used by OR algorithm (default: 0). Actual lead time in CSV may differ.')
    parser.add_argument('--policy', type=str, choices=['vanilla', 'capped'], default='capped',
                       help='OR policy type: vanilla (standard base-stock) or capped (smoothed orders). Default: capped')
    args = parser.parse_args()
    
    # Create environment
    env = ta.make(env_id="VendingMachine-v0")
    
    # Load CSV demand player (auto-detects items)
    try:
        csv_player = CSVDemandPlayer(args.demand_file, initial_samples=None)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Get item configurations from CSV
    item_configs = csv_player.get_initial_item_configs()
    
    # Add items to environment
    for config in item_configs:
        env.add_item(**config)
    
    # Generate initial demand samples for all items (unified across all products)
    # Using the same historical samples regardless of item type
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    print(f"\nUsing unified initial samples for all items: {unified_samples}")
    print(f"Promised lead time (used by OR algorithm): {args.promised_lead_time} days")
    print(f"Note: Actual lead times in CSV may differ, creating a test scenario for OR robustness.")
    
    # Set NUM_DAYS based on CSV
    from textarena.envs.VendingMachine import env as vm_env_module
    original_num_days = vm_env_module.NUM_DAYS
    original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
    vm_env_module.NUM_DAYS = csv_player.get_num_days()
    print(f"Set NUM_DAYS to {vm_env_module.NUM_DAYS} based on CSV")
    
    # Add news from CSV
    news_schedule = csv_player.get_news_schedule()
    for day, news in news_schedule.items():
        env.add_news(day, news)
    
    # Create OR agent (uses promised lead time, not actual CSV lead time)
    or_items_config = {
        config['item_id']: {
            'lead_time': args.promised_lead_time,  # Use promised value instead of CSV value
            'profit': config['profit'],
            'holding_cost': config['holding_cost']
        }
        for config in item_configs
    }
    or_agent = ORAgent(or_items_config, initial_samples, policy=args.policy)
    
    # Reset environment
    env.reset(num_players=2)
    
    # Run game
    done = False
    current_day = 1
    last_demand = {}  # Track demand to update OR agent
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (OR algorithm)
            # Update item configurations for current day (supports dynamic changes)
            has_inf_lead_time = False
            for item_id in csv_player.get_item_ids():
                config = csv_player.get_day_item_config(current_day, item_id)
                env.update_item_config(
                    item_id=item_id,
                    lead_time=config['lead_time'],
                    profit=config['profit'],
                    holding_cost=config['holding_cost'],
                    description=config['description']
                )
                # Check if any item has lead_time=inf (supplier unavailable)
                if config['lead_time'] == float('inf'):
                    has_inf_lead_time = True
            
            # If supplier unavailable (lead_time=inf), skip OR decision
            if has_inf_lead_time:
                # Create action with order=0 for all items
                zero_orders = {item_id: 0 for item_id in csv_player.get_item_ids()}
                action = json.dumps({"action": zero_orders}, indent=2)
                
                print(f"\nWARNING Day {current_day}: Supplier unavailable (lead_time=inf)")
                print(f"Day {current_day} OR Decision: {action} (automatically set to 0)")
                
                # Skip to demand turn
                done, _ = env.step(action=action)
                continue
            
            action, stats = or_agent.get_action(observation)
            
            # Print detailed statistics for each item
            print(f"\n{'='*70}")
            print(f"Day {current_day} OR Decision ({args.policy.upper()} Policy):")
            print(f"{'='*70}")
            for item_id, item_stats in stats.items():
                print(f"\n{item_id}:")
                print(f"  Empirical mean: {item_stats['empirical_mean']:.2f}")
                print(f"  Empirical std: {item_stats['empirical_std']:.2f}")
                print(f"  Lead time (L): {item_stats['L']}")
                _safe_print(f"  mu_hat (μ̂): {item_stats['mu_hat']:.2f}")
                _safe_print(f"  sigma_hat (σ̂): {item_stats['sigma_hat']:.2f}")
                print(f"  Critical fractile (q): {item_stats['q']:.4f}")
                _safe_print(f"  z*: {item_stats['z_star']:.4f}")
                print(f"  Base stock: {item_stats['base_stock']:.2f}")
                print(f"  Current inventory: {item_stats['current_inventory']}")
                if args.policy == 'capped':
                    print(f"  Cap value: {item_stats['cap']:.2f}")
                    print(f"  Order (capped): {item_stats['order']}")
                    print(f"  Order (uncapped): {item_stats['order_uncapped']}")
                else:
                    print(f"  Order: {item_stats['order']}")
            
            _safe_print(f"\n{action}")
            print(f"{'='*70}")
        else:  # Demand from CSV
            action = csv_player.get_action(current_day)
            
            # Parse demand to update OR agent's history
            demand_data = json.loads(action)
            last_demand = demand_data['action']
            
            print(f"\nDay {current_day} Demand: {action}")
            
            # Update OR agent with observed demand
            for item_id, qty in last_demand.items():
                or_agent.update_demand_observation(item_id, qty)
            
            current_day += 1
        
        done, _ = env.step(action=action)
    
    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]
    
    print("\n" + "="*60)
    print("=== Final Results (OR Algorithm Baseline) ===")
    print("="*60)
    
    # Per-item statistics
    total_ordered = vm_info.get('total_ordered', {})
    total_sold = vm_info.get('total_sold', {})
    ending_inventory = vm_info.get('ending_inventory', {})
    items = vm_info.get('items', {})
    
    print("\nPer-Item Statistics:")
    for item_id, item_info in items.items():
        ordered = total_ordered.get(item_id, 0)
        sold = total_sold.get(item_id, 0)
        ending = ending_inventory.get(item_id, 0)
        profit = item_info['profit']
        holding_cost = item_info['holding_cost']
        
        total_profit = profit * sold
        print(f"\n{item_id} ({item_info['description']}):")
        print(f"  Ordered: {ordered}, Sold: {sold}, Ending: {ending}")
        print(f"  Profit/unit: ${profit}, Holding: ${holding_cost}/unit/day")
        print(f"  Total Profit: ${total_profit}")
    
    # Daily breakdown
    print("\n" + "="*60)
    print("Daily Breakdown:")
    print("="*60)
    for day_log in vm_info.get('daily_logs', []):
        day = day_log['day']
        news = day_log.get('news', None)
        profit = day_log['daily_profit']
        holding = day_log['daily_holding_cost']
        reward = day_log['daily_reward']
        
        news_str = f" [NEWS: {news}]" if news else ""
        print(f"Day {day}{news_str}: Profit=${profit:.2f}, Holding=${holding:.2f}, Reward=${reward:.2f}")
    
    # Totals
    total_reward = vm_info.get('total_reward', 0)
    total_profit = vm_info.get('total_sales_profit', 0)
    total_holding = vm_info.get('total_holding_cost', 0)
    
    print("\n" + "="*60)
    print("=== TOTAL SUMMARY ===")
    print("="*60)
    print(f"Total Profit from Sales: ${total_profit:.2f}")
    print(f"Total Holding Cost: ${total_holding:.2f}")
    print(f"\n>>> Total Reward (OR Baseline): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)
    
    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = original_initial_inventory


if __name__ == "__main__":
    main()

