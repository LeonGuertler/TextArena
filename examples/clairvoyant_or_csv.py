"""
Clairvoyant OR baseline demo with CSV-driven demand.

This demo uses:
- VM Agent: Clairvoyant OR baseline (knows distribution shift schedule)
- Demand: Fixed demand patterns loaded from CSV file

The clairvoyant OR agent uses a base-stock policy:
  base_stock = μ_hat + z*sigma_hat
  order = max(base_stock - current_inventory, 0)

where μ_hat and σ_hat are provided directly by the clairvoyant schedule.

Usage:
  python clairvoyant_or_csv.py --demand-file path/to/demands.csv --instance 1
"""

import os
import sys
import argparse
import json
import math
import unicodedata
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


def _schedule_instance1(day: int, _item_id: str) -> tuple[float, float]:
    return 100.0, 25.0


def _schedule_instance2(day: int, _item_id: str) -> tuple[float, float]:
    if day <= 15:
        return 100.0, 25.0
    return 200.0, 25.0 * math.sqrt(2.0)


def _schedule_instance3(day: int, _item_id: str) -> tuple[float, float]:
    t = max(day, 1)
    return 100.0 * t, 25.0 * math.sqrt(float(t))


def _schedule_instance4(day: int, _item_id: str) -> tuple[float, float]:
    if day <= 15:
        return 100.0, 25.0
    return 100.0, 50.0


def _schedule_instance4_1(day: int, _item_id: str) -> tuple[float, float]:
    if day <= 15:
        return 100.0, 25.0
    return 100.0, 0.0


def _schedule_instance5(day: int, _item_id: str) -> tuple[float, float]:
    return 500.0, 25.0 * math.sqrt(5.0)


def _schedule_instance6(day: int, _item_id: str) -> tuple[float, float]:
    return 500.0, 25.0 * math.sqrt(5.0)


def _schedule_instance7(day: int, _item_id: str) -> tuple[float, float]:
    if day <= 15:
        return 500.0, 25.0 * math.sqrt(5.0)
    return 700.0, 25.0 * math.sqrt(7.0)


def _schedule_instance8(day: int, _item_id: str) -> tuple[float, float]:
    return 511.0, 25.0 * math.sqrt(5.11)


CLAIRVOYANT_SCHEDULES: dict[str, tuple] = {
    "1": (_schedule_instance1, "Instance 1: mu_hat=100, sigma_hat=25 (stationary)."),
    "2": (_schedule_instance2, "Instance 2: mu_hat/sigma_hat shift at day 16 (100->200, 25->25*sqrt(2))."),
    "3": (_schedule_instance3, "Instance 3: mu_hat=100*t, sigma_hat=25*sqrt(t) (increasing demand)."),
    "4": (_schedule_instance4, "Instance 4: sigma_hat doubles after day 15 (variance increase)."),
    "4_1": (_schedule_instance4_1, "Instance 4_1: sigma_hat -> 0 after day 15 (variance decrease to zero)."),
    "5": (_schedule_instance5, "Instance 5: mu_hat=500, sigma_hat=25*sqrt(5) (L=4)."),
    "6": (_schedule_instance6, "Instance 6: mu_hat=500, sigma_hat=25*sqrt(5) (random lead time)."),
    "7": (_schedule_instance7, "Instance 7: shift to mu_hat=700, sigma_hat=25*sqrt(7) after day 15."),
    "8": (_schedule_instance8, "Instance 8: mu_hat=511, sigma_hat=25*sqrt(5.11) (intermittent supplier)."),
}


class ClairvoyantSchedule:
    """Provides clairvoyant μ̂ and σ̂ values for each instance."""

    def __init__(self, instance_id: str):
        if instance_id not in CLAIRVOYANT_SCHEDULES:
            available = ", ".join(sorted(CLAIRVOYANT_SCHEDULES.keys()))
            raise ValueError(f"Unsupported instance {instance_id}. Available: {available}")
        self.instance_id = instance_id
        self.schedule_fn, self.description = CLAIRVOYANT_SCHEDULES[instance_id]

    def describe(self) -> str:
        return self.description

    def get_params(self, day: int, item_id: str) -> tuple[float, float]:
        if day < 1:
            raise ValueError("Day index must be ≥ 1 for clairvoyant schedule.")
        mu_hat, sigma_hat = self.schedule_fn(day, item_id)
        return float(mu_hat), float(sigma_hat)


class ClairvoyantORAgent:
    """
    Clairvoyant OR baseline agent using predetermined μ̂/σ̂ schedules.
    
    Supports two policies:
    1. Vanilla: order_quantity = max(base_stock - current_inventory, 0)
       where base_stock = μ̂ + z*σ̂
    2. Capped: order_quantity = max(min(base_stock - current_inventory, cap), 0)
       where cap = μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
    
    μ̂, σ̂ come from the clairvoyant schedule.
    L is determined by instance number and current day (hard-coded for each instance).
    """

    def __init__(self, items_config: dict, schedule: ClairvoyantSchedule, policy: str = 'capped'):
        self.items_config = items_config
        self.schedule = schedule
        self.policy = policy
        self.current_day = 1
        self.observed_demands = {item_id: [] for item_id in items_config}

        print(f"\n=== Clairvoyant OR Agent Initialized (Policy: {policy.upper()}) ===")
        print(f"Schedule: {self.schedule.describe()}")
        print(f"Instance {self.schedule.instance_id}: Lead time L is hard-coded per instance specification")
        for item_id, config in items_config.items():
            p = config['profit']
            h = config['holding_cost']
            q = p / (p + h)
            z_star = norm.ppf(q)
            print(f"{item_id}:")
            print(f"  Profit (p): {p}, Holding cost (h): {h}")
            print(f"  Critical fractile (q): {q:.4f}")
            _safe_print(f"  z* = Φ^(-1)(q): {z_star:.4f}")

    def set_day(self, day: int):
        self.current_day = max(1, day)
    
    def _get_lead_time(self) -> float:
        """
        Get lead time L for current instance and day.
        
        Hard-coded based on instance characteristics:
        - Instances 1-4, 4_1: L = 0 (always)
        - Instances 5, 6, 8: L = 4 (always)
        - Instance 7: L = 4 (days 1-15), L = 6 (days 16+)
        
        Returns:
            Lead time value for current instance and day
        """
        instance = self.schedule.instance_id
        
        if instance in ["1", "2", "3", "4", "4_1"]:
            return 0.0
        elif instance in ["5", "6", "8"]:
            return 4.0
        elif instance == "7":
            # Days 1-15: L=4, Days 16+: L=6
            return 4.0 if self.current_day <= 15 else 6.0
        else:
            # Fallback (should not happen)
            return 0.0

    def _parse_inventory_from_observation(self, observation: str, item_id: str) -> int:
        """
        Parse current total inventory (on-hand + in-transit) from observation.
        """
        try:
            lines = observation.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{item_id}"):
                    if i + 1 < len(lines):
                        inventory_line = lines[i + 1]
                        if "On-hand:" in inventory_line and "In-transit:" in inventory_line:
                            on_hand_start = inventory_line.find("On-hand:") + len("On-hand:")
                            on_hand_end = inventory_line.find(",", on_hand_start)
                            on_hand = int(inventory_line[on_hand_start:on_hand_end].strip())

                            in_transit_start = inventory_line.find("In-transit:") + len("In-transit:")
                            in_transit_str = inventory_line[in_transit_start:].strip()
                            in_transit_str = in_transit_str.replace("units", "").strip()
                            in_transit = int(in_transit_str)

                            return on_hand + in_transit
            print(f"Warning: Could not parse inventory for {item_id}, assuming 0")
            return 0
        except Exception as exc:
            print(f"Error parsing inventory for {item_id}: {exc}")
            return 0

    def _calculate_order(
        self,
        item_id: str,
        current_inventory: int,
    ) -> dict:
        """
        Calculate order quantity using clairvoyant μ̂/σ̂.

        Returns:
            Dict with keys: order, mu_hat, sigma_hat, L, z_star, q, base_stock,
                           cap (if capped), order_uncapped (if capped)
        """
        config = self.items_config[item_id]
        L = self._get_lead_time()  # Hard-coded L based on instance and current day
        p = config['profit']
        h = config['holding_cost']
        mu_hat, sigma_hat = self.schedule.get_params(self.current_day, item_id)

        q = p / (p + h)
        z_star = norm.ppf(q)
        base_stock = mu_hat + z_star * sigma_hat
        
        result = {
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
            order = max(int(math.ceil(base_stock - current_inventory)), 0)
            result['order'] = order
        else:  # capped policy
            # Vanilla order (for logging)
            order_uncapped = max(int(math.ceil(base_stock - current_inventory)), 0)
            
            # Cap calculation: μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
            cap_z = norm.ppf(0.95)
            cap = mu_hat / (1 + L) + cap_z * sigma_hat / math.sqrt(1 + L)
            
            # Capped order
            order = max(min(int(math.ceil(base_stock - current_inventory)), int(math.ceil(cap))), 0)
            
            result['order'] = order
            result['order_uncapped'] = order_uncapped
            result['cap'] = cap
        
        return result

    def update_demand_observation(self, item_id: str, observed_demand: int):
        """Track observed demand for logging purposes."""
        self.observed_demands[item_id].append(observed_demand)

    def get_action(self, observation: str) -> tuple[str, dict]:
        """
        Generate ordering decision in JSON format with detailed statistics.
        
        Returns:
            Tuple of (action_json_string, statistics_dict)
            statistics_dict contains all calculation details for logging
        """
        action_dict = {}
        rationale_parts = []
        statistics = {}

        for item_id in self.items_config:
            current_inventory = self._parse_inventory_from_observation(observation, item_id)
            calc_result = self._calculate_order(item_id, current_inventory)
            action_dict[item_id] = calc_result['order']
            statistics[item_id] = calc_result

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

        policy_name = "Clairvoyant OR (capped)" if self.policy == 'capped' else "Clairvoyant OR (vanilla)"
        rationale = f"{policy_name}: " + "; ".join(rationale_parts)
        result = {
            "action": action_dict,
            "rationale": rationale
        }
        return json.dumps(result, indent=2), statistics


def main():
    parser = argparse.ArgumentParser(description='Run Clairvoyant OR baseline with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--instance', type=str, choices=sorted(CLAIRVOYANT_SCHEDULES.keys()), default="1",
                       help='Clairvoyant benchmark instance (1, 2, 3, 4, 4_1, 5, 6, 7, 8).')
    parser.add_argument('--policy', type=str, choices=['vanilla', 'capped'], default='capped',
                       help='OR policy type: vanilla (standard base-stock) or capped (smoothed orders). Default: capped')
    args = parser.parse_args()

    schedule = ClairvoyantSchedule(args.instance)
    _safe_print(f"\nSelected instance {args.instance}: {schedule.describe()}")
    print("Lead time L is hard-coded for each instance:")
    if args.instance in ["1", "2", "3", "4", "4_1"]:
        print("  L = 0 (always)")
    elif args.instance in ["5", "6", "8"]:
        print("  L = 4 (always)")
    elif args.instance == "7":
        print("  L = 4 (days 1-15), L = 6 (days 16+)")

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

    print("\nNote: Actual lead times in CSV may differ from the hard-coded L used by OR agent.")
    print("      The hard-coded L values match the implicit lead times in the clairvoyant schedules.")

    # Set NUM_DAYS based on CSV
    from textarena.envs.VendingMachine import env as vm_env_module
    original_num_days = vm_env_module.NUM_DAYS
    vm_env_module.NUM_DAYS = csv_player.get_num_days()
    print(f"Set NUM_DAYS to {vm_env_module.NUM_DAYS} based on CSV")

    # Add news from CSV
    news_schedule = csv_player.get_news_schedule()
    for day, news in news_schedule.items():
        env.add_news(day, news)

    # Create clairvoyant OR agent (L is hard-coded based on instance)
    or_items_config = {
        config['item_id']: {
            'profit': config['profit'],
            'holding_cost': config['holding_cost']
        }
        for config in item_configs
    }
    or_agent = ClairvoyantORAgent(or_items_config, schedule, policy=args.policy)

    # Reset environment
    env.reset(num_players=2)

    # Run game
    done = False
    current_day = 1
    last_demand = {}  # Track demand to update OR agent

    while not done:
        pid, observation = env.get_observation()

        if pid == 0:  # VM agent (clairvoyant OR algorithm)
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
                if config['lead_time'] == float('inf'):
                    has_inf_lead_time = True

            # If supplier unavailable (lead_time=inf), skip OR decision
            if has_inf_lead_time:
                zero_orders = {item_id: 0 for item_id in csv_player.get_item_ids()}
                action = json.dumps({"action": zero_orders}, indent=2)

                print(f"\nWARNING Day {current_day}: Supplier unavailable (lead_time=inf)")
                print(f"Day {current_day} OR Decision: {action} (automatically set to 0)")

                done, _ = env.step(action=action)
                continue

            or_agent.set_day(current_day)
            action, stats = or_agent.get_action(observation)
            
            # Print detailed statistics for each item
            print(f"\n{'='*70}")
            print(f"Day {current_day} Clairvoyant OR Decision ({args.policy.upper()} Policy):")
            print(f"{'='*70}")
            for item_id, item_stats in stats.items():
                print(f"\n{item_id}:")
                _safe_print(f"  mu_hat (μ̂) from schedule: {item_stats['mu_hat']:.2f}")
                _safe_print(f"  sigma_hat (σ̂) from schedule: {item_stats['sigma_hat']:.2f}")
                print(f"  Lead time (L): {item_stats['L']}")
                print(f"  Critical fractile (q): {item_stats['q']:.4f}")
                print(f"  z*: {item_stats['z_star']:.4f}")
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

            demand_data = json.loads(action)
            last_demand = demand_data['action']

            print(f"\nDay {current_day} Demand: {action}")

            for item_id, qty in last_demand.items():
                or_agent.update_demand_observation(item_id, qty)

            current_day += 1

        done, _ = env.step(action=action)

    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]

    print("\n" + "="*60)
    print("=== Final Results (Clairvoyant OR Baseline) ===")
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
    print(f"\n>>> Total Reward (Clairvoyant OR Baseline): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)

    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days


if __name__ == "__main__":
    main()

