"""
Hybrid Strategy: LLM Agent + OR Algorithm Recommendation

This demo combines:
- OR Algorithm: Provides data-driven baseline recommendations (good for normal days)
- LLM Agent: Makes final decisions considering OR + news schedule (reacts to events)

The OR algorithm calculates optimal orders using base-stock policy but cannot
react to news. The LLM agent sees OR recommendations and adjusts based on news.

Usage:
  python agent&or_csv_demo.py --demand-file path/to/demands.csv
"""

import os
import sys
import argparse
import json
import re
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import norm
import textarena as ta
from textarena.core import Agent


WEEK_CONCLUDED_PATTERN = re.compile(r'^(\s*Week\s+(\d+)\s+concluded:)(.*)$')


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
    print(_sanitize_text(str(text)))


class GPT5MiniAgent(Agent):
    """Lightweight agent wrapper that uses the OpenAI Responses API with gpt-5-mini."""

    def __init__(
        self,
        system_prompt: str,
        reasoning_effort: str = "low",
        text_verbosity: str = "low",
    ):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package is required for GPT5MiniAgent. Install it with: pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.model_name = "gpt-5-mini"
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.client = OpenAI(api_key=api_key)

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")

        request_payload = {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": observation}]},
            ],
        }

        if self.reasoning_effort:
            request_payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.text_verbosity:
            request_payload["text"] = {"verbosity": self.text_verbosity}

        response = self.client.responses.create(**request_payload)
        return response.output_text.strip()


def inject_carry_over_insights(observation: str, insights: dict) -> str:
    """
    Insert carry-over insights at the top of observation.
    
    Format:
    ========================================
    CARRY-OVER INSIGHTS (Key Discoveries):
    ========================================
    Day 5: Demand increased by 50% after Day 3 sports event (avg: 100->150)
    Day 12: Lead time changed from 2 to 4 days starting Day 10
    ========================================
    
    [Original Observation]
    """
    if not insights:
        return observation
    
    # Sort insights by week number
    sorted_insights = sorted(insights.items())
    
    # Build insights section at the top
    insights_section = "=" * 70 + "\n"
    insights_section += "CARRY-OVER INSIGHTS (Key Discoveries):\n"
    insights_section += "=" * 70 + "\n"
    
    for week_num, memo in sorted_insights:
        insights_section += f"Week {week_num}: {memo}\n"
    
    insights_section += "=" * 70 + "\n\n"
    
    # Prepend insights section to observation
    return insights_section + observation


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
        
        print(f"Loaded CSV with {len(self.df)} weeks of demand data")
        print(f"Detected {len(self.item_ids)} items: {self.item_ids}")
        if self.has_news:
            news_weeks = self.df[self.df['news'].notna()]['week'].tolist()
            print(f"News scheduled for weeks: {news_weeks}")
    
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
        Get item configuration for a specific week (supports dynamic changes).
        
        Args:
            day: Week number (1-indexed)
            item_id: Item identifier
            
        Returns:
            Dict with keys: description, lead_time, profit, holding_cost
        """
        if day < 1 or day > len(self.df):
            raise ValueError(f"Week {day} out of range (1-{len(self.df)})")
        
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
            week = int(row['week'])
            if pd.notna(row['news']) and str(row['news']).strip():
                news_schedule[week] = str(row['news']).strip()
        
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

    mu_hat = (1+L) * empirical_mean
    sigma_hat = sqrt(1+L) * empirical_std
    z* = Phi^(-1)(q), where q = profit / (profit + holding_cost)
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
        
        print(f"\n=== OR Agent Initialized for recommendations (Policy: {policy.upper()}) ===")
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
            _safe_print(f"  z* = Phi^(-1)(q): {z_star:.4f}")
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
        
        # Calculate mu_hat and sigma_hat for lead time + review period
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
    
    def get_recommendation(self, observation: str) -> tuple[dict, dict]:
        """
        Generate OR algorithm recommendations with detailed statistics.
        
        Args:
            observation: Current game observation
            
        Returns:
            Tuple of (recommendations_dict, statistics_dict)
            recommendations_dict: {"item_id": order_quantity, ...}
            statistics_dict: {"item_id": {calculation_details}, ...}
        """
        recommendations = {}
        statistics = {}
        
        for item_id in self.items_config:
            # Parse current inventory from observation
            current_inventory = self._parse_inventory_from_observation(observation, item_id)
            
            # Calculate order quantity with full statistics
            calc_result = self._calculate_order(item_id, current_inventory)
            recommendations[item_id] = calc_result['order']
            statistics[item_id] = calc_result
        
        return recommendations, statistics


def make_hybrid_vm_agent(initial_samples: dict = None, promised_lead_time: int = 0,
                         human_feedback_enabled: bool = False, guidance_enabled: bool = False):
    """Create hybrid VM agent that considers both OR recommendations and news."""
    
    # Extract item IDs to show in prompt
    available_items = list(initial_samples.keys()) if initial_samples else []
    items_str = ", ".join([f'"{item}"' for item in available_items])
    
    system = (
        "You are the Vending Machine controller (VM). "
        "You manage multiple items, each with unit profit and holding costs. "
        "Objective: Maximize total reward = sum of weekly rewards R_t. "
        "Weekly reward: R_t = Profit * Sold - HoldingCost * EndingInventory. "
        "\n\n"
        f"AVAILABLE ITEMS: {items_str}\n"
        "WARNING: You MUST use these EXACT item IDs (with parentheses and all special characters) in your action!\n"
        "\n"
        "Key mechanics:\n"
        f"- Supplier-promised lead time: {promised_lead_time} weeks\n"
        "- Orders placed this week arrive after a LEAD TIME (number of weeks until delivery)\n"
        "- IMPORTANT: Actual lead time may differ from promised and may change over time!\n"
        "- Lead time is NOT directly revealed. You must INFER it from arrival records.\n"
        "- When goods arrive, you'll see: 'arrived=X units (ordered on Week Y, lead_time was Z weeks)'\n"
        "- Use this information to track actual lead time and adjust your strategy\n"
        "\n"
        "Inventory visibility:\n"
        "- On-hand: Current inventory available for sale this week\n"
        "- In-transit: Total units you ordered that haven't arrived yet (but you don't know WHEN they'll arrive)\n"
        "- You must track your own orders and infer when they'll arrive based on inferred lead_time\n"
        "- IMPORTANT: Initial inventory on Week 1: Each item starts with 0 units on-hand\n"
        "\n"
        "- Holding cost is charged on ending inventory each week\n"
        "- Weekly sequence: order submission happens first, then any scheduled shipments arrive, and customer demand is realized last\n"
        "- News events may affect future demand\n"
        "\n"
        "NEWS INFORMATION:\n"
        "- You must analyze whether these events correlate with demand changes\n"
        "- Not all news necessarily impact demand - use historical data to assess\n"
        "- If no news is present for a week, the field will be empty\n"
        "\n"
    )
    
    # Add human feedback mode explanation if enabled
    if human_feedback_enabled:
        system += (
            "HUMAN-IN-THE-LOOP MODE:\n"
            "You will interact with a human supervisor in a two-stage process:\n"
            "  Stage 1: You provide your initial rationale and decision (full JSON with rationale + action)\n"
            "  Stage 2 (if human provides feedback): You receive the human's feedback and output ONLY the final action (no rationale needed)\n"
            "\n"
            "The human supervisor has domain expertise and may:\n"
            "  - Suggest adjustments based on information you don't have access to\n"
            "  - Point out considerations you might have missed\n"
            "  - Provide strategic insights about demand patterns\n"
            "\n"
            "When you receive human feedback in Stage 2, incorporate it thoughtfully and output only the action JSON.\n"
            "\n"
        )
    
    # Add guidance mode explanation if enabled
    if guidance_enabled:
        system += (
            "STRATEGIC GUIDANCE:\n"
            "You may receive strategic guidance from a human supervisor that should inform your decisions. "
            "This guidance will appear at the top of your observations and should be followed consistently.\n"
            "\n"
        )
    
    # Add historical demand data if provided
    if initial_samples:
        system += "HISTORICAL DEMAND DATA (for reference):\n"
        system += "You have access to the following historical demand samples to help you estimate future demand:\n\n"
        for item_id, samples in initial_samples.items():
            system += f"{item_id}:\n"
            system += f"  Past demands: {samples}\n"
            system += "\n"
        system += "Use this data to inform your ordering decisions, especially on Week 1.\n\n"
    
    system += (
        "OR ALGORITHM ASSISTANCE:\n"
        "You will receive recommendations from an Operations Research (OR) algorithm each week.\n"
        "The OR algorithm uses a base-stock policy based on statistical analysis:\n"
        "- It calculates optimal orders using historical demand patterns\n"
        f"- IMPORTANT: OR uses the PROMISED lead time ({promised_lead_time} weeks) in its calculations\n"
        "- Uses profit and holding cost in its calculations\n"
        "- Aims to balance service level (meeting demand) vs holding costs\n"
        "- Works well for NORMAL, STABLE demand patterns\n"
        "\n"
        "IMPORTANT LIMITATIONS:\n"
        "1. The OR algorithm CANNOT react to news events. It only looks at historical data.\n"
        "2. The OR algorithm uses PROMISED lead time, not actual lead time.\n"
        "This is where YOU come in!\n"
        "\n"
        "Your Strategy:\n"
        "1. INFER actual lead time from arrival records in game history (look for 'lead_time was X weeks')\n"
        "2. Compare inferred actual lead time with promised lead time to detect discrepancies\n"
        "3. Track your own orders and when they should arrive based on inferred actual lead_time\n"
        "4. Use 'In-transit' to see total goods coming, but remember you must infer WHEN they arrive\n"
        "5. Use OR recommendation as your BASELINE for normal weeks (OR uses statistical analysis)\n"
        "6. Adjust OR recommendations if actual lead time differs from promised lead time\n"
        "7. React to THIS WEEK'S NEWS as it happens, considering inferred actual lead time\n"
        "8. Learn from past news events to understand their impact on demand\n"
        "9. Balance between data-driven OR approach and news-driven/lead-time-adjusted insights\n"
        "\n"
        "CARRY-OVER INSIGHTS:\n"
        "- If carry-over insights exist, they will appear at the TOP of your observation in a dedicated section.\n"
        "- Focus on the MOST RECENT insights as trends evolve over time. Older insights may be outdated.\n"
        "- Use insights as quick references, but always verify against current game data.\n"
        "\n"
        "STRICT RULES for writing carry_over_insight:\n"
        "⚠️ DEFAULT: Return empty string \"\" (most weeks should have NO new insight)\n"
        "\n"
        "ONLY write a new insight when ALL of these conditions are met:\n"
        "  1. You observe a SIGNIFICANT, SUSTAINED change (not temporary fluctuation):\n"
        "     - Demand mean shift (e.g., sustained 30%+ change over 3+ weeks)\n"
        "     - Variance pattern change (e.g., volatility doubled/halved)\n"
        "     - Lead time structural change (e.g., changed from 2 to 4 weeks)\n"
        "     - Major news impact with lasting effect\n"
        "\n"
        "  2. You have CONCRETE EVIDENCE with specific numbers:\n"
        "     - Week ranges (e.g., \"Weeks 8-12 avg: 150 vs Weeks 1-7 avg: 100\")\n"
        "     - Statistical measures (mean, std, lead_time values)\n"
        "     - Specific news events and their timing\n"
        "\n"
        "  3. NO similar insight exists in CARRY-OVER INSIGHTS section above:\n"
        "     - Check if the change is already documented\n"
        "     - If updating an existing insight, reference the old one\n"
        "     - If change is temporary/reversed, note that it ended\n"
        "\n"
        "  4. The insight will be USEFUL for future decisions (not just describing history)\n"
        "\n"
        "EXAMPLES of when to write:\n"
        "  ✅ \"Demand increased 50% after Week 5 sports event; new baseline: 150 units (was 100)\"\n"
        "  ✅ \"Lead time changed from 2 to 4 weeks starting Week 10 (observed in Weeks 10-13 arrivals)\"\n"
        "  ✅ \"Variance doubled after Week 15; demand now fluctuates 80-220 (was 90-110)\"\n"
        "  ❌ \"This week's demand was high\" (not sustained, no evidence)\n"
        "  ❌ \"Sales continue as before\" (no change, unnecessary)\n"
        "  ❌ \"Demand is volatile\" (already documented in previous insight)\n"
        "\n"
        "Remember: Insights are for PERSISTENT changes only. Temporary fluctuations go in rationale, not insights.\n"
        "\n"
        "Example decision process:\n"
        "- Step 1: Check recent arrivals to infer current lead_time (e.g., 'lead_time was 2 weeks')\n"
        "- Step 2: Review OR recommends: 200 units\n"
        "- Step 3: THIS WEEK'S NEWS: Major sports event finale\n"
        "- Step 4: Your analysis: Sports events may increase demand significantly in 2 weeks (lead_time!)\n"
        "- Step 5: Your decision: Adjust order upward based on your reasoning\n"
        "\n"
    )
    
    # Create example format with actual item IDs
    if available_items:
        example_action = ", ".join([f'"{item}": 100' for item in available_items[:2]])  # Show up to 2 items
        if len(available_items) > 2:
            example_action += ", ..."
    else:
        example_action = '"item_id": quantity, ...'
    
    system += (
        "IMPORTANT: Think step by step, then decide.\n"
        "You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "rationale": "First, explain your reasoning: (1) infer current lead_time from recent arrivals, '
        '(2) review OR algorithm recommendations, '
        '(3) analyze current inventory (on-hand + in-transit) and demand patterns, '
        '(4) evaluate this week\'s news and learn from past events, '
        '(5) consider lead_time when adjusting OR recommendations (goods arrive after lead_time!), '
        '(6) decide: follow OR baseline or adjust based on news/analysis, '
        '(7) explain your final ordering strategy",\n'
        '  "carry_over_insight": "Only if NEW sustained change observed with specific evidence; otherwise \"\" (must check if already exists above)",\n'
        f'  "action": {{{example_action}}}\n'
        "}\n"
        "\n"
        f"REMEMBER: Use EXACT item IDs: {items_str}\n"
        "\n"
        "Think through your rationale BEFORE making the final order decision.\n"
        "Do NOT include any other text outside the JSON."
    )
    # return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)
    return GPT5MiniAgent(system_prompt=system)


def main():
    parser = argparse.ArgumentParser(description='Run hybrid strategy (LLM + OR) with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time used by OR and shown to LLM (default: 0). Actual lead time in CSV may differ.')
    parser.add_argument('--policy', type=str, choices=['vanilla', 'capped'], default='capped',
                       help='OR policy type: vanilla (standard base-stock) or capped (smoothed orders). Default: capped')
    parser.add_argument('--human-feedback', action='store_true',
                       help='Enable daily human feedback on agent decisions (Mode 1)')
    parser.add_argument('--guidance-frequency', type=int, default=0,
                       help='Collect strategic guidance every N days (Mode 2). 0=disabled')
    parser.add_argument('--real-instance-train', type=str, default=None,
                       help='Path to train.csv for real instances (extracts initial samples from weeks 1-10). If not provided, uses default unified samples.')
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print('Example: export OPENAI_API_KEY="sk-your-key-here"')
        sys.exit(1)
    
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
    
    # Generate initial demand samples
    if args.real_instance_train:
        # Load from real instance train.csv
        try:
            train_df = pd.read_csv(args.real_instance_train)
            # Use all weeks (1-10) from train.csv
            train_samples = train_df[train_df['week_number'] >= 1]['demand'].tolist()
            initial_samples = {item_id: train_samples for item_id in csv_player.get_item_ids()}
            print(f"\nUsing initial samples from real instance train.csv: {args.real_instance_train}")
            print(f"  Samples (weeks 1-10): {train_samples}")
            print(f"  Mean: {sum(train_samples)/len(train_samples):.1f}, Count: {len(train_samples)}")
        except Exception as e:
            print(f"Error loading train.csv: {e}")
            print("Falling back to default unified samples")
            unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
            initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    else:
        # Use default unified samples for synthetic instances
        unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
        initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
        print(f"\nUsing default unified initial samples: {unified_samples}")
    
    print(f"Promised lead time (used by OR, shown to LLM): {args.promised_lead_time} days")
    print(f"Note: OR uses promised lead time for recommendations. Actual lead times in CSV may differ.")
    print(f"      LLM must infer actual lead time from arrivals and adjust OR recommendations accordingly.")
    
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
    
    # Create OR agent (for recommendations only - uses promised lead time)
    or_items_config = {
        config['item_id']: {
            'lead_time': args.promised_lead_time,  # Use promised value instead of CSV value
            'profit': config['profit'],
            'holding_cost': config['holding_cost']
        }
        for config in item_configs
    }
    or_agent = ORAgent(or_items_config, initial_samples, policy=args.policy)
    
    # Create hybrid VM agent
    base_agent = make_hybrid_vm_agent(
        initial_samples=initial_samples,
        promised_lead_time=args.promised_lead_time,
        human_feedback_enabled=args.human_feedback,
        guidance_enabled=(args.guidance_frequency > 0)
    )
    
    # Wrap with HumanFeedbackAgent if human-in-the-loop modes are enabled
    if args.human_feedback or args.guidance_frequency > 0:
        print("\n" + "="*70)
        print("HUMAN-IN-THE-LOOP MODE ACTIVATED")
        print("="*70)
        if args.human_feedback:
            print("Mode 1: Daily feedback on agent decisions is ENABLED")
        if args.guidance_frequency > 0:
            print(f"Mode 2: Strategic guidance every {args.guidance_frequency} days is ENABLED")
        print("="*70 + "\n")
        
        vm_agent = ta.agents.HumanFeedbackAgent(
            base_agent=base_agent,
            enable_daily_feedback=args.human_feedback,
            guidance_frequency=args.guidance_frequency
        )
    else:
        vm_agent = base_agent
    
    # Reset environment
    env.reset(num_players=2)
    
    # Run game
    done = False
    current_day = 1
    last_demand = {}  # Track demand to update OR agent
    carry_over_insights = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (Hybrid: LLM + OR)
            observation = inject_carry_over_insights(observation, carry_over_insights)
            # Update item configurations for current day (supports dynamic changes)
            for item_id in csv_player.get_item_ids():
                config = csv_player.get_day_item_config(current_day, item_id)
                env.update_item_config(
                    item_id=item_id,
                    lead_time=config['lead_time'],
                    profit=config['profit'],
                    holding_cost=config['holding_cost'],
                    description=config['description']
                )
            
            # Get OR recommendations with statistics
            or_recommendations, or_stats = or_agent.get_recommendation(observation)
            
            # Print detailed OR statistics
            print(f"\n{'='*70}")
            print(f"OR ALGORITHM RECOMMENDATIONS ({args.policy.upper()} Policy):")
            print(f"{'='*70}")
            for item_id, item_stats in or_stats.items():
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
                    print(f"  OR recommends (capped): {item_stats['order']}")
                    print(f"  OR recommends (uncapped): {item_stats['order_uncapped']}")
                else:
                    print(f"  OR recommends: {item_stats['order']}")
            print(f"{'='*70}\n")
            
            # Format OR recommendations for LLM
            or_text = "\n" + "="*60 + "\n"
            or_text += f"OR ALGORITHM RECOMMENDATIONS ({args.policy} policy):\n"
            for item_id, rec_qty in or_recommendations.items():
                or_text += f"  {item_id}: {rec_qty} units\n"
            or_text += "\nNote: OR algorithm uses statistical base-stock policy based on historical demand.\n"
            or_text += "It does NOT consider news events. You should adjust if needed.\n"
            or_text += "="*60 + "\n"
            
            # Enhance observation with OR recommendations
            enhanced_observation = observation + or_text
            
            # LLM agent makes final decision
            action = vm_agent(enhanced_observation)
            
            # Print complete JSON output with proper formatting
            print(f"\nDay {current_day} Hybrid Decision:")
            print("="*60)
            try:
                # Remove markdown code block markers if present
                # Strip markdown code fences (```json or ``` at start/end)
                cleaned_action = action.strip()
                # Remove ```json or ``` from the beginning
                cleaned_action = re.sub(r'^```(?:json)?\s*', '', cleaned_action)
                # Remove ``` from the end
                cleaned_action = re.sub(r'\s*```$', '', cleaned_action)
                
                # Parse and pretty print
                action_dict = json.loads(cleaned_action)
                
                formatted_json = json.dumps(action_dict, indent=2, ensure_ascii=False)
                _safe_print(formatted_json)
                
                carry_memo = action_dict.get("carry_over_insight")
                if isinstance(carry_memo, str):
                    carry_memo = carry_memo.strip()
                else:
                    carry_memo = None
                
                if carry_memo:
                    carry_over_insights[current_day] = carry_memo
                    _safe_print(f"Carry-over insight: {carry_memo}")
                else:
                    if current_day in carry_over_insights:
                        del carry_over_insights[current_day]
                    print("Carry-over insight: (empty)")
                
                # Flush to ensure complete output to file
                sys.stdout.flush()
            except Exception as e:
                # Fallback to raw output if JSON parsing fails
                print(f"[DEBUG: JSON parsing failed: {e}]")
                _safe_print(action)
                sys.stdout.flush()
            print("="*60)
            print(f"  (OR recommended: {or_recommendations})")
            sys.stdout.flush()
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
    print("=== Final Results (Hybrid: LLM + OR) ===")
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
    print(f"\n>>> Total Reward (Hybrid Strategy): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)
    
    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = original_initial_inventory


if __name__ == "__main__":
    main()











