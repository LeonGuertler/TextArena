"""
Hybrid Strategy: LLM Agent + OR Algorithm Recommendation

This demo combines:
- OR Algorithm: Provides data-driven baseline recommendations (good for normal days)
- LLM Agent: Makes final decisions considering OR

The OR algorithm calculates optimal orders using base-stock policy. 
The LLM agent sees OR recommendations and adjusts based on the current situation.
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
from typing import Dict, List, Tuple
import textarena as ta
from textarena.core import Agent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()




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
    
    # Sort insights by period index
    sorted_insights = sorted(insights.items())
    
    # Build insights section at the top
    insights_section = "=" * 70 + "\n"
    insights_section += "CARRY-OVER INSIGHTS (Key Discoveries):\n"
    insights_section += "=" * 70 + "\n"
    
    for period_num, memo in sorted_insights:
        insights_section += f"Period {period_num}: {memo}\n"
    
    insights_section += "=" * 70 + "\n\n"
    
    # Prepend insights section to observation
    return insights_section + observation


_TIMELINE_TERM_SUBS = [
    (re.compile(r'\bWeek\s+(\d+)\s+concluded:'), r'Period \1 conclude:'),
    (re.compile(r'\bweek\s+(\d+)\s+concluded:'), r'period \1 conclude:'),
    (re.compile(r'\bWeeks\b'), 'Periods'),
    (re.compile(r'\bweeks\b'), 'periods'),
    (re.compile(r'\bWeek\b'), 'Period'),
    (re.compile(r'\bweek\b'), 'period'),
    (re.compile(r'\bDay\b'), 'Period'),
    (re.compile(r'\bDays\b'), 'Periods'),
]


def _normalize_timeline_terms(text: str) -> str:
    normalized = text
    for pattern, replacement in _TIMELINE_TERM_SUBS:
        normalized = pattern.sub(replacement, normalized)
    return normalized




class CSVDemandPlayer:
    """
    Simulates demand agent by reading from CSV file.
    Supports dynamic item configurations that can change per period.
    Uses exact dates (e.g., 2019-07-01) with 14-day periods.
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
        
        # Extract exact dates for each item
        self.dates = self._extract_dates()
        
        print(f"Loaded CSV with {len(self.df)} periods of demand data")
        print(f"Detected {len(self.item_ids)} items: {self.item_ids}")
        if self.dates:
            print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
    
    def _extract_item_ids(self) -> list:
        """Extract item IDs from CSV columns that start with 'demand_'."""
        item_ids = []
        for col in self.df.columns:
            if col.startswith('demand_'):
                item_id = col[len('demand_'):]
                item_ids.append(item_id)
        return item_ids
    
    def _extract_dates(self) -> list:
        """Extract dates from the first item's exact_dates column."""
        if not self.item_ids:
            return []
        first_item = self.item_ids[0]
        date_col = f'exact_dates_{first_item}'
        if date_col in self.df.columns:
            return self.df[date_col].tolist()
        return []
    
    def _validate_item_columns(self):
        """Validate that CSV has all required columns for each item."""
        # Required: exact_dates and demand columns
        for item_id in self.item_ids:
            if f'exact_dates_{item_id}' not in self.df.columns:
                raise ValueError(f"CSV missing required column: exact_dates_{item_id}")
            if f'demand_{item_id}' not in self.df.columns:
                raise ValueError(f"CSV missing required column: demand_{item_id}")
            
            # Optional: description, lead_time, profit, holding_cost (only in test.csv)
            # These are validated when accessed
    
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
                'description': str(first_row.get(f'description_{item_id}', item_id)),
                'lead_time': lead_time,
                'profit': float(first_row[f'profit_{item_id}']),
                'holding_cost': float(first_row[f'holding_cost_{item_id}'])
            }
            configs.append(config)
        
        return configs
    
    def get_period_item_config(self, period_index: int, item_id: str) -> dict:
        """
        Get item configuration for a specific period (supports dynamic changes).
        
        Args:
            period_index: Period number (1-indexed)
            item_id: Item identifier
            
        Returns:
            Dict with keys: description, lead_time, profit, holding_cost, exact_date
        """
        if period_index < 1 or period_index > len(self.df):
            raise ValueError(f"Period {period_index} out of range (1-{len(self.df)})")
        
        if item_id not in self.item_ids:
            raise ValueError(f"Unknown item_id: {item_id}")
        
        row = self.df.iloc[period_index - 1]
        
        # Get exact date
        exact_date = str(row[f'exact_dates_{item_id}'])
        
        # Handle lead_time - could be int or "inf"
        lead_time_col = f'lead_time_{item_id}'
        if lead_time_col in row:
            lead_time_val = row[lead_time_col]
            if isinstance(lead_time_val, str) and lead_time_val.lower() == 'inf':
                lead_time = float('inf')
            elif isinstance(lead_time_val, float) and lead_time_val == float('inf'):
                lead_time = float('inf')
            else:
                lead_time = int(lead_time_val)
        else:
            lead_time = 1  # Default if not specified
        
        # Get other configs (may not exist in train.csv)
        description = str(row.get(f'description_{item_id}', item_id))
        profit = float(row.get(f'profit_{item_id}', 2.0))
        holding_cost = float(row.get(f'holding_cost_{item_id}', 1.0))
        
        return {
            'description': description,
            'lead_time': lead_time,
            'profit': profit,
            'holding_cost': holding_cost,
            'exact_date': exact_date
        }
    
    def get_num_periods(self) -> int:
        """Return number of periods in CSV."""
        return len(self.df)
    
    def get_exact_date(self, period_index: int) -> str:
        """Get exact date for a specific period."""
        if period_index < 1 or period_index > len(self.df):
            return f"Period_{period_index}"
        if self.dates:
            return str(self.dates[period_index - 1])
        return f"Period_{period_index}"
    
    def get_action(self, period_index: int) -> str:
        """
        Generate buy action for given period based on CSV data in JSON format.
        
        Args:
            period_index: Current period (1-indexed)
            
        Returns:
            JSON string like '{"action": {"351484002": 622, ...}}'
        """
        # Get row for this period (period_index is 1-indexed, df is 0-indexed)
        if period_index < 1 or period_index > len(self.df):
            raise ValueError(f"Period {period_index} out of range (1-{len(self.df)})")
        
        row = self.df.iloc[period_index - 1]
        
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
    OR algorithm baseline agent using a capped base-stock policy.
    
    order_quantity = max(min(base_stock - current_inventory, cap), 0)
    where base_stock = μ̂ + z*σ̂,  cap = μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L),
    μ̂ = (1+L) × empirical_mean,  σ̂ = sqrt(1+L) × empirical_std,  z* = Φ⁻¹(q),
    q = profit / (profit + holding_cost).
    """
    
    def __init__(self, items_config: dict, initial_samples: dict = None):
        """
        Args:
            items_config: Dict of {item_id: {'lead_time': L, 'profit': p, 'holding_cost': h}}
            initial_samples: Optional dict of {item_id: [list of initial demand samples]}
                           If None or empty, will use only observed demands
        """
        self.items_config = items_config
        self.initial_samples = initial_samples if initial_samples else {}
        
        # Store observed demands (will be updated each day)
        # Format: {item_id: [demand_day1, demand_day2, ...]}
        self.observed_demands = {item_id: [] for item_id in items_config}
        
        print("\n=== OR Agent Initialized for recommendations (CAPPED POLICY) ===")
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
          chips(Regular) (...): Profit=$2/unit, Holding=$1/unit/period
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
        
        # Calculate order quantity (capped policy only)
        order_uncapped = max(int(np.ceil(base_stock - current_inventory)), 0)
        cap_z = norm.ppf(0.95)
        cap = mu_hat / (1 + L) + cap_z * sigma_hat / np.sqrt(1 + L)
        order = max(min(order_uncapped, int(np.ceil(cap))), 0)
        
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
    """Create hybrid VM agent that considers OR recommendations with exact dates."""
    
    # Extract item IDs to show in prompt
    available_items = list(initial_samples.keys()) if initial_samples else []
    items_str = ", ".join([f'"{item}"' for item in available_items])
    primary_item = available_items[0] if available_items else "item_id"
    
    system = (
        "=== ROLE & OBJECTIVE ===\n"
        f"You control the vending machine for a single SKU \"{primary_item}\" while collaborating with an OR baseline. "
        "Maximize total reward R_t = Profit × units_sold − HoldingCost × ending_inventory over total periods.\n"
        "\n"
        "=== GAME MECHANISM: PERIOD EXECUTION SEQUENCE ===\n"
        "Each period follows this strict execution order:\n"
        "  1. VM Decision Phase: You receive observation (including OR recommendation) and place orders for Period N\n"
        "  2. Arrival Resolution: Orders scheduled to arrive in Period N are added to on-hand inventory\n"
        "  3. Demand Resolution: Customer demand is satisfied from on-hand inventory\n"
        "  4. Period Conclusion: System generates 'Period N conclude' message (visible in Period N+1)\n"
        "\n"
        "Important: Steps 2-4 happen AFTER your decision. You will see their results in the next period.\n"
        "\n"
        "=== LEAD TIME DEFINITION ===\n"
        f"Promised lead time: {promised_lead_time} period(s). 'Lead time = L periods' means:\n"
        "1. Order placed in Period N's decision phase\n"
        "2. Order arrives during Period (N+L)'s arrival resolution phase\n"
        "3. Arrival becomes visible in 'Period (N+L) conclude' message\n"
        "4. You read this message at the start of Period (N+L+1)'s decision phase\n"
        "\n"
        "Note: There is always a 1-period observation delay between when orders physically arrive\n"
        "and when you can observe the arrival in the 'conclude' message.\n"
        "\n"
        "=== CRITICAL TIMING EXAMPLE ===\n"
        "SCENARIO A: Actual lead_time = 1 period\n"
        "  • Period 1: You see OR recommendation and place your order. No history yet, so no conclude to read.\n"
        "  • Period 2 START: You read 'Period 1 conclude: arrived=0'. This is NORMAL!\n"
        "    The Period 1 order arrives DURING Period 2 (after your Period 2 decision), not before.\n"
        "  • Period 3 START: You read 'Period 2 conclude: arrived=X (ordered Period 1, lead_time was 1 periods)'.\n"
        "    NOW you have confirmation that actual lead_time = 1.\n"
        "\n"
        "SCENARIO B: Actual lead_time = 0 periods (same-period arrival)\n"
        "  • Period 1: You see OR recommendation and place your order.\n"
        "  • Period 2 START: You read 'Period 1 conclude: arrived=Y (ordered Period 1, lead_time was 0 periods)'.\n"
        "    With lead_time = 0, the order arrives within the same period it was placed.\n"
        "\n"
        "KEY INSIGHT: Do NOT conclude that 'actual lead_time ≠ promised' just because 'Period N conclude' shows arrived=0.\n"
        "When actual lead_time ≥ 1, the order placed in Period N arrives DURING Period N+lead_time, and you only\n"
        "see confirmation in 'Period N+lead_time conclude' (read at Period N+lead_time+1).\n"
        "\n"
        "Lost orders never produce a conclude statement—they remain in 'In-transit' indefinitely.\n"
        "Prolonged absence (multiple periods past promised lead_time with no conclude) signals a lost shipment.\n"
        "\n"
        "=== KEY IMPLICATIONS ===\n"
        "- When deciding for Period N, you see 'Period N-1 conclude' message\n"
        "- Period N's arrivals happen during Period N but are only visible in Period N+1\n"
        "- Only use CONCLUDED period messages to infer actual lead time\n"
        "- Actual lead time may differ from promised lead time; orders may also be lost\n"
        "- Your order decision should ensure: order + on-hand + in-transit covers (L+1) periods of demand\n"
        "  (L+1 because current period's demand occurs after your decision)\n"
        "\n"
        "=== ENVIRONMENT SNAPSHOT ===\n"
        "- Period information and full history are provided.\n"
        "- Calendar dates and product descriptions may or may not be provided in context.\n"
        "- When dates are available, ACTIVELY apply calendar + world knowledge:\n"
        "  * Identify major retail/cultural calendar events\n"
        "  * Recognize seasonal demand drivers\n"
        "- When product description is available, match it to seasonal relevance.\n"
        "- When calendar dates are available, demand can spike or drop significantly around key calendar events—anticipate proactively.\n"
        "- On-hand inventory starts at 0 and incurs holding cost every period. \"In-transit\" shows total undelivered units, but you must infer ETAs.\n"
        f"- Supplier-promised lead time is {promised_lead_time} period(s); actual lead time can drift and must be inferred from CONCLUDED periods only.\n"
        "- Orders may also never CONCLUDE.\n"
        "\n"
        "=== OR BASELINE (CAPPED): MATHEMATICAL DETAILS ===\n"
        "The OR agent uses a base-stock policy with the following components:\n"
        "\n"
        "1. DEMAND ESTIMATION (from historical samples ξ₁, ξ₂, ..., ξₙ):\n"
        "   - Empirical mean: μ̄ = (1/n) Σᵢ ξᵢ\n"
        "   - Empirical std dev: σ̄ = √[(1/(n-1)) Σᵢ (ξᵢ - μ̄)²]\n"
        "   - Total demand over review+lead period (1+L):\n"
        "     μ̂ = (1 + L) × μ̄\n"
        "     σ̂ = √(1 + L) × σ̄\n"
        "   (assumes i.i.d. demands; √(1+L) from variance summation)\n"
        "\n"
        "2. CRITICAL FRACTILE & SAFETY FACTOR:\n"
        "   - Critical fractile: q = profit / (profit + holding_cost)\n"
        "   - Safety factor: z* = Φ⁻¹(q), where Φ is the standard normal CDF\n"
        "   (Higher q → higher z* → more safety stock to avoid stockouts)\n"
        "\n"
        "3. BASE STOCK LEVEL:\n"
        "   - base_stock = μ̂ + z* × σ̂\n"
        "   (Balances expected demand μ̂ with safety stock z*σ̂)\n"
        "\n"
        "4. CAPPED ORDER QUANTITY:\n"
        "   - Uncapped: order_raw = base_stock − pipeline_inventory\n"
        "   - Cap formula: cap = μ̂/(1+L) + Φ⁻¹(0.95) × σ̂/√(1+L)\n"
        "     (μ̂/(1+L) = single-period mean; Φ⁻¹(0.95)≈1.645 for 95% service)\n"
        "   - Final order: order = max(0, min(order_raw, cap))\n"
        "   (Cap smooths large swings when pipeline is low)\n"
        "\n"
        "5. OR LIMITATIONS:\n"
        "   - Uses promised lead time L (not actual observed)\n"
        "   - Uses ALL historical samples equally (no recency weighting)\n"
        "   - Cannot see lost orders, or actual arrival patterns\n"
        "   - Assumes i.i.d. demand (no regime shifts or seasonality)\n"
        "\n"
        "YOUR ROLE: The OR recommendation is a data-driven baseline. You can override it by considering:\n"
        "- Actual vs. promised lead time (from concluded periods)\n"
        "- Demand regime changes (detected from recent history)\n"
        "- Seasonality/world knowledge (from calendar dates and product description when available)\n"
        "- Lost shipments or pipeline anomalies\n"
        "\n"
        "=== COLLABORATION STRATEGY ===\n"
        "1. Read the OR recommendation (quantity + stats) and treat it as the starting point.\n"
        "2. Compare OR's assumptions to reality: inferred demand regimes, arrivals, and missing shipments.\n"
        "3. Decide whether to follow, scale, or override OR's quantity. Explain the adjustment path explicitly in your rationale.\n"
        "\n"
        "=== DEMAND & LEAD-TIME REASONING ===\n"
        "Lead-Time Inference:\n"
        "- ONLY use 'Period X conclude' messages from history to infer actual lead time\n"
        "- Message format: 'arrived=Y units (ordered on Period Z, lead_time was W periods)'\n"
        "- Actual lead time calculation: W = X - Z\n"
        "- NEVER infer lead-time from current period's observations (you haven't seen arrivals yet)\n"
        "- If orders don't arrive for many periods beyond promised lead time, they may be lost\n"
        "\n"
        "Demand Forecasting:\n"
        "- When product description and/or calendar dates are available, use them as PRIMARY forecasting anchors:\n"
        "  * What product category is this? (if description available)\n"
        "  * What time of year is it? (if dates available)\n"
        "  * Are there upcoming or recent calendar events that affect this category? (if dates available)\n"
        "- Historical samples provide initial intuition, but demand can shift suddenly\n"
        "- Combine calendar knowledge with actual demand patterns to inform your forecast\n"
        "- Confirm sustained mean/variance changes before reacting to apparent regime shifts\n"
        "\n"
    )
    
    # Add human feedback mode explanation if enabled
    if human_feedback_enabled:
        system += (
            "HUMAN-IN-THE-LOOP MODE:\n"
            "You will interact with a human supervisor in a two-stage process:\n"
            "  Stage 1: You provide your initial rationale and decision (full JSON with rationale + short_rationale_for_human + action)\n"
            "  Stage 2 (if human provides feedback): You receive the human's feedback and output ONLY the final action (no rationale needed)\n"
            "\n"
            "The human supervisor has domain expertise and may:\n"
            "  - Suggest adjustments based on information you don't have access to\n"
            "  - Point out considerations you might have missed\n"
            "  - Provide strategic insights about demand patterns\n"
            "\n"
            "IMPORTANT: The 'short_rationale_for_human' field is what the human will actually read on their screen.\n"
            "Make it concise, actionable, and focused on YOUR key decision point (e.g., 'Following OR's 450 units because demand is stable' or 'Increased OR's 300 to 400 units due to upcoming holiday').\n"
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
        system += "Use this data to inform your ordering decisions, especially in Period 1.\n\n"
    
    system += (
        "=== DECISION CHECKLIST ===\n"
        "1. When available, use world knowledge and product description to compare to historical demand for this SKU.\n"
        "2. Reconcile on-hand + pipeline with expected arrivals; highlight overdue/lost shipments.\n"
        "3. Inspect the OR recommendation (quantity + stats) and decide how to adapt it.\n"
        "4. Justify your final quantity by tying it to demand outlook, lead-time belief, and OR's baseline.\n"
        "\n"
        "=== RATIONALE GUIDELINES ===\n"
        "You must provide TWO types of rationale:\n"
        "1. FULL RATIONALE ('rationale' field): Complete step-by-step analysis covering all factors\n"
        "2. SHORT RATIONALE ('short_rationale_for_human' field): 1-3 sentences for human decision-maker\n"
        "   - Focus ONLY on your key adjustment decision\n"
        "   - Example: 'Following OR recommendation of 450 units - demand pattern is stable'\n"
        "   - Example: 'Increased OR's 300 to 400 units - anticipating 30% holiday demand surge'\n"
        "   - Example: 'Reduced OR's 500 to 350 units - recent demand dropped 25% and lead time is shorter than expected'\n"
        "   - BE SPECIFIC with numbers and reasons\n"
        "\n"
        "=== CARRY-OVER INSIGHTS ===\n"
        "- Only record NEW, non-duplicated insights for sustained, evidence-backed shifts (demand mean/variance, lead time, seasonality) that future decisions must remember.\n"
        "- Stay conservative: if the observation is already captured or not yet material, leave the field empty instead of restating it.\n"
        "- Provide concrete stats (date ranges, averages, lead-time values). List multiple insights separated by '; ' or newline. Remove insights once they stop being true. Default output is \"\".\n"
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
        "=== OUTPUT FORMAT ===\n"
        "Return valid JSON only:\n"
        "{\n"
        '  "rationale": "Explain step by step: lead-time inference, inventory & demand analysis, final strategy.",\n'
        '  "short_rationale_for_human": "Brief summary (1-3 sentences) explaining your key reasoning: why you adjusted OR recommendations or why you followed them unchanged.",\n'
        '  "carry_over_insight": "Summaries of NEW sustained changes with evidence, or \\"\\".",\n'
        f'  "action": {{{example_action}}}\n'
        "}\n"
        f"Use the exact item ID(s) when writing \"action\" (current ID(s): {items_str or primary_item}). "
        "No extra commentary outside the JSON."
    )
    # return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)
    return GPT5MiniAgent(system_prompt=system)


def main():
    parser = argparse.ArgumentParser(description='Run hybrid strategy (LLM + OR) with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time used by OR and shown to LLM in periods (default: 0). Actual lead time in CSV may differ.')
    parser.add_argument('--human-feedback', action='store_true',
                       help='Enable periodic human feedback on agent decisions (Mode 1)')
    parser.add_argument('--guidance-frequency', type=int, default=0,
                       help='Collect strategic guidance every N periods (Mode 2). 0=disabled')
    parser.add_argument('--real-instance-train', type=str, default=None,
                       help='Path to train.csv for real instances (extracts initial samples). If not provided, uses default unified samples.')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Maximum number of periods to run (limits NUM_DAYS). If None, uses all periods from CSV.')
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
            # Extract demand samples from train.csv (H&M format: exact_dates_{item_id}, demand_{item_id})
            item_ids = csv_player.get_item_ids()
            if item_ids:
                first_item = item_ids[0]
                demand_col = f'demand_{first_item}'
                if demand_col in train_df.columns:
                    train_samples = train_df[demand_col].tolist()
                    initial_samples = {item_id: train_samples for item_id in item_ids}
                    print(f"\nUsing initial samples from train.csv: {args.real_instance_train}")
                    print(f"  Samples: {train_samples}")
                    print(f"  Mean: {sum(train_samples)/len(train_samples):.1f}, Count: {len(train_samples)}")
                else:
                    raise ValueError(f"Column {demand_col} not found in train.csv")
            else:
                raise ValueError("No items detected in test CSV")
        except Exception as e:
            print(f"Error loading train.csv: {e}")
            print("Falling back to default unified samples")
            unified_samples = [112, 97, 116, 138, 94]
            initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    else:
        # Use default unified samples for synthetic instances
        unified_samples = [112, 97, 116, 138, 94]
        initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
        print(f"\nUsing default unified initial samples: {unified_samples}")
    
    print(f"Promised lead time (used by OR, shown to LLM): {args.promised_lead_time} periods")
    print(f"Note: OR uses promised lead time for recommendations. Actual lead times in CSV may differ.")
    print(f"      LLM must infer actual lead time from arrivals and adjust OR recommendations accordingly.")
    
    # Determine number of periods to run
    total_periods = csv_player.get_num_periods()
    num_periods = total_periods
    if args.max_periods is not None:
        num_periods = min(args.max_periods, total_periods)
        print(f"Limiting run to {num_periods} periods (CSV has {total_periods})")
    else:
        print(f"Running full CSV horizon: {num_periods} periods")
    
    # Create OR agent (for recommendations only - uses promised lead time)
    or_items_config = {
        config['item_id']: {
            'lead_time': args.promised_lead_time,  # Use promised value instead of CSV value
            'profit': config['profit'],
            'holding_cost': config['holding_cost']
        }
        for config in item_configs
    }
    or_agent = ORAgent(or_items_config, initial_samples)
    
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
    
    # Reset environment with explicit horizon
    env.reset(num_players=2, num_days=num_periods, initial_inventory_per_item=0)
    
    # Run game
    done = False
    current_period = 1
    last_demand = {}  # Track demand to update OR agent
    carry_over_insights = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (Hybrid: LLM + OR)
            # Get exact date for current period
            exact_date = csv_player.get_exact_date(current_period)
            
            # Inject exact date into observation's CURRENT STATUS section
            # Format is "PERIOD N / TOTAL", use robust regex (case-insensitive, flexible whitespace)
            import re
            period_pattern = re.compile(
                rf'PERIOD\s+{current_period}\s+/\s+\d+',
                re.IGNORECASE
            )
            observation = period_pattern.sub(
                f'PERIOD {current_period} (Date: {exact_date}) / {csv_player.get_num_periods()}',
                observation
            )
            
            # Inject exact dates into GAME HISTORY section
            if "=== GAME HISTORY ===" in observation or "GAME HISTORY" in observation:
                for p in range(1, current_period):
                    p_date = csv_player.get_exact_date(p)
                    # Use robust regex (case-insensitive, flexible whitespace)
                    history_pattern = re.compile(
                        rf'Period\s+{p}\s+conclude:',
                        re.IGNORECASE
                    )
                    observation = history_pattern.sub(
                        f'Period {p} (Date: {p_date}) conclude:',
                        observation
                    )
            
            observation = _normalize_timeline_terms(observation)
            observation = inject_carry_over_insights(observation, carry_over_insights)
            # Update item configurations for current period (supports dynamic changes)
            for item_id in csv_player.get_item_ids():
                config = csv_player.get_period_item_config(current_period, item_id)
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
            print(f"Period {current_period} ({exact_date}) OR ALGORITHM RECOMMENDATIONS (CAPPED POLICY):")
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
                print(f"  Cap value: {item_stats['cap']:.2f}")
                print(f"  OR recommends (capped): {item_stats['order']}")
                print(f"  OR recommends (uncapped): {item_stats['order_uncapped']}")
            print(f"{'='*70}\n")
            
            # Format OR recommendations for LLM
            or_text = "\n" + "="*60 + "\n"
            or_text += "OR ALGORITHM RECOMMENDATIONS (capped policy):\n"
            for item_id, rec_qty in or_recommendations.items():
                or_text += f"  {item_id}: {rec_qty} units\n"
            or_text += "\nNote: OR uses the promised lead time and historical demand only.\n"
            or_text += "It cannot see lost shipments, or actual lead-time shifts—adjust accordingly.\n"
            or_text += "="*60 + "\n"
            
            # Enhance observation with OR recommendations
            enhanced_observation = observation + or_text
            
            # LLM agent makes final decision
            action = vm_agent(enhanced_observation)
            
            # Print complete JSON output with proper formatting
            print(f"\nPeriod {current_period} ({exact_date}) Hybrid Decision:")
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
                    carry_over_insights[current_period] = carry_memo
                    _safe_print(f"Carry-over insight: {carry_memo}")
                else:
                    if current_period in carry_over_insights:
                        del carry_over_insights[current_period]
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
            exact_date = csv_player.get_exact_date(current_period)
            action = csv_player.get_action(current_period)
            
            # Parse demand to update OR agent's history
            demand_data = json.loads(action)
            last_demand = demand_data['action']
            
            print(f"\nPeriod {current_period} ({exact_date}) Demand: {action}")
            
            # Update OR agent with observed demand
            for item_id, qty in last_demand.items():
                or_agent.update_demand_observation(item_id, qty)
            
            current_period += 1
        
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
        print(f"  Profit/unit: ${profit}, Holding: ${holding_cost}/unit/period")
        print(f"  Total Profit: ${total_profit}")
    
    # Period breakdown
    print("\n" + "="*60)
    print("Period Breakdown:")
    print("="*60)
    for day_log in vm_info.get('daily_logs', []):
        period = day_log['day']
        exact_date = csv_player.get_exact_date(period)
        profit = day_log['daily_profit']
        holding = day_log['daily_holding_cost']
        reward = day_log['daily_reward']
        
        print(f"Period {period} ({exact_date}): Profit=${profit:.2f}, Holding=${holding:.2f}, Reward=${reward:.2f}")
    
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
    
if __name__ == "__main__":
    main()











