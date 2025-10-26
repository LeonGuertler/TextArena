"""
LLM→OR Strategy: LLM Proposes OR Parameters

This demo uses:
- LLM Agent: Analyzes game state and proposes OR algorithm parameters (L, μ̂, σ̂)
- OR Calculator: Uses LLM-proposed parameters to compute optimal orders

The LLM evaluates current conditions (inventory, news, demand patterns) and
selects appropriate parameter estimation methods or explicit values. The backend
then computes orders using the standard OR base-stock formula.

Usage:
  python llm_to_or_csv_demo.py --demand-file path/to/demands.csv
"""

import os
import sys
import argparse
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple
import textarena as ta


DAY_CONCLUDED_PATTERN = re.compile(r'^(\s*Day\s+(\d+)\s+concluded:)(.*)$')


def inject_carry_over_insights(observation: str, insights: Dict[int, str]) -> str:
    """Insert previously recorded insights into matching day summaries."""
    if not insights:
        return observation

    lines = observation.splitlines()
    augmented: List[str] = []

    for line in lines:
        match = DAY_CONCLUDED_PATTERN.match(line)
        if match:
            day_num = int(match.group(2))
            memo = insights.get(day_num)
            if memo:
                if "Insight:" in match.group(3):
                    augmented.append(line)
                else:
                    augmented.append(f"{match.group(1)}{match.group(3)} | Insight: {memo}")
                continue
        augmented.append(line)

    return "\n".join(augmented)


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


# ============================================================================
# Helper Functions for Parsing and Computation
# ============================================================================

def parse_total_inventory(observation: str, item_id: str) -> int:
    """
    Parse total inventory (on-hand + in-transit) from observation for a specific item.
    
    Observation format:
      chips(Regular) (...): Profit=$2/unit, Holding=$1/unit/day
        On-hand: 5, In-transit: 20 units
    
    Returns:
        Total inventory across all pipeline stages (on-hand + in-transit)
    """
    try:
        lines = observation.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{item_id}"):
                # Next line should have the inventory info
                if i + 1 < len(lines):
                    inventory_line = lines[i + 1]
                    
                    # Parse: "  On-hand: 5, In-transit: 20 units"
                    if "On-hand:" in inventory_line and "In-transit:" in inventory_line:
                        # Extract on-hand value
                        on_hand_match = re.search(r'On-hand:\s*(\d+)', inventory_line)
                        # Extract in-transit value
                        in_transit_match = re.search(r'In-transit:\s*(\d+)', inventory_line)
                        
                        if on_hand_match and in_transit_match:
                            on_hand = int(on_hand_match.group(1))
                            in_transit = int(in_transit_match.group(1))
                            total_inventory = on_hand + in_transit
                            return total_inventory
        
        print(f"Warning: Could not parse inventory for {item_id}, assuming 0")
        return 0
    except Exception as e:
        print(f"Warning: Could not parse inventory for {item_id}: {e}")
        return 0


def parse_arrivals_from_history(observation: str) -> Dict[str, List[int]]:
    """
    Parse observed lead times from arrival records in game history.
    
    Looks for patterns like: "arrived=X units (ordered on Day Y, lead_time was Z days)"
    
    Returns:
        Dict of {item_id: [list of observed lead times]}
    """
    observed_lead_times = {}
    
    # Look for patterns like "chips(Regular): ordered=X, arrived=Y units (ordered on Day Z, lead_time was W days)"
    # More specific pattern to avoid matching "concluded:" or other false positives
    # Pattern: item_id: ordered=... lead_time was X days
    pattern = r'(\S+?):\s+ordered=.*?lead_time was (\d+) day'
    
    matches = re.findall(pattern, observation)
    for item_id, lead_time_str in matches:
        lead_time = int(lead_time_str)
        if item_id not in observed_lead_times:
            observed_lead_times[item_id] = []
        observed_lead_times[item_id].append(lead_time)
    
    return observed_lead_times


def compute_L(method: str, params: dict, observed_lead_times: List[int], promised_lead_time: float) -> float:
    """
    Compute lead time L based on method.
    
    Args:
        method: "default", "calculate", or "explicit"
        params: Dict that may contain "value" for explicit method
        observed_lead_times: List of observed lead times from arrivals
        promised_lead_time: The promised lead time from supplier
        
    Returns:
        Computed lead time value
        
    Raises:
        ValueError: If method is invalid or required data is missing
    """
    if method == "default":
        return promised_lead_time
    elif method == "calculate":
        if not observed_lead_times:
            raise ValueError("Cannot calculate lead time: no observed arrivals yet")
        return float(np.mean(observed_lead_times))
    elif method == "explicit":
        if "value" not in params:
            raise ValueError("Method 'explicit' for L requires 'value' field")
        return float(params["value"])
    else:
        raise ValueError(f"Invalid method for L: {method}")


def compute_mu_hat(method: str, params: dict, samples: List[float], L: float) -> float:
    """
    Compute expected demand μ̂ based on method.
    
    Args:
        method: "default", "recent_N", "EWMA_gamma", or "explicit"
        params: Dict that may contain "N", "gamma", or "value"
        samples: List of historical demand samples
        L: Lead time value
        
    Returns:
        Computed μ̂ value
        
    Raises:
        ValueError: If method is invalid or required data is missing
    """
    if method == "explicit":
        if "value" not in params:
            raise ValueError("Method 'explicit' for mu_hat requires 'value' field")
        return float(params["value"])
    
    # For non-explicit methods, we need samples
    if not samples:
        return 0.0  # No samples yet, return 0
    
    if method == "default":
        empirical_mean = np.mean(samples)
        return (1 + L) * empirical_mean
    elif method == "recent_N":
        if "N" not in params:
            raise ValueError("Method 'recent_N' for mu_hat requires 'N' field")
        N = int(params["N"])
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        recent_samples = samples[-N:] if len(samples) >= N else samples
        empirical_mean = np.mean(recent_samples)
        return (1 + L) * empirical_mean
    elif method == "EWMA_gamma":
        if "gamma" not in params:
            raise ValueError("Method 'EWMA_gamma' for mu_hat requires 'gamma' field")
        gamma = float(params["gamma"])
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        
        # EWMA: (1+L) × (ξ_{t-1} + γ×ξ_{t-2} + γ²×ξ_{t-3} + ...) / (1 + γ + γ² + ...)
        # ξ_t are the samples in reverse chronological order
        numerator = 0.0
        denominator = 0.0
        for i, sample in enumerate(reversed(samples)):
            weight = gamma ** i
            numerator += weight * sample
            denominator += weight
        
        if denominator == 0:
            return 0.0
        
        weighted_mean = numerator / denominator
        return (1 + L) * weighted_mean
    else:
        raise ValueError(f"Invalid method for mu_hat: {method}")


def compute_sigma_hat(method: str, params: dict, samples: List[float], L: float) -> float:
    """
    Compute standard deviation σ̂ based on method.
    
    Args:
        method: "default", "recent_N", or "explicit"
        params: Dict that may contain "N" or "value"
        samples: List of historical demand samples
        L: Lead time value
        
    Returns:
        Computed σ̂ value
        
    Raises:
        ValueError: If method is invalid or required data is missing
    """
    if method == "explicit":
        if "value" not in params:
            raise ValueError("Method 'explicit' for sigma_hat requires 'value' field")
        return float(params["value"])
    
    # For non-explicit methods, we need samples
    if not samples or len(samples) < 2:
        return 0.0  # Not enough samples, return 0
    
    if method == "default":
        empirical_std = np.std(samples, ddof=1)
        return np.sqrt(1 + L) * empirical_std
    elif method == "recent_N":
        if "N" not in params:
            raise ValueError("Method 'recent_N' for sigma_hat requires 'N' field")
        N = int(params["N"])
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        recent_samples = samples[-N:] if len(samples) >= N else samples
        if len(recent_samples) < 2:
            return 0.0
        empirical_std = np.std(recent_samples, ddof=1)
        return np.sqrt(1 + L) * empirical_std
    else:
        raise ValueError(f"Invalid method for sigma_hat: {method}")


def validate_parameters_json(params_json: dict, item_ids: List[str], current_configs: Dict[str, dict]):
    """
    Validate the parameters JSON structure and required fields.
    
    Args:
        params_json: The parsed JSON from LLM
        item_ids: List of expected item IDs
        current_configs: Dict of {item_id: config} with current item configurations
        
    Raises:
        ValueError: If JSON structure is invalid or required fields are missing
    """
    if "parameters" not in params_json:
        raise ValueError("JSON must contain 'parameters' field")
    
    parameters = params_json["parameters"]
    
    # Check all items are present
    for item_id in item_ids:
        if item_id not in parameters:
            raise ValueError(f"Missing parameters for item: {item_id}")
        
        item_params = parameters[item_id]
        
        # Check L parameter
        if "L" not in item_params:
            raise ValueError(f"Missing 'L' parameter for item {item_id}")
        L_param = item_params["L"]
        if "method" not in L_param:
            raise ValueError(f"Missing 'method' in L parameter for item {item_id}")
        
        L_method = L_param["method"]
        if L_method not in ["default", "calculate", "explicit"]:
            raise ValueError(f"Invalid L method for item {item_id}: {L_method}")
        if L_method == "explicit" and "value" not in L_param:
            raise ValueError(f"Method 'explicit' for L requires 'value' field for item {item_id}")
        
        # Check mu_hat parameter
        if "mu_hat" not in item_params:
            raise ValueError(f"Missing 'mu_hat' parameter for item {item_id}")
        mu_param = item_params["mu_hat"]
        if "method" not in mu_param:
            raise ValueError(f"Missing 'method' in mu_hat parameter for item {item_id}")
        
        mu_method = mu_param["method"]
        if mu_method not in ["default", "recent_N", "EWMA_gamma", "explicit"]:
            raise ValueError(f"Invalid mu_hat method for item {item_id}: {mu_method}")
        if mu_method == "recent_N" and "N" not in mu_param:
            raise ValueError(f"Method 'recent_N' for mu_hat requires 'N' field for item {item_id}")
        if mu_method == "EWMA_gamma" and "gamma" not in mu_param:
            raise ValueError(f"Method 'EWMA_gamma' for mu_hat requires 'gamma' field for item {item_id}")
        if mu_method == "explicit" and "value" not in mu_param:
            raise ValueError(f"Method 'explicit' for mu_hat requires 'value' field for item {item_id}")
        
        # Check sigma_hat parameter
        if "sigma_hat" not in item_params:
            raise ValueError(f"Missing 'sigma_hat' parameter for item {item_id}")
        sigma_param = item_params["sigma_hat"]
        if "method" not in sigma_param:
            raise ValueError(f"Missing 'method' in sigma_hat parameter for item {item_id}")
        
        sigma_method = sigma_param["method"]
        if sigma_method not in ["default", "recent_N", "explicit"]:
            raise ValueError(f"Invalid sigma_hat method for item {item_id}: {sigma_method}")
        if sigma_method == "recent_N" and "N" not in sigma_param:
            raise ValueError(f"Method 'recent_N' for sigma_hat requires 'N' field for item {item_id}")
        if sigma_method == "explicit" and "value" not in sigma_param:
            raise ValueError(f"Method 'explicit' for sigma_hat requires 'value' field for item {item_id}")


# ============================================================================
# LLM Agent Creation
# ============================================================================

def make_llm_to_or_agent(initial_samples: dict, current_configs: dict, 
                         promised_lead_time: int,
                         human_feedback_enabled: bool = False, 
                         guidance_enabled: bool = False):
    """
    Create LLM agent that proposes OR parameters.
    
    Args:
        initial_samples: Dict of {item_id: [samples]}
        current_configs: Dict of {item_id: config} with current item configurations
        promised_lead_time: The lead time promised by supplier (shown to LLM)
        human_feedback_enabled: Whether human feedback mode is enabled
        guidance_enabled: Whether guidance mode is enabled
    """
    system = (
        "You are the Vending Machine controller (VM) using an LLM→OR strategy.\n"
        "Your role: Analyze the current situation and propose parameters for an OR algorithm.\n"
        "The OR algorithm will use your parameters to compute optimal orders.\n"
        "\n"
        "Objective: Maximize total reward = sum of daily rewards R_t.\n"
        "Daily reward: R_t = Profit × Sold - HoldingCost × EndingInventory.\n"
        "\n"
        "=== CURRENT ITEM CONFIGURATIONS ===\n"
    )
    
    # Add current item configurations (promised lead time, profit, holding cost)
    for item_id, config in current_configs.items():
        profit = config['profit']
        holding_cost = config['holding_cost']
        description = config.get('description', item_id)
        
        system += f"\n{item_id} ({description}):\n"
        system += f"  Supplier-promised lead time: {promised_lead_time} days\n"
        system += f"  Profit: ${profit}/unit\n"
        system += f"  Holding cost: ${holding_cost}/unit/day\n"
    
    system += (
        "\n=== KEY MECHANICS ===\n"
        "- Actual lead time may differ from promised lead time and may change over time\n"
        "- You can infer actual lead time from arrival records: 'arrived=X units (ordered on Day Y, lead_time was Z days)'\n"
        "- Inventory visibility:\n"
        "  * On-hand: Current inventory available for sale today\n"
        "  * In-transit: Total units ordered that haven't arrived yet\n"
        "  * ⚠️ Initial inventory on Day 1: Each item starts with 0 units on-hand\n"
        "- Holding cost is charged on ending inventory each day\n"
        "- carry_over_insight: Reserve this for clear, sustained shifts or structural changes that future days must remember; otherwise output \"\"\n"
        "- News events may affect future demand\n"
        "\n"
    )
    
    # Add human feedback mode explanation if enabled
    if human_feedback_enabled:
        system += (
            "=== HUMAN-IN-THE-LOOP MODE ===\n"
            "You will interact with a human supervisor in a two-stage process:\n"
            "  Stage 1: You provide your initial rationale and parameters\n"
            "  Stage 2 (if human provides feedback): You receive feedback and output only final parameters\n"
            "\n"
        )
    
    # Add guidance mode explanation if enabled
    if guidance_enabled:
        system += (
            "=== STRATEGIC GUIDANCE ===\n"
            "You may receive strategic guidance from a human supervisor.\n"
            "This guidance will appear at the top of your observations.\n"
            "\n"
        )
    
    # Add historical demand data
    if initial_samples:
        system += "=== HISTORICAL DEMAND DATA ===\n"
        for item_id, samples in initial_samples.items():
            system += f"{item_id}:\n"
            system += f"  Past demands: {samples}\n"
        system += "\n"
    
    system += (
        "=== OR ALGORITHM PARAMETERS ===\n"
        "You must propose three parameters for each item:\n"
        "\n"
        "1. L (lead time for current order):\n"
        "   - default: Use the supplier-promised lead time shown above\n"
        "   - calculate: Use average of all observed lead times from past arrivals\n"
        "   - explicit: Provide your own predicted value\n"
        "   Example: {\"method\": \"calculate\"} or {\"method\": \"explicit\", \"value\": 3}\n"
        "\n"
        "2. mu_hat (expected total demand over lead time period):\n"
        "   - default: (1+L) × mean of all historical samples\n"
        "   - recent_N: (1+L) × mean of last N samples (must specify N)\n"
        "   - EWMA_gamma: (1+L) × exponentially weighted moving average (must specify gamma in [0,1])\n"
        "   - explicit: Provide your own prediction\n"
        "   Example: {\"method\": \"recent_N\", \"N\": 5} or {\"method\": \"explicit\", \"value\": 250}\n"
        "   When you choose recent_N, think about trend strength:\n"
        "     • Strong, freshly emerging trends or news-driven shifts → favor a smaller N that focuses on only the most recent days.\n"
        "     • Mild fluctuations or noisy data with no clear trend → favor a larger N to smooth noise.\n"
        "     • Ensure N does not exceed the number of available samples; if it would, adjust the window or pick another method.\n"
        "\n"
        "3. sigma_hat (standard deviation of demand over lead time period):\n"
        "   - default: sqrt(1+L) × std of all historical samples\n"
        "   - recent_N: sqrt(1+L) × std of last N samples (must specify N)\n"
        "   - explicit: Provide your own prediction\n"
        "   Example: {\"method\": \"default\"} or {\"method\": \"explicit\", \"value\": 15}\n"
        "\n"
        "The OR algorithm will compute orders using:\n"
        "  order = max(mu_hat + Φ^(-1)(q) × sigma_hat - pipeline_inventory, 0)\n"
        "  where q = profit / (profit + holding_cost) [critical fractile]\n"
        "\n"
        "=== YOUR STRATEGY ===\n"
        "1. Analyze current inventory (on-hand + in-transit)\n"
        "2. Review demand patterns from game history\n"
        "3. Infer actual lead time from arrival records\n"
        "4. Consider today's news and its potential impact\n"
        "5. Choose appropriate parameter methods:\n"
        "   - Use 'default' or 'calculate' for stable conditions\n"
        "   - Use 'recent_N' to react to trend changes\n"
        "   - Use 'explicit' when you have strong predictions (e.g., news impact)\n"
        "   - When using 'recent_N', justify the window length (trend duration, news horizon, volatility) and remember carry_over_insight stays empty unless that shift truly persists\n"
        "6. Explain your reasoning clearly\n"
        "\n"
        "=== OUTPUT FORMAT ===\n"
        "You MUST respond with valid JSON in this EXACT format:\n"
        "{\n"
        "  \"rationale\": \"Explain your analysis and parameter choices for each item\",\n"
        "  \"carry_over_insight\": \"Short memo for future days (\"\" when no durable change is present)\",\n"
        "  \"parameters\": {\n"
        "    \"item_id_1\": {\n"
        "      \"L\": {\"method\": \"...\", \"value\": ...},\n"
        "      \"mu_hat\": {\"method\": \"...\", \"N\": ..., \"gamma\": ..., \"value\": ...},\n"
        "      \"sigma_hat\": {\"method\": \"...\", \"N\": ..., \"value\": ...}\n"
        "    },\n"
        "    \"item_id_2\": { ... }\n"
        "  }\n"
        "}\n"
        "\n"
        "IMPORTANT:\n"
        "- Include ONLY the fields required for each method\n"
        "- For 'recent_N', include 'N' (integer >= 1)\n"
        "- For 'EWMA_gamma', include 'gamma' (float in [0, 1])\n"
        "- For 'explicit', include 'value' (numeric)\n"
        "- Do NOT include any text outside the JSON\n"
        "- Think carefully about your parameter choices\n"
    )
    
    return ta.agents.OpenAIAgent(model_name="gpt-4o", system_prompt=system, temperature=0)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LLM→OR strategy with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time to show to LLM (default: 1). This is what supplier promises, not the actual lead time in CSV.')
    parser.add_argument('--human-feedback', action='store_true',
                       help='Enable daily human feedback on agent decisions (Mode 1)')
    parser.add_argument('--guidance-frequency', type=int, default=0,
                       help='Collect strategic guidance every N days (Mode 2). 0=disabled')
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
    
    # Generate initial demand samples for all items (unified across all products)
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    print(f"\nUsing unified initial samples for all items: {unified_samples}")
    print(f"Promised lead time (shown to LLM): {args.promised_lead_time} days")
    print(f"Note: Actual lead times in CSV may differ and will be inferred by LLM from arrivals.")
    
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
    
    # Initialize tracking data structures
    observed_demands = {item_id: list(initial_samples[item_id]) for item_id in csv_player.get_item_ids()}
    observed_lead_times = {item_id: [] for item_id in csv_player.get_item_ids()}
    current_item_configs = {
        config['item_id']: {
            'lead_time': config['lead_time'],
            'profit': config['profit'],
            'holding_cost': config['holding_cost'],
            'description': config['description']
        }
        for config in item_configs
    }
    
    # Create LLM agent
    base_agent = make_llm_to_or_agent(
        initial_samples=initial_samples,
        current_configs=current_item_configs,
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
    carry_over_insights: Dict[int, str] = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (LLM→OR)
            observation = inject_carry_over_insights(observation, carry_over_insights)
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
                # Update tracking
                current_item_configs[item_id] = config
                # Check if any item has lead_time=inf (supplier unavailable)
                if config['lead_time'] == float('inf'):
                    has_inf_lead_time = True
            
            # If supplier unavailable (lead_time=inf), skip LLM→OR decision
            if has_inf_lead_time:
                # Create action with order=0 for all items
                zero_orders = {item_id: 0 for item_id in csv_player.get_item_ids()}
                action = json.dumps({"action": zero_orders}, indent=2)
                
                print(f"\nWARNING Day {current_day}: Supplier unavailable (lead_time=inf)")
                print("VM did not place order (automatically set to 0)")
                print("\n" + "="*70)
                print("Final Order Action:")
                print(action)
                print("="*70)
                sys.stdout.flush()
                
                # Skip to demand turn
                done, _ = env.step(action=action)
                continue
            
            # Parse observed lead times from arrivals in history
            new_lead_times = parse_arrivals_from_history(observation)
            for item_id, lt_list in new_lead_times.items():
                observed_lead_times[item_id].extend(lt_list)
            
            # Recreate agent with updated configs (for system prompt)
            base_agent = make_llm_to_or_agent(
                initial_samples=initial_samples,
                current_configs=current_item_configs,
                promised_lead_time=args.promised_lead_time,
                human_feedback_enabled=args.human_feedback,
                guidance_enabled=(args.guidance_frequency > 0)
            )
            
            if args.human_feedback or args.guidance_frequency > 0:
                vm_agent = ta.agents.HumanFeedbackAgent(
                    base_agent=base_agent,
                    enable_daily_feedback=args.human_feedback,
                    guidance_frequency=args.guidance_frequency
                )
            else:
                vm_agent = base_agent
            
            # Get LLM response
            llm_response = vm_agent(observation)
            
            # Parse JSON response
            try:
                # Clean markdown code fences
                cleaned_response = llm_response.strip()
                cleaned_response = re.sub(r'^```(?:json)?\s*', '', cleaned_response)
                cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
                
                params_json = json.loads(cleaned_response)
                
                # Validate JSON structure
                validate_parameters_json(params_json, csv_player.get_item_ids(), current_item_configs)
                
                carry_memo = params_json.get("carry_over_insight")
                if isinstance(carry_memo, str):
                    carry_memo = carry_memo.strip()
                else:
                    carry_memo = None
                if carry_memo:
                    carry_over_insights[current_day] = carry_memo
                elif current_day in carry_over_insights:
                    del carry_over_insights[current_day]
                
                print(f"\nDay {current_day} LLM→OR Decision:")
                print("="*70)
                print("LLM Rationale:")
                print(params_json.get("rationale", "(no rationale provided)"))
                print("\n" + "="*70)
                
            except json.JSONDecodeError as e:
                print(f"\nERROR: Failed to parse LLM output as JSON: {e}")
                print(f"Raw output:\n{llm_response}")
                sys.exit(1)
            except ValueError as e:
                print(f"\nERROR: Invalid parameter specification: {e}")
                print(f"Raw output:\n{llm_response}")
                sys.exit(1)
            
            # Compute orders using OR formula with LLM-proposed parameters
            orders = {}
            
            for item_id in csv_player.get_item_ids():
                item_params = params_json["parameters"][item_id]
                config = current_item_configs[item_id]
                
                print(f"\n{item_id}:")
                
                try:
                    # Compute L
                    L = compute_L(
                        method=item_params["L"]["method"],
                        params=item_params["L"],
                        observed_lead_times=observed_lead_times[item_id],
                        promised_lead_time=args.promised_lead_time
                    )
                    print(f"  L method: {item_params['L']['method']}, computed L = {L:.2f}")
                    
                    # Compute mu_hat
                    mu_hat = compute_mu_hat(
                        method=item_params["mu_hat"]["method"],
                        params=item_params["mu_hat"],
                        samples=observed_demands[item_id],
                        L=L
                    )
                    print(f"  mu_hat method: {item_params['mu_hat']['method']}, computed mu_hat = {mu_hat:.2f}")
                    
                    # Compute sigma_hat
                    sigma_hat = compute_sigma_hat(
                        method=item_params["sigma_hat"]["method"],
                        params=item_params["sigma_hat"],
                        samples=observed_demands[item_id],
                        L=L
                    )
                    print(f"  sigma_hat method: {item_params['sigma_hat']['method']}, computed sigma_hat = {sigma_hat:.2f}")
                    
                    # Get total inventory (on-hand + in-transit)
                    total_inventory = parse_total_inventory(observation, item_id)
                    print(f"  Total inventory (on-hand + in-transit): {total_inventory}")
                    
                    # Compute critical fractile
                    p = config['profit']
                    h = config['holding_cost']
                    q = p / (p + h)
                    z_star = norm.ppf(q)
                    print(f"  Critical fractile q = {q:.4f}, z* = {z_star:.4f}")
                    
                    # Compute order using OR formula
                    base_stock = mu_hat + z_star * sigma_hat
                    order = max(int(np.ceil(base_stock - total_inventory)), 0)
                    print(f"  Base stock = {base_stock:.2f}")
                    print(f"  Computed order: {order}")
                    
                    orders[item_id] = order
                    
                except ValueError as e:
                    print(f"  ERROR computing order: {e}")
                    sys.exit(1)
            
            # Create action JSON
            action_json = {"action": orders}
            action = json.dumps(action_json, indent=2)
            
            print("\n" + "="*70)
            print("Final Order Action:")
            print(action)
            print("="*70)
            sys.stdout.flush()
            
        else:  # Demand from CSV
            action = csv_player.get_action(current_day)
            
            # Parse demand to update observed demands
            demand_data = json.loads(action)
            for item_id, qty in demand_data['action'].items():
                observed_demands[item_id].append(qty)
            
            print(f"\nDay {current_day} Demand: {action}")
            current_day += 1
        
        done, _ = env.step(action=action)
    
    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]
    
    print("\n" + "="*70)
    print("=== Final Results (LLM→OR Strategy) ===")
    print("="*70)
    
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
    print("\n" + "="*70)
    print("Daily Breakdown:")
    print("="*70)
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
    
    print("\n" + "="*70)
    print("=== TOTAL SUMMARY ===")
    print("="*70)
    print(f"Total Profit from Sales: ${total_profit:.2f}")
    print(f"Total Holding Cost: ${total_holding:.2f}")
    print(f"\n>>> Total Reward (LLM→OR Strategy): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*70)
    
    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = original_initial_inventory


if __name__ == "__main__":
    main()

