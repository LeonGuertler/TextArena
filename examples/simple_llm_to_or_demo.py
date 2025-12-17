"""
Simple LLM->OR Strategy: LLM as Pure Demand Forecaster

This simplified demo uses:
- LLM Agent: ONLY predicts single-period demand (no inventory/game knowledge)
- OR Calculator: Uses LLM prediction to compute optimal orders via capped base-stock policy

The LLM receives only demand-relevant information (historical demands, dates, product description)
and outputs a single demand prediction. The backend then computes:
  μ̂ = (1 + L) × predicted_demand
  σ̂ = √(1+L) × empirical_std (default method)

Usage:
  python simple_llm_to_or_demo.py --demand-file path/to/test.csv --real-instance-train path/to/train.csv
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
from typing import Dict, List, Tuple, Any, Optional
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


# ============================================================================
# GPT5MiniAgent Class (from original)
# ============================================================================

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


# ============================================================================
# Utility Functions
# ============================================================================

def inject_carry_over_insights(observation: str, insights: Dict[int, str]) -> str:
    """
    Insert carry-over insights at the top of observation.
    
    Format:
    ======================================================================
    CARRY-OVER INSIGHTS (Key Discoveries):
    ======================================================================
    Period 5: Demand regime shift: avg increased from 280 to 365 (+30%)
    Period 12: Seasonal pattern: December shows 40% higher demand
    ======================================================================
    
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


def robust_parse_json(text: str, required_field: str = "predicted_demand") -> dict:
    """
    Robustly parse JSON from LLM output, attempting to fix common formatting errors.
    Handles cases where LLM outputs multiple concatenated JSON objects (restart mid-stream).
    """
    # Step 1: Clean markdown code fences
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Step 2: Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try to find ALL potential JSON objects and pick the valid one
    # This handles cases where LLM restarts and outputs multiple JSONs concatenated
    brace_positions = []
    for i, char in enumerate(cleaned):
        if char == '{':
            brace_positions.append(('open', i))
        elif char == '}':
            brace_positions.append(('close', i))
    
    # Try to extract valid JSON objects, prioritizing ones with required field
    valid_jsons = []
    for i, (type_i, pos_i) in enumerate(brace_positions):
        if type_i != 'open':
            continue
        # Try pairing with each subsequent closing brace
        for j, (type_j, pos_j) in enumerate(brace_positions[i+1:], start=i+1):
            if type_j != 'close':
                continue
            candidate = cleaned[pos_i:pos_j + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    valid_jsons.append(parsed)
                    # If this JSON has the required field, prefer it immediately
                    if required_field and required_field in parsed:
                        return parsed
            except json.JSONDecodeError:
                continue
    
    # Return the last valid JSON if any (often the complete one when LLM restarts)
    if valid_jsons:
        # Prefer one with required field, else return last one
        for j in reversed(valid_jsons):
            if required_field and required_field in j:
                return j
        return valid_jsons[-1]
    
    # Step 4: Try first-to-last brace extraction with repair
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace:last_brace + 1]
        # Remove trailing commas
        repaired = re.sub(r',(\s*[}\]])', r'\1', json_candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    
    # Step 5: Final attempt - remove trailing commas on full text
    repaired = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON after repair attempts. Original error: {e.msg}",
            e.doc,
            e.pos
        )


# ============================================================================
# CSVDemandPlayer Class (from original, unchanged)
# ============================================================================

class CSVDemandPlayer:
    """
    Simulates demand agent by reading from CSV file.
    Supports dynamic item configurations that can change per period.
    """
    def __init__(self, csv_path: str, initial_samples: dict = None):
        """
        Args:
            csv_path: Path to CSV file
            initial_samples: Optional dict of {item_id: [historical demand samples]}
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
        for item_id in self.item_ids:
            if f'exact_dates_{item_id}' not in self.df.columns:
                raise ValueError(f"CSV missing required column: exact_dates_{item_id}")
            if f'demand_{item_id}' not in self.df.columns:
                raise ValueError(f"CSV missing required column: demand_{item_id}")
    
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
        """Get initial item configurations from first row of CSV."""
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
        """Get item configuration for a specific period."""
        if period_index < 1 or period_index > len(self.df):
            raise ValueError(f"Period {period_index} out of range (1-{len(self.df)})")
        
        if item_id not in self.item_ids:
            raise ValueError(f"Unknown item_id: {item_id}")
        
        row = self.df.iloc[period_index - 1]
        exact_date = str(row[f'exact_dates_{item_id}'])
        
        # Handle lead_time
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
            lead_time = 1
        
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
        """Generate buy action for given period based on CSV data in JSON format."""
        if period_index < 1 or period_index > len(self.df):
            raise ValueError(f"Period {period_index} out of range (1-{len(self.df)})")
        
        row = self.df.iloc[period_index - 1]
        
        action_dict = {}
        for item_id in self.item_ids:
            col_name = f'demand_{item_id}'
            qty = int(row[col_name])
            action_dict[item_id] = qty
        
        result = {"action": action_dict}
        return json.dumps(result, indent=2)


# ============================================================================
# Demand Forecaster Agent (NEW)
# ============================================================================

def make_demand_forecaster_agent(
    item_id: str,
    description: str,
    initial_samples: List[int],
) -> GPT5MiniAgent:
    """
    Create LLM agent that ONLY forecasts demand.
    No knowledge of inventory, lead time, or game mechanics.
    """
    
    system = (
        "=== ROLE & OBJECTIVE ===\n"
        f"You are a demand forecaster for SKU \"{item_id}\". "
        "Your ONLY task is to predict the demand for the NEXT period based on "
        "historical demand data, calendar dates, and product information.\n"
        "\n"
        "=== PRODUCT INFORMATION ===\n"
        f"{description}\n"
        "\n"
        "=== HISTORICAL DEMAND DATA (Training Period) ===\n"
        f"Past demands: {initial_samples}\n"
        "\n"
        "=== FORECASTING GUIDELINES ===\n"
        "Calendar dates and product descriptions may or may not be provided in context.\n"
        "\n"
        "1. When dates are available, ACTIVELY apply calendar + world knowledge:\n"
        "   - Identify major retail/cultural calendar events\n"
        "   - Recognize seasonal demand drivers\n"
        "   - Demand can spike or drop significantly around key calendar events—anticipate proactively\n"
        "2. When product description is available:\n"
        "   - Match product category to seasonal relevance\n"
        "3. Detect trends:\n"
        "   - Look for sustained increases or decreases in demand\n"
        "   - Identify regime changes (sudden shifts in mean or variance)\n"
        "4. Compare recent vs historical:\n"
        "   - Weight recent observations appropriately\n"
        "   - Validate apparent changes with sufficient evidence\n"
        "\n"
        "=== CARRY-OVER INSIGHTS ===\n"
        "This is a critical mechanism for cross-period memory.\n"
        "\n"
        "PURPOSE: Record NEW, sustained, actionable demand pattern shifts that "
        "future periods must remember for accurate forecasting.\n"
        "\n"
        "WHAT TO RECORD:\n"
        "- Confirmed demand regime changes (mean/variance shifts)\n"
        "- Seasonal patterns with evidence (e.g., 'Holiday demand spike confirmed')\n"
        "- Trend confirmations (e.g., 'Upward trend: avg increased from 280 to 365')\n"
        "- Any observation helpful for future demand prediction\n"
        "\n"
        "FORMAT REQUIREMENTS:\n"
        "- Include concrete numerical evidence (date ranges, averages, percentages)\n"
        "- **CRITICAL - BE CONSERVATIVE**: Only record if the signal is SIGNIFICANT and SUSTAINED "
        "(at least 3+ periods of consistent evidence). When in doubt, output empty string.\n"
        "- Do NOT repeat insights already captured in previous periods\n"
        "- If multiple changes exist, separate with '; ' or newline\n"
        "- Retire/update insights when they no longer hold\n"
        "- Output empty string \"\" if no new significant pattern detected\n"
        "\n"
        "EXAMPLES:\n"
        "- \"Demand regime shift at Period 5: avg increased from 280 to 365 (+30%)\"\n"
        "- \"Seasonal peak confirmed: Dec weeks show 40% higher demand\"\n"
        "- \"Downward trend since Period 10: weekly decline of ~5%\"\n"
        "- \"\" (empty - no new pattern)\n"
        "\n"
        "=== OUTPUT FORMAT ===\n"
        "Return EXACTLY ONE valid JSON object, nothing else:\n"
        "{\n"
        '  "rationale": "Your step-by-step demand analysis: (1) historical pattern summary, '
        '(2) seasonal/calendar considerations for the current date, (3) recent trend analysis, (4) final forecast logic.",\n'
        '  "carry_over_insight": "NEW sustained demand pattern with evidence, or empty string.",\n'
        '  "predicted_demand": <integer: your forecast for next period demand>\n'
        "}\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the JSON object above - no text before or after\n"
        "- Do NOT restart or output multiple JSON objects\n"
        "- predicted_demand MUST be a numeric integer (e.g., 50), NOT a word (e.g., 'Fifty')\n"
        "- Focus ONLY on demand forecasting - you have no knowledge of inventory or orders\n"
    )
    
    return GPT5MiniAgent(system_prompt=system)


# ============================================================================
# Context Builder (NEW)
# ============================================================================

def build_forecasting_context(
    current_period: int,
    current_date: str,
    total_periods: int,
    item_id: str,
    observed_demands: List[Tuple[int, str, int]],  # [(period, date, demand), ...]
    initial_samples: List[int],
) -> str:
    """
    Build demand-only context for LLM forecaster.
    NO inventory, NO lead time, NO game mechanics.
    """
    lines = []
    
    # Current status
    lines.append("=== CURRENT FORECASTING TASK ===")
    lines.append(f"Current Period: {current_period}")
    lines.append(f"Current Date: {current_date}")
    lines.append(f"Total Periods: {total_periods}")
    lines.append(f"Task: Predict demand for Period {current_period}")
    lines.append("")
    
    # Test phase observed demands
    if observed_demands:
        lines.append("=== OBSERVED DEMAND (Test Phase) ===")
        for period, date, demand in observed_demands:
            lines.append(f"  Period {period} ({date}): {demand} units")
        lines.append("")
    
    lines.append("Provide your demand forecast as JSON.")
    
    return "\n".join(lines)


# ============================================================================
# Inventory Parser (needed for OR computation)
# ============================================================================

def parse_total_inventory(observation: str, item_id: str) -> int:
    """
    Parse total inventory (on-hand + in-transit) from environment observation.
    """
    try:
        lines = observation.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{item_id}"):
                if i + 1 < len(lines):
                    inventory_line = lines[i + 1]
                    if "On-hand:" in inventory_line and "In-transit:" in inventory_line:
                        on_hand_match = re.search(r'On-hand:\s*(\d+)', inventory_line)
                        in_transit_match = re.search(r'In-transit:\s*(\d+)', inventory_line)
                        
                        if on_hand_match and in_transit_match:
                            on_hand = int(on_hand_match.group(1))
                            in_transit = int(in_transit_match.group(1))
                            return on_hand + in_transit
        
        print(f"Warning: Could not parse inventory for {item_id}, assuming 0")
        return 0
    except Exception as e:
        print(f"Warning: Could not parse inventory for {item_id}: {e}")
        return 0


# ============================================================================
# OR Computation (Simplified)
# ============================================================================

def compute_order_from_prediction(
    predicted_demand: float,
    promised_lead_time: int,
    all_samples: List[int],
    profit: float,
    holding_cost: float,
    total_inventory: int,
) -> Dict[str, Any]:
    """
    Compute capped base-stock order from LLM's demand prediction.
    
    Args:
        predicted_demand: LLM's forecast for single-period demand
        promised_lead_time: L from command line argument
        all_samples: All historical demands for sigma calculation
        profit: Item profit per unit
        holding_cost: Item holding cost per unit per period
        total_inventory: Current on-hand + in-transit inventory
    
    Returns:
        Dict with all computation details for logging
    """
    L = promised_lead_time
    
    # mu_hat from LLM prediction
    mu_hat = (1 + L) * predicted_demand
    
    # sigma_hat using default method (all samples)
    if len(all_samples) > 1:
        empirical_std = np.std(all_samples, ddof=1)
    else:
        empirical_std = 0.0
    sigma_hat = np.sqrt(1 + L) * empirical_std
    
    # Critical fractile
    q = profit / (profit + holding_cost)
    z_star = norm.ppf(q)
    
    # Base stock
    base_stock = mu_hat + z_star * sigma_hat
    
    # Uncapped order
    order_uncapped = max(int(np.ceil(base_stock - total_inventory)), 0)
    
    # Cap calculation
    cap_z = norm.ppf(0.95)
    if L >= 0 and (1 + L) > 0:
        cap = mu_hat / (1 + L) + cap_z * sigma_hat / np.sqrt(1 + L)
    else:
        cap = predicted_demand + cap_z * empirical_std
    
    # Final capped order
    order = max(min(order_uncapped, int(np.ceil(cap))), 0)
    
    return {
        "predicted_demand": predicted_demand,
        "L": L,
        "mu_hat": mu_hat,
        "empirical_std": empirical_std,
        "sigma_hat": sigma_hat,
        "q": q,
        "z_star": z_star,
        "base_stock": base_stock,
        "total_inventory": total_inventory,
        "cap": cap,
        "order_uncapped": order_uncapped,
        "order": order,
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Simple LLM->OR strategy (LLM as demand forecaster)')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data (test.csv)')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time used by OR algorithm in periods (default: 0)')
    parser.add_argument('--real-instance-train', type=str, default=None,
                       help='Path to train.csv for initial samples. If not provided, uses default samples.')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Maximum number of periods to run. If None, uses all periods from CSV.')
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print('Example: export OPENAI_API_KEY="sk-your-key-here"')
        sys.exit(1)
    
    # Create environment
    env = ta.make(env_id="VendingMachine-v0")
    
    # Load CSV demand player
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
    
    # Load initial samples from train.csv
    initial_samples = {}
    
    if args.real_instance_train:
        try:
            train_df = pd.read_csv(args.real_instance_train)
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
        unified_samples = [112, 97, 116, 138, 94]
        initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
        print(f"\nUsing default unified initial samples: {unified_samples}")
    
    print(f"Promised lead time (used by OR): {args.promised_lead_time} periods")
    
    # Determine number of periods to run
    total_periods = csv_player.get_num_periods()
    num_periods = total_periods
    if args.max_periods is not None:
        num_periods = min(args.max_periods, total_periods)
        print(f"Limiting run to {num_periods} periods (CSV has {total_periods})")
    else:
        print(f"Running full CSV horizon: {num_periods} periods")
    
    # Get item description for agent
    item_ids = csv_player.get_item_ids()
    primary_item = item_ids[0]
    first_config = csv_player.get_period_item_config(1, primary_item)
    item_description = first_config['description']
    
    # Create demand forecaster agent
    forecaster_agent = make_demand_forecaster_agent(
        item_id=primary_item,
        description=item_description,
        initial_samples=initial_samples[primary_item],
    )
    
    # Initialize tracking
    observed_demands: Dict[str, List[Tuple[int, str, int]]] = {item_id: [] for item_id in item_ids}
    all_demand_samples: Dict[str, List[int]] = {item_id: list(initial_samples[item_id]) for item_id in item_ids}
    carry_over_insights: Dict[int, str] = {}
    
    # Get current configs for OR computation
    current_item_configs = {
        config['item_id']: {
            'lead_time': config['lead_time'],
            'profit': config['profit'],
            'holding_cost': config['holding_cost'],
            'description': config['description']
        }
        for config in item_configs
    }
    
    # Reset environment
    env.reset(num_players=2, num_days=num_periods, initial_inventory_per_item=0)
    
    # Run game
    done = False
    current_period = 1
    
    print(f"\n{'='*70}")
    print("SIMPLE LLM->OR STRATEGY (LLM as Demand Forecaster)")
    print(f"{'='*70}\n")
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (LLM forecaster -> OR)
            exact_date = csv_player.get_exact_date(current_period)
            
            # Update item configurations for current period
            for item_id in item_ids:
                config = csv_player.get_period_item_config(current_period, item_id)
                env.update_item_config(
                    item_id=item_id,
                    lead_time=config['lead_time'],
                    profit=config['profit'],
                    holding_cost=config['holding_cost'],
                    description=config['description']
                )
                current_item_configs[item_id] = config
            
            # Build forecasting context (demand-only, no inventory info)
            forecasting_context = build_forecasting_context(
                current_period=current_period,
                current_date=exact_date,
                total_periods=csv_player.get_num_periods(),
                item_id=primary_item,
                observed_demands=observed_demands[primary_item],
                initial_samples=initial_samples[primary_item],
            )
            
            # Inject carry-over insights
            forecasting_context = inject_carry_over_insights(forecasting_context, carry_over_insights)
            
            # Get LLM forecast
            llm_response = forecaster_agent(forecasting_context)
            
            # Parse JSON response
            try:
                params_json = robust_parse_json(llm_response)
                
                # Validate predicted_demand exists
                if "predicted_demand" not in params_json:
                    raise ValueError("Missing 'predicted_demand' field in LLM response")
                
                predicted_demand = int(params_json["predicted_demand"])
                if predicted_demand < 0:
                    raise ValueError(f"predicted_demand must be non-negative, got {predicted_demand}")
                
                # Print LLM forecast
                _safe_print(f"\nPeriod {current_period} ({exact_date}) LLM Demand Forecast:")
                print("="*70)
                print("LLM Rationale:")
                _safe_print(params_json.get("rationale", "(no rationale provided)"))
                
                # Handle carry-over insight
                carry_memo = params_json.get("carry_over_insight")
                if isinstance(carry_memo, str):
                    carry_memo = carry_memo.strip()
                else:
                    carry_memo = None
                
                if carry_memo:
                    carry_over_insights[current_period] = carry_memo
                    _safe_print(f"\nCarry-over insight: {carry_memo}")
                else:
                    if current_period in carry_over_insights:
                        del carry_over_insights[current_period]
                    print("\nCarry-over insight: (empty)")
                
                print(f"\nPredicted demand: {predicted_demand} units")
                print("="*70)
                
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"\nERROR: Failed to parse LLM output: {e}"
                print(error_msg, file=sys.stderr)
                print(error_msg)
                _safe_print(f"Raw output:\n{llm_response}")
                print("="*70)
                print("=== ERROR SUMMARY ===")
                print(f"Period: {current_period}")
                print(f"Error: {e}")
                print("="*70)
                sys.exit(1)
            
            # Parse inventory from environment observation (needed for OR)
            # observation is already a string from the environment
            total_inventory = parse_total_inventory(observation, primary_item)
            
            # Compute OR order from prediction
            config = current_item_configs[primary_item]
            or_result = compute_order_from_prediction(
                predicted_demand=predicted_demand,
                promised_lead_time=args.promised_lead_time,
                all_samples=all_demand_samples[primary_item],
                profit=config['profit'],
                holding_cost=config['holding_cost'],
                total_inventory=total_inventory,
            )
            
            # Print OR computation details
            print(f"\n{'='*70}")
            _safe_print(f"Period {current_period} ({exact_date}) OR Backend Computation (CAPPED POLICY):")
            print(f"{'='*70}")
            print(f"\n{primary_item}:")
            print(f"  LLM predicted single-period demand: {or_result['predicted_demand']}")
            print(f"  Promised lead time (L): {or_result['L']}")
            _safe_print(f"  mu_hat = (1+L) × prediction = {1+or_result['L']} × {or_result['predicted_demand']} = {or_result['mu_hat']:.2f}")
            print(f"  empirical_std (all {len(all_demand_samples[primary_item])} samples): {or_result['empirical_std']:.2f}")
            _safe_print(f"  sigma_hat = sqrt(1+L) × std = {np.sqrt(1+or_result['L']):.3f} × {or_result['empirical_std']:.2f} = {or_result['sigma_hat']:.2f}")
            _safe_print(f"  Critical fractile q = {or_result['q']:.4f}, z* = {or_result['z_star']:.4f}")
            print(f"  Base stock = {or_result['base_stock']:.2f}")
            print(f"  Total inventory (on-hand + in-transit): {or_result['total_inventory']}")
            print(f"  Cap value: {or_result['cap']:.2f}")
            print(f"  Order (uncapped): {or_result['order_uncapped']}")
            print(f"  Order (capped): {or_result['order']}")
            
            # Create action JSON
            orders = {primary_item: or_result['order']}
            action_json = {"action": orders}
            action = json.dumps(action_json, indent=2)
            
            print("\n" + "="*70)
            print("Final Order Action:")
            _safe_print(action)
            print("="*70)
            sys.stdout.flush()
            
        else:  # Demand from CSV
            exact_date = csv_player.get_exact_date(current_period)
            action = csv_player.get_action(current_period)
            
            # Parse demand to update tracking
            demand_data = json.loads(action)
            for item_id, qty in demand_data['action'].items():
                observed_demands[item_id].append((current_period, exact_date, qty))
                all_demand_samples[item_id].append(qty)
            
            print(f"\nPeriod {current_period} ({exact_date}) Demand: {action}")
            current_period += 1
        
        done, _ = env.step(action=action)
    
    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]
    
    print("\n" + "="*70)
    _safe_print("=== Final Results (Simple LLM->OR Strategy) ===")
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
        print(f"  Profit/unit: ${profit}, Holding: ${holding_cost}/unit/period")
        print(f"  Total Profit: ${total_profit}")
    
    # Period breakdown
    print("\n" + "="*70)
    print("Period Breakdown:")
    print("="*70)
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
    
    print("\n" + "="*70)
    print("=== TOTAL SUMMARY ===")
    print("="*70)
    print(f"Total Profit from Sales: ${total_profit:.2f}")
    print(f"Total Holding Cost: ${total_holding:.2f}")
    print(f"\n>>> Total Reward (Simple LLM->OR Strategy): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

