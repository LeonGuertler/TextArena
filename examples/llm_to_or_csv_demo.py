"""
LLM->OR Strategy: LLM Proposes OR Parameters

This demo uses:
- LLM Agent: Analyzes game state and proposes OR algorithm parameters (L, mu_hat, sigma_hat)
- OR Calculator: Uses LLM-proposed parameters to compute optimal orders

The LLM evaluates current conditions (inventory, demand patterns) and
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


def inject_carry_over_insights(observation: str, insights: Dict[int, str]) -> str:
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


# ============================================================================
# Helper Functions for Parsing and Computation
# ============================================================================

def parse_total_inventory(observation: str, item_id: str) -> int:
    """
    Parse total inventory (on-hand + in-transit) from observation for a specific item.
    
    Observation format:
      chips(Regular) (...): Profit=$2/unit, Holding=$1/unit/period
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
    
    Looks for patterns like: "arrived=X units (ordered on Period Y, lead_time was Z periods)"
    
    Returns:
        Dict of {item_id: [list of observed lead times]}
    """
    observed_lead_times = {}
    
    # Pattern 1: Match "item_id: ... arrived=X units (ordered on Period Y, lead_time was Z periods)"
    # This handles both numeric item_ids (e.g., "706016001") and non-numeric (e.g., "chips(Regular)")
    # More flexible pattern: matches item_id (can contain parentheses), then looks for "arrived=... units (ordered on Period X, lead_time was Y periods)"
    # Example: "chips(Regular): ordered=140, arrived=141 units (ordered on Period 1, lead_time was 4 periods)"
    pattern1 = r'([^\s:]+(?:\([^)]+\))?):\s+[^:]*?arrived=\d+\s+units\s+\(ordered\s+on\s+Period\s+\d+,\s+lead_time\s+was\s+(\d+)\s+periods?\)'
    
    # Pattern 2: Match "item_id: ordered=... lead_time was X periods" (fallback for numeric item_ids)
    # This handles cases where format is different
    pattern2 = r'(\d\S*?):\s+ordered=.*?lead_time\s+was\s+(\d+)\s+periods?'
    
    # Try pattern 1 first (more specific, handles non-numeric item_ids)
    matches = re.findall(pattern1, observation, re.IGNORECASE)
    for item_id, lead_time_str in matches:
        # Exclude common non-item-id words
        item_id_lower = item_id.lower().strip()
        if item_id_lower in ['conclude', 'concluded', 'period', 'date', 'summary', 'error']:
            continue
        # Skip if item_id is just a number (likely a period number, not an item_id)
        if item_id.strip().isdigit():
            continue
        try:
            lead_time = int(lead_time_str)
            if item_id not in observed_lead_times:
                observed_lead_times[item_id] = []
            observed_lead_times[item_id].append(lead_time)
        except ValueError:
            continue
    
    # Try pattern 2 as fallback (only if pattern 1 didn't match this item_id)
    matches2 = re.findall(pattern2, observation, re.IGNORECASE)
    for item_id, lead_time_str in matches2:
        # Skip if already found by pattern 1
        if item_id in observed_lead_times:
            continue
        # Exclude common non-item-id words
        if item_id.lower() in ['conclude', 'concluded', 'period', 'date', 'summary', 'error']:
            continue
        try:
            lead_time = int(lead_time_str)
            if item_id not in observed_lead_times:
                observed_lead_times[item_id] = []
            observed_lead_times[item_id].append(lead_time)
        except ValueError:
            continue
    
    return observed_lead_times


def compute_L(method: str, params: dict, observed_lead_times: List[int], promised_lead_time: float) -> float:
    """
    Compute lead time L based on method.
    
    Args:
        method: "default", "calculate", "recent_N", or "explicit"
        params: Dict that may contain "N" for recent_N or "value" for explicit
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
    elif method == "recent_N":
        if not observed_lead_times:
            raise ValueError("Cannot compute recent_N lead time: no observed arrivals yet")
        if "N" not in params:
            raise ValueError("Method 'recent_N' for L requires 'N' field")
        N = int(params["N"])
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        recent_samples = observed_lead_times[-N:] if len(observed_lead_times) >= N else observed_lead_times
        return float(np.mean(recent_samples))
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


def robust_parse_json(text: str) -> dict:
    """
    Robustly parse JSON from LLM output, attempting to fix common formatting errors.
    
    Common issues fixed:
    - Extra quotes after numeric values (e.g., "value": 15.739624" -> "value": 15.739624)
    - Missing quotes around string values
    - Trailing commas
    - Markdown code fences
    
    Args:
        text: Raw text from LLM
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after repair attempts
    """
    # Step 1: Clean markdown code fences
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Step 1.5: Fix invalid escape sequences in JSON strings
    # JSON only supports: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # Fix common invalid escapes:
    # - \$ -> $ (dollar sign doesn't need escaping in JSON)
    # - \xXX -> convert to \u00XX (hex escape to unicode escape)
    def fix_invalid_escapes(text):
        # Fix \$ -> $ (but not \\$ which is a literal backslash followed by $)
        text = re.sub(r'(?<!\\)\\\$', '$', text)
        # Fix \xXX -> \u00XX (convert hex escapes to unicode escapes)
        def hex_to_unicode(match):
            hex_val = match.group(1)
            try:
                # Convert hex to unicode code point
                code_point = int(hex_val, 16)
                # Convert to \uXXXX format
                return f'\\u{code_point:04X}'
            except ValueError:
                return match.group(0)  # Keep original if invalid
        text = re.sub(r'\\x([0-9a-fA-F]{2})', hex_to_unicode, text)
        return text
    
    cleaned = fix_invalid_escapes(cleaned)
    
    # Step 2: Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # Continue to repair attempts
    
    # Step 3: Fix common issues - extra quotes after numeric values
    # Pattern 1: "value": number" -> "value": number (handles cases like "value": 15.739624" or "value": 17.606")
    # This also handles cases without space: "value":15.739624" or "value":15.739624"}}
    repaired = re.sub(r'"value"\s*:\s*([+-]?\d+\.?\d*)"([,}\s\]])', r'"value": \1\2', cleaned)
    # Also fix invalid escapes in repaired version
    repaired = fix_invalid_escapes(repaired)
    
    # Pattern 2: More aggressive - fix any numeric value followed by quote before closing brace/bracket
    # Handles: "value": number"}} or "value": number"}
    repaired = re.sub(r':\s*([+-]?\d+\.?\d*)"([,}\s]*[}\]]+)', r': \1\2', repaired)
    
    # Pattern 3: Handle cases where value is already quoted but shouldn't be (for numeric values)
    # Pattern: "value": "number" -> "value": number (only if it's a valid number)
    def fix_quoted_number(match):
        key = match.group(1)
        num_str = match.group(2)
        suffix = match.group(3)
        try:
            # Try to parse as number
            float(num_str)
            return f'"{key}": {num_str}{suffix}'
        except ValueError:
            # Keep as string if not a valid number
            return match.group(0)
    
    repaired = re.sub(r'"value"\s*:\s*"([+-]?\d+\.?\d*)"([,}\s\]])', fix_quoted_number, repaired)
    
    # Step 4: Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
    
    # Step 5: Try parsing repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Step 6: Try to extract JSON object from text (in case there's extra text)
    # Find the first { and last }
    first_brace = repaired.find('{')
    last_brace = repaired.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = repaired[first_brace:last_brace + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
    
    # Step 7: Additional fix for any remaining quote issues after numeric values
    # This is a catch-all for any pattern we might have missed
    repaired2 = re.sub(r':\s*([+-]?\d+\.?\d*)"([,}\s\]])', r': \1\2', repaired)
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError:
        pass
    
    # Step 8: Final attempt - raise original error with helpful message
    try:
        json.loads(cleaned)  # This will raise the original error
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON after repair attempts. Original error: {e.msg}",
            e.doc,
            e.pos
        )


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
        if L_method not in ["default", "calculate", "recent_N", "explicit"]:
            raise ValueError(f"Invalid L method for item {item_id}: {L_method}")
        if L_method == "recent_N" and "N" not in L_param:
            raise ValueError(f"Method 'recent_N' for L requires 'N' field for item {item_id}")
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
    Create LLM agent that proposes OR parameters with exact dates.
    
    Args:
        initial_samples: Dict of {item_id: [samples]}
        current_configs: Dict of {item_id: config} with current item configurations
        promised_lead_time: The lead time promised by supplier (shown to LLM)
        human_feedback_enabled: Whether human feedback mode is enabled
        guidance_enabled: Whether guidance mode is enabled
    """
    item_ids = list(current_configs.keys())
    primary_item = item_ids[0] if item_ids else "item_id"
    
    system = (
        "=== ROLE & OBJECTIVE ===\n"
        f"You run an LLM→OR controller for a single SKU \"{primary_item}\". "
        "Your job is to translate the observation into OR parameters so the backend can compute the order. "
        "Maximize total reward R_t = Profit × units_sold − HoldingCost × ending_inventory each total periods.\n"
        "\n"
        "=== GAME MECHANISM: PERIOD EXECUTION SEQUENCE ===\n"
        "Each period follows this strict execution order:\n"
        "  1. VM Decision Phase: You receive observation and propose OR parameters for Period N\n"
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
        "  • Period 1: You emit OR parameters. No history yet, so no conclude to read.\n"
        "  • Period 2 START: You read 'Period 1 conclude: arrived=0'. This is NORMAL!\n"
        "    The order arrives DURING Period 2 (after your Period 2 decision), not before.\n"
        "  • Period 3 START: You read 'Period 2 conclude: arrived=X (ordered Period 1, lead_time was 1 periods)'.\n"
        "    NOW you have confirmation that actual lead_time = 1.\n"
        "\n"
        "SCENARIO B: Actual lead_time = 0 periods (same-period arrival)\n"
        "  • Period 1: You emit OR parameters.\n"
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
        "- Your parameters should ensure: order + on-hand + in-transit covers (L+1) periods of demand\n"
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
        "- Inventory view: on-hand starts at 0, holding cost applies every period, and \"in-transit\" shows total undelivered units.\n"
        f"- Promised lead time is {promised_lead_time} period(s) but actual lead time can drift and must be inferred from CONCLUDED periods only.\n"
        "- Orders may also never CONCLUDE.\n"
        "\n"
        "=== OR BACKEND RECAP ===\n"
        "- The OR engine treats your parameters as follows (single-SKU base stock):\n"
        "    base_stock = μ̂ + z*·σ̂,  where z* = Φ⁻¹(q) and q = profit / (profit + holding_cost).\n"
        "- It always runs the capped policy: final order = min(base_stock − pipeline_inventory, cap), "
        "with cap = μ̂/(1+L) + Φ⁻¹(0.95)·σ̂/√(1+L).\n"
        "- The OR engine only knows the promised lead time and historical demand statistics; it has no awareness of lost orders, or actual lead-time shifts. "
        "Your parameters must bridge that gap.\n"
        "\n"
        "=== LEAD-TIME INFERENCE ===\n"
        "ONLY use 'Period X conclude' messages from history to infer actual lead time:\n"
        "- Message format: 'arrived=Y units (ordered on Period Z, lead_time was W periods)'\n"
        "- Actual lead time calculation: W = X - Z\n"
        "- NEVER infer lead-time from current period's observations (you haven't seen arrivals yet)\n"
        "- If orders don't arrive for many periods beyond promised lead time, they may be lost\n"
        "\n"
        "=== DEMAND & LEAD-TIME ANALYSIS ===\n"
        "- When product description and/or calendar dates are available, use them as PRIMARY forecasting anchors:\n"
        "  * What product category is this? (if description available)\n"
        "  * What time of year is it? (if dates available)\n"
        "  * Are there upcoming or recent calendar events that affect this category? (if dates available)\n"
        "- Compare historical demand segments to confirm mean/variance changes before altering μ̂/σ̂.\n"
        "- Historical samples seed your prior, but demand can shift abruptly—validate each changepoint with evidence.\n"
        "- Combine calendar knowledge with actual demand patterns to inform your parameter choices.\n"
        "- Promised lead time may fail any period; reconcile expected vs. actual arrivals (including possible lost shipments).\n"
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
        "=== PARAMETER MENU ===\n"
        "You output L, μ̂, and σ̂ for the single SKU:\n"
        "1. L (lead time this period):\n"
        "   • default → promised lead time.\n"
        "   • calculate → average of all observed lead times.\n"
        "   • recent_N → average of the last N observed lead times (you choose N).\n"
        "   • explicit → your best estimate (use when missing shipments suggest a longer lead time).\n"
        "2. mu_hat (demand across review+lead period):\n"
        "   • default → (1+L) × mean of all samples.\n"
        "   • recent_N → (1+L) × mean of last N samples (N chosen per detected regime).\n"
        "   • EWMA_gamma → (1+L) × exponentially weighted mean (specify gamma ∈ [0,1]).\n"
        "   • explicit → (1+L) × your forecast based on seasonality.\n"
        "3. sigma_hat:\n"
        "   • default → sqrt(1+L) × std of all samples.\n"
        "   • recent_N → sqrt(1+L) × std of last N samples.\n"
        "   • explicit → your volatility estimate.\n"
        "\n"
        "When using recent_N:\n"
        "   - Detect the most recent changepoint for that parameter (demand or lead time).\n"
        "   - N = max(min(regime_length, 20), 3), capped by available sample count.\n"
        "   - Document the changepoint evidence and chosen N in your rationale.\n"
        "\n"
    )
    
    system += (
        "=== DECISION CHECKLIST ===\n"
        "1. Summarize current date + demand context in your rationale.\n"
        "2. Reconcile on-hand + pipeline against the orders you expect; flag overdue shipments or losses.\n"
        "3. Decide how to set L, μ̂, σ̂ (method + parameters) based on detected changepoints.\n"
        "4. Explain how your parameters help the OR backend balance service level vs. holding cost.\n"
        "\n"
        "=== CARRY-OVER INSIGHTS ===\n"
        "This is a critical mechanism for cross-period memory.\n"
        "\n"
        "PURPOSE: Record NEW, sustained, actionable pattern shifts that "
        "future periods must remember for accurate parameter selection.\n"
        "\n"
        "WHAT TO RECORD:\n"
        "- Confirmed demand regime changes (mean/variance shifts)\n"
        "- Lead time changes with evidence (e.g., 'Actual lead time is 3, not promised 2')\n"
        "- Seasonal patterns with evidence (e.g., 'Holiday demand spike confirmed')\n"
        "- Missing/delayed shipment patterns\n"
        "- Any observation helpful for future OR parameter decisions\n"
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
        "- \"Lead time confirmed as 3 periods (observed: P1 order arrived P4)\"\n"
        "- \"Seasonal peak confirmed: Dec weeks show 40% higher demand\"\n"
        "- \"\" (empty - no new pattern)\n"
        "\n"
        "=== OUTPUT FORMAT ===\n"
        "Return valid JSON only:\n"
        "{\n"
        '  "rationale": "Explain current context, changepoint evidence, chosen methods/values, and how they address missing shipments.",\n'
        '  "carry_over_insight": "Summaries of NEW sustained changes with evidence, or \\"\\".",\n'
        '  "parameters": {\n'
        f'    "{primary_item}": {{\n'
        '      "L": {"method": "..."},\n'
        '      "mu_hat": {"method": "..."},\n'
        '      "sigma_hat": {"method": "..."}\n'
        "    }}\n"
        "  }\n"
        "}\n"
        "\n"
        "=== CRITICAL: METHOD VALUES MUST BE EXACT STRINGS ===\n"
        "The 'method' field for each parameter MUST be one of the exact strings listed below. "
        "DO NOT use descriptive text, explanations, or variations. Use ONLY the exact method names.\n"
        "\n"
        "=== FIELD REQUIREMENTS BY METHOD ===\n"
        "IMPORTANT: Only include fields required by your chosen method. DO NOT include 'value' field unless using 'explicit' method.\n"
        "\n"
        "For L parameter:\n"
        '  - "default": Only include {"method": "default"} (backend uses promised lead time)\n'
        '  - "calculate": Only include {"method": "calculate"} (backend computes average from observed lead times)\n'
        '  - "recent_N": Include {"method": "recent_N", "N": <integer>} (backend computes average of last N lead times)\n'
        '  - "explicit": Include {"method": "explicit", "value": <number>} (ONLY method that requires "value")\n'
        "\n"
        "For mu_hat parameter:\n"
        '  - "default": Only include {"method": "default"} (backend computes (1+L)×mean of all samples)\n'
        '  - "recent_N": Include {"method": "recent_N", "N": <integer>} (backend computes (1+L)×mean of last N samples)\n'
        '  - "EWMA_gamma": Include {"method": "EWMA_gamma", "gamma": <float 0-1>} (backend computes (1+L)×EWMA)\n'
        '  - "explicit": Include {"method": "explicit", "value": <number>} (ONLY method that requires "value")\n'
        "\n"
        "For sigma_hat parameter:\n"
        '  - "default": Only include {"method": "default"} (backend computes sqrt(1+L)×std of all samples)\n'
        '  - "recent_N": Include {"method": "recent_N", "N": <integer>} (backend computes sqrt(1+L)×std of last N samples)\n'
        '  - "explicit": Include {"method": "explicit", "value": <number>} (ONLY method that requires "value")\n'
        "\n"
        "=== EXAMPLES ===\n"
        "CORRECT example (using recent_N for mu_hat, default for others):\n"
        '  "mu_hat": {"method": "recent_N", "N": 5}  ✓ (no value field)\n'
        '  "sigma_hat": {"method": "default"}  ✓ (no value field)\n'
        "\n"
        "INCORRECT example (DO NOT include value when not using explicit):\n"
        '  "mu_hat": {"method": "recent_N", "N": 5, "value": 604}  ✗ (remove value)\n'
        '  "sigma_hat": {"method": "recent_N", "N": 3, "value": 112.33}  ✗ (remove value)\n'
        "\n"
        "CORRECT example (using explicit method):\n"
        '  "mu_hat": {"method": "explicit", "value": 604}  ✓ (value required for explicit)\n'
        "\n"
        "=== GENERAL RULES ===\n"
        "- Include ONLY the fields required by your chosen method.\n"
        "- DO NOT include 'value' field unless method is 'explicit'.\n"
        "- DO NOT include 'N' field unless method is 'recent_N'.\n"
        "- DO NOT include 'gamma' field unless method is 'EWMA_gamma'.\n"
        "- All numeric values must be floats/ints; all N values are integers ≥ 1.\n"
        "- No extra commentary outside the JSON.\n"
        "- The method field must be an exact match to one of the valid strings listed above.\n"
    )
    
    # return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)
    return GPT5MiniAgent(system_prompt=system)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LLM->OR strategy with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time to show to LLM in periods (default: 0). This is what supplier promises, not the actual lead time in CSV.')
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
    
    print(f"Promised lead time (shown to LLM): {args.promised_lead_time} periods")
    print(f"Note: Actual lead times in CSV may differ and will be inferred by LLM from arrivals.")
    
    # Determine number of periods to run
    total_periods = csv_player.get_num_periods()
    num_periods = total_periods
    if args.max_periods is not None:
        num_periods = min(args.max_periods, total_periods)
        print(f"Limiting run to {num_periods} periods (CSV has {total_periods})")
    else:
        print(f"Running full CSV horizon: {num_periods} periods")
    
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
    
    # Reset environment with explicit horizon
    env.reset(num_players=2, num_days=num_periods, initial_inventory_per_item=0)
    
    # Run game
    done = False
    current_period = 1
    carry_over_insights: Dict[int, str] = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (LLM→OR)
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
                # Update tracking
                current_item_configs[item_id] = config
            
            # Parse observed lead times from arrivals in history
            new_lead_times = parse_arrivals_from_history(observation)
            for item_id, lt_list in new_lead_times.items():
                # Safety check: only process item_ids that exist in observed_lead_times
                if item_id in observed_lead_times:
                    observed_lead_times[item_id].extend(lt_list)
                else:
                    # Skip invalid item_ids (e.g., false positives from regex matching)
                    print(f"Warning: Skipping invalid item_id '{item_id}' from arrival parsing")
            
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
            
            # Parse JSON response with robust parser
            try:
                params_json = robust_parse_json(llm_response)
                
                # Validate JSON structure
                validate_parameters_json(params_json, csv_player.get_item_ids(), current_item_configs)
                
                _safe_print(f"\nPeriod {current_period} ({exact_date}) LLM->OR Decision:")
                print("="*70)
                print("LLM Rationale:")
                _safe_print(params_json.get("rationale", "(no rationale provided)"))
                
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
                
                print("\n" + "="*70)
                
            except json.JSONDecodeError as e:
                error_msg = f"\nERROR: Failed to parse LLM output as JSON: {e}"
                print(error_msg, file=sys.stderr)
                print(error_msg)
                _safe_print(f"Raw output:\n{llm_response}")
                print("\n" + "="*70)
                print("=== ERROR SUMMARY ===")
                print("="*70)
                print(f"Period: {current_period}")
                print(f"Error: JSON parsing failed")
                print(f"Details: {e}")
                print("="*70)
                sys.exit(1)
            except ValueError as e:
                error_msg = f"\nERROR: Invalid parameter specification: {e}"
                print(error_msg, file=sys.stderr)
                print(error_msg)
                _safe_print(f"Raw output:\n{llm_response}")
                print("\n" + "="*70)
                print("=== ERROR SUMMARY ===")
                print("="*70)
                print(f"Period: {current_period}")
                print(f"Error: Parameter validation failed")
                print(f"Details: {e}")
                print("="*70)
                sys.exit(1)
            
            # Compute orders using OR formula with LLM-proposed parameters
            orders = {}
            
            print(f"\n{'='*70}")
            _safe_print(f"Period {current_period} ({exact_date}) LLM->OR Backend Computation (CAPPED POLICY):")
            print(f"{'='*70}")
            
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
                    l_method = item_params['L']['method']
                    l_extra = ""
                    if l_method == 'explicit' and 'value' in item_params['L']:
                        l_extra = f", value={item_params['L']['value']}"
                    elif l_method == 'recent_N' and 'N' in item_params['L']:
                        l_extra = f", N={int(item_params['L']['N'])}"
                    elif l_method == 'calculate':
                        l_extra = f", observed_samples={len(observed_lead_times[item_id])}"
                    print(f"  L method: {l_method}{l_extra}, computed L = {L:.2f}")
                    
                    # Compute mu_hat
                    mu_hat = compute_mu_hat(
                        method=item_params["mu_hat"]["method"],
                        params=item_params["mu_hat"],
                        samples=observed_demands[item_id],
                        L=L
                    )
                    mu_method = item_params['mu_hat']['method']
                    mu_extra = ""
                    if 'N' in item_params['mu_hat']:
                        mu_extra = f", N={int(item_params['mu_hat']['N'])}"
                    elif 'gamma' in item_params['mu_hat']:
                        mu_extra = f", gamma={float(item_params['mu_hat']['gamma']):.3f}"
                    elif 'value' in item_params['mu_hat']:
                        mu_extra = f", value={item_params['mu_hat']['value']}"
                    print(f"  mu_hat method: {mu_method}{mu_extra}, computed mu_hat = {mu_hat:.2f}")
                    
                    # Compute sigma_hat
                    sigma_hat = compute_sigma_hat(
                        method=item_params["sigma_hat"]["method"],
                        params=item_params["sigma_hat"],
                        samples=observed_demands[item_id],
                        L=L
                    )
                    sig_method = item_params['sigma_hat']['method']
                    sig_extra = ""
                    if 'N' in item_params['sigma_hat']:
                        sig_extra = f", N={int(item_params['sigma_hat']['N'])}"
                    elif 'value' in item_params['sigma_hat']:
                        sig_extra = f", value={item_params['sigma_hat']['value']}"
                    print(f"  sigma_hat method: {sig_method}{sig_extra}, computed sigma_hat = {sigma_hat:.2f}")
                    
                    # Get total inventory (on-hand + in-transit)
                    total_inventory = parse_total_inventory(observation, item_id)
                    print(f"  Total inventory (on-hand + in-transit): {total_inventory}")
                    
                    # Compute critical fractile
                    p = config['profit']
                    h = config['holding_cost']
                    q = p / (p + h)
                    z_star = norm.ppf(q)
                    _safe_print(f"  Critical fractile q = {q:.4f}, z* = {z_star:.4f}")
                    
                    # Compute base stock and capped order
                    base_stock = mu_hat + z_star * sigma_hat
                    print(f"  Base stock = {base_stock:.2f}")
                    
                    order_uncapped = max(int(np.ceil(base_stock - total_inventory)), 0)
                    
                    cap_z = norm.ppf(0.95)
                    cap = mu_hat / (1 + L) + cap_z * sigma_hat / np.sqrt(1 + L)
                    order = max(min(order_uncapped, int(np.ceil(cap))), 0)
                    
                    print(f"  Cap value: {cap:.2f}")
                    print(f"  Order (capped): {order}")
                    print(f"  Order (uncapped): {order_uncapped}")
                    
                    orders[item_id] = order
                    
                except ValueError as e:
                    error_msg = f"  ERROR computing order for {item_id}: {e}"
                    print(error_msg, file=sys.stderr)
                    print(error_msg)
                    print("\n" + "="*70)
                    print("=== ERROR SUMMARY ===")
                    print("="*70)
                    print(f"Period: {current_period}")
                    print(f"Item: {item_id}")
                    print(f"Error: Order computation failed")
                    print(f"Details: {e}")
                    print("="*70)
                    sys.exit(1)
            
            # Create action JSON
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
            
            # Parse demand to update observed demands
            demand_data = json.loads(action)
            for item_id, qty in demand_data['action'].items():
                observed_demands[item_id].append(qty)
            
            print(f"\nPeriod {current_period} ({exact_date}) Demand: {action}")
            current_period += 1
        
        done, _ = env.step(action=action)
    
    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]
    
    print("\n" + "="*70)
    _safe_print("=== Final Results (LLM->OR Strategy) ===")
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
    print(f"\n>>> Total Reward (LLM->OR Strategy): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*70)
    
if __name__ == "__main__":
    main()

