"""
Vending Machine demo with CSV-driven demand.

This demo uses:
- VM Agent: OpenAI LLM generates ordering decisions
- Demand: Fixed demand patterns loaded from CSV file

CSV Format:
  - Column A: exact_dates_{item_id}  (e.g., exact_dates_cola, exact_dates_chips)
  - Column B+: demand_{item_id} (e.g., demand_cola, demand_chips)
  - Last column: news (optional, can be empty)

Usage:
  python vending_csv_demo.py --demand-file path/to/demands.csv
"""

import os
import sys
import argparse
import json
import re
import unicodedata
import pandas as pd
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
        
        print(f"Loaded CSV with {len(self.df)} periods of demand data (14-day periods)")
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
                'description': str(first_row[f'description_{item_id}']),
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
        import json
        
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


def make_vm_agent(initial_samples: dict = None, promised_lead_time: int = 0, 
                  human_feedback_enabled: bool = False, guidance_enabled: bool = False):
    """Create VM agent with updated prompt for profit-based system with exact dates."""
    
    # Extract item IDs to show in prompt
    available_items = list(initial_samples.keys()) if initial_samples else []
    items_str = ", ".join([f'"{item}"' for item in available_items])
    primary_item = available_items[0] if available_items else "item_id"
    
    system = (
        "=== ROLE & OBJECTIVE ===\n"
        f"You control a single vending SKU \"{primary_item}\". "
        "Maximize total reward (R_t = Profit × units_sold − HoldingCost × ending_inventory) over 14‑day periods.\n"
        "\n"
        "=== TIMELINE & DATA ===\n"
        "- Observations contain the exact period dates plus complete history to date; there is no future news feed.\n"
        "- Use calendar + world knowledge (seasonality, holidays) whenever the description/date imply it.\n"
        "\n"
        "=== CRITICAL: PERIOD SEQUENCING ===\n"
        "Each period follows this strict order:\n"
        "  1. YOU MAKE DECISION for Period N (this is when you see the observation)\n"
        "  2. Arrivals occur (orders placed in Period N-LT arrive now)\n"
        "  3. Demand occurs\n"
        "  4. Period N concludes and is added to history\n"
        "\n"
        "IMPORTANT IMPLICATIONS:\n"
        f"- When deciding for Period N, you CANNOT see Period N's arrivals yet (even with LT={promised_lead_time}).\n"
        "- Example: If LT=1, an order placed in Period 5 arrives in Period 6, but you won't see it until Period 7's decision.\n"
        "- Remember that shipments can arrive AFTER your ordering decision and BEFORE the demand for the period. If the inventory you ordered yesterday (with a promised lead time of 1) has not arrived yet, it does not necessarily mean that it will be delayed and won't arrive today.\n"
        "- Only use PAST period conclusions (\"Period X conclude: ... arrived=Y\") to infer lead time.\n"
        "- Your order decision in every round should be made such that the order quantity + on-hand inventory + in-transit inventory covers L+1 periods worth of demand. It is L+1 and not L because the current round's demand realizes after you make your order decision. For example, if L = 1, then your order quantity + on-hand inventory + in-transit inventory should cover 2 periods worth of demand.\n"
        "\n"
        "=== INVENTORY & ORDERS ===\n"
        "- On-hand inventory starts at 0 in Period 1 and is charged holding cost every period.\n"
        "- \"In-transit\" shows total units not yet delivered; you must infer when each shipment should arrive.\n"
        f"- Supplier-promised lead time is {promised_lead_time} period(s), but actual lead time can drift and must be inferred from CONCLUDED periods only.\n"
        "- Orders may also never CONCLUDE.\n"
        "\n"
        "=== DEMAND REASONING ===\n"
        "- Treat the SKU description plus exact date as anchors for applying seasonality/world knowledge.\n"
        "- Compare historical demand segments to detect sustained mean/variance changes or new regimes.\n"
        "- Historical samples seed your prior, but demand can shift abruptly—confirm each change with evidence.\n"
        "\n"
        "=== LEAD-TIME INFERENCE PLAYBOOK ===\n"
        "ONLY use \"Period X conclude: ... arrived=Y units (ordered on Period Z, lead_time was W)\" from history to infer lead time.\n"
        "NEVER infer lead-time changes from current period's missing arrivals—you haven't seen them yet due to sequencing.\n"
        "Example: In Period 6 decision, if history shows \"Period 5 conclude: ... arrived=900 (ordered on Period 4, lead_time was 1)\", then LT=1.\n"
        "If the order does not arrive for many periods longer than the promised lead time, then it may be lost.\n"
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
        system += "=== HISTORICAL DEMAND SAMPLES ===\n"
        system += "Use these unified samples to ground your prior before any real demand is observed:\n"
        for item_id, samples in initial_samples.items():
            system += f"- {item_id}: {samples}\n"
        system += "\n"
    
    # Create example format with actual item IDs
    if available_items:
        example_action = ", ".join([f'"{item}": 100' for item in available_items[:2]])  # Show up to 2 items
        if len(available_items) > 2:
            example_action += ", ..."
    else:
        example_action = '"item_id": quantity, ...'
    
    system += (
        "=== DECISION CHECKLIST ===\n"
        "1. Note the current date/news and compare to relevant seasonal history.\n"
        "2. Reconcile on-hand + in-transit vs. expected arrivals; flag overdue orders.\n"
        "3. Infer lead time (or order loss) from arrivals/absences and adjust safety stock.\n"
        "4. Forecast demand using calendar knowledge plus recent data regimes.\n"
        "5. Place an order that balances service level vs. holding cost while respecting pipeline.\n"
        "\n"
        "=== CARRY-OVER INSIGHTS ===\n"
        "- Only log NEW, non-duplicated insights when a sustained, evidence-backed shift occurs (demand mean/variance, lead time, seasonality confirmation) and future periods need that reminder.\n"
        "- Stay conservative: if the effect is already recorded or not yet significant, leave the field empty instead of restating it.\n"
        "- Provide concrete stats (date ranges, averages, lead-time values). If multiple independent changes exist, list each separated by '; ' or newlines.\n"
        "- Remove or update insights when the effect ends. Default output is an empty string \"\".\n"
        "\n"
        "=== OUTPUT FORMAT ===\n"
        "Respond with valid JSON only:\n"
        "{\n"
        '  "rationale": "Step-by-step reasoning covering (a) date/news context, (b) demand regime analysis, '
        ' (c) lead_time vs. missing orders, (d) inventory & pipeline assessment, (e) final order logic.",\n'
        '  "carry_over_insight": "Summarize all NEW sustained changes with evidence, or \\"\\" if none.",\n'
        f'  "action": {{{example_action}}}\n'
        "}\n"
        "\n"
        f"Use the exact item ID when populating \"action\" (current ID(s): {items_str or primary_item}). "
        "Do not output extra text outside the JSON."
    )
    # return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)
    return GPT5MiniAgent(system_prompt=system)


def main():
    parser = argparse.ArgumentParser(description='Run vending machine with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    parser.add_argument('--promised-lead-time', type=int, default=0,
                       help='Promised lead time shown to LLM in periods (default: 0, where 1 period = 14 days). Actual lead time in CSV may differ.')
    parser.add_argument('--human-feedback', action='store_true',
                       help='Enable daily human feedback on agent decisions (Mode 1)')
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
    print(f"Promised lead time (shown to LLM): {args.promised_lead_time} periods (1 period = 14 days)")
    print(f"Note: Actual lead times in CSV may differ. LLM must infer actual lead time from arrivals.")
    
    # Set NUM_DAYS based on CSV (each period = 14 days)
    from textarena.envs.VendingMachine import env as vm_env_module
    original_num_days = vm_env_module.NUM_DAYS
    original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
    num_periods = csv_player.get_num_periods()
    if args.max_periods is not None:
        num_periods = min(args.max_periods, num_periods)
    vm_env_module.NUM_DAYS = num_periods
    print(f"Set NUM_DAYS to {vm_env_module.NUM_DAYS} periods based on CSV" + 
          (f" (limited from {csv_player.get_num_periods()})" if args.max_periods is not None else ""))
    
    # Create VM agent with historical data
    base_agent = make_vm_agent(
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
            print("✅ Mode 1: Daily feedback on agent decisions is ENABLED")
        if args.guidance_frequency > 0:
            print(f"✅ Mode 2: Strategic guidance every {args.guidance_frequency} days is ENABLED")
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
    current_period = 1
    carry_over_insights = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent
            # Get exact date for current period
            exact_date = csv_player.get_exact_date(current_period)
            
            # Inject exact date into observation's CURRENT STATUS section
            observation = observation.replace(
                f"PERIOD {current_period} / ",
                f"PERIOD {current_period} (Date: {exact_date}) / "
            )
            
            # Inject exact dates into GAME HISTORY section
            if "=== GAME HISTORY ===" in observation:
                for p in range(1, current_period):
                    p_date = csv_player.get_exact_date(p)
                    observation = observation.replace(
                        f"Period {p} conclude:",
                        f"Period {p} (Date: {p_date}) conclude:"
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
            
            # Get VM action (even if lead_time=inf - agent doesn't know about supply issues)
            action = vm_agent(observation)
            
            # Print complete JSON output with proper formatting
            print(f"\nPeriod {current_period} ({exact_date}) VM Action:")
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
                
                carry_memo = action_dict.get("carry_over_insight")
                if isinstance(carry_memo, str):
                    carry_memo = carry_memo.strip()
                else:
                    carry_memo = None
                if carry_memo:
                    carry_over_insights[current_period] = carry_memo
                elif current_period in carry_over_insights:
                    del carry_over_insights[current_period]
                
                formatted_json = json.dumps(action_dict, indent=2, ensure_ascii=False)
                _safe_print(formatted_json)
                # Flush to ensure complete output to file
                sys.stdout.flush()
            except Exception as e:
                # Fallback to raw output if JSON parsing fails
                print(f"[DEBUG: JSON parsing failed: {e}]")
                _safe_print(action)
                sys.stdout.flush()
            print("="*60)
            sys.stdout.flush()
        else:  # Demand from CSV
            exact_date = csv_player.get_exact_date(current_period)
            action = csv_player.get_action(current_period)
            print(f"Period {current_period} ({exact_date}) Demand: {action}")
            current_period += 1
        
        done, _ = env.step(action=action)
    
    # Display results
    rewards, game_info = env.close()
    vm_info = game_info[0]
    
    print("\n" + "="*60)
    print("=== Final Results ===")
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
    print(f"\n>>> Total Reward: ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)
    
    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = original_initial_inventory


if __name__ == "__main__":
    main()
