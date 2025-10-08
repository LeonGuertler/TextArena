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
import numpy as np
import pandas as pd
from scipy.stats import norm
import textarena as ta


class CSVDemandPlayer:
    """
    Simulates demand agent by reading from CSV file.
    """
    def __init__(self, csv_path: str, item_ids: list):
        """
        Args:
            csv_path: Path to CSV file
            item_ids: List of item IDs (e.g., ['chips(Regular)', 'chips(BBQ)'])
        """
        self.df = pd.read_csv(csv_path)
        self.item_ids = item_ids
        
        # Validate CSV has required columns
        required_cols = ['day'] + [f'demand_{item_id}' for item_id in item_ids]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Extract news if available
        self.has_news = 'news' in self.df.columns
        
        print(f"Loaded CSV with {len(self.df)} days of demand data")
        if self.has_news:
            news_days = self.df[self.df['news'].notna()]['day'].tolist()
            print(f"News scheduled for days: {news_days}")
    
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
    
    Policy: order_quantity = max(base_stock - current_inventory, 0)
    where base_stock = μ̂ + z*σ̂
    
    μ̂ = (1+L) × empirical_mean
    σ̂ = sqrt(1+L) × empirical_std
    z* = Φ^(-1)(q), where q = profit/(profit + holding_cost)
    """
    
    def __init__(self, items_config: dict, initial_samples: dict):
        """
        Args:
            items_config: Dict of {item_id: {'lead_time': L, 'profit': p, 'holding_cost': h}}
            initial_samples: Dict of {item_id: [list of initial demand samples]}
        """
        self.items_config = items_config
        self.initial_samples = initial_samples
        
        # Store observed demands (will be updated each day)
        # Format: {item_id: [demand_day1, demand_day2, ...]}
        self.observed_demands = {item_id: [] for item_id in items_config}
        
        print("\n=== OR Agent Initialized (for recommendations) ===")
        for item_id, config in items_config.items():
            L = config['lead_time']
            p = config['profit']
            h = config['holding_cost']
            q = p / (p + h)
            z_star = norm.ppf(q)
            
            samples = initial_samples[item_id]
            print(f"{item_id}:")
            print(f"  Lead time (L): {L}")
            print(f"  Profit (p): {p}, Holding cost (h): {h}")
            print(f"  Critical fractile (q): {q:.4f}")
            print(f"  z* = Phi^(-1)(q): {z_star:.4f}")
            print(f"  Initial samples: {samples}")
    
    def _parse_inventory_from_observation(self, observation: str, item_id: str) -> int:
        """
        Parse current total inventory (on-hand + in-transit) from observation.
        
        Observation format:
          chips(Regular) (...): Profit=$2.5/unit, Holding=$0.1/unit/day, Lead=2d
            On-hand: 5, Pipeline: [10, 0]
        
        Returns:
            Total inventory across all pipeline stages (on-hand + sum of pipeline)
        """
        try:
            lines = observation.split('\n')
            for i, line in enumerate(lines):
                # Find the item header line
                if line.strip().startswith(f"{item_id}"):
                    # Next line should have the inventory info
                    if i + 1 < len(lines):
                        inventory_line = lines[i + 1]
                        
                        # Parse on-hand: "  On-hand: 5, Pipeline: [10, 0]"
                        if "On-hand:" in inventory_line and "Pipeline:" in inventory_line:
                            # Extract on-hand value
                            on_hand_start = inventory_line.find("On-hand:") + len("On-hand:")
                            on_hand_end = inventory_line.find(",", on_hand_start)
                            on_hand = int(inventory_line[on_hand_start:on_hand_end].strip())
                            
                            # Extract pipeline array
                            pipeline_start = inventory_line.find("[")
                            pipeline_end = inventory_line.find("]") + 1
                            if pipeline_start != -1 and pipeline_end > pipeline_start:
                                pipeline_str = inventory_line[pipeline_start:pipeline_end]
                                pipeline = json.loads(pipeline_str)
                                in_transit = sum(pipeline)
                                
                                total_inventory = on_hand + in_transit
                                return total_inventory
            
            # If not found, return 0 (shouldn't happen in normal operation)
            print(f"Warning: Could not parse inventory for {item_id}, assuming 0")
            return 0
        except Exception as e:
            print(f"Error parsing inventory for {item_id}: {e}")
            return 0
    
    def _calculate_order(self, item_id: str, current_inventory: int) -> int:
        """
        Calculate order quantity using OR base-stock policy.
        
        Args:
            item_id: Item identifier
            current_inventory: Current total inventory (on-hand + in-transit)
            
        Returns:
            Order quantity (non-negative integer)
        """
        config = self.items_config[item_id]
        L = config['lead_time']
        p = config['profit']
        h = config['holding_cost']
        
        # Collect all demand samples
        all_samples = self.initial_samples[item_id] + self.observed_demands[item_id]
        
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
        
        # Calculate order quantity
        order = max(int(np.ceil(base_stock - current_inventory)), 0)
        
        return order
    
    def update_demand_observation(self, item_id: str, observed_demand: int):
        """
        Update observed demand history for an item.
        
        Args:
            item_id: Item identifier
            observed_demand: The true demand observed on this day (requested quantity)
        """
        self.observed_demands[item_id].append(observed_demand)
    
    def get_recommendation(self, observation: str) -> dict:
        """
        Generate OR algorithm recommendations (not final action, just recommendation).
        
        Args:
            observation: Current game observation
            
        Returns:
            Dict like {"chips(Regular)": 250, "chips(BBQ)": 180}
        """
        recommendations = {}
        
        for item_id in self.items_config:
            # Parse current inventory from observation
            current_inventory = self._parse_inventory_from_observation(observation, item_id)
            
            # Calculate order quantity
            order = self._calculate_order(item_id, current_inventory)
            recommendations[item_id] = order
        
        return recommendations


def make_hybrid_vm_agent(initial_samples: dict = None):
    """Create hybrid VM agent that considers both OR recommendations and news."""
    system = (
        "You are the Vending Machine controller (VM). "
        "You manage multiple items, each with unit profit and holding costs. "
        "Objective: Maximize total reward = sum of daily rewards R_t. "
        "Daily reward: R_t = Profit × Sold - HoldingCost × EndingInventory. "
        "\n\n"
        "Key mechanics:\n"
        "- Orders placed today arrive after the item's lead time\n"
        "- You see on-hand inventory and pipeline for each item\n"
        "- Holding cost is charged on ending inventory each day\n"
        "- DAILY NEWS: News events are revealed each day (if any). You will NOT know future news in advance.\n"
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
        system += "Use this data to inform your ordering decisions, especially on Day 1.\n\n"
    
    system += (
        "🔍 OR ALGORITHM ASSISTANCE:\n"
        "You will receive recommendations from an Operations Research (OR) algorithm each day.\n"
        "The OR algorithm uses a base-stock policy based on statistical analysis:\n"
        "- It calculates optimal orders using historical demand patterns\n"
        "- Uses lead time, profit, and holding cost in its calculations\n"
        "- Aims to balance service level (meeting demand) vs holding costs\n"
        "- Works well for NORMAL, STABLE demand patterns\n"
        "\n"
        "⚠️ IMPORTANT LIMITATION:\n"
        "The OR algorithm CANNOT react to news events. It only looks at historical data.\n"
        "This is where YOU come in!\n"
        "\n"
        "Your Strategy:\n"
        "1. Use OR recommendation as your BASELINE for normal days\n"
        "2. React to TODAY'S NEWS as it happens, considering lead time\n"
        "3. Learn from past news events to understand their impact on demand\n"
        "4. Balance between data-driven OR approach and news-driven insights\n"
        "\n"
        "Example decision process:\n"
        "- OR recommends: 200 units\n"
        "- TODAY'S NEWS: Major sports event finale\n"
        "- Your analysis: Sports events may increase demand significantly\n"
        "- Your decision: Adjust order upward based on your reasoning\n"
        "\n"
        "IMPORTANT: Think step by step, then decide.\n"
        "You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "rationale": "First, explain your reasoning: review OR algorithm recommendations, '
        'analyze current inventory and demand patterns, evaluate today\'s news (if any) and learn from past news, '
        'consider lead time constraints, decide whether to follow OR baseline or adjust based on news, '
        'and explain your final ordering strategy",\n'
        '  "action": {"item_id": quantity, "item_id": quantity, ...}\n'
        "}\n"
        "\n"
        "Think through your rationale BEFORE making the final order decision.\n"
        "\n"
        "Example format:\n"
        "{\n"
        '  "rationale": "[Review OR recommendations] → [Analyze inventory/demand] → [Consider today\'s news] → [Decide: follow OR or adjust],\n'
        '  "action": {"item_id_1": quantity, "item_id_2": quantity, ...}\n'
        "}\n"
        "\n"
        "Do NOT include any other text outside the JSON."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-4o", system_prompt=system)


def main():
    parser = argparse.ArgumentParser(description='Run hybrid strategy (LLM + OR) with CSV demand')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data')
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print('Example: export OPENAI_API_KEY="sk-your-key-here"')
        sys.exit(1)
    
    # Create environment
    env = ta.make(env_id="VendingMachine-v0")
    
    # Define items (match CSV column names)
    item_configs = [
        {"item_id": "chips(Regular)", "description": "Potato Chips (Regular), 10oz bag", "lead_time": 1, "profit": 2, "holding_cost": 1},
        {"item_id": "chips(BBQ)", "description": "Potato Chips (BBQ), 20oz bag", "lead_time": 1, "profit": 3, "holding_cost": 2},
    ]
    
    item_ids = []
    for config in item_configs:
        env.add_item(**config)
        item_ids.append(config['item_id'])
    
    # Initial demand samples (same as pure OR baseline for fair comparison)
    initial_samples = {
        "chips(Regular)": [200, 202, 202, 205, 188, 201, 204, 198, 201, 195],
        "chips(BBQ)": [142, 155, 126, 115, 166, 176, 127, 131, 155, 164]
    }
    
    # Load CSV demand player
    try:
        csv_player = CSVDemandPlayer(args.demand_file, item_ids)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Set NUM_DAYS based on CSV
    from textarena.envs.VendingMachine import env as vm_env_module
    original_num_days = vm_env_module.NUM_DAYS
    vm_env_module.NUM_DAYS = csv_player.get_num_days()
    print(f"Set NUM_DAYS to {vm_env_module.NUM_DAYS} based on CSV")
    
    # Add news from CSV
    news_schedule = csv_player.get_news_schedule()
    for day, news in news_schedule.items():
        env.add_news(day, news)
    
    # Create OR agent (for recommendations only)
    or_items_config = {
        config['item_id']: {
            'lead_time': config['lead_time'],
            'profit': config['profit'],
            'holding_cost': config['holding_cost']
        }
        for config in item_configs
    }
    or_agent = ORAgent(or_items_config, initial_samples)
    
    # Create hybrid VM agent
    vm_agent = make_hybrid_vm_agent(initial_samples)
    
    # Reset environment
    env.reset(num_players=2)
    
    # Run game
    done = False
    current_day = 1
    last_demand = {}  # Track demand to update OR agent
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent (Hybrid: LLM + OR)
            # Get OR recommendations
            or_recommendations = or_agent.get_recommendation(observation)
            
            # Format OR recommendations for display
            or_text = "\n" + "="*60 + "\n"
            or_text += "📊 OR ALGORITHM RECOMMENDATIONS (for your reference):\n"
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
                import json
                import re
                
                # Strip markdown code fences (```json or ``` at start/end)
                cleaned_action = action.strip()
                # Remove ```json or ``` from the beginning
                cleaned_action = re.sub(r'^```(?:json)?\s*', '', cleaned_action)
                # Remove ``` from the end
                cleaned_action = re.sub(r'\s*```$', '', cleaned_action)
                
                # Parse and pretty print
                action_dict = json.loads(cleaned_action)
                formatted_json = json.dumps(action_dict, indent=2, ensure_ascii=False)
                print(formatted_json)
                # Flush to ensure complete output to file
                sys.stdout.flush()
            except Exception as e:
                # Fallback to raw output if JSON parsing fails
                print(f"[DEBUG: JSON parsing failed: {e}]")
                print(action)
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


if __name__ == "__main__":
    main()


