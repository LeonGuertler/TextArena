"""
Vending Machine demo with CSV-driven demand.

This demo uses:
- VM Agent: OpenAI LLM generates ordering decisions
- Demand: Fixed demand patterns loaded from CSV file

CSV Format:
  - Column A: day (1, 2, 3, ...)
  - Column B+: demand_{item_id} (e.g., demand_cola, demand_chips)
  - Last column: news (optional, can be empty)

Usage:
  python vending_csv_demo.py --demand-file path/to/demands.csv
"""

import os
import sys
import argparse
import pandas as pd
import textarena as ta


class CSVDemandPlayer:
    """
    Simulates demand agent by reading from CSV file.
    """
    def __init__(self, csv_path: str, item_ids: list):
        """
        Args:
            csv_path: Path to CSV file
            item_ids: List of item IDs (e.g., ['cola', 'chips'])
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
            JSON string like '{"action": {"cola": 10, "chips": 5}}'
        """
        import json
        
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


def make_vm_agent(initial_samples: dict = None):
    """Create VM agent with updated prompt for profit-based system."""
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
            mean = sum(samples) / len(samples)
            system += f"{item_id}:\n"
            system += f"  Past demands: {samples}\n"
            #system += f"  Average: {mean:.1f} units/day\n"
            system += "\n"
        system += "Use this data to inform your ordering decisions, especially on Day 1.\n\n"
    
    system += (
        "Strategy:\n"
        "- Study demand patterns from game history\n"
        "- React to TODAY'S NEWS as it happens, considering lead time for orders\n"
        "- Learn from past news events to understand their impact on demand\n"
        "- Balance profit vs holding cost (don't overstock)\n"
        "\n"
        "IMPORTANT: Think step by step, then decide.\n"
        "You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "rationale": "First, explain your reasoning: analyze current inventory and demand patterns, '
        'evaluate today\'s news (if any) and learn from past news, consider lead time constraints, '
        'and explain your ordering strategy",\n'
        '  "action": {"item_id": quantity, "item_id": quantity, ...}\n'
        "}\n"
        "\n"
        "Think through your rationale BEFORE making the final order decision.\n"
        "\n"
        "Example format:\n"
        "{\n"
        '  "rationale": "[Analyze current situation] → [Consider news/history] → [Explain strategy] → [State order decisions]",\n'
        '  "action": {"item_id_1": quantity, "item_id_2": quantity, ...}\n'
        "}\n"
        "\n"
        "Do NOT include any other text outside the JSON."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-4o", system_prompt=system, temperature=0)


def main():
    parser = argparse.ArgumentParser(description='Run vending machine with CSV demand')
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
    
    # Initial demand samples (same as OR baseline for fair comparison)
    initial_samples = {
        "chips(Regular)": [200, 202, 202, 205, 188, 201, 204, 198, 201, 195],
        "chips(BBQ)": [142, 155, 126, 115, 166, 176, 127, 131, 155, 164]
    }
    
    # Create VM agent with historical data
    vm_agent = make_vm_agent(initial_samples)
    
    # Reset environment
    env.reset(num_players=2)
    
    # Run game
    done = False
    current_day = 1
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent
            action = vm_agent(observation)
            
            # Print complete JSON output with proper formatting
            print(f"\nDay {current_day} VM Action:")
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
            sys.stdout.flush()
        else:  # Demand from CSV
            action = csv_player.get_action(current_day)
            print(f"Day {current_day} Demand: {action}")
            current_day += 1
        
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
    print(f"\n>>> Total Reward: ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)
    
    # Restore original NUM_DAYS
    vm_env_module.NUM_DAYS = original_num_days


if __name__ == "__main__":
    main()

