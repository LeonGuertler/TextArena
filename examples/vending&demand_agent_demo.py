"""
Vending Machine demo with dual LLM agents.

This demo uses:
- VM Agent: OpenAI LLM controls inventory ordering
- Demand Agent: OpenAI LLM simulates customer demand

Both agents are powered by LLMs and interact strategically.
"""

import os
import sys
import json
import re
import textarena as ta


DAY_CONCLUDED_PATTERN = re.compile(r'^(\s*Day\s+(\d+)\s+concluded:)(.*)$')


def inject_carry_over_insights(observation: str, insights) -> str:
    """Insert prior carry-over insights into the day summaries of the observation."""
    if not insights:
        return observation

    lines = observation.splitlines()
    augmented = []

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


def make_vm_agent():
    """Create VM agent with updated prompt for multi-item, lead time, and holding cost."""
    system = (
        "You are the Vending Machine controller (VM). "
        "You manage multiple items, each with unit profit and holding costs. "
        "Objective: Maximize total reward = sum of daily rewards R_t. "
        "Daily reward: R_t = Profit × Sold - HoldingCost × EndingInventory. "
        "\n\n"
        "Key mechanics:\n"
        "- Orders placed today arrive after the item's lead time (e.g., lead=2 means arrives in 2 days)\n"
        "- You see on-hand inventory and pipeline for each item\n"
        "- Holding cost is charged on ending inventory each day (incentive to keep inventory low)\n"
        "- DAILY NEWS: News events are revealed each day (if any). You will NOT know future news in advance.\n"
        "\n"
        "Strategy:\n"
        "- Study demand patterns from game history for each item\n"
        "- React to TODAY'S NEWS as it happens, considering lead time for orders\n"
        "- Learn from past news events to understand their impact on demand\n"
        "- Order enough to cover demand during lead time + buffer, but minimize holding costs\n"
        "- Consider profit margins and holding costs when prioritizing which items to stock\n"
        "- When you spot patterns future days should remember, summarize them in 'carry_over_insight' (keep it \"\" unless the shift is clearly persistent)\n"
        "\n"
        "IMPORTANT: Think step by step, then decide.\n"
        "You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "rationale": "First, explain your reasoning: analyze current inventory and demand patterns for each item, '
        'evaluate today\'s news (if any) and learn from past news, consider different lead times, '
        'and explain your ordering strategy for each item",\n'
        '  "carry_over_insight": "Short memo for future days (\"\" when there is no meaningful change)",\n'
        '  "action": {"item_id": quantity, "item_id": quantity, ...}\n'
        "}\n"
        "\n"
        "Think through your rationale BEFORE making the final order decision.\n"
        "\n"
        "Example format:\n"
        "{\n"
        '  "rationale": "[Analyze each item\'s inventory/demand] → [Consider news impact by item type] → [Account for different lead times] → [Explain ordering strategy per item]",\n'
        '  "action": {"item_id_1": quantity, "item_id_2": quantity, ...}\n'
        "}\n"
        "\n"
        "Do NOT include any other text outside the JSON."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)


def make_demand_agent():
    """Create Demand agent with updated prompt for multi-item purchasing."""
    system = (
        "You are a Demand agent representing customers. "
        "You can request to purchase multiple items each day. "
        "Objective: Simulate realistic customer demand patterns. "
        "\n\n"
        "Behavior guidelines:\n"
        "- Base demand on typical consumption patterns\n"
        "- React to NEWS events (e.g., increase demand during promotions, holidays)\n"
        "- Vary demand slightly day-to-day to simulate natural fluctuations\n"
        "- You see game history showing requested vs sold quantities (but not VM's inventory or orders)\n"
        "- Consider whether previous requests were fully satisfied\n"
        "\n"
        "IMPORTANT: You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "action": {"item_id": quantity, "item_id": quantity, ...}\n'
        "}\n"
        "\n"
        "Example:\n"
        "{\n"
        '  "action": {"cola": 12, "chips": 8, "water": 10, "popcorn": 3}\n'
        "}\n"
        "\n"
        "Do NOT include any other text outside the JSON."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)


def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print('Example: export OPENAI_API_KEY="sk-your-key-here"')
        sys.exit(1)
    
    # Create environment
    env = ta.make(env_id="VendingMachine-v0")
    from textarena.envs.VendingMachine import env as vm_env_module
    original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
    
    # Define items with lead times, profits, and holding costs
    env.add_item(item_id="cola", description="Coca-Cola 12oz Can", lead_time=1, profit=1.5, holding_cost=0.05)
    env.add_item(item_id="chips", description="Potato Chips 1oz Bag", lead_time=2, profit=2.0, holding_cost=0.08)
    env.add_item(item_id="water", description="Bottled Water 16oz", lead_time=0, profit=1.0, holding_cost=0.03)
    env.add_item(item_id="popcorn", description="Popcorn 2oz Bag", lead_time=3, profit=2.5, holding_cost=0.10)
    
    # Add news events
    env.add_news(day=2, news="Weekend Sale: All drinks 20% off!")
    env.add_news(day=5, news="Movie Marathon Event: Expect high snack demand")
    env.add_news(day=6, news="Holiday: Office closed, low traffic expected")
    
    # Create agents
    vm_agent = make_vm_agent()
    demand_agent = make_demand_agent()
    
    # Reset environment
    env.reset(num_players=2)
    
    # Run game
    done = False
    current_day = 1
    carry_over_insights = {}
    
    while not done:
        pid, observation = env.get_observation()
        
        if pid == 0:  # VM agent
            observation = inject_carry_over_insights(observation, carry_over_insights)
            action = vm_agent(observation)
            
            # Print complete JSON output with proper formatting
            print(f"\nDay {current_day} VM Action:")
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
                    carry_over_insights[current_day] = carry_memo
                    carry_text = carry_memo
                else:
                    if current_day in carry_over_insights:
                        del carry_over_insights[current_day]
                    carry_text = "(empty)"

                formatted_json = json.dumps(action_dict, indent=2, ensure_ascii=False)
                print(formatted_json)
                print(f"Carry-over insight: {carry_text}")
                # Flush to ensure complete output to file
                sys.stdout.flush()
            except Exception as e:
                # Fallback to raw output if JSON parsing fails
                print(f"[DEBUG: JSON parsing failed: {e}]")
                print(action)
                sys.stdout.flush()
            print("="*60)
            sys.stdout.flush()
        else:  # Demand agent
            action = demand_agent(observation)
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
        
        revenue = profit * sold
        print(f"\n{item_id} ({item_info['description']}):")
        print(f"  Ordered: {ordered}, Sold: {sold}, Ending: {ending}")
        print(f"  Profit/unit: ${profit}, Holding: ${holding_cost}/unit/day")
        print(f"  Revenue from sales: ${revenue}")
    
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
        print(f"Day {day}{news_str}: Profit=${profit:.2f}, Holding=${holding:.2f}, Reward(R_t)=${reward:.2f}")
    
    # Totals
    total_reward = vm_info.get('total_reward', 0)
    total_profit = vm_info.get('total_sales_profit', 0)
    total_holding = vm_info.get('total_holding_cost', 0)
    
    print("\n" + "="*60)
    print("=== TOTAL SUMMARY ===")
    print("="*60)
    print(f"Total Profit from Sales: ${total_profit:.2f}")
    print(f"Total Holding Cost: ${total_holding:.2f}")
    print(f"\n>>> Total Reward (Profit - Holding): ${total_reward:.2f} <<<")
    print(f"VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)
    
    vm_env_module.INITIAL_INVENTORY_PER_ITEM = original_initial_inventory


if __name__ == "__main__":
    main()
