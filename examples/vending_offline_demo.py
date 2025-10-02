"""Minimal offline demo for VendingMachineEnv using OpenAI gpt-5-mini agents.

Multi-item vending machine with lead times and inventory pipeline.

Instructions:
- Set your OpenAI API key as environment variable: export OPENAI_API_KEY="your-key-here"
- Run: python examples/vending_offline_demo.py
"""

import os
import textarena as ta

# --- Configure OpenAI API key ---
# Set your OpenAI API key as an environment variable: export OPENAI_API_KEY="your-key-here"
# Or create a .env file with: OPENAI_API_KEY=your-key-here
if not os.getenv("OPENAI_API_KEY"):
    print("Error: Please set your OPENAI_API_KEY environment variable")
    print('Example: $env:OPENAI_API_KEY="sk-your-actual-key-here"')
    exit(1)


def make_vm_agent():
    """Vending Machine agent (OpenAI gpt-5-mini)."""
    system = (
        "You are the Vending Machine controller (VM). "
        "You manage multiple items, each with different prices, costs, lead times, and holding costs. "
        "Objective: Maximize total reward = sum of daily rewards R_t. "
        "Daily reward: R_t = (Price - Cost) × Sold - HoldingCost × EndingInventory. "
        "\n\n"
        "Key mechanics:\n"
        "- Orders placed today arrive after the item's lead time (e.g., lead=2 means arrives in 2 days)\n"
        "- You see on-hand inventory and pipeline for each item\n"
        "- Holding cost is charged on ending inventory each day (incentive to keep inventory low)\n"
        "- NEWS SCHEDULE: You can see all scheduled news events (e.g., holidays, promotions) for future days\n"
        "\n"
        "Strategy:\n"
        "- Study demand patterns from game history for each item\n"
        "- PAY ATTENTION to the news schedule and plan inventory accordingly\n"
        "- Order enough to cover demand during lead time + buffer, but minimize holding costs\n"
        "- Consider profit margins and holding costs when prioritizing which items to stock\n"
        "- Anticipate demand changes based on scheduled news events\n"
        "\n"
        "Action format: [Order] item_id:qty=N, item_id:qty=N\n"
        "Example: [Order] cola:qty=15, chips:qty=10\n"
        "You can order any non-negative quantity for each item."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-5-mini", system_prompt=system)


def make_demand_agent():
    """Demand agent (OpenAI gpt-5-mini)."""
    system = (
        "You are the Demand agent (customer). "
        "You can purchase multiple items each day. "
        "You do NOT see current inventory levels. "
        "\n"
        "Key information:\n"
        "- You can see item prices and lead times\n"
        "- NEWS SCHEDULE: You can see all scheduled news events that may affect your demand\n"
        "- You should adjust your purchasing behavior based on news events\n"
        "\n"
        "Strategy:\n"
        "- Choose purchase quantities based on item prices and your preferences\n"
        "- PAY ATTENTION to news events (e.g., holidays, sales) and adjust demand accordingly\n"
        "- Add some randomness to simulate realistic demand\n"
        "- Adjust slightly based on recent history to create patterns\n"
        "\n"
        "Action format: [Buy] item_id:qty=N, item_id:qty=N\n"
        "Example: [Buy] cola:qty=5, chips:qty=3\n"
        "You can request any non-negative quantity for each item."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-5-mini", system_prompt=system)


def main():
    agents = {
        0: make_vm_agent(),
        1: make_demand_agent(),
    }

    env = ta.make(env_id="VendingMachine-v0")
    
    # Add items to the vending machine
    # add_item(item_id, description, lead_time, price, cost, holding_cost)
    env.add_item(item_id="cola", description="Cola", lead_time=1, price=7, cost=4, holding_cost=0.5)
    env.add_item(item_id="chips", description="Chips", lead_time=2, price=5, cost=3, holding_cost=0.3)
    env.add_item(item_id="water", description="Water", lead_time=0, price=3, cost=2, holding_cost=0.2)
    env.add_item(item_id="popcorn", description="Popcorn", lead_time=3, price=8, cost=5, holding_cost=0.4)
    
    # Add news for specific days (optional)
    # Both agents see the complete news schedule from the start
    env.add_news(day=2, news="Weekend Sale: Expect 30% higher demand for all items")
    env.add_news(day=3, news="Post-weekend: Demand returns to normal levels")
    env.add_news(day=6, news="Baseball final game: Expect 50% higher demand for popcorn")
    
    env.reset(num_players=2)

    done = False
    while not done:
        pid, observation = env.get_observation()
        
        # Get agent action (no output during gameplay)
        action = agents[pid](observation)
        done, _ = env.step(action=action)

    rewards, game_info = env.close()
    print("\n" + "="*60)
    print("=== Final Results ===")
    print("="*60)
    vm_info = game_info[0]
    
    # Display results per item
    total_ordered = vm_info.get('total_ordered', {})
    total_sold = vm_info.get('total_sold', {})
    ending_inventory = vm_info.get('ending_inventory', {})
    items = vm_info.get('items', {})
    
    print("\nPer-Item Statistics:")
    total_revenue = 0
    total_cogs = 0  # Cost of Goods Sold
    for item_id, item_info in items.items():
        ordered = total_ordered.get(item_id, 0)
        sold = total_sold.get(item_id, 0)
        ending = ending_inventory.get(item_id, 0)
        price = item_info['price']
        cost = item_info['cost']
        holding_cost = item_info['holding_cost']
        
        revenue = price * sold
        cogs = cost * sold  # Cost of goods actually sold
        margin_profit = revenue - cogs
        
        total_revenue += revenue
        total_cogs += cogs
        
        print(f"\n{item_id} ({item_info['description']}):")
        print(f"  Ordered: {ordered} units, Sold: {sold} units, Ending: {ending} units")
        print(f"  Price: ${price}, Cost: ${cost}, Holding: ${holding_cost}/unit/day")
        print(f"  Revenue: ${revenue}, COGS: ${cogs}, Margin: ${margin_profit}")
    
    # Display daily breakdown
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
    
    # Display totals
    total_reward = vm_info.get('total_reward', 0)
    total_margin_profit = vm_info.get('total_sales_profit', 0)  # This is margin profit from daily calculations
    total_holding_cost = vm_info.get('total_holding_cost', 0)
    total_procurement_cost = vm_info.get('total_procurement_cost', 0)
    
    print("\n" + "="*60)
    print("=== TOTAL SUMMARY ===")
    print("="*60)
    print(f"Total Revenue (Price × Sold): ${total_revenue:.2f}")
    print(f"Total COGS (Cost × Sold): ${total_cogs:.2f}")
    print(f"Total Margin Profit (Revenue - COGS): ${total_margin_profit:.2f}")
    print(f"Total Holding Cost: ${total_holding_cost:.2f}")
    print(f"\n>>> Total Reward (Margin - Holding): ${total_reward:.2f} <<<")
    print(f"\nAdditional Info:")
    print(f"  Total Procurement Cost (Cost × Ordered): ${total_procurement_cost:.2f}")
    print(f"  VM Final Reward: {rewards.get(0, 0):.2f}")
    print("="*60)


if __name__ == "__main__":
    main()


