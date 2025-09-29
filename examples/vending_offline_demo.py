"""Minimal offline demo for VendingMachineEnv using OpenAI gpt-5-mini agents.

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
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)


def make_vm_agent():
    """Vending Machine agent (OpenAI gpt-5-mini)."""
    system = (
        "You are the Vending Machine controller (VM). Each day you may restock any non-negative integer quantity. "
        "Objective: Maximize end-of-episode profit. Profit = 7 * TotalSold - 5 * TotalRestocked. "
        "Each unit costs $5 to restock and sells for $7, so profit per unit is $2. "
        "Strategy: Stock enough to meet expected demand. Look at demand patterns from game history. "
        "If demand was 10-15 units yesterday, stock 12-16 units today to capture most sales. "
        "Output exactly one bracketed action at the end: [Restock] qty=INTEGER."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-5-mini", system_prompt=system)


def make_demand_agent():
    """Demand agent (OpenAI gpt-5-mini)."""
    system = (
        "You are the Demand agent. Each day choose a purchase quantity (0..20). "
        "You do NOT know current inventory. Randomize within 0..20 and adjust mildly using recent history to avoid repeating the exact same number every day. "
        "Output exactly one bracketed action at the end: [Buy] qty=INTEGER."
    )
    return ta.agents.OpenAIAgent(model_name="gpt-5-mini", system_prompt=system)


def main():
    agents = {
        0: make_vm_agent(),
        1: make_demand_agent(),
    }

    env = ta.make(env_id="VendingMachine-v0")
    env.reset(num_players=2)

    done = False
    while not done:
        pid, observation = env.get_observation()
        
        # Get agent action (no output during gameplay)
        action = agents[pid](observation)
        done, _ = env.step(action=action)

    rewards, game_info = env.close()
    print("\n=== Final Results ===")
    vm_info = game_info[0]
    unit_price = vm_info.get('unit_price', 7)
    unit_cost = vm_info.get('unit_cost', 5)
    total_sold = vm_info.get('total_sold', 0)
    total_restocked = vm_info.get('total_restocked', 0)
    
    print(f"Total Restocked: {total_restocked} units")
    print(f"Total Sold: {total_sold} units") 
    print(f"Total Requested: {vm_info.get('total_requested', 0)} units")
    print(f"Ending Inventory: {vm_info.get('ending_inventory', 0)} units")
    print(f"Revenue: ${unit_price * total_sold}")
    print(f"Cost: ${unit_cost * total_restocked}")
    print(f"Profit: ${vm_info.get('total_profit', 0)}")
    print(f"VM Reward: {rewards.get(0, 0)}")


if __name__ == "__main__":
    main()


