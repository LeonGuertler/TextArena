"""
Quick manual test for _inject_carry_over_insights helper.

Run:
    python examples/fullstack_demo/backend/test_carry_over_injection.py
"""

from simulation_current import _inject_carry_over_insights


def main():
    observation = (
        "=== CURRENT STATUS ===\n"
        "PERIOD 3 / 5\n"
        "=== GAME HISTORY ===\n"
        "Period 1 conclude:\n"
        "  Sample text\n"
        "Period 2 conclude:\n"
        "  Sample text\n"
        "Period 3 conclude:\n"
        "  Sample text\n"
    )
    insights = {
        2: "Lead time confirmed at 1 period.",
        3: "Demand regime shifted up."
    }
    augmented = _inject_carry_over_insights(observation, insights)
    print("Original observation:\n", observation)
    print("\nAugmented observation:\n", augmented)


if __name__ == "__main__":
    main()

