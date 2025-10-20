"""
Simple test script for human feedback functionality.
This script tests the HumanFeedbackAgent wrapper without requiring full game execution.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import textarena as ta


def test_wrapper_creation():
    """Test that HumanFeedbackAgent can be created and wraps correctly."""
    print("Testing HumanFeedbackAgent creation...")
    
    # Create a base agent (using a mock if OpenAI key not available)
    try:
        base_agent = ta.agents.OpenAIAgent(
            model_name="gpt-4o-mini",
            system_prompt="You are a test agent."
        )
        print("[OK] Base OpenAIAgent created")
    except Exception as e:
        print(f"[WARN] Could not create OpenAIAgent (API key may be missing): {e}")
        print("  This is expected if OPENAI_API_KEY is not set.")
        return
    
    # Test Mode 1 only
    wrapper1 = ta.agents.HumanFeedbackAgent(
        base_agent=base_agent,
        enable_daily_feedback=True,
        guidance_frequency=0
    )
    print("[OK] HumanFeedbackAgent created with Mode 1 (daily feedback)")
    
    # Test Mode 2 only
    wrapper2 = ta.agents.HumanFeedbackAgent(
        base_agent=base_agent,
        enable_daily_feedback=False,
        guidance_frequency=5
    )
    print("[OK] HumanFeedbackAgent created with Mode 2 (periodic guidance)")
    
    # Test both modes
    wrapper3 = ta.agents.HumanFeedbackAgent(
        base_agent=base_agent,
        enable_daily_feedback=True,
        guidance_frequency=5
    )
    print("[OK] HumanFeedbackAgent created with both modes")
    
    # Test guidance scheduling
    wrapper3.current_day = 5
    assert wrapper3.should_collect_guidance() == True
    print("[OK] Guidance collection scheduled correctly (day 5)")
    
    wrapper3.current_day = 6
    assert wrapper3.should_collect_guidance() == False
    print("[OK] Guidance collection skipped correctly (day 6)")
    
    wrapper3.current_day = 10
    assert wrapper3.should_collect_guidance() == True
    print("[OK] Guidance collection scheduled correctly (day 10)")
    
    print("\n[SUCCESS] All tests passed!")


def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    
    # Test that HumanFeedbackAgent is available
    assert hasattr(ta.agents, 'HumanFeedbackAgent')
    print("[OK] HumanFeedbackAgent is available in ta.agents")
    
    # Test that it's in __all__
    from textarena.agents import __all__
    assert 'HumanFeedbackAgent' in __all__
    print("[OK] HumanFeedbackAgent is exported in __all__")
    
    print("\n[SUCCESS] All imports successful!")


if __name__ == "__main__":
    print("="*70)
    print("Human Feedback Agent Test Suite")
    print("="*70)
    print()
    
    try:
        test_imports()
        print()
        test_wrapper_creation()
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)

