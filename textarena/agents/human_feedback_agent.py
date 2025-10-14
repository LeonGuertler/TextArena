"""
Human-in-the-Loop Agent Wrapper

This module provides a wrapper for LLM agents that enables two modes of human feedback:
- Mode 1 (Daily Feedback): Human can provide feedback on agent's initial rationale/decision
  for immediate adjustment. This feedback is ephemeral and only affects that day's decision.
- Mode 2 (Periodic Guidance): Human provides strategic guidance every N days that persists
  in the context window and influences all subsequent decisions.
"""

import json
import re
import sys
from typing import Optional, List
from textarena.core import Agent


class HumanFeedbackAgent(Agent):
    """
    Wrapper agent that adds human-in-the-loop feedback capabilities.
    
    Args:
        base_agent: The underlying LLM agent (e.g., OpenAIAgent)
        enable_daily_feedback: If True, enables Mode 1 (daily feedback on decisions)
        guidance_frequency: How often to collect strategic guidance (Mode 2). 
                           0 = disabled, N = every N days
    """
    
    def __init__(self, base_agent, enable_daily_feedback: bool = True, 
                 guidance_frequency: int = 0):
        super().__init__()
        self.base_agent = base_agent
        self.enable_daily_feedback = enable_daily_feedback
        self.guidance_frequency = guidance_frequency
        self.accumulated_guidance: List[str] = []  # Mode 2 persistent guidance
        self.current_day = 1
        
        # Access the underlying OpenAI client for multi-turn conversations
        if hasattr(base_agent, 'client'):
            self.client = base_agent.client
            self.model_name = base_agent.model_name
            self.system_prompt = base_agent.system_prompt
        else:
            raise ValueError("Base agent must have 'client', 'model_name', and 'system_prompt' attributes")
    
    def should_collect_guidance(self) -> bool:
        """Check if we should collect Mode 2 guidance on current day."""
        if self.guidance_frequency <= 0:
            return False
        return (self.current_day % self.guidance_frequency) == 0
    
    def format_game_history_for_human(self, observation: str) -> str:
        """
        Extract and format game history from observation for human display.
        
        Shows complete game history including:
        - Daily summaries (orders, sales, stock, profit, holding cost)
        - News events
        - Current inventory status
        """
        # The observation already contains formatted game history
        # We'll extract and present it in a clean way
        
        lines = []
        lines.append("=" * 70)
        lines.append("GAME STATE SUMMARY FOR YOUR REVIEW")
        lines.append("=" * 70)
        
        # Extract current day info
        if "DAY" in observation:
            day_match = re.search(r'DAY (\d+) / (\d+)', observation)
            if day_match:
                current_day = day_match.group(1)
                total_days = day_match.group(2)
                lines.append(f"\nCurrent Day: {current_day} / {total_days}")
        
        # Extract news (if present)
        if "TODAY'S NEWS" in observation or "Past News" in observation:
            lines.append("\n--- NEWS ---")
            news_section = self._extract_section(observation, "NEWS")
            if news_section:
                lines.append(news_section)
        
        # Extract current status
        if "CURRENT STATUS" in observation or "ITEMS" in observation:
            lines.append("\n--- CURRENT INVENTORY STATUS ---")
            status_section = self._extract_section(observation, "ITEMS")
            if status_section:
                lines.append(status_section)
        
        # Extract game history
        if "GAME HISTORY" in observation:
            lines.append("\n--- PAST DAYS SUMMARY ---")
            history_section = self._extract_section(observation, "GAME HISTORY")
            if history_section:
                lines.append(history_section)
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from the observation text."""
        lines = text.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if section_name in line:
                in_section = True
                continue
            if in_section:
                if line.startswith('===') and section_name not in line:
                    break
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    def collect_daily_feedback(self, initial_response: str, game_state: str) -> str:
        """
        Mode 1: Collect human feedback on agent's initial decision.
        
        Args:
            initial_response: The agent's initial rationale and decision
            game_state: Formatted game state for human review
            
        Returns:
            Human feedback text (can be empty)
        """
        print("\n" + "=" * 70)
        print("AGENT'S INITIAL DECISION (Mode 1 - Daily Feedback)")
        print("=" * 70)
        
        # Display the agent's raw response directly - ensure complete output
        print("\n" + initial_response)
        sys.stdout.flush()  # Force flush to ensure complete output
        
        print("\n" + "=" * 70)
        print("YOUR TURN: Provide feedback on this decision (or press Enter to accept)")
        print("=" * 70)
        print("You can:")
        print("- Suggest adjustments to the order quantities")
        print("- Point out considerations the agent might have missed")
        print("- Just press Enter to accept the agent's decision as-is")
        print("-" * 70)
        sys.stdout.flush()
        
        feedback = input("\nYour feedback: ").strip()
        
        if feedback:
            print(f"\n[OK] Feedback recorded. Asking agent to reconsider...")
        else:
            print(f"\n[OK] No feedback. Using agent's initial decision.")
        
        sys.stdout.flush()
        return feedback
    
    def collect_periodic_guidance(self) -> str:
        """
        Mode 2: Collect strategic guidance that persists in context.
        
        Returns:
            Strategic guidance text
        """
        print("\n" + "=" * 70)
        print(f"STRATEGIC GUIDANCE REQUEST (Mode 2 - Day {self.current_day})")
        print("=" * 70)
        print("Provide strategic guidance for the agent to follow in upcoming decisions.")
        print("This guidance will be remembered and applied to all future decisions.")
        print()
        print("Examples:")
        print("- 'Be more conservative with ordering to reduce holding costs'")
        print("- 'When you see news about events, increase orders by 50%'")
        print("- 'Focus on maintaining higher stock levels for chips(Regular)'")
        print("-" * 70)
        
        guidance = input("\nYour strategic guidance: ").strip()
        
        if guidance:
            print(f"\n✓ Guidance recorded and will persist in future decisions.")
        else:
            print(f"\n✓ No guidance provided.")
        
        return guidance
    
    def inject_guidance(self, observation: str) -> str:
        """
        Inject accumulated Mode 2 guidance into the observation.
        
        Args:
            observation: The original game observation
            
        Returns:
            Enhanced observation with guidance prepended
        """
        if not self.accumulated_guidance:
            return observation
        
        guidance_section = "\n\n" + "=" * 70 + "\n"
        guidance_section += "HUMAN STRATEGIC GUIDANCE (apply to your decisions)\n"
        guidance_section += "=" * 70 + "\n"
        
        for i, guidance in enumerate(self.accumulated_guidance, 1):
            guidance_section += f"\nGuidance {i}: {guidance}\n"
        
        guidance_section += "=" * 70 + "\n"
        
        return guidance_section + observation
    
    def construct_feedback_conversation(self, observation: str, 
                                       initial_response: str, 
                                       feedback: str) -> List[dict]:
        """
        Construct a multi-turn conversation for the second LLM call.
        
        This creates the conversation structure:
        - System: background knowledge + game rules
        - User: past records + news schedule
        - Assistant: agent's initial rationale and decision
        - User: human feedback (requesting only action output)
        
        Args:
            observation: The game observation
            initial_response: Agent's initial response
            feedback: Human feedback
            
        Returns:
            List of message dicts for OpenAI API
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": observation
            },
            {
                "role": "assistant",
                "content": initial_response
            },
            {
                "role": "user",
                "content": f"HUMAN FEEDBACK: {feedback}\n\nBased on this feedback, provide your final decision. Output ONLY the action in JSON format: {{\"action\": {{\"item_id\": quantity, ...}}}}. Do NOT include rationale."
            }
        ]
        
        return messages
    
    def __call__(self, observation: str) -> str:
        """
        Main entry point. Coordinates both feedback modes.
        
        Args:
            observation: The game observation from environment
            
        Returns:
            Final action string (JSON format)
        """
        # Mode 2: Check if we should collect strategic guidance
        if self.should_collect_guidance():
            guidance = self.collect_periodic_guidance()
            if guidance:
                self.accumulated_guidance.append(guidance)
        
        # Inject accumulated guidance into observation
        enhanced_observation = self.inject_guidance(observation)
        
        # Mode 1: Two-stage decision process with daily feedback
        if self.enable_daily_feedback:
            # Format game state for human
            game_state = self.format_game_history_for_human(observation)
            
            # Stage 1: Get initial decision from agent
            print(f"\n{'='*70}")
            print(f"Day {self.current_day} - Stage 1: Agent is thinking...")
            print(f"{'='*70}")
            sys.stdout.flush()
            initial_response = self.base_agent(enhanced_observation)
            
            # Collect human feedback
            feedback = self.collect_daily_feedback(initial_response, game_state)
            
            if feedback.strip():
                # Stage 2: Re-query agent with feedback
                print(f"\n{'='*70}")
                print(f"Day {self.current_day} - Stage 2: Agent reconsidering with your feedback...")
                print(f"{'='*70}")
                sys.stdout.flush()
                
                messages = self.construct_feedback_conversation(
                    enhanced_observation, initial_response, feedback
                )
                
                # Make the second API call
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        n=1,
                        stop=None
                    )
                    final_response = completion.choices[0].message.content.strip()
                    
                    print("\nAgent's Final Action (after feedback):")
                    print("-" * 70)
                    print(final_response)
                    print("-" * 70)
                    sys.stdout.flush()
                    
                    self.current_day += 1
                    return final_response
                except Exception as e:
                    print(f"\nError in second API call: {e}")
                    print("Falling back to initial decision.")
                    self.current_day += 1
                    return initial_response
            else:
                # No feedback, use initial decision
                self.current_day += 1
                return initial_response
        else:
            # Mode 1 disabled, just use base agent
            response = self.base_agent(enhanced_observation)
            self.current_day += 1
            return response

