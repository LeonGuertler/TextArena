"""
Refactored simulation logic with completely separate Mode 1 and Mode 2 implementations.

Mode 1: Multi-turn chat with AI advisor, human makes final decision
Mode 2: AI auto-play with periodic strategic guidance (照搬 llm_csv_demo.py)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import textarena as ta

ModeLiteral = Literal["mode1", "mode2"]


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    mode: ModeLiteral
    demand_file: str
    promised_lead_time: int = 4
    guidance_frequency: int = 5  # Only used for mode2
    max_days: int = 50


@dataclass
class TranscriptEvent:
    """Single event in the simulation transcript."""
    kind: str
    payload: Dict[str, Any]


class SimulationTranscript:
    """Records simulation events."""
    
    def __init__(self):
        self.events: List[TranscriptEvent] = []
    
    def append(self, kind: str, payload: Dict[str, Any]):
        self.events.append(TranscriptEvent(kind=kind, payload=payload))
    
    def to_list(self) -> List[Dict[str, Any]]:
        return [{"kind": e.kind, "payload": e.payload} for e in self.events]


class CSVDemandPlayer:
    """Simulates demand agent by reading from CSV file."""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.item_ids = self._extract_item_ids()
        if not self.item_ids:
            raise ValueError("No item columns found in CSV")
        self._validate_item_columns()
        self.has_news = 'news' in self.df.columns
    
    def _extract_item_ids(self) -> list:
        item_ids = []
        for col in self.df.columns:
            if col.startswith('demand_'):
                item_ids.append(col[len('demand_'):])
        return item_ids
    
    def _validate_item_columns(self):
        required_suffixes = ['demand', 'description', 'lead_time', 'profit', 'holding_cost']
        for item_id in self.item_ids:
            for suffix in required_suffixes:
                col_name = f'{suffix}_{item_id}'
                if col_name not in self.df.columns:
                    raise ValueError(f"CSV missing required column: {col_name}")
    
    def get_item_ids(self) -> list:
        return self.item_ids.copy()
    
    def get_initial_item_configs(self) -> list:
        if len(self.df) == 0:
            raise ValueError("CSV is empty")
        
        first_row = self.df.iloc[0]
        configs = []
        
        for item_id in self.item_ids:
            lead_time_val = first_row[f'lead_time_{item_id}']
            if isinstance(lead_time_val, str) and lead_time_val.lower() == 'inf':
                lead_time = float('inf')
            elif isinstance(lead_time_val, float) and lead_time_val == float('inf'):
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
    
    def get_day_item_config(self, day: int, item_id: str) -> dict:
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range")
        if item_id not in self.item_ids:
            raise ValueError(f"Unknown item_id: {item_id}")
        
        row = self.df.iloc[day - 1]
        lead_time_val = row[f'lead_time_{item_id}']
        if isinstance(lead_time_val, str) and lead_time_val.lower() == 'inf':
            lead_time = float('inf')
        elif isinstance(lead_time_val, float) and lead_time_val == float('inf'):
            lead_time = float('inf')
        else:
            lead_time = int(lead_time_val)
        
        return {
            'description': str(row[f'description_{item_id}']),
            'lead_time': lead_time,
            'profit': float(row[f'profit_{item_id}']),
            'holding_cost': float(row[f'holding_cost_{item_id}'])
        }
    
    def get_num_days(self) -> int:
        return len(self.df)
    
    def get_news_schedule(self) -> dict:
        if not self.has_news:
            return {}
        news_schedule = {}
        for _, row in self.df.iterrows():
            day = int(row['day'])
            if pd.notna(row['news']) and str(row['news']).strip():
                news_schedule[day] = str(row['news']).strip()
        return news_schedule
    
    def get_action(self, day: int) -> str:
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range")
        row = self.df.iloc[day - 1]
        action_dict = {}
        for item_id in self.item_ids:
            col_name = f'demand_{item_id}'
            qty = int(row[col_name])
            action_dict[item_id] = qty
        return json.dumps({"action": action_dict}, indent=2)


def _default_initial_samples() -> dict:
    """Default historical demand samples."""
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    return unified_samples


def _make_mode1_agent(initial_samples: dict, promised_lead_time: int):
    """
    Create Mode 1 AI advisor agent.
    
    The agent acts as an advisor, providing rationale and suggested actions,
    but the human makes the final decision.
    """
    available_items = list(initial_samples.keys())
    items_str = ", ".join([f'"{item}"' for item in available_items])
    
    system = (
        "You are an AI advisor for a Vending Machine controller. "
        "Your role is to provide recommendations to a human decision-maker who will make the final ordering decisions.\n\n"
        
        "IMPORTANT: You are NOT the final decision-maker. The human will review your advice and make their own choice.\n\n"
        
        f"AVAILABLE ITEMS: {items_str}\n"
        "⚠️ CRITICAL: Always use these EXACT item IDs (with parentheses and all special characters) in your suggestions!\n\n"
        
        "Game Mechanics:\n"
        f"- Supplier-promised lead time: {promised_lead_time} days\n"
        "- Orders arrive after a LEAD TIME (actual may differ from promised)\n"
        "- You must INFER actual lead time from arrival records\n"
        "- Daily reward: R_t = Profit × Sold - HoldingCost × EndingInventory\n"
        "- Initial inventory: 0 units per item on Day 1\n\n"
        
        "Multi-Turn Conversation Mode:\n"
        "- You will have multiple exchanges with the human before they make their final decision\n"
        "- Each time, provide your rationale and recommended action\n"
        "- The human may ask questions, request clarifications, or suggest adjustments\n"
        "- Always respond with JSON containing 'rationale' and 'action' keys\n"
        "- Be responsive to the human's feedback and questions\n\n"
    )
    
    if initial_samples:
        system += "HISTORICAL DEMAND DATA:\n"
        for item_id, samples in initial_samples.items():
            system += f"{item_id}: Past demands = {samples}\n"
        system += "\n"
    
    # Build action example
    if available_items:
        action_example = ", ".join([f'"{item}": quantity' for item in available_items[:2]])
        if len(available_items) > 2:
            action_example += ", ..."
    else:
        action_example = '"item_id": quantity, ...'
    
    system += (
        "Response Format (ALWAYS use this format):\n"
        "{\n"
        '  "rationale": "Explain your reasoning in <=5 sentences: (1) inferred lead time, '
        '(2) current inventory analysis, (3) demand patterns and news impact, '
        '(4) your recommendation",\n'
        f'  "action": {{{action_example}}}\n'
        "}\n\n"
        "Remember: You are an ADVISOR. Provide clear recommendations but acknowledge that the human will make the final call.\n"
        "Only return JSON. No Markdown fences or extra commentary."
    )
    
    return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)


def _make_mode2_agent(initial_samples: dict, promised_lead_time: int):
    """
    Create Mode 2 auto-play agent (照搬 llm_csv_demo.py 的 prompt).
    
    The agent runs automatically and will receive strategic guidance prepended to observations.
    """
    available_items = list(initial_samples.keys())
    items_str = ", ".join([f'"{item}"' for item in available_items])
    
    system = (
        "You are the Vending Machine controller (VM). "
        "You manage multiple items, each with unit profit and holding costs. "
        "Objective: Maximize total reward = sum of daily rewards R_t. "
        "Daily reward: R_t = Profit × Sold - HoldingCost × EndingInventory. "
        "\n\n"
        f"AVAILABLE ITEMS: {items_str}\n"
        "⚠️ CRITICAL: You MUST use these EXACT item IDs (with parentheses and all special characters) in your action!\n"
        "\n"
        "Key mechanics:\n"
        f"- Supplier-promised lead time: {promised_lead_time} days\n"
        "- Orders placed today arrive after a LEAD TIME (number of days until delivery)\n"
        "- IMPORTANT: Actual lead time may differ from promised and may change over time!\n"
        "- Lead time is NOT directly revealed. You must INFER it from arrival records.\n"
        "- When goods arrive, you'll see: 'arrived=X units (ordered on Day Y, lead_time was Z days)'\n"
        "- Use this information to track actual lead time and adjust your strategy\n"
        "\n"
        "Inventory visibility:\n"
        "- On-hand: Current inventory available for sale today\n"
        "- In-transit: Total units you ordered that haven't arrived yet (but you don't know WHEN they'll arrive)\n"
        "- You must track your own orders and infer when they'll arrive based on inferred lead_time\n"
        "- IMPORTANT: Initial inventory on Day 1: Each item starts with 0 units on-hand\n"
        "\n"
        "- Holding cost is charged on ending inventory each day\n"
        "- DAILY NEWS: News events are revealed each day (if any). You will NOT know future news in advance.\n"
        "\n"
        "STRATEGIC GUIDANCE:\n"
        "You may receive strategic guidance from a human supervisor that should inform your decisions. "
        "This guidance will appear at the top of your observations and should be followed consistently.\n"
        "\n"
    )
    
    if initial_samples:
        system += "HISTORICAL DEMAND DATA (for reference):\n"
        system += "You have access to the following historical demand samples to help you estimate future demand:\n\n"
        for item_id, samples in initial_samples.items():
            system += f"{item_id}:\n"
            system += f"  Past demands: {samples}\n\n"
        system += "Use this data to inform your ordering decisions, especially on Day 1.\n\n"
    
    example_action = ", ".join([f'"{item}": 100' for item in available_items[:2]])
    if len(available_items) > 2:
        example_action += ", ..."
    
    system += (
        "Strategy:\n"
        "- INFER lead time from arrival records in game history (look for 'lead_time was X days')\n"
        "- Track your own orders and when they should arrive based on inferred lead_time\n"
        "- Use 'In-transit' to see total goods coming, but remember you must infer WHEN they arrive\n"
        "- Study demand patterns from game history\n"
        "- React to TODAY'S NEWS as it happens, accounting for inferred lead time\n"
        "- Learn from past news events to understand their impact on demand\n"
        "- Balance profit vs holding cost (don't overstock)\n"
        "\n"
        "IMPORTANT: Think step by step, then decide.\n"
        "You MUST respond with valid JSON in this exact format:\n"
        "{\n"
        '  "rationale": "First, explain your reasoning: (1) infer current lead_time from recent arrivals, '
        '(2) analyze current inventory (on-hand + in-transit) and demand patterns, '
        '(3) evaluate today\'s news and learn from past events, '
        '(4) consider lead_time when placing orders (goods won\'t arrive immediately!), '
        '(5) explain your ordering strategy",\n'
        '  "action": {' + example_action + '}\n'
        "}\n"
        "\n"
        f"⚠️ REMEMBER: Use EXACT item IDs: {items_str}\n"
        "\n"
        "Think through your rationale BEFORE making the final order decision.\n"
        "Do NOT include any other text outside the JSON."
    )
    
    return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)


class Mode1Session:
    """
    Mode 1: Multi-turn chat with AI advisor, human makes final decision.
    
    Each day:
    1. User can chat multiple times with AI advisor
    2. AI provides rationale + suggested action each time
    3. User submits final decision (two numbers)
    4. User's decision is executed directly
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.transcript = SimulationTranscript()
        self.csv_player = CSVDemandPlayer(config.demand_file)
        self._pending_human_decision: Optional[Dict[str, int]] = None
        
        # Setup environment
        self.env = ta.make(env_id="VendingMachine-v0")
        item_configs = self.csv_player.get_initial_item_configs()
        for item_config in item_configs:
            self.env.add_item(**item_config)
        
        # Add news
        news_schedule = self.csv_player.get_news_schedule()
        for day, news in news_schedule.items():
            self.env.add_news(day, news)
        
        # Set NUM_DAYS and initial inventory
        from textarena.envs.VendingMachine import env as vm_env_module
        self._vm_env_module = vm_env_module
        self._original_num_days = vm_env_module.NUM_DAYS
        self._original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
        vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
        vm_env_module.NUM_DAYS = min(config.max_days, self.csv_player.get_num_days())
        
        # Create AI advisor
        unified_samples = _default_initial_samples()
        initial_samples = {item_id: unified_samples.copy() for item_id in self.csv_player.get_item_ids()}
        self.agent = _make_mode1_agent(initial_samples, config.promised_lead_time)
        
        # Conversation history (for multi-turn chat)
        self.conversation: List[Dict[str, str]] = []  # [{role: "human"|"assistant", content: "..."}]
        
        # Game state
        self.current_day = 1
        self.completed = False
        self.final_reward: Optional[float] = None
        self._pid = -1  # Current player ID waiting for action
        self._current_observation: Optional[str] = None
        
        # Start game
        self.env.reset(num_players=2)
        self._advance_to_vm_turn()
    
    def _advance_to_vm_turn(self):
        """Advance game until it's VM's turn (pid=0) or game ends."""
        while not self.completed:
            pid, observation = self.env.get_observation()
            self._pid = pid
            self._current_observation = observation
            
            if pid == 0:  # VM's turn
                # Update item configs for current day
                for item_id in self.csv_player.get_item_ids():
                    config = self.csv_player.get_day_item_config(self.current_day, item_id)
                self.env.update_item_config(
                    item_id=item_id,
                    lead_time=config['lead_time'],
                    profit=config['profit'],
                    holding_cost=config['holding_cost'],
                    description=config['description']
                )
                break
            else:  # Demand player's turn (pid=1)
                action = self.csv_player.get_action(self.current_day)
                done, _ = self.env.step(action=action)
                
                # Capture structured day summary
                self._record_day_summary(human_decision=self._pending_human_decision)
                self._pending_human_decision = None
                
                if done:
                    self._finalize()
                    break
                
                self.current_day += 1
    
    def add_human_message(self, message: str) -> Dict[str, Any]:
        """
        Add a human message to the conversation and get AI's response.
        
        Returns dict with AI's response (rationale + action).
        """
        if self.completed:
            raise RuntimeError("Game is already completed")
        if self._pid != 0:
            raise RuntimeError("Not VM's turn")
        
        # Add human message to conversation
        self.conversation.append({"role": "human", "content": message})
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": self._current_observation}
        ]
        
        # Add conversation history
        for msg in self.conversation:
            if msg["role"] == "human":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Call LLM
        try:
            completion = self.agent.client.chat.completions.create(
                model=self.agent.model_name,
                messages=messages,
                n=1,
                stop=None
            )
            response = completion.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
        
        # Clean and parse response
        cleaned_response = self._clean_json(response)
        try:
            parsed = json.loads(cleaned_response)
            rationale = parsed.get("rationale", "")
            action = parsed.get("action", {})
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            rationale = response
            action = {}
        
        # Add AI response to conversation
        self.conversation.append({"role": "assistant", "content": response})
        
        return {
            "rationale": rationale,
            "action": action,
            "raw_response": response
        }
    
    def submit_final_decision(self, action_json: str) -> Dict[str, Any]:
        """
        Submit human's final decision and execute it.
        
        action_json: e.g., '{"chips(Regular)": 100, "chips(BBQ)": 50}'
        """
        if self.completed:
            raise RuntimeError("Game is already completed")
        if self._pid != 0:
            raise RuntimeError("Not VM's turn")
        
        # Parse and validate action
        try:
            cleaned_action = self._clean_json(action_json)
            data = json.loads(cleaned_action)
            
            if isinstance(data, dict) and "action" in data:
                action_dict = data["action"]
            elif isinstance(data, dict):
                action_dict = data
            else:
                raise ValueError("Invalid JSON format")
            
            # Validate quantities
            for item_id, quantity in action_dict.items():
                if not isinstance(quantity, (int, float)) or quantity < 0:
                    raise ValueError(f"Invalid quantity for {item_id}: {quantity}")
                action_dict[item_id] = int(quantity)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error processing decision: {e}")
        
        # Record final decision
        self._pending_human_decision = action_dict.copy()
        
        # Execute action
        payload = json.dumps({"action": action_dict})
        done, _ = self.env.step(action=payload)
        
        if done:
            self._finalize()
        else:
            self._advance_to_vm_turn()
        
        return self.serialize_state()
    
    def _record_day_summary(self, *, human_decision: Optional[Dict[str, int]]):
        """Record a structured day summary using environment logs."""
        if not hasattr(self.env, "daily_logs") or not self.env.daily_logs:
            return
        
        day_log = self.env.daily_logs[-1]
        day_number = int(day_log.get("day", self.current_day))
        
        summary = {
            "day": day_number,
            "news": day_log.get("news"),
            "starting_inventory": {item: int(qty) for item, qty in day_log.get("starting_inventory", {}).items()},
            "orders": {item: int(qty) for item, qty in day_log.get("orders", {}).items()},
            "arrivals": {
                item: [
                    {
                        "quantity": int(entry[0]),
                        "ordered_day": int(entry[1]),
                        "actual_lead_time": max(day_number - int(entry[1]), 0),
                    }
                    for entry in entries
                ]
                for item, entries in day_log.get("arrivals", {}).items()
            },
            "demand": {item: int(qty) for item, qty in day_log.get("requests", {}).items()},
            "sales": {item: int(qty) for item, qty in day_log.get("sales", {}).items()},
            "ending_inventory": {item: int(qty) for item, qty in day_log.get("ending_inventory", {}).items()},
            "reward": {
                "daily_reward": round(float(day_log.get("daily_reward", 0.0)), 2),
                "profit": round(float(day_log.get("daily_profit", 0.0)), 2),
                "holding_cost": round(float(day_log.get("daily_holding_cost", 0.0)), 2),
            },
        }
        
        if human_decision:
            summary["human_decision"] = {item: int(qty) for item, qty in human_decision.items()}
        
        self.transcript.append("day_summary", summary)
        self.conversation = []  # Clear prior chat context once day wraps
    
    def _finalize(self):
        """Finalize the game and calculate rewards."""
        rewards, game_info = self.env.close()
        self.completed = True
        self.final_reward = rewards.get(0, 0.0)
        self.transcript.append("game_complete", {
            "final_reward": self.final_reward,
            "game_info": game_info[0] if game_info else {}
        })
        self._vm_env_module.NUM_DAYS = self._original_num_days
        self._vm_env_module.INITIAL_INVENTORY_PER_ITEM = self._original_initial_inventory
    
    def _clean_json(self, text: str) -> str:
        """Remove markdown fences from LLM output."""
        cleaned = text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return cleaned.strip()
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize current state for API responses."""
        return {
            "mode": self.config.mode,
            "current_day": self.current_day,
            "completed": self.completed,
            "final_reward": self.final_reward,
            "waiting_for_final_action": (not self.completed and self._pid == 0),
            "conversation": self.conversation,
            "transcript": self.transcript.to_list()
        }


class Mode2Session:
    """
    Mode 2: AI auto-play with periodic strategic guidance.
    
    Guidance is prepended to observations (照搬 llm_csv_demo.py).
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.transcript = SimulationTranscript()
        self.csv_player = CSVDemandPlayer(config.demand_file)
        
        # Setup environment
        self.env = ta.make(env_id="VendingMachine-v0")
        item_configs = self.csv_player.get_initial_item_configs()
        for item_config in item_configs:
            self.env.add_item(**item_config)
        
        # Add news
        news_schedule = self.csv_player.get_news_schedule()
        for day, news in news_schedule.items():
            self.env.add_news(day, news)
        
        # Set NUM_DAYS and initial inventory
        from textarena.envs.VendingMachine import env as vm_env_module
        self._vm_env_module = vm_env_module
        self._original_num_days = vm_env_module.NUM_DAYS
        self._original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
        vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
        vm_env_module.NUM_DAYS = min(config.max_days, self.csv_player.get_num_days())
        
        # Create AI agent
        unified_samples = _default_initial_samples()
        initial_samples = {item_id: unified_samples.copy() for item_id in self.csv_player.get_item_ids()}
        self.agent = _make_mode2_agent(initial_samples, config.promised_lead_time)
        
        # Guidance management (Mode 2)
        self.accumulated_guidance: List[Dict[str, Any]] = []
        self.pending_guidance_day: Optional[int] = None  # Day waiting for guidance
        self._last_guidance_request_day: Optional[int] = None
        
        # Game state
        self.current_day = 1
        self.completed = False
        self.final_reward: Optional[float] = None
        
        # Start game
        self.env.reset(num_players=2)
        
        # Always request initial guidance before starting
        self.pending_guidance_day = 0  # Day 0 = before game starts
        self.transcript.append("initial_guidance_request", {"day": 0})
        self._last_guidance_request_day = 0
    
    def _should_collect_guidance(self, day: int) -> bool:
        """Check if we should collect guidance on this day."""
        if self.config.guidance_frequency <= 0:
            return False
        if self._last_guidance_request_day == day:
            return False
        # Guidance at day 0 (before start), then every guidance_frequency days
        # If frequency=5: day 0, day 5, day 10, etc.
        return (day % self.config.guidance_frequency) == 0
    
    def submit_guidance(self, guidance: str) -> Dict[str, Any]:
        """Submit strategic guidance and continue running."""
        if self.completed:
            raise RuntimeError("Game is already completed")
        if self.pending_guidance_day is None:
            raise RuntimeError("Not waiting for guidance")
        
        trimmed_guidance = guidance.strip()
        if trimmed_guidance:
            self.accumulated_guidance.append({
                "day": self.pending_guidance_day,
                "guidance": trimmed_guidance,
            })
        
        self.transcript.append("guidance_submitted", {
            "day": self.pending_guidance_day,
            "guidance": guidance
        })
        
        self.pending_guidance_day = None
        
        # Continue running
        self._run_until_pause_or_complete()
        
        return self.serialize_state()
    
    def _run_until_pause_or_complete(self):
        """Run the simulation until we need guidance or game ends."""
        while not self.completed:
            pid, observation = self.env.get_observation()
            
            if pid == 0:  # VM's turn
                # Check if we should pause for guidance
                if self._should_collect_guidance(self.current_day):
                    self.pending_guidance_day = self.current_day
                    self._last_guidance_request_day = self.current_day
                    self.transcript.append("guidance_request", {"day": self.current_day})
                    return  # Pause here
                
                # Update item configs
                for item_id in self.csv_player.get_item_ids():
                    config = self.csv_player.get_day_item_config(self.current_day, item_id)
                    self.env.update_item_config(
                        item_id=item_id,
                        lead_time=config['lead_time'],
                        profit=config['profit'],
                        holding_cost=config['holding_cost'],
                        description=config['description']
                    )
                
                # Inject guidance into observation
                enhanced_obs = self._inject_guidance(observation)
                
                # Get AI action
                action = self.agent(enhanced_obs)
                
                done, _ = self.env.step(action=action)
                if done:
                    self._finalize()
                    return
                # Continue to next turn (demand player)
            
            else:  # Demand player's turn
                action = self.csv_player.get_action(self.current_day)
                done, _ = self.env.step(action=action)
                if done:
                    self._finalize()
                    return
                
                # Record daily summary
                self._record_daily_summary()
                self.current_day += 1
    
    def _record_daily_summary(self):
        """Record a structured daily summary, including guidance context."""
        if not hasattr(self.env, "daily_logs") or not self.env.daily_logs:
            return
        
        day_log = self.env.daily_logs[-1]
        day_number = int(day_log.get("day", self.current_day))
        
        summary = {
            "day": day_number,
            "news": day_log.get("news"),
            "starting_inventory": {item: int(qty) for item, qty in day_log.get("starting_inventory", {}).items()},
            "orders": {item: int(qty) for item, qty in day_log.get("orders", {}).items()},
            "arrivals": {
                item: [
                    {
                        "quantity": int(entry[0]),
                        "ordered_day": int(entry[1]),
                        "actual_lead_time": max(day_number - int(entry[1]), 0),
                    }
                    for entry in entries
                ]
                for item, entries in day_log.get("arrivals", {}).items()
            },
            "demand": {item: int(qty) for item, qty in day_log.get("requests", {}).items()},
            "sales": {item: int(qty) for item, qty in day_log.get("sales", {}).items()},
            "ending_inventory": {item: int(qty) for item, qty in day_log.get("ending_inventory", {}).items()},
            "reward": {
                "daily_reward": round(float(day_log.get("daily_reward", 0.0)), 2),
                "profit": round(float(day_log.get("daily_profit", 0.0)), 2),
                "holding_cost": round(float(day_log.get("daily_holding_cost", 0.0)), 2),
            },
            "guidance_in_effect": [dict(entry) for entry in self.accumulated_guidance],
        }
        
        self.transcript.append("day_summary", summary)
    
    def _inject_guidance(self, observation: str) -> str:
        """Prepend accumulated guidance to observation."""
        if not self.accumulated_guidance:
            return observation
        
        guidance_section = "\n\n" + "=" * 70 + "\n"
        guidance_section += "HUMAN STRATEGIC GUIDANCE (apply to your decisions)\n"
        guidance_section += "=" * 70 + "\n"
        
        for idx, entry in enumerate(self.accumulated_guidance, 1):
            day = entry.get("day")
            message = entry.get("guidance", "")
            quoted = json.dumps(message)
            if day is not None:
                guidance_section += f"\nGuidance {idx} (Day {day}): {quoted}\n"
            else:
                guidance_section += f"\nGuidance {idx}: {quoted}\n"
        
        guidance_section += "=" * 70 + "\n"
        
        return guidance_section + observation
    
    def _finalize(self):
        """Finalize the game."""
        rewards, game_info = self.env.close()
        self.completed = True
        self.final_reward = rewards.get(0, 0.0)
        self.transcript.append("game_complete", {
            "final_reward": self.final_reward,
            "game_info": game_info[0] if game_info else {}
        })
        self._vm_env_module.NUM_DAYS = self._original_num_days
        self._vm_env_module.INITIAL_INVENTORY_PER_ITEM = self._original_initial_inventory
    
    def serialize_state(self) -> Dict[str, Any]:
        """Serialize current state for API responses."""
        return {
            "mode": self.config.mode,
            "current_day": self.current_day,
            "completed": self.completed,
            "final_reward": self.final_reward,
            "waiting_for_guidance": (self.pending_guidance_day is not None),
            "transcript": self.transcript.to_list()
        }


def load_simulation(config: SimulationConfig):
    """Factory function to create appropriate session based on mode."""
    if config.mode == "mode1":
        return Mode1Session(config)
    elif config.mode == "mode2":
        return Mode2Session(config)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

