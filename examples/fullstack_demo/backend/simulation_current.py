"""Simulation helpers for fullstack demo - imports external test scripts."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import textarena as ta

# Add examples directory to path for importing external scripts
EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(EXAMPLES_DIR))

# Import from external test scripts
from or_csv_demo import CSVDemandPlayer, ORAgent
from or_to_llm_csv_demo import GPT5MiniAgent, make_hybrid_vm_agent


def _sanitize_text(text: str) -> str:
    """Normalize to NFKC and escape remaining non-ASCII characters."""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.encode("ascii", "backslashreplace").decode("ascii")


def _safe_print(text: str) -> None:
    """Print text with encoding fallback for Windows."""
    print(_sanitize_text(str(text)))


ModeLiteral = Literal["modeA", "modeB", "modeC"]


@dataclass
class SimulationConfig:
    mode: ModeLiteral
    demand_file: str  # test.csv path
    train_file: str   # train.csv path for initial samples
    promised_lead_time: int = 1  # Fixed to 1
    guidance_frequency: int = 5
    enable_or: bool = True  # Flag to enable/disable OR recommendations


@dataclass
class TranscriptEvent:
    kind: str
    payload: Dict[str, Any]


@dataclass
class SimulationTranscript:
    events: List[TranscriptEvent] = field(default_factory=list)
    completed: bool = False
    final_reward: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def append(self, kind: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(TranscriptEvent(kind=kind, payload=payload))


DAY_CONCLUDED_PATTERN = re.compile(r'^(\s*Day\s+(\d+)\s+concluded:)(.*)$')


def _inject_carry_over_insights(observation: str, insights: Dict[int, str]) -> str:
    if not insights:
        return observation

    lines = observation.splitlines()
    augmented: List[str] = []

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


def _make_base_agent(
    *,
    item_ids: List[str],
    initial_samples: Dict[str, List[int]],
    promised_lead_time: int,
    human_feedback_enabled: bool,
    guidance_enabled: bool,
    or_enabled: bool = False,
) -> ta.Agent:
    """Create base agent using external script's agent creation function."""
    # Use the external make_hybrid_vm_agent from or_to_llm_csv_demo
    return make_hybrid_vm_agent(
        initial_samples=initial_samples,
        promised_lead_time=promised_lead_time,
        human_feedback_enabled=human_feedback_enabled,
        guidance_enabled=guidance_enabled
    )


class SimulationSession:
    """Stateful session managing game state and human interaction."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.csv_player = CSVDemandPlayer(config.demand_file, initial_samples=None)
        self.transcript = SimulationTranscript()

        self.current_day = 1
        self._guidance_messages: Dict[int, str] = {}
        self._guidance_history: List[Tuple[int, str]] = []
        self._pending_guidance_day: Optional[int] = None
        self._carry_over_insights: Dict[int, str] = {}
        self._ui_daily_logs: List[Dict[str, Any]] = []
        self._running_reward: float = 0.0
        self._streaming_text: str = ""  # Store partial LLM response for streaming
        self._streaming_lock: threading.Lock = threading.Lock()  # Lock for streaming text access
        self._step_logging_callback: Optional[Callable[[Dict[str, Any]], None]] = None  # Callback for logging steps
        
        # Extract initial samples from train.csv
        self._initial_samples, self._initial_sample_dates = self._load_initial_samples()
        
        # Print initialization info to terminal
        print(f"\n{'='*70}")
        print(f"SIMULATION INITIALIZATION - {config.mode.upper()}")
        print(f"{'='*70}")
        print(f"Total periods: {self.csv_player.get_num_periods()}")
        print(f"Promised lead time: {config.promised_lead_time} periods")
        print(f"Items: {', '.join(self.csv_player.get_item_ids())}")
        print(f"Initial samples: {self._initial_samples}")
        if config.mode == "modeC":
            print(f"Guidance frequency: Every {config.guidance_frequency} periods")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        # Initialize OR agent if enabled
        self._or_agent: Optional[ORAgent] = None
        self._or_recommendations: Dict[str, int] = {}
        self._or_statistics: Dict[str, Dict[str, Any]] = {}
        self._final_inventory_snapshot: Optional[List[Dict[str, Any]]] = None
        
        if config.enable_or and config.mode in ("modeA", "modeB", "modeC"):
            # Create OR agent configuration using promised lead time
            or_items_config = {}
            for item_config in self.csv_player.get_initial_item_configs():
                or_items_config[item_config['item_id']] = {
                    'lead_time': config.promised_lead_time,
                    'profit': item_config['profit'],
                    'holding_cost': item_config['holding_cost']
                }
            self._or_agent = ORAgent(or_items_config, self._initial_samples, policy='capped')
            print(f"\nOR Agent initialized with CAPPED policy")
            sys.stdout.flush()

        # Determine if we need LLM agent
        need_llm = config.mode in ("modeB", "modeC")
        
        if need_llm:
            self._agent = _make_base_agent(
                item_ids=self.csv_player.get_item_ids(),
                initial_samples=self._initial_samples,
                promised_lead_time=config.promised_lead_time,
                human_feedback_enabled=(config.mode == "modeB"),
                guidance_enabled=(config.mode == "modeC"),
                or_enabled=config.enable_or,
            )
        else:
            # modeA doesn't need LLM agent
            self._agent = None

        self._env = ta.make(env_id="VendingMachine-v0")
        self._base_env = self._resolve_base_env(self._env)
        from textarena.envs.VendingMachine import env as vm_env_module

        self._vm_env_module = vm_env_module
        self._original_num_days = vm_env_module.NUM_DAYS
        self._original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
        vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
        self._total_days = self.csv_player.get_num_periods()
        vm_env_module.NUM_DAYS = self._total_days

        self._setup_environment()
        # Ensure day 1 item configurations reflect CSV
        self._apply_day_item_configs(self.current_day)
        self._pid, initial_observation = self._env.get_observation()
        self._observation = _inject_carry_over_insights(initial_observation, self._carry_over_insights)
        if self._pid != 0:
            raise RuntimeError("VM should act first")

        if self.config.mode == "modeA":
            self._bootstrap_modeA()
        elif self.config.mode == "modeB":
            self._bootstrap_modeB()
        else:  # modeC
            self._bootstrap_modeC()

    def _load_initial_samples(self) -> tuple[Dict[str, List[int]], List[str]]:
        """Load initial demand samples and dates from train.csv."""
        train_df = pd.read_csv(self.config.train_file)
        item_ids = self.csv_player.get_item_ids()
        
        if not item_ids:
            raise ValueError("No items detected in CSV")
        
        first_item = item_ids[0]
        demand_col = f'demand_{first_item}'
        date_col = f'exact_dates_{first_item}'
        
        if demand_col not in train_df.columns:
            raise ValueError(f"Column {demand_col} not found in train.csv")
        
        train_samples = train_df[demand_col].tolist()
        initial_samples = {item_id: train_samples for item_id in item_ids}
        
        # Extract dates if available
        sample_dates = []
        if date_col in train_df.columns:
            sample_dates = train_df[date_col].tolist()
        else:
            # Fallback: use sample numbers if dates not available
            sample_dates = [f"Sample {i+1}" for i in range(len(train_samples))]
        
        print(f"Loaded {len(train_samples)} initial samples from train.csv")
        print(f"  Samples: {train_samples}")
        print(f"  Dates: {sample_dates}")
        print(f"  Mean: {sum(train_samples)/len(train_samples):.1f}")
        
        return initial_samples, sample_dates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def serialize_state(self) -> Dict[str, Any]:
        # Build period dates map for frontend lookup
        num_periods = self.csv_player.get_num_periods()
        period_dates = {}
        for period in range(1, num_periods + 1):
            date = self.csv_player.get_exact_date(period)
            # Only include if it's a real date (not "Period_X" format)
            if date and not date.startswith("Period_"):
                period_dates[period] = date
        
        state: Dict[str, Any] = {
            "mode": self.config.mode,
            "guidance_frequency": self.config.guidance_frequency,
            "current_day": self.current_day,
            "current_period_date": self.csv_player.get_exact_date(self.current_day),
            "total_periods": num_periods,  # Total number of periods in the simulation
            "player_id": self._pid,
            "observation": self._observation,
            "transcript": [
                {"kind": evt.kind, "payload": evt.payload} 
                for evt in (self.transcript.events.copy() if hasattr(self.transcript, '_lock') else self.transcript.events)
            ],
            "completed": self.transcript.completed,
            "final_reward": self.transcript.final_reward,
            "status_cards": self._build_status_cards(),
            "daily_logs": copy.deepcopy(self._ui_daily_logs),
            "initial_samples": self._initial_samples,
            "initial_sample_dates": self._initial_sample_dates,
            "period_dates": period_dates,  # Map of period number -> date string
            "item_description": self._get_item_description(),
            "streaming_text": self.get_streaming_text(),  # Include streaming text for real-time updates
        }
        
        # Add OR recommendations if available
        if self._or_recommendations:
            state["or_recommendation"] = {
                "recommendations": self._or_recommendations,
                "statistics": self._or_statistics
            }
        
        if self.config.mode in ("modeA", "modeB"):
            state["waiting_for_final_action"] = self._pid == 0 and not self.transcript.completed
        elif self.config.mode == "modeC":
            state["waiting_for_guidance"] = self._pending_guidance_day is not None
            state["guidance_history"] = [
                {"day": day, "message": message} for day, message in self._guidance_history
            ]
            latest = self._latest_agent_proposal()
            if latest is not None:
                state["latest_agent_proposal"] = latest
        return state

    def _get_item_description(self) -> str:
        """Get item description from current period."""
        item_ids = self.csv_player.get_item_ids()
        if not item_ids:
            return ""
        
        # Get description from current period
        try:
            config = self.csv_player.get_period_item_config(self.current_day, item_ids[0])
            return config.get('description', '')
        except:
            return ""

    def submit_final_action(self, action_json: str) -> Dict[str, Any]:
        if self.config.mode not in ("modeA", "modeB"):
            raise RuntimeError("Final action only available in Mode A and B")
        if self._pid != 0:
            raise RuntimeError("Not waiting for VM turn")
        action_dict, carry_memo = self._parse_action_json(action_json)
        self._store_carry_over_insight(self.current_day, carry_memo)
        payload = json.dumps({"action": action_dict})
        
        # Print human decision to terminal
        print(f"\n{'='*70}")
        print(f"Period {self.current_day} - HUMAN DECISION:")
        print(f"{'='*70}")
        print(f"Human Action:")
        for item_id, qty in action_dict.items():
            or_rec = self._or_recommendations.get(item_id, "N/A") if self._or_recommendations else "N/A"
            print(f"  {item_id}: {qty} units (OR recommended: {or_rec})")
        if carry_memo:
            print(f"\nCarry-over Insight: {carry_memo}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self.transcript.append(
            "final_action",
            {"day": self.current_day, "content": action_dict, "source": "human"},
        )
        return self._advance_with_vm_action(payload)

    def submit_guidance(self, message: str) -> Dict[str, Any]:
        if self.config.mode != "modeC":
            raise RuntimeError("Guidance only available in Mode C")
        if self._pending_guidance_day is None:
            raise RuntimeError("Not waiting for guidance")
        trimmed = message.strip()
        self._guidance_messages[self._pending_guidance_day] = trimmed
        self._guidance_history.append((self._pending_guidance_day, trimmed))
        
        # Print human guidance to terminal
        print(f"\n{'='*70}")
        print(f"Period {self._pending_guidance_day} - HUMAN GUIDANCE:")
        print(f"{'='*70}")
        print(f"{trimmed}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self.transcript.append(
            "guidance",
            {"day": self._pending_guidance_day, "content": trimmed},
        )
        self._pending_guidance_day = None
        self._run_until_pause_or_complete()
        return self.serialize_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_environment(self) -> None:
        for config in self.csv_player.get_initial_item_configs():
            self._env.add_item(**config)

        self.transcript.append("initial_samples", {"samples": self._initial_samples})

        # News功能已移除 - H&M数据集没有news列
        # for day, news in self.csv_player.get_news_schedule().items():
        #     self._env.add_news(day, news)

        self._env.reset(num_players=2)
        self._reset_ui_tracking()

    def _apply_day_item_configs(self, day: int) -> None:
        """Update environment item configs to match CSV for the specified day."""
        if day < 1 or day > self.csv_player.get_num_periods():
            return

        for item_id in self.csv_player.get_item_ids():
            try:
                config = self.csv_player.get_period_item_config(day, item_id)
            except ValueError:
                continue
            self._env.update_item_config(
                item_id=item_id,
                lead_time=config.get("lead_time"),
                profit=config.get("profit"),
                holding_cost=config.get("holding_cost"),
                description=config.get("description"),
            )

    def _bootstrap_modeA(self) -> None:
        """Bootstrap for OR-only mode."""
        self.transcript.append("observation", {"day": self.current_day, "content": self._observation})
        # Get OR recommendation
        self._update_or_recommendation()
    
    def _bootstrap_modeB(self) -> None:
        """Bootstrap for OR + LLM mode."""
        self.transcript.append("observation", {"day": self.current_day, "content": self._observation})
        # Get OR recommendation first
        if self._or_agent:
            self._update_or_recommendation()
        # LLM proposal will be generated asynchronously in the background
        # This allows the frontend to show OR recommendation and "thinking" state immediately
    
    def _trigger_llm_proposal_async(self) -> None:
        """Trigger LLM proposal generation in a background thread with streaming support."""
        def generate_proposal():
            try:
                # Clear streaming text at start
                with self._streaming_lock:
                    self._streaming_text = ""
                
                # Generate proposal with streaming callback
                self._agent_proposal_with_history_streaming()
            except Exception as e:
                print(f"Error generating LLM proposal: {e}", file=sys.stderr)
                sys.stderr.flush()
                with self._streaming_lock:
                    self._streaming_text = ""
        
        thread = threading.Thread(target=generate_proposal, daemon=True)
        thread.start()
    
    def get_streaming_text(self) -> str:
        """Get current streaming text (partial LLM response)."""
        with self._streaming_lock:
            return self._streaming_text

    def _bootstrap_modeC(self) -> None:
        """Bootstrap for Mode C with LLM and optional OR."""
        self.transcript.append("observation", {"day": self.current_day, "content": self._observation})
        self._run_until_pause_or_complete()
    
    def _update_or_recommendation(self) -> None:
        """Update OR recommendations for current observation."""
        if self._or_agent:
            action_json_str, self._or_statistics = self._or_agent.get_action(self._observation)
            # Parse the JSON string to extract action dict
            action_data = json.loads(action_json_str)
            self._or_recommendations = action_data.get('action', {})
            
            # Print detailed OR statistics to terminal
            print(f"\n{'='*70}")
            print(f"Period {self.current_day} - OR ALGORITHM RECOMMENDATIONS (CAPPED Policy):")
            print(f"{'='*70}")
            for item_id, item_stats in self._or_statistics.items():
                print(f"\n{item_id}:")
                print(f"  Empirical mean: {item_stats['empirical_mean']:.2f}")
                print(f"  Empirical std: {item_stats['empirical_std']:.2f}")
                print(f"  Lead time (L): {item_stats['L']}")
                _safe_print(f"  mu_hat (μ̂): {item_stats['mu_hat']:.2f}")
                _safe_print(f"  sigma_hat (σ̂): {item_stats['sigma_hat']:.2f}")
                print(f"  Critical fractile (q): {item_stats['q']:.4f}")
                _safe_print(f"  z*: {item_stats['z_star']:.4f}")
                print(f"  Base stock: {item_stats['base_stock']:.2f}")
                print(f"  Current inventory: {item_stats['current_inventory']}")
                if 'cap' in item_stats:
                    print(f"  Cap value: {item_stats['cap']:.2f}")
                    print(f"  OR recommends (capped): {item_stats['order']}")
                    print(f"  OR recommends (uncapped): {item_stats['order_uncapped']}")
                else:
                    print(f"  OR recommends: {item_stats['order']}")
            print(f"{'='*70}")
            sys.stdout.flush()
            
            self.transcript.append(
                "or_recommendation",
                {
                    "day": self.current_day,
                    "recommendations": self._or_recommendations,
                    "statistics": self._or_statistics
                }
            )

    def _agent_proposal_with_history(self) -> Dict[str, Any]:
        """Generate agent proposal without conversation history (direct observation only)."""
        prompt = self._format_prompt_without_conversation()
        action_text = self._agent(prompt)
        cleaned = self._clean_json(action_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"rationale": cleaned, "action": {}}
        
        # Print LLM reasoning to terminal
        print(f"\n{'='*70}")
        print(f"Period {self.current_day} - LLM DECISION:")
        print(f"{'='*70}")
        
        rationale = data.get("rationale", "")
        if rationale:
            print(f"\nLLM Rationale:")
            print(f"{rationale}")
        
        action = data.get("action", {})
        if action:
            print(f"\nLLM Action:")
            for item_id, qty in action.items():
                or_rec = self._or_recommendations.get(item_id, "N/A")
                print(f"  {item_id}: {qty} units (OR recommended: {or_rec})")
        
        carry_over = data.get("carry_over_insight", "")
        if carry_over:
            print(f"\nCarry-over Insight: {carry_over}")
        else:
            print(f"\nCarry-over Insight: (empty)")
        
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self._store_carry_over_insight(self.current_day, data.get("carry_over_insight"))
        self.transcript.append(
            "agent_proposal", {"day": self.current_day, "content": data}
        )
        return data
    
    def _agent_proposal_with_history_streaming(self) -> Dict[str, Any]:
        """Generate agent proposal with streaming support - updates partial text as it generates."""
        prompt = self._format_prompt_without_conversation()
        
        # Clear streaming text
        with self._streaming_lock:
            self._streaming_text = ""
        
        # Create a streaming wrapper around the agent
        # Since OpenAI Responses API may not support streaming, we'll simulate it
        # by updating text as it's generated using a wrapper
        
        def streaming_agent_call(agent, observation):
            """Wrapper that updates streaming text during agent call with true token streaming."""
            # Try to unwrap agent if it's wrapped (e.g., HumanFeedbackAgent)
            base_agent = agent
            if hasattr(agent, 'base_agent'):
                base_agent = agent.base_agent
            
            # Debug: print agent type
            print(f"DEBUG: Agent type: {type(agent)}, Base agent type: {type(base_agent)}")
            print(f"DEBUG: Base agent has client: {hasattr(base_agent, 'client')}")
            if hasattr(base_agent, 'client'):
                print(f"DEBUG: Base agent client has responses: {hasattr(base_agent.client, 'responses')}")
            
            # Check if base agent is GPT5MiniAgent and supports streaming
            if hasattr(base_agent, 'client') and hasattr(base_agent.client, 'responses'):
                # Try true streaming using OpenAI Responses streaming interface
                try:
                    request_payload = {
                        "model": base_agent.model_name,
                        "input": [
                            {"role": "system", "content": [{"type": "input_text", "text": base_agent.system_prompt}]},
                            {"role": "user", "content": [{"type": "input_text", "text": observation}]},
                        ],
                    }
                    if getattr(base_agent, "reasoning_effort", None):
                        request_payload["reasoning"] = {"effort": base_agent.reasoning_effort}
                    if getattr(base_agent, "text_verbosity", None):
                        request_payload["text"] = {"verbosity": base_agent.text_verbosity}

                    import time

                    print("DEBUG: Attempting streaming call via responses.stream")
                    full_text = ""

                    # Use streaming context manager so we can get final response afterwards
                    with base_agent.client.responses.stream(**request_payload) as stream:
                        for event in stream:
                            event_type = getattr(event, "type", None)
                            if event_type == "response.output_text.delta":
                                # Extract delta text safely
                                chunk_text = ""
                                delta_obj = getattr(event, "delta", None)
                                if isinstance(delta_obj, str):
                                    chunk_text = delta_obj
                                elif delta_obj is not None:
                                    # delta may expose .text or .content depending on SDK version
                                    if hasattr(delta_obj, "text"):
                                        chunk_text = delta_obj.text
                                    elif hasattr(delta_obj, "content"):
                                        chunk_text = delta_obj.content
                                if chunk_text:
                                    full_text += chunk_text
                                    with self._streaming_lock:
                                        self._streaming_text = full_text
                            elif event_type == "response.completed":
                                break
                            # Avoid tight loop to give time for frontend polling
                            time.sleep(0.01)

                        # After stream ends, grab the final response from SDK helper
                        final_response = stream.get_final_response()
                        if hasattr(final_response, "output_text"):
                            final_text = final_response.output_text
                            if isinstance(final_text, list):
                                final_text = "".join(final_text)
                            if isinstance(final_text, str):
                                full_text = final_text or full_text

                    if full_text:
                        return full_text.strip()

                except Exception as stream_error:
                    print(f"DEBUG: Streaming attempt failed ({type(stream_error).__name__}): {stream_error}")
                    # Fall back to non-streaming call below

                # Fallback to non-streaming call
                try:
                    response = base_agent.client.responses.create(**request_payload)
                    result = response.output_text.strip() if hasattr(response, 'output_text') else str(response).strip()
                    with self._streaming_lock:
                        self._streaming_text = result
                    return result
                except Exception as e:
                    print(f"DEBUG: Non-streaming call failed ({type(e).__name__}): {e}")
                    raise
            else:
                # Regular agent call - simulate streaming by updating text incrementally after call completes
                # This provides visual feedback even if true streaming isn't available
                import time
                import re
                
                # Clear streaming text at start
                with self._streaming_lock:
                    self._streaming_text = ""
                
                # Call agent - this might take a while
                # Show a placeholder while waiting
                with self._streaming_lock:
                    self._streaming_text = "Thinking..."
                
                result = agent(observation)
                
                if not result:
                    with self._streaming_lock:
                        self._streaming_text = ""
                    return ""
                
                # Simulate streaming by revealing text incrementally
                # This gives the appearance of streaming even though we have the full result
                # Split into chunks and reveal progressively
                chunk_size = max(10, len(result) // 30)  # Update ~30 times for smooth effect
                accumulated = ""
                
                for i in range(0, len(result), chunk_size):
                    accumulated = result[:i + chunk_size]
                    with self._streaming_lock:
                        self._streaming_text = accumulated
                    # Small delay to allow frontend polling (50ms for smooth streaming)
                    time.sleep(0.05)
                
                # Ensure final text is set
                with self._streaming_lock:
                    self._streaming_text = result
                
                return result
        
        # Call agent with streaming wrapper
        print(f"DEBUG: Calling streaming agent with prompt length: {len(prompt)}")
        action_text = streaming_agent_call(self._agent, prompt)
        print(f"DEBUG: Agent call completed, response length: {len(action_text) if action_text else 0}")
        print(f"DEBUG: Response preview: {action_text[:200] if action_text else 'EMPTY'}")
        
        cleaned = self._clean_json(action_text)
        print(f"DEBUG: Cleaned response length: {len(cleaned)}")
        try:
            data = json.loads(cleaned)
            print(f"DEBUG: Parsed JSON successfully, keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode failed: {e}, using cleaned text as rationale")
            data = {"rationale": cleaned, "action": {}}
        
        # Keep streaming text until proposal is added to transcript
        # Don't clear it here - it will be cleared when the next proposal starts
        
        # Print LLM reasoning to terminal
        print(f"\n{'='*70}")
        print(f"Period {self.current_day} - LLM DECISION:")
        print(f"{'='*70}")
        
        rationale = data.get("rationale", "")
        if rationale:
            print(f"\nLLM Rationale:")
            print(f"{rationale}")
        
        action = data.get("action", {})
        if action:
            print(f"\nLLM Action:")
            for item_id, qty in action.items():
                or_rec = self._or_recommendations.get(item_id, "N/A")
                print(f"  {item_id}: {qty} units (OR recommended: {or_rec})")
        
        carry_over = data.get("carry_over_insight", "")
        if carry_over:
            print(f"\nCarry-over Insight: {carry_over}")
        else:
            print(f"\nCarry-over Insight: (empty)")
        
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self._store_carry_over_insight(self.current_day, data.get("carry_over_insight"))
        self.transcript.append(
            "agent_proposal", {"day": self.current_day, "content": data}
        )
        
        # Clear streaming text once proposal is complete - frontend will show short_rationale_for_human instead
        with self._streaming_lock:
            self._streaming_text = ""
        
        return data

    def _format_prompt_without_conversation(self) -> str:
        """Format prompt with current observation and OR recommendations only (no conversation history)."""
        lines = ["CURRENT OBSERVATION:", self._observation.strip(), ""]
        
        # Add OR recommendations if available
        if self._or_recommendations and self.config.enable_or:
            lines.append("=" * 70)
            lines.append("OR ALGORITHM RECOMMENDATIONS:")
            lines.append("=" * 70)
            for item_id, quantity in self._or_recommendations.items():
                stats = self._or_statistics.get(item_id, {})
                lines.append(f"{item_id}: {quantity} units")
                if stats:
                    lines.append(f"  Base stock level: {stats.get('base_stock', 0):.2f}")
                    lines.append(f"  Current inventory (on-hand + in-transit): {stats.get('current_inventory', 0)}")
                    lines.append(f"  Empirical mean demand: {stats.get('empirical_mean', 0):.2f}")
                    lines.append(f"  Empirical std demand: {stats.get('empirical_std', 0):.2f}")
                    if 'cap' in stats:
                        lines.append(f"  Order cap: {stats.get('cap', 0):.2f}")
            lines.append("=" * 70)
            lines.append("")
        
        lines.append("Provide your JSON proposal.")
        return "\n".join(lines)

    def _store_carry_over_insight(self, day: int, memo_value) -> None:
        if isinstance(memo_value, str):
            memo = memo_value.strip()
        else:
            memo = None
        if memo:
            self._carry_over_insights[day] = memo
        elif day in self._carry_over_insights:
            del self._carry_over_insights[day]

    def _advance_with_vm_action(self, action: str) -> Dict[str, Any]:
        done, _ = self._env.step(action=action)
        if done:
            return self._finalize_session()

        self._pid, next_observation = self._env.get_observation()
        self._observation = _inject_carry_over_insights(next_observation, self._carry_over_insights)
        if self._pid != 1:
            raise RuntimeError("Expected demand turn after VM action")

        demand_action = self.csv_player.get_action(self.current_day)
        demand_dict = json.loads(demand_action)
        
        # Print demand to terminal
        print(f"\n{'='*70}")
        print(f"Period {self.current_day} - ACTUAL DEMAND:")
        print(f"{'='*70}")
        for item_id, qty in demand_dict.get("action", {}).items():
            print(f"  {item_id}: {qty} units")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self.transcript.append(
            "demand_action", {"day": self.current_day, "content": demand_dict}
        )
        
        # Update OR agent with observed demand
        if self._or_agent:
            for item_id, observed_demand in demand_dict.get("action", {}).items():
                self._or_agent.update_demand_observation(item_id, observed_demand)
        
        done, _ = self._env.step(action=demand_action)
        self._sync_ui_daily_logs()
        if done:
            return self._finalize_session()

        self.current_day += 1
        # Apply new day item configurations before fetching the next observation
        self._apply_day_item_configs(self.current_day)
        self._pid, next_observation = self._env.get_observation()
        self._observation = _inject_carry_over_insights(next_observation, self._carry_over_insights)
        self.transcript.append(
            "observation", {"day": self.current_day, "content": self._observation}
        )

        if self.config.mode in ("modeA", "modeB"):
            # Get OR recommendation for new day
            if self._or_agent:
                self._update_or_recommendation()
            
            if self.config.mode == "modeB":
                # Trigger LLM proposal generation asynchronously
                # This allows the frontend to return immediately and show "thinking" state
                self._trigger_llm_proposal_async()
                # Return state immediately with OR recommendation, LLM will update via polling
                return {"next_observation": self._observation, "completed": False}
            else:
                # modeA: just return state with OR recommendation
                return {"next_observation": self._observation, "completed": False}

        self._run_until_pause_or_complete()
        return self.serialize_state()

    def _run_until_pause_or_complete(self) -> None:
        while not self.transcript.completed:
            guidance_day = self._guidance_due_for_day(self.current_day)
            if guidance_day is not None and guidance_day not in self._guidance_messages:
                self._pending_guidance_day = guidance_day
                # Get OR recommendation before pausing
                if self._or_agent:
                    self._update_or_recommendation()
                print(f"\n{'='*70}")
                print(f"Period {guidance_day} - WAITING FOR HUMAN GUIDANCE")
                print(f"{'='*70}")
                sys.stdout.flush()
                break

            # Get OR recommendation before LLM
            if self._or_agent:
                self._update_or_recommendation()

            prompt = self._format_guided_prompt()
            agent_output = self._agent(prompt)
            cleaned = self._clean_json(agent_output)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                data = {"rationale": cleaned, "action": {}}
            
            # Print LLM decision for modeC
            print(f"\n{'='*70}")
            print(f"Period {self.current_day} - LLM AUTO-PLAY DECISION:")
            print(f"{'='*70}")
            
            rationale = data.get("rationale", "")
            if rationale:
                print(f"\nLLM Rationale:")
                print(f"{rationale}")
            
            action = data.get("action", {})
            if action:
                print(f"\nLLM Action:")
                for item_id, qty in action.items():
                    or_rec = self._or_recommendations.get(item_id, "N/A") if self._or_recommendations else "N/A"
                    print(f"  {item_id}: {qty} units (OR recommended: {or_rec})")
            
            carry_over = data.get("carry_over_insight", "")
            if carry_over:
                print(f"\nCarry-over Insight: {carry_over}")
            else:
                print(f"\nCarry-over Insight: (empty)")
            
            print(f"{'='*70}")
            sys.stdout.flush()
            
            self._store_carry_over_insight(self.current_day, data.get("carry_over_insight"))
            self.transcript.append(
                "agent_proposal", {"day": self.current_day, "content": data}
            )

            # Log this step if callback is set (for modeC automatic decisions)
            if self._step_logging_callback:
                try:
                    action_dict = data.get("action", {})
                    current_reward = self._running_reward if hasattr(self, "_running_reward") else 0.0
                    or_rec = self._or_recommendations if hasattr(self, "_or_recommendations") else None
                    
                    # Extract prompts
                    input_prompt = prompt  # The formatted prompt sent to LLM
                    output_prompt = data.get("rationale", "")
                    
                    self._step_logging_callback({
                        "period": self.current_day,
                        "inventory_decision": action_dict,
                        "total_reward": current_reward,
                        "input_prompt": input_prompt,
                        "output_prompt": output_prompt,
                        "or_recommendation": or_rec,
                    })
                except Exception as e:
                    # Don't fail the game if logging fails
                    print(f"Warning: Failed to log step: {e}", file=sys.stderr)
                    sys.stderr.flush()

            action_payload = json.dumps({"action": data.get("action", {})})
            result = self._advance_with_vm_action(action_payload)
            if isinstance(result, dict) and result.get("completed"):
                break
            if self.transcript.completed:
                break
            if self._pending_guidance_day is not None:
                break

    def _format_guided_prompt(self) -> str:
        lines = ["CURRENT OBSERVATION:", self._observation.strip(), ""]
        
        # Add OR recommendations if available
        if self._or_recommendations and self.config.enable_or:
            lines.append("=" * 70)
            lines.append("OR ALGORITHM RECOMMENDATIONS:")
            lines.append("=" * 70)
            for item_id, quantity in self._or_recommendations.items():
                stats = self._or_statistics.get(item_id, {})
                lines.append(f"{item_id}: {quantity} units")
                if stats:
                    lines.append(f"  Base stock level: {stats.get('base_stock', 0):.2f}")
                    lines.append(f"  Current inventory (on-hand + in-transit): {stats.get('current_inventory', 0)}")
                    lines.append(f"  Empirical mean demand: {stats.get('empirical_mean', 0):.2f}")
                    lines.append(f"  Empirical std demand: {stats.get('empirical_std', 0):.2f}")
                    if 'cap' in stats:
                        lines.append(f"  Order cap: {stats.get('cap', 0):.2f}")
            lines.append("=" * 70)
            lines.append("")
        
        if self._guidance_history:
            lines.append("HUMAN GUIDANCE HISTORY:")
            for day, message in self._guidance_history:
                quoted = json.dumps(message)
                lines.append(f"Day {day} human message: {quoted}")
            lines.append("")
        lines.append("Return JSON proposal only.")
        return "\n".join(lines)

    def _guidance_due_for_day(self, day: int) -> Optional[int]:
        if self.config.mode != "modeC":
            return None
        frequency = max(self.config.guidance_frequency, 1)
        if ((day - 1) % frequency) == 0:
            return day
        return None

    def _clean_json(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        return cleaned.strip()

    def _parse_action_json(self, action_json: str) -> Tuple[Dict[str, int], Optional[str]]:
        cleaned = self._clean_json(action_json)
        data = json.loads(cleaned)
        memo: Optional[str] = None
        if isinstance(data, dict):
            memo_val = data.get("carry_over_insight")
            if isinstance(memo_val, str):
                memo = memo_val.strip() or None
        if isinstance(data, dict) and "action" in data and isinstance(data["action"], dict):
            action_dict = data["action"]
        elif isinstance(data, dict):
            action_dict = data
        else:
            raise ValueError("Final decision must be a JSON object")
        result: Dict[str, int] = {}
        for item_id, quantity in action_dict.items():
            if not isinstance(quantity, (int, float)) or quantity < 0:
                raise ValueError(f"Invalid quantity for {item_id}: {quantity}")
            result[item_id] = int(quantity)
        return result, memo

    def _finalize_session(self) -> Dict[str, Any]:
        self._sync_ui_daily_logs()
        
        # Cache the final inventory snapshot before closing environment
        self._final_inventory_snapshot = self._build_inventory_snapshot()
        
        rewards, game_info = self._env.close()
        vm_info = game_info[0]
        total_reward = vm_info.get("total_reward", 0.0)
        self.transcript.final_reward = float(total_reward)
        self.transcript.completed = True
        
        # Print final results to terminal
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETED - {self.config.mode.upper()}")
        print(f"{'='*70}")
        print(f"\nTotal periods completed: {self.current_day - 1}")
        print(f"Total Profit from Sales: ${vm_info.get('total_sales_profit', 0.0):.2f}")
        print(f"Total Holding Cost: ${vm_info.get('total_holding_cost', 0.0):.2f}")
        print(f"\n>>> TOTAL REWARD: ${total_reward:.2f} <<<")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        self.transcript.append(
            "final_summary",
            {
                "total_reward": total_reward,
                "vm_reward": rewards.get(0, 0.0),
                "totals": {
                    "sales_profit": vm_info.get("total_sales_profit", 0.0),
                    "holding_cost": vm_info.get("total_holding_cost", 0.0),
                },
            },
        )
        self._vm_env_module.NUM_DAYS = self._original_num_days
        self._vm_env_module.INITIAL_INVENTORY_PER_ITEM = self._original_initial_inventory
        return {"completed": True, "final_reward": self.transcript.final_reward}

    @staticmethod
    def _resolve_base_env(env: Any) -> Any:
        base = env
        while hasattr(base, "env"):
            base = base.env
        return base

    def _reset_ui_tracking(self) -> None:
        self._ui_daily_logs.clear()
        self._running_reward = 0.0

    def _sync_ui_daily_logs(self) -> None:
        env_logs = getattr(self._base_env, "daily_logs", None)
        if not env_logs:
            return
        start = len(self._ui_daily_logs)
        if start >= len(env_logs):
            return
        for raw_log in env_logs[start:]:
            reward = float(raw_log.get("daily_reward", 0.0))
            self._running_reward += reward
            self._ui_daily_logs.append(self._sanitize_day_log(raw_log, env_logs))

    def _sanitize_day_log(self, log: Dict[str, Any], all_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        log_day = int(log.get("day", 0))
        sanitized: Dict[str, Any] = {
            "day": log_day,
            "date": self.csv_player.get_exact_date(log_day),  # Add exact date for this period
            "daily_profit": float(log.get("daily_profit", 0.0)),
            "daily_holding_cost": float(log.get("daily_holding_cost", 0.0)),
            "daily_reward": float(log.get("daily_reward", 0.0)),
        }
        for field_name in ("orders", "starting_inventory", "requests", "sales", "ending_inventory"):
            field_val = log.get(field_name, {})
            sanitized[field_name] = {item: int(value) for item, value in field_val.items()}
        arrivals = {}
        for item, entries in log.get("arrivals", {}).items():
            arrivals[item] = [
                {"quantity": int(entry[0]), "order_day": int(entry[1])} for entry in entries
            ]
        sanitized["arrivals"] = arrivals
        
        # Build order_status: track each order's arrival status
        order_status = []
        orders = log.get("orders", {})
        log_day = int(log.get("day", 0))
        
        # Get pending_orders from base_env to find lead_time and arrival_day
        base_env = self._base_env
        pending_orders = getattr(base_env, "pending_orders", [])
        current_day = getattr(base_env, "current_day", self.current_day)
        
        # Build a map of orders by (order_day, item_id) for quick lookup
        order_map = {}
        for order in pending_orders:
            order_day = order.get("order_day")
            item_id = order.get("item_id")
            if order_day == log_day:
                order_map[item_id] = {
                    "quantity": order.get("quantity", 0),
                    "lead_time": order.get("original_lead_time", 0),
                    "arrival_day": order.get("arrival_day", float("inf")),
                }
        
        # Process each order from this day
        for item_id, quantity in orders.items():
            if quantity <= 0:
                continue
            
            # Try to find order info from pending_orders
            order_info = order_map.get(item_id)
            lead_time = 0
            arrival_day = float("inf")
            
            if order_info:
                lead_time = order_info.get("lead_time", 0)
                arrival_day = order_info.get("arrival_day", float("inf"))
            else:
                # Check if order arrived in the same day (lead_time=0 case)
                current_arrivals = log.get("arrivals", {})
                item_arrivals = current_arrivals.get(item_id, [])
                for arrival_entry in item_arrivals:
                    if isinstance(arrival_entry, (list, tuple)) and len(arrival_entry) >= 2:
                        arrival_order_day = int(arrival_entry[1])
                        if arrival_order_day == log_day:
                            arrival_day = log_day
                            lead_time = 0
                            break
                
                # If not found in current log, check subsequent logs
                if arrival_day == float("inf"):
                    for future_log in all_logs:
                        future_day = int(future_log.get("day", 0))
                        if future_day <= log_day:
                            continue
                        future_arrivals = future_log.get("arrivals", {})
                        item_arrivals = future_arrivals.get(item_id, [])
                        for arrival_entry in item_arrivals:
                            if isinstance(arrival_entry, (list, tuple)) and len(arrival_entry) >= 2:
                                arrival_order_day = int(arrival_entry[1])
                                if arrival_order_day == log_day:
                                    arrival_day = future_day
                                    lead_time = future_day - log_day
                                    break
                        if arrival_day != float("inf"):
                            break
            
            # Determine if order has arrived
            arrived = arrival_day != float("inf") and arrival_day <= current_day
            arrival_period = int(arrival_day) if arrival_day != float("inf") and arrived else None
            
            order_status.append({
                "item_id": item_id,
                "quantity": int(quantity),
                "order_day": log_day,
                "lead_time": float(lead_time) if lead_time != float("inf") else 0.0,
                "arrival_day": float(arrival_day) if arrival_day != float("inf") else None,
                "arrived": arrived,
                "arrival_period": arrival_period,
            })
        
        sanitized["order_status"] = order_status
        return sanitized

    def _latest_agent_proposal(self) -> Optional[Dict[str, Any]]:
        events_copy = self.transcript.events.copy() if hasattr(self.transcript, '_lock') else self.transcript.events
        for event in reversed(events_copy):
            if event.kind == "agent_proposal" and isinstance(event.payload, dict):
                payload = event.payload
                content = payload.get("content")
                if isinstance(content, dict):
                    result: Dict[str, Any] = {"day": payload.get("day")}
                    result.update(content)
                    return result
        return None

    def _build_status_cards(self) -> Dict[str, Any]:
        latest_log = self._ui_daily_logs[-1] if self._ui_daily_logs else None
        inventory_snapshot = self._build_inventory_snapshot()
        return {
            "progress": {
                "current_day": self.current_day,
                "total_days": self._total_days,
                "waiting_for_vm_action": self._pid == 0,
            },
            "reward": {
                "to_date": self._running_reward,
                "final": self.transcript.final_reward,
            },
            "cashflow": {
                "day": latest_log["day"] if latest_log else None,
                "profit": latest_log["daily_profit"] if latest_log else 0.0,
                "holding_cost": latest_log["daily_holding_cost"] if latest_log else 0.0,
                "reward": latest_log["daily_reward"] if latest_log else 0.0,
            },
            "inventory": inventory_snapshot,
        }

    def _build_inventory_snapshot(self) -> List[Dict[str, Any]]:
        # If game is completed, return cached snapshot
        if self.transcript.completed and self._final_inventory_snapshot is not None:
            return self._final_inventory_snapshot
        
        base_env = self._base_env
        items = getattr(base_env, "items", {})
        on_hand = getattr(base_env, "on_hand_inventory", {})
        pending = getattr(base_env, "pending_orders", [])
        current = getattr(base_env, "current_day", self.current_day)
        snapshot: List[Dict[str, Any]] = []
        for item_id, info in items.items():
            in_transit = 0
            for order in pending:
                if order.get("item_id") != item_id:
                    continue
                arrival_day = order.get("arrival_day", float("inf"))
                if arrival_day >= current:
                    in_transit += int(order.get("quantity", 0))
            snapshot.append(
                {
                    "item_id": item_id,
                    "description": info.get("description"),
                    "profit": float(info.get("profit", 0.0)),
                    "holding_cost": float(info.get("holding_cost", 0.0)),
                    "on_hand": int(on_hand.get(item_id, 0)),
                    "in_transit": in_transit,
                }
            )
        return snapshot


class ModeASession(SimulationSession):
    """Mode A: OR → Human Decision"""
    def __init__(self, config: SimulationConfig):
        if config.mode != "modeA":
            raise ValueError("ModeASession requires mode='modeA'")
        super().__init__(config)

    def submit_final_decision(self, action_json: str) -> Dict[str, Any]:
        result = self.submit_final_action(action_json)
        if result.get("completed"):
            return self.serialize_state()
        return self.serialize_state()


class ModeBSession(SimulationSession):
    """Mode B: OR → LLM → Human Decision"""
    def __init__(self, config: SimulationConfig):
        if config.mode != "modeB":
            raise ValueError("ModeBSession requires mode='modeB'")
        super().__init__(config)

    def submit_final_decision(self, action_json: str) -> Dict[str, Any]:
        result = self.submit_final_action(action_json)
        if result.get("completed"):
            return self.serialize_state()
        return self.serialize_state()


class ModeCSession(SimulationSession):
    """Mode C: OR → LLM → Guidance"""
    def __init__(self, config: SimulationConfig):
        if config.mode != "modeC":
            raise ValueError("ModeCSession requires mode='modeC'")
        super().__init__(config)


def load_simulation(config: SimulationConfig) -> SimulationSession:
    if not os.path.exists(config.demand_file):
        raise FileNotFoundError(f"Demand file not found: {config.demand_file}")
    if not os.path.exists(config.train_file):
        raise FileNotFoundError(f"Train file not found: {config.train_file}")
    
    if config.mode == "modeA":
        return ModeASession(config)
    if config.mode == "modeB":
        return ModeBSession(config)
    if config.mode == "modeC":
        return ModeCSession(config)
    raise ValueError(f"Unsupported mode: {config.mode}")


__all__ = [
    "CSVDemandPlayer",
    "ORAgent",
    "SimulationConfig",
    "SimulationTranscript",
    "SimulationSession",
    "ModeASession",
    "ModeBSession",
    "ModeCSession",
    "load_simulation",
]
