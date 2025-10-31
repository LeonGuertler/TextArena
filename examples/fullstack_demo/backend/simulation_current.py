"""Legacy-style simulation helpers retaining llm_csv_demo-style transcripts."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd
from openai import OpenAI
import textarena as ta


ModeLiteral = Literal["mode1", "mode2"]


@dataclass
class SimulationConfig:
    mode: ModeLiteral
    demand_file: str
    promised_lead_time: int = 0
    guidance_frequency: int = 5


@dataclass
class TranscriptEvent:
    kind: str
    payload: Dict[str, Any]


@dataclass
class SimulationTranscript:
    events: List[TranscriptEvent] = field(default_factory=list)
    completed: bool = False
    final_reward: Optional[float] = None

    def append(self, kind: str, payload: Dict[str, Any]) -> None:
        self.events.append(TranscriptEvent(kind=kind, payload=payload))


class CSVDemandPlayer:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        if "day" not in self.df.columns:
            raise ValueError("CSV must contain a 'day' column")
        self.item_ids = self._extract_item_ids()
        if not self.item_ids:
            raise ValueError("CSV missing demand_* columns")
        self._validate_item_columns()
        self.has_news = "news" in self.df.columns

    def _extract_item_ids(self) -> List[str]:
        item_ids: List[str] = []
        for col in self.df.columns:
            if col.startswith("demand_"):
                item_ids.append(col[len("demand_") :])
        return item_ids

    def _validate_item_columns(self) -> None:
        required = ["demand", "description", "lead_time", "profit", "holding_cost"]
        for item_id in self.item_ids:
            for suffix in required:
                col = f"{suffix}_{item_id}"
                if col not in self.df.columns:
                    raise ValueError(f"CSV missing required column: {col}")

    def get_item_ids(self) -> List[str]:
        return list(self.item_ids)

    def get_initial_item_configs(self) -> List[Dict[str, Any]]:
        if self.df.empty:
            raise ValueError("CSV is empty")
        first_row = self.df.iloc[0]
        configs: List[Dict[str, Any]] = []
        for item_id in self.item_ids:
            configs.append(
                {
                    "item_id": item_id,
                    "description": str(first_row[f"description_{item_id}"]),
                    "lead_time": self._normalize_lead_time(first_row[f"lead_time_{item_id}"]),
                    "profit": float(first_row[f"profit_{item_id}"]),
                    "holding_cost": float(first_row[f"holding_cost_{item_id}"]),
                }
            )
        return configs

    def get_day_item_config(self, day: int, item_id: str) -> Dict[str, Any]:
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range (1-{len(self.df)})")
        if item_id not in self.item_ids:
            raise ValueError(f"Unknown item_id: {item_id}")
        row = self.df.iloc[day - 1]
        return {
            "description": str(row[f"description_{item_id}"]),
            "lead_time": self._normalize_lead_time(row[f"lead_time_{item_id}"]),
            "profit": float(row[f"profit_{item_id}"]),
            "holding_cost": float(row[f"holding_cost_{item_id}"]),
        }

    def get_num_days(self) -> int:
        return len(self.df)

    def get_news_schedule(self) -> Dict[int, str]:
        if not self.has_news:
            return {}
        schedule: Dict[int, str] = {}
        for _, row in self.df.iterrows():
            day = int(row["day"])
            raw = row.get("news")
            if pd.notna(raw) and str(raw).strip():
                schedule[day] = str(raw).strip()
        return schedule

    def get_demand_action(self, day: int) -> str:
        if day < 1 or day > len(self.df):
            raise ValueError(f"Day {day} out of range (1-{len(self.df)})")
        row = self.df.iloc[day - 1]
        payload = {item_id: int(row[f"demand_{item_id}"]) for item_id in self.item_ids}
        return json.dumps({"action": payload}, indent=2)

    @staticmethod
    def _normalize_lead_time(value: Any) -> float:
        if isinstance(value, str) and value.lower() == "inf":
            return float("inf")
        try:
            numeric = float(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid lead time value: {value}") from exc
        if numeric == float("inf"):
            return float("inf")
        return int(numeric)


def _default_initial_samples(item_ids: Iterable[str]) -> Dict[str, List[int]]:
    historical = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    return {item_id: historical.copy() for item_id in item_ids}


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
    item_ids: Iterable[str],
    initial_samples: Dict[str, List[int]],
    promised_lead_time: int,
    human_feedback_enabled: bool,
    guidance_enabled: bool,
) -> ta.Agent:
    available_items = [str(item) for item in item_ids]
    items_str = ", ".join(f'"{item}"' for item in available_items) or "None"

    system_parts: List[str] = [
        "You are the Vending Machine controller (VM). You manage multiple items, each with unit profit and holding costs.",
        "Objective: Maximize total reward = sum of daily rewards R_t, where R_t = Profit\*Sold - HoldingCost\*EndingInventory.",
        "",
        f"AVAILABLE ITEMS: {items_str}",
        "CRITICAL: Use these exact item IDs (including parentheses/punctuation) in your action JSON.",
        "",
        "Core mechanics:",
        f"- Supplier-promised lead time (displayed to you): {promised_lead_time} days.",
        "- Actual lead time may differ and can change; infer it from arrival records.",
        "- Arrival log format: 'arrived=X units (ordered on Day Y, lead_time was Z days)'.",
        "- Track your orders and infer when each shipment will arrive.",
        "- On-hand inventory is immediately available; In-transit totals goods en route (arrival timing must be inferred).",
        "- Initial on-hand inventory on Day 1 is 0 for every item.",
        "- Holding cost applies to ending inventory each day.",
        "- Daily news is revealed day by day; you cannot see future news.",
        "- Daily sequence: you submit the order first, previously scheduled shipments arrive next, and customer demand happens last.",
    ]

    if human_feedback_enabled:
        system_parts.extend(
            [
                "",
                "HUMAN COLLABORATION:",
                "- You may exchange messages with a human supervisor before submitting the final action.",
                "- Provide thorough rationale and proposed orders when prompted.",
                "- Incorporate any human feedback carefully before your final decision.",
            ]
        )

    if guidance_enabled:
        system_parts.extend(
            [
                "",
                "STRATEGIC GUIDANCE:",
                "- Periodically you will receive strategic guidance from the human supervisor.",
                "- Treat new guidance as authoritative until superseded; reference it explicitly in your rationale.",
            ]
        )

    if initial_samples:
        system_parts.append("")
        system_parts.append("HISTORICAL DEMAND SAMPLES (reference):")
        for item_id, samples in initial_samples.items():
            system_parts.append(f"- {item_id}: {samples}")
        system_parts.append("Use these samples to ground early decisions, especially on Day 1.")

    if available_items:
        example_action = ", ".join(f'"{item}": 100' for item in available_items[:2])
        if len(available_items) > 2:
            example_action += ", ..."
    else:
        example_action = '"item_id": quantity, ...'

    system_parts.extend(
        [
            "",
            "Strategy checklist:",
            "- Infer current lead time from recent arrivals and update when evidence changes.",
            "- Track on-hand plus in-transit inventory against demand patterns.",
            "- Respond to today's news and remember how similar events affected demand before.",
            "- Balance profit potential against holding costs; avoid unnecessary overstocking.",
            "- Maintain carry_over_insight only for NEW, sustained shifts:",
            "    * Cite concrete evidence (specific days, averages, or news).",
            "    * If the observation already exists, output an empty string \"\".",
            "- Keep rationale CONCISE, output only insights that materially affect decision.",
            "",
            "RESPONSE FORMAT (valid JSON only):",
            "{",
            '  "rationale": "Explain step by step: lead-time inference, inventory & demand analysis, news impact, final strategy.",',
            '  "carry_over_insight": "New sustained insight with evidence, or empty string \"\".",',
            f'  "action": {{{example_action}}}',
            "}",
            "",
            f"Remember: use the exact item IDs: {items_str}. Do not output any text outside the JSON.",
        ]
    )

    system_prompt = "\n".join(system_parts)

    return GPT5MiniAgent(
        system_prompt=system_prompt,
        temperature=0.0,
        reasoning_effort="minimal",
        verbosity="low",
    )


class GPT5MiniAgent(ta.Agent):
    """Wrapper around the OpenAI Responses API for gpt-5-mini."""

    def __init__(
        self,
        *,
        system_prompt: str,
        temperature: float = 0.0,
        reasoning_effort: str = "minimal",
        verbosity: str = "low",
    ) -> None:
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self._client = OpenAI(api_key=api_key)
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Expected observation to be a string, received {type(observation)}")
        response = self._client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self._system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": observation}],
                },
            ],
            reasoning={"effort": self._reasoning_effort},
            text={"verbosity": self._verbosity},
        )
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        fragments: List[str] = []
        for block in getattr(response, "output", []) or []:
            for content in getattr(block, "content", []) or []:
                text_obj = getattr(content, "text", None)
                if text_obj and hasattr(text_obj, "value"):
                    fragments.append(text_obj.value)
        if fragments:
            return "\n".join(fragments).strip()
        raise RuntimeError("gpt-5-mini response did not include text content")


class SimulationSession:
    """Stateful session retaining llm_csv_demo style transcript events."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.csv_player = CSVDemandPlayer(config.demand_file)
        self.transcript = SimulationTranscript()

        self.current_day = 1
        self._conversation: List[Dict[str, str]] = []
        self._guidance_messages: Dict[int, str] = {}
        self._guidance_history: List[Tuple[int, str]] = []
        self._pending_guidance_day: Optional[int] = None
        self._carry_over_insights: Dict[int, str] = {}
        self._ui_daily_logs: List[Dict[str, Any]] = []
        self._running_reward: float = 0.0
        self._initial_samples = _default_initial_samples(self.csv_player.get_item_ids())

        self._agent = _make_base_agent(
            item_ids=self.csv_player.get_item_ids(),
            initial_samples=self._initial_samples,
            promised_lead_time=config.promised_lead_time,
            human_feedback_enabled=(config.mode == "mode1"),
            guidance_enabled=(config.mode == "mode2"),
        )

        self._env = ta.make(env_id="VendingMachine-v0")
        self._base_env = self._resolve_base_env(self._env)
        from textarena.envs.VendingMachine import env as vm_env_module

        self._vm_env_module = vm_env_module
        self._original_num_days = vm_env_module.NUM_DAYS
        self._original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
        vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
        self._total_days = self.csv_player.get_num_days()
        vm_env_module.NUM_DAYS = self._total_days

        self._setup_environment()
        self._pid, initial_observation = self._env.get_observation()
        self._observation = _inject_carry_over_insights(initial_observation, self._carry_over_insights)
        if self._pid != 0:
            raise RuntimeError("VM should act first")

        if self.config.mode == "mode1":
            self._bootstrap_mode1()
        else:
            self._bootstrap_mode2()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def serialize_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "mode": self.config.mode,
            "guidance_frequency": self.config.guidance_frequency,
            "current_day": self.current_day,
            "player_id": self._pid,
            "observation": self._observation,
            "transcript": [
                {"kind": evt.kind, "payload": evt.payload} for evt in self.transcript.events
            ],
            "completed": self.transcript.completed,
            "final_reward": self.transcript.final_reward,
            "status_cards": self._build_status_cards(),
            "daily_logs": copy.deepcopy(self._ui_daily_logs),
            "news": self._build_news_schedule(),
        }
        if self.config.mode == "mode1":
            state["conversation"] = list(self._conversation)
            state["waiting_for_final_action"] = self._pid == 0 and not self.transcript.completed
        else:
            state["waiting_for_guidance"] = self._pending_guidance_day is not None
            state["guidance_history"] = [
                {"day": day, "message": message} for day, message in self._guidance_history
            ]
            latest = self._latest_agent_proposal()
            if latest is not None:
                state["latest_agent_proposal"] = latest
        return state

    def add_human_message(self, message: str) -> Dict[str, Any]:
        if self.config.mode != "mode1":
            raise RuntimeError("Human chat only available in Mode 1")
        self._conversation.append({"role": "human", "content": message})
        self.transcript.append("human_message", {"day": self.current_day, "content": message})
        return self._agent_proposal_with_history()

    def submit_final_action(self, action_json: str) -> Dict[str, Any]:
        if self.config.mode != "mode1":
            raise RuntimeError("Final action only available in Mode 1")
        if self._pid != 0:
            raise RuntimeError("Not waiting for VM turn")
        action_dict, carry_memo = self._parse_action_json(action_json)
        self._store_carry_over_insight(self.current_day, carry_memo)
        payload = json.dumps({"action": action_dict})
        self.transcript.append(
            "final_action",
            {"day": self.current_day, "content": action_dict, "source": "human"},
        )
        return self._advance_with_vm_action(payload)

    def submit_guidance(self, message: str) -> Dict[str, Any]:
        if self.config.mode != "mode2":
            raise RuntimeError("Guidance only available in Mode 2")
        if self._pending_guidance_day is None:
            raise RuntimeError("Not waiting for guidance")
        trimmed = message.strip()
        self._guidance_messages[self._pending_guidance_day] = trimmed
        self._guidance_history.append((self._pending_guidance_day, trimmed))
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

        for day, news in self.csv_player.get_news_schedule().items():
            self._env.add_news(day, news)

        self._env.reset(num_players=2)
        self._reset_ui_tracking()

    def _bootstrap_mode1(self) -> None:
        self._conversation.clear()
        self.transcript.append("observation", {"day": self.current_day, "content": self._observation})
        self._agent_proposal_with_history()

    def _bootstrap_mode2(self) -> None:
        self.transcript.append("observation", {"day": self.current_day, "content": self._observation})
        self._run_until_pause_or_complete()

    def _agent_proposal_with_history(self) -> Dict[str, Any]:
        prompt = self._format_prompt_with_conversation()
        action_text = self._agent(prompt)
        cleaned = self._clean_json(action_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"rationale": cleaned, "action": {}}
        self._store_carry_over_insight(self.current_day, data.get("carry_over_insight"))
        self._conversation.append({"role": "agent", "content": cleaned})
        self.transcript.append(
            "agent_proposal", {"day": self.current_day, "content": data}
        )
        return data

    def _format_prompt_with_conversation(self) -> str:
        lines = ["CURRENT OBSERVATION:", self._observation.strip(), ""]
        if self._conversation:
            lines.append("CONVERSATION SO FAR:")
            for turn in self._conversation:
                role = turn["role"].upper()
                lines.append(f"{role}: {turn['content']}")
            lines.append("")
        lines.append("Provide your next JSON proposal.")
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

        demand_action = self.csv_player.get_demand_action(self.current_day)
        self.transcript.append(
            "demand_action", {"day": self.current_day, "content": json.loads(demand_action)}
        )
        done, _ = self._env.step(action=demand_action)
        self._sync_ui_daily_logs()
        if done:
            return self._finalize_session()

        self.current_day += 1
        self._pid, next_observation = self._env.get_observation()
        self._observation = _inject_carry_over_insights(next_observation, self._carry_over_insights)
        self.transcript.append(
            "observation", {"day": self.current_day, "content": self._observation}
        )

        if self.config.mode == "mode1":
            self._conversation.clear()
            proposal = self._agent_proposal_with_history()
            return {"next_observation": self._observation, "proposal": proposal, "completed": False}

        self._run_until_pause_or_complete()
        return self.serialize_state()

    def _run_until_pause_or_complete(self) -> None:
        while not self.transcript.completed:
            guidance_day = self._guidance_due_for_day(self.current_day)
            if guidance_day is not None and guidance_day not in self._guidance_messages:
                self._pending_guidance_day = guidance_day
                break

            prompt = self._format_guided_prompt()
            agent_output = self._agent(prompt)
            cleaned = self._clean_json(agent_output)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                data = {"rationale": cleaned, "action": {}}
            self._store_carry_over_insight(self.current_day, data.get("carry_over_insight"))
            self.transcript.append(
                "agent_proposal", {"day": self.current_day, "content": data}
            )

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
        if self._guidance_history:
            lines.append("HUMAN GUIDANCE HISTORY:")
            for day, message in self._guidance_history:
                quoted = json.dumps(message)
                lines.append(f"Day {day} human message: {quoted}")
            lines.append("")
        lines.append("Return JSON proposal only.")
        return "\n".join(lines)

    def _guidance_due_for_day(self, day: int) -> Optional[int]:
        if self.config.mode != "mode2":
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
        rewards, game_info = self._env.close()
        vm_info = game_info[0]
        total_reward = vm_info.get("total_reward", 0.0)
        self.transcript.final_reward = float(total_reward)
        self.transcript.completed = True
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
            self._ui_daily_logs.append(self._sanitize_day_log(raw_log))

    @staticmethod
    def _sanitize_day_log(log: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {
            "day": int(log.get("day", 0)),
            "news": log.get("news"),
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
        return sanitized

    def _latest_agent_proposal(self) -> Optional[Dict[str, Any]]:
        for event in reversed(self.transcript.events):
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

    def _build_news_schedule(self) -> Dict[str, List[Dict[str, Any]]]:
        schedule = self.csv_player.get_news_schedule()
        if not schedule:
            return {"today": [], "upcoming": [], "past": []}
        today: List[Dict[str, Any]] = []
        past: List[Dict[str, Any]] = []
        for day in sorted(schedule.keys()):
            entry = {"day": day, "content": schedule[day]}
            if day == self.current_day:
                today.append(entry)
            elif day < self.current_day:
                past.append(entry)
        return {"today": today, "upcoming": [], "past": past}


class Mode1Session(SimulationSession):
    def __init__(self, config: SimulationConfig):
        if config.mode != "mode1":
            raise ValueError("Mode1Session requires mode='mode1'")
        super().__init__(config)

    def submit_final_decision(self, action_json: str) -> Dict[str, Any]:
        self.submit_final_action(action_json)
        return self.serialize_state()


class Mode2Session(SimulationSession):
    def __init__(self, config: SimulationConfig):
        if config.mode != "mode2":
            raise ValueError("Mode2Session requires mode='mode2'")
        super().__init__(config)


def load_simulation(config: SimulationConfig) -> SimulationSession:
    if not os.path.exists(config.demand_file):
        raise FileNotFoundError(f"Demand file not found: {config.demand_file}")
    if config.mode == "mode1":
        return Mode1Session(config)
    if config.mode == "mode2":
        return Mode2Session(config)
    raise ValueError(f"Unsupported mode: {config.mode}")


__all__ = [
    "CSVDemandPlayer",
    "SimulationConfig",
    "SimulationTranscript",
    "SimulationSession",
    "Mode1Session",
    "Mode2Session",
    "load_simulation",
]
