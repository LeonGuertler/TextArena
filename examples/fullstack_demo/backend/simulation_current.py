"""Legacy-style simulation helpers retaining llm_csv_demo-style transcripts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd
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


def _make_base_agent(*, promised_lead_time: int, guidance_enabled: bool) -> ta.agents.OpenAIAgent:
    items_placeholder = "Use listed item IDs"
    system = (
        "You are the Vending Machine controller (VM).\n"
        "Objective: maximize total reward = Σ (profit · sold - holding_cost · ending_inventory).\n"
        f"Supplier-promised lead time: {promised_lead_time} days.\n"
        "Actual lead time may differ; infer from arrivals.\n"
        "Provide JSON only: {\"rationale\": \"...\", \"carry_over_insight\": \"...\", \"action\": {\"item\": qty, ...}}\n"
        "If no new insight, set carry_over_insight to \"\"—only populate it when you detect a meaningful, lasting shift.\n"
    )
    if guidance_enabled:
        system += (
            "\nYou may receive human strategic guidance. Incorporate ALL guidance faithfully.\n"
        )
    system += f"\nAvailable items: {items_placeholder}\n"
    return ta.agents.OpenAIAgent(model_name="gpt-4o-mini", system_prompt=system, temperature=0)


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

        self._agent = _make_base_agent(
            promised_lead_time=config.promised_lead_time,
            guidance_enabled=(config.mode == "mode2"),
        )

        self._env = ta.make(env_id="VendingMachine-v0")
        from textarena.envs.VendingMachine import env as vm_env_module

        self._vm_env_module = vm_env_module
        self._original_num_days = vm_env_module.NUM_DAYS
        self._original_initial_inventory = vm_env_module.INITIAL_INVENTORY_PER_ITEM
        vm_env_module.INITIAL_INVENTORY_PER_ITEM = 0
        vm_env_module.NUM_DAYS = self.csv_player.get_num_days()

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
        }
        if self.config.mode == "mode1":
            state["conversation"] = list(self._conversation)
            state["waiting_for_final_action"] = self._pid == 0 and not self.transcript.completed
        else:
            state["waiting_for_guidance"] = self._pending_guidance_day is not None
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

        unified_samples = _default_initial_samples(self.csv_player.get_item_ids())
        self.transcript.append("initial_samples", {"samples": unified_samples})

        for day, news in self.csv_player.get_news_schedule().items():
            self._env.add_news(day, news)

        self._env.reset(num_players=2)

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
