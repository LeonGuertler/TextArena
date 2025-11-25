"""Supabase client helpers for logging simulation runs."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from supabase import Client, create_client


def _create_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise RuntimeError("Supabase environment variables not set")
    return create_client(url, service_key)


@dataclass
class SupabaseLogger:
    client: Client
    table_name: str = "game_runs"
    step_table_name: str = "game_steps"
    completion_table_name: str = "game_completions"

    def log_run(
        self,
        *,
        user_id: str,
        mode: str,
        final_reward: Optional[float],
        log_text: str,
        guidance_frequency: Optional[int],
        run_id: Optional[str],
    ) -> None:
        payload = {
            "user_id": user_id,
            "mode": mode,
            "final_reward": final_reward,
            "log_text": log_text,
            "guidance_frequency": guidance_frequency,
            "run_id": run_id,
        }
        self.client.table(self.table_name).insert(payload).execute()

    def log_step(
        self,
        *,
        user_index: int,
        user_uuid: str,
        instance: str,
        mode: str,
        period: int,
        inventory_decision: Dict[str, int],
        total_reward: float,
        input_prompt: Optional[str] = None,
        output_prompt: Optional[str] = None,
        or_recommendation: Optional[Dict[str, int]] = None,
        run_id: Optional[str] = None,
        step_type: str = "decision",  # "decision" or "guidance"
    ) -> None:
        """Log a single game step/decision."""
        payload = {
            "user_index": user_index,
            "user_uuid": user_uuid,
            "instance": instance,
            "mode": mode,
            "period": period,
            "inventory_decision": inventory_decision,
            "total_reward": total_reward,
            "input_prompt": input_prompt,
            "output_prompt": output_prompt,
            "or_recommendation": or_recommendation,
            "run_id": run_id,
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.client.table(self.step_table_name).insert(payload).execute()

    def log_guidance(
        self,
        *,
        user_index: int,
        user_uuid: str,
        instance: str,
        mode: str,
        period: int,
        guidance_message: str,
        total_reward: float,
        run_id: Optional[str] = None,
    ) -> None:
        """Log human guidance in Mode C."""
        payload = {
            "user_index": user_index,
            "user_uuid": user_uuid,
            "instance": instance,
            "mode": mode,
            "period": period,
            "inventory_decision": {},  # Empty dict for guidance entries
            "total_reward": total_reward,
            "input_prompt": None,
            "output_prompt": None,
            "or_recommendation": None,
            "guidance_message": guidance_message,  # Separate column for guidance
            "run_id": run_id,
            "step_type": "guidance",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.client.table(self.step_table_name).insert(payload).execute()

    def log_game_completion(
        self,
        *,
        user_index: int,
        user_uuid: str,
        instance: str,
        mode: str,
        total_reward: float,
        run_id: Optional[str] = None,
    ) -> None:
        """Log game completion summary."""
        payload = {
            "user_index": user_index,
            "user_uuid": user_uuid,
            "instance": instance,
            "mode": mode,
            "total_reward": total_reward,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.client.table(self.completion_table_name).insert(payload).execute()


def get_supabase_logger() -> SupabaseLogger:
    client = _create_supabase_client()
    return SupabaseLogger(client=client)


@dataclass
class SupabaseUserManager:
    """Manage user records and cohort index in Supabase.

    Table schema (recommended):
      - uuid: uuid/text, unique
      - user_id: text
      - name: text
      - index: integer, auto-increment (SERIAL/identity)
    """

    client: Client
    table_name: str = "users"  # Default, will be overridden in __post_init__
    
    def __post_init__(self):
        """Set table_name from environment variable (reads at instance creation time)."""
        # Always read from environment to ensure it's current
        env_table_name = os.getenv("SUPABASE_USER_TABLE")
        if env_table_name:
            self.table_name = env_table_name

    def _canonical_key(self, user_id: str, name: str) -> str:
        """Build canonical string from user_id and name."""
        return (user_id.strip() + name.strip()).lower()

    def _hash_to_uuid(self, user_id: str, name: str) -> str:
        """Deterministically hash user_id+name to a UUID string."""
        canonical = self._canonical_key(user_id, name)
        # Use a fixed namespace to make mapping stable
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, canonical)
        return str(user_uuid)

    def get_or_create_user_index(self, user_id: str, name: str) -> Dict[str, Any]:
        """Return deterministic UUID and cohort index for this (user_id, name).

        If the UUID already exists in the user table, return its index.
        Otherwise insert a new record and return the newly assigned index.
        """
        if not user_id or not name:
            raise ValueError("user_id and name are required")

        user_uuid = self._hash_to_uuid(user_id, name)

        # 1) Check if the user already exists
        existing = (
            self.client.table(self.table_name)
            .select("index")
            .eq("uuid", user_uuid)
            .limit(1)
            .execute()
        )

        data = getattr(existing, "data", None) or []
        if data:
            # Existing record; just return its index
            return {"uuid": user_uuid, "index": data[0]["index"]}

        # 2) Insert new record; let DB assign index via SERIAL/identity
        payload = {
            "uuid": user_uuid,
            "user_id": user_id,
            "name": name,
        }

        # Insert the record - execute() returns the inserted row with all columns including index
        inserted = (
            self.client.table(self.table_name)
            .insert(payload)
            .execute()
        )
        
        inserted_data = getattr(inserted, "data", None) or []
        if not inserted_data:
            raise RuntimeError("Failed to insert user record into Supabase")
        
        # The inserted data should contain the index (SERIAL columns are returned by default)
        inserted_row = inserted_data[0]
        if "index" not in inserted_row:
            # If index wasn't returned, query it back
            result = (
                self.client.table(self.table_name)
                .select("index")
                .eq("uuid", user_uuid)
                .limit(1)
                .execute()
            )
            result_data = getattr(result, "data", None) or []
            if not result_data:
                raise RuntimeError("Failed to retrieve index after insert")
            index = result_data[0]["index"]
        else:
            index = inserted_row["index"]

        return {"uuid": user_uuid, "index": index}


def get_supabase_user_manager() -> SupabaseUserManager:
    client = _create_supabase_client()
    return SupabaseUserManager(client=client)

