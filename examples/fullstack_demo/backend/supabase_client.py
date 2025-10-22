"""Supabase client helpers for logging simulation runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends
from supabase import Client, create_client


@dataclass
class SupabaseLogger:
    client: Client
    table_name: str = "game_runs"

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


def get_supabase_logger() -> SupabaseLogger:
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise RuntimeError("Supabase environment variables not set")
    client = create_client(url, service_key)
    return SupabaseLogger(client=client)

