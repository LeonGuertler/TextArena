"""Optional Supabase logging for development/demo mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol


# Check if we're in development mode (no Supabase configured)
DEMO_MODE = not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


class SupabaseLogger(Protocol):
    """Protocol for Supabase logger - compatible with real and mock implementations."""
    
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
        ...


@dataclass
class MockSupabaseLogger:
    """Mock Supabase logger for development."""
    
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
        """Log to console instead of Supabase in demo mode."""
        print(f"[DEMO MODE] Game run completed:")
        print(f"  User: {user_id}")
        print(f"  Mode: {mode}")
        print(f"  Final Reward: {final_reward}")
        print(f"  Run ID: {run_id}")


def get_supabase_logger_optional():
    """
    Get Supabase logger.
    Falls back to mock logger in demo mode.
    """
    if DEMO_MODE:
        # Demo mode: log to console
        print("[DEMO MODE] Supabase not configured, using mock logger")
        return MockSupabaseLogger()
    
    # Production mode: use real Supabase
    from .supabase_client import get_supabase_logger as real_logger
    return real_logger()

