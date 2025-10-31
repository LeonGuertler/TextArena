"""Optional authentication for development/demo mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Header


# Check if we're in development mode (no Supabase configured)
DEMO_MODE = not os.getenv("SUPABASE_JWT_SECRET")


@dataclass
class AuthContext:
    """Authentication context - compatible with real and mock auth."""
    user_id: Optional[str]
    token: str


def get_auth_context_optional(authorization: Optional[str] = Header(None)) -> AuthContext:
    """
    Extract and verify JWT from Authorization header.
    Falls back to mock auth in demo mode.
    """
    if DEMO_MODE:
        # Demo mode: no authentication required
        return AuthContext(user_id="demo-user", token="demo-token")
    
    # Production mode: require real authentication
    from .token_verifier import get_auth_context as real_auth
    return real_auth(authorization)

