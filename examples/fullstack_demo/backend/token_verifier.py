"""Supabase JWT verification utilities for FastAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, Header, status


def _get_jwt_secret() -> str:
    """Get Supabase JWT secret for token verification."""
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET must be set in .env file")
    return secret


@dataclass
class AuthContext:
    user_id: Optional[str]
    token: str


def get_auth_context(authorization: Optional[str] = Header(None)) -> AuthContext:
    """Extract and verify JWT from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    token = authorization.split(" ", 1)[1]
    payload = _decode_token(token)
    user_id = payload.get("sub")
    return AuthContext(user_id=user_id, token=token)


def _decode_token(token: str) -> Dict[str, Any]:
    """Decode and verify Supabase JWT using the JWT secret."""
    secret = _get_jwt_secret()
    
    try:
        # Supabase uses HS256 algorithm with JWT secret
        payload = jwt.decode(
            token,
            key=secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}") from exc

