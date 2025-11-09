"""FastAPI app powering the fullstack vending machine demo."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from .simulation_current import (
    SimulationConfig,
    load_simulation,
)
from .supabase_client import SupabaseLogger, get_supabase_logger
from .token_verifier import AuthContext, get_auth_context


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="TextArena VM Demo")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend not found")


@app.get("/mode1.html")
def serve_mode1():
    mode1_path = FRONTEND_DIR / "mode1.html"
    if mode1_path.exists():
        return FileResponse(mode1_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mode 1 page not found")


@app.get("/mode2.html")
def serve_mode2():
    mode2_path = FRONTEND_DIR / "mode2.html"
    if mode2_path.exists():
        return FileResponse(mode2_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mode 2 page not found")


@app.get("/config.js")
def config_js() -> Response:
    supabase_url = os.getenv("SUPABASE_URL", "")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    body = (
        f"window.SUPABASE_URL = \"{supabase_url}\";\n"
        f"window.SUPABASE_ANON_KEY = \"{anon_key}\";\n"
    )
    return Response(content=body, media_type="application/javascript")


class StartRunPayload(BaseModel):
    mode: str = Field(pattern="^(mode1|mode2)$")
    promised_lead_time: int = 0
    guidance_frequency: Optional[int] = Field(default=5, ge=1)


class MessagePayload(BaseModel):
    message: Optional[str] = None


class FinalActionPayload(BaseModel):
    action_json: str = Field(min_length=2)


@dataclass
class RunEntry:
    session: Any  # Can be Mode1Session or Mode2Session
    owner_id: str


RUN_STORE: Dict[str, RunEntry] = {}

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


@app.post("/runs")
def start_run(
    payload: StartRunPayload,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
):
    _ensure_mode_choice(payload.mode, auth)

    # Construct absolute path to demand_case1_iid_normal.csv
    backend_dir = Path(__file__).resolve().parent
    demand_csv_path = backend_dir.parent.parent / "demand_case1_iid_normal.csv"
    
    config = SimulationConfig(
        mode=payload.mode,  # type: ignore[arg-type]
        demand_file=str(demand_csv_path),
        promised_lead_time=payload.promised_lead_time,
        guidance_frequency=payload.guidance_frequency or 5,
    )

    session = load_simulation(config)
    run_id = str(uuid.uuid4())
    RUN_STORE[run_id] = RunEntry(session=session, owner_id=auth.user_id)

    state = session.serialize_state()
    state.update({"run_id": run_id})
    _maybe_persist(run_id, session, state, auth, supabase_logger)
    return state


@app.get("/runs/{run_id}")
def get_run(run_id: str, auth: AuthContext = Depends(get_auth_context)):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    return entry.session.serialize_state()


@app.post("/runs/{run_id}/messages")
def send_message(
    run_id: str,
    payload: MessagePayload,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    session = entry.session

    message = (payload.message or "").strip()

    if session.config.mode == "mode1":
        if not message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Message cannot be empty",
            )
        # Mode 1: Add human message to conversation, get AI response
        result = session.add_human_message(message)
        state = session.serialize_state()
        state["ai_response"] = result
        return state
    
    # Mode 2: Submit guidance
    result = session.submit_guidance(message)
    if result.get("completed"):
        _maybe_persist(run_id, session, result, auth, supabase_logger)
    return result


@app.post("/runs/{run_id}/final-action")
def submit_final_action(
    run_id: str,
    payload: FinalActionPayload,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    session = entry.session

    if session.config.mode != "mode1":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only Mode 1 supports final actions")

    result = session.submit_final_decision(payload.action_json)
    
    # Only persist when game is completed
    if result.get("completed"):
        _maybe_persist(run_id, session, result, auth, supabase_logger)
    
    return result


def _get_entry(run_id: str) -> RunEntry:
    if run_id not in RUN_STORE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RUN_STORE[run_id]


def _ensure_mode_choice(mode: str, auth: AuthContext) -> None:
    if not auth.user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing user context")
    if mode not in {"mode1", "mode2"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode")


def _ensure_user_access(entry: RunEntry, auth: AuthContext) -> None:
    if not auth.user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing user context")
    if entry.owner_id != auth.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Run owned by another user")


def _maybe_persist(
    run_id: str,
    session: Any,
    state: Dict[str, Any],
    auth: AuthContext,
    supabase_logger: SupabaseLogger,
) -> None:
    # Only persist when game is completed
    if not state.get("completed"):
        return
        
    transcript = state.get("transcript", [])
    raw_output = json.dumps(transcript)
    final_reward = state.get("final_reward", 0.0)
    
    supabase_logger.log_run(
        user_id=auth.user_id,
        mode=state.get("mode"),
        final_reward=final_reward,
        log_text=raw_output,
        guidance_frequency=session.config.guidance_frequency if session.config.mode == "mode2" else None,
        run_id=run_id,
    )

