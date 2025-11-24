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


@app.get("/modeA.html")
def serve_modeA():
    modeA_path = FRONTEND_DIR / "modeA.html"
    if modeA_path.exists():
        return FileResponse(modeA_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mode A page not found")


@app.get("/modeB.html")
def serve_modeB():
    modeB_path = FRONTEND_DIR / "modeB.html"
    if modeB_path.exists():
        return FileResponse(modeB_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mode B page not found")


@app.get("/modeC.html")
def serve_modeC():
    modeC_path = FRONTEND_DIR / "modeC.html"
    if modeC_path.exists():
        return FileResponse(modeC_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mode C page not found")


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
    mode: str = Field(pattern="^(modeA|modeB|modeC)$")
    guidance_frequency: Optional[int] = Field(default=4, ge=1)
    enable_or: bool = True  # The convenient switch for OR on/off


class FinalActionPayload(BaseModel):
    action_json: str = Field(min_length=2)


@dataclass
class RunEntry:
    session: Any  # Can be Mode1Session or Mode2Session
    owner_id: str


RUN_STORE: Dict[str, RunEntry] = {}


@app.post("/runs")
def start_run(
    payload: StartRunPayload,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
):
    _ensure_mode_choice(payload.mode, auth)

    # Construct paths to H&M instance 568601006
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / "568601006"
    test_csv_path = instance_dir / "test.csv"
    train_csv_path = instance_dir / "train.csv"
    
    if not test_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test CSV not found: {test_csv_path}"
        )
    if not train_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Train CSV not found: {train_csv_path}"
        )
    
    config = SimulationConfig(
        mode=payload.mode,  # type: ignore[arg-type]
        demand_file=str(test_csv_path),
        train_file=str(train_csv_path),
        promised_lead_time=1,  # Fixed to 1
        guidance_frequency=payload.guidance_frequency or 4,
        enable_or=payload.enable_or,
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

    if session.config.mode not in ("modeA", "modeB"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only Mode A and B support final actions")

    result = session.submit_final_decision(payload.action_json)
    
    # Only persist when game is completed
    if result.get("completed"):
        _maybe_persist(run_id, session, result, auth, supabase_logger)
    
    return result


class GuidancePayload(BaseModel):
    message: str = Field(min_length=0)  # Allow empty guidance - user can submit blank to let AI continue autonomously


@app.post("/runs/{run_id}/guidance")
def submit_guidance(
    run_id: str,
    payload: GuidancePayload,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    session = entry.session

    if session.config.mode != "modeC":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only Mode C supports guidance")

    result = session.submit_guidance(payload.message)
    
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
    if mode not in {"modeA", "modeB", "modeC"}:
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
        guidance_frequency=session.config.guidance_frequency if session.config.mode == "mode2_llm" else None,
        run_id=run_id,
    )


# Mount static files after all API routes to avoid conflicts
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
