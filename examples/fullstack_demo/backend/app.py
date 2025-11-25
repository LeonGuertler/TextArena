"""FastAPI app powering the fullstack vending machine demo."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from .simulation_current import (
    SimulationConfig,
    load_simulation,
)
from .supabase_client import (
    SupabaseLogger,
    SupabaseUserManager,
    get_supabase_logger,
    get_supabase_user_manager,
)
from .token_verifier import AuthContext, get_auth_context


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logger = logging.getLogger(__name__)

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


class UserIndexPayload(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)


class UserIndexResponse(BaseModel):
    uuid: str
    index: int


@app.post("/user-index", response_model=UserIndexResponse)
def create_or_get_user_index(
    payload: UserIndexPayload,
    user_manager: SupabaseUserManager = Depends(get_supabase_user_manager),
):
    """Create (or fetch) a stable UUID + index for this user.

    The index can later be used to decide which modes a user sees.
    """
    try:
        result = user_manager.get_or_create_user_index(
            user_id=payload.user_id,
            name=payload.name,
        )
        return UserIndexResponse(uuid=result["uuid"], index=int(result["index"]))
    except ValueError as exc:
        logger.warning(f"Invalid input for user-index: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(f"Failed to get or create user index for user_id={payload.user_id}, name={payload.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get or create user index: {str(exc)}",
        ) from exc


class StartRunPayload(BaseModel):
    mode: str = Field(pattern="^(modeA|modeB|modeC)$")
    guidance_frequency: Optional[int] = Field(default=4, ge=1)
    enable_or: bool = True  # The convenient switch for OR on/off
    instance: int = Field(default=0, ge=0, le=3)  # Instance number: 0=tutorial, 1=568601006, 2=599580017, 3=706016001


class FinalActionPayload(BaseModel):
    action_json: str = Field(min_length=2)


@dataclass
class RunEntry:
    session: Any  # Can be Mode1Session or Mode2Session
    owner_id: str
    user_index: Optional[int] = None
    user_uuid: Optional[str] = None
    instance: Optional[str] = None  # Instance folder name (e.g., "tutorial", "568601006")


RUN_STORE: Dict[str, RunEntry] = {}


def _get_user_info(user_id: str, user_manager: SupabaseUserManager) -> Dict[str, Any]:
    """Get user index and uuid from users table by user_id."""
    try:
        # Query users table by user_id to get uuid and index
        result = (
            user_manager.client.table(user_manager.table_name)
            .select("uuid, index")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = getattr(result, "data", None) or []
        if data:
            return {"uuid": data[0]["uuid"], "index": data[0]["index"]}
        # If not found, return None values (logging will be skipped)
        return {"uuid": None, "index": None}
    except Exception as e:
        logging.warning(f"Failed to get user info for user_id={user_id}: {e}")
        return {"uuid": None, "index": None}


@app.post("/runs")
def start_run(
    payload: StartRunPayload,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_auth_context),
    supabase_logger: SupabaseLogger = Depends(get_supabase_logger),
    user_manager: SupabaseUserManager = Depends(get_supabase_user_manager),
):
    _ensure_mode_choice(payload.mode, auth)

    # Map instance number to folder name
    instance_folders = {
        0: "tutorial",
        1: "568601006",
        2: "599580017",
        3: "706016001",
    }
    instance_folder = instance_folders[payload.instance]
    
    # Construct paths to H&M instance based on instance number
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / instance_folder
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
    
    # Get user info for logging
    user_info = _get_user_info(auth.user_id or "anonymous", user_manager)
    
    RUN_STORE[run_id] = RunEntry(
        session=session,
        owner_id=auth.user_id or "anonymous",
        user_index=user_info.get("index"),
        user_uuid=user_info.get("uuid"),
        instance=instance_folder,
    )

    # Set up step logging callback for modeC (automatic decisions)
    if payload.mode == "modeC" and user_info.get("index") is not None and user_info.get("uuid") and instance_folder:
        def log_step_callback(step_data: Dict[str, Any]) -> None:
            try:
                supabase_logger.log_step(
                    user_index=user_info["index"],
                    user_uuid=user_info["uuid"],
                    instance=instance_folder,
                    mode=payload.mode,
                    period=step_data["period"],
                    inventory_decision=step_data["inventory_decision"],
                    total_reward=step_data["total_reward"],
                    input_prompt=step_data.get("input_prompt"),
                    output_prompt=step_data.get("output_prompt"),
                    or_recommendation=step_data.get("or_recommendation"),
                    run_id=run_id,
                )
            except Exception as e:
                logging.warning(f"Failed to log step in modeC: {e}", exc_info=True)
        
        session._step_logging_callback = log_step_callback

    # For modeB, trigger LLM proposal generation in the background
    # This allows the frontend to show OR recommendation and "thinking" state immediately
    if payload.mode == "modeB" and hasattr(session, "_trigger_llm_proposal_async"):
        background_tasks.add_task(session._trigger_llm_proposal_async)

    state = session.serialize_state()
    state.update({"run_id": run_id})
    _maybe_persist(run_id, session, state, auth, supabase_logger)
    return state


@app.get("/runs/{run_id}")
def get_run(run_id: str, auth: AuthContext = Depends(get_auth_context)):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    return entry.session.serialize_state()


@app.get("/instances/{instance_num}/image")
def get_instance_image(instance_num: int):
    """Get the product image for a specific instance."""
    instance_folders = {
        0: "tutorial",
        1: "568601006",
        2: "599580017",
        3: "706016001",
    }
    
    if instance_num not in instance_folders:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid instance number")
    
    instance_folder = instance_folders[instance_num]
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / instance_folder
    image_path = instance_dir / "image.jpg"
    
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/instances/{instance_num}/description")
def get_instance_description(instance_num: int):
    """Get the product description for a specific instance."""
    instance_folders = {
        0: "tutorial",
        1: "568601006",
        2: "599580017",
        3: "706016001",
    }
    
    if instance_num not in instance_folders:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid instance number")
    
    instance_folder = instance_folders[instance_num]
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / instance_folder
    description_path = instance_dir / "description.csv"
    
    if not description_path.exists():
        return {"product": "", "description": ""}
    
    # Read description.csv
    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        product = ""
        description = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Product:"):
                product = line.replace("Product:", "").strip()
            elif line.startswith("Product description:"):
                description = line.replace("Product description:", "").strip()
        
        return {"product": product, "description": description}
    except Exception as e:
        return {"product": "", "description": f"Error reading description: {str(e)}"}


def _extract_prompts_from_transcript(session: Any, period: int) -> Dict[str, Optional[str]]:
    """Extract input and output prompts for a given period from transcript."""
    input_prompt = None
    output_prompt = None
    
    # Look for agent_proposal events for this period
    for evt in session.transcript.events:
        if evt.kind == "agent_proposal" and evt.payload.get("day") == period:
            content = evt.payload.get("content", {})
            # Input prompt would be the observation, output would be the rationale
            output_prompt = content.get("rationale")
            # Try to find the observation for this period
            for obs_evt in session.transcript.events:
                if obs_evt.kind == "observation" and obs_evt.payload.get("day") == period:
                    input_prompt = obs_evt.payload.get("content", "")
                    break
            break
    
    return {"input_prompt": input_prompt, "output_prompt": output_prompt}


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

    # Extract decision before submitting
    import json
    try:
        action_dict = json.loads(payload.action_json)
        if isinstance(action_dict, dict) and "action" in action_dict:
            action_dict = action_dict["action"]
    except json.JSONDecodeError:
        # If parsing fails, try to extract from the raw string
        action_dict = {}
    
    # Get current period and reward
    current_period = session.current_day
    current_reward = session._running_reward if hasattr(session, "_running_reward") else 0.0
    
    # Get OR recommendation
    or_recommendation = session._or_recommendations if hasattr(session, "_or_recommendations") else None
    
    # Extract prompts
    prompts = _extract_prompts_from_transcript(session, current_period)
    
    result = session.submit_final_decision(payload.action_json)
    
    # Log the step (non-blocking - don't fail if logging fails)
    if entry.user_index is not None and entry.user_uuid and entry.instance:
        try:
            supabase_logger.log_step(
                user_index=entry.user_index,
                user_uuid=entry.user_uuid,
                instance=entry.instance,
                mode=session.config.mode,
                period=current_period,
                inventory_decision=action_dict,
                total_reward=current_reward,
                input_prompt=prompts.get("input_prompt"),
                output_prompt=prompts.get("output_prompt"),
                or_recommendation=or_recommendation,
                run_id=run_id,
            )
        except Exception as e:
            logging.warning(f"Failed to log step to Supabase: {e}", exc_info=True)
    
    # Only persist when game is completed (non-blocking - don't fail if persistence fails)
    if result.get("completed"):
        try:
            _maybe_persist(run_id, session, result, auth, supabase_logger)
        except Exception as e:
            # Log persistence error but don't fail the request
            logging.warning(f"Failed to persist game completion to Supabase: {e}", exc_info=True)
    
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

    # Log the human guidance before processing
    if entry.user_index is not None and entry.user_uuid and entry.instance:
        try:
            # Get current period (the pending guidance day) and reward
            guidance_period = session._pending_guidance_day if hasattr(session, "_pending_guidance_day") and session._pending_guidance_day else session.current_day
            current_reward = session._running_reward if hasattr(session, "_running_reward") else 0.0
            
            supabase_logger.log_guidance(
                user_index=entry.user_index,
                user_uuid=entry.user_uuid,
                instance=entry.instance,
                mode=session.config.mode,
                period=guidance_period,
                guidance_message=payload.message.strip(),
                total_reward=current_reward,
                run_id=run_id,
            )
        except Exception as e:
            logging.warning(f"Failed to log guidance to Supabase: {e}", exc_info=True)

    # Set up step logging callback for modeC automatic decisions
    if entry.user_index is not None and entry.user_uuid and entry.instance:
        def log_step_callback(step_data: Dict[str, Any]) -> None:
            try:
                supabase_logger.log_step(
                    user_index=entry.user_index,
                    user_uuid=entry.user_uuid,
                    instance=entry.instance,
                    mode=session.config.mode,
                    period=step_data["period"],
                    inventory_decision=step_data["inventory_decision"],
                    total_reward=step_data["total_reward"],
                    input_prompt=step_data.get("input_prompt"),
                    output_prompt=step_data.get("output_prompt"),
                    or_recommendation=step_data.get("or_recommendation"),
                    run_id=run_id,
                )
            except Exception as e:
                logging.warning(f"Failed to log step in modeC: {e}", exc_info=True)
        
        session._step_logging_callback = log_step_callback
    else:
        session._step_logging_callback = None

    result = session.submit_guidance(payload.message)
    
    # Only persist when game is completed (non-blocking - don't fail if persistence fails)
    if result.get("completed"):
        try:
            _maybe_persist(run_id, session, result, auth, supabase_logger)
        except Exception as e:
            # Log persistence error but don't fail the request
            logging.warning(f"Failed to persist game completion to Supabase: {e}", exc_info=True)
    
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
        
    final_reward = state.get("final_reward", 0.0)
    
    # Get entry to access user info and instance
    entry = RUN_STORE.get(run_id)
    
    # Log game completion with user index/uuid if available
    if entry and entry.user_index is not None and entry.user_uuid and entry.instance:
        try:
            supabase_logger.log_game_completion(
                user_index=entry.user_index,
                user_uuid=entry.user_uuid,
                instance=entry.instance,
                mode=state.get("mode"),
                total_reward=float(final_reward),
                run_id=run_id,
            )
        except Exception as e:
            logging.warning(f"Failed to log game completion to Supabase: {e}", exc_info=True)


# Mount static files after all API routes to avoid conflicts
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
