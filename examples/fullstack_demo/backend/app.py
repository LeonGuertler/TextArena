"""FastAPI app powering the fullstack vending machine demo."""

from __future__ import annotations

import json
import logging
import os
import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency for memory monitoring

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
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

# Track active LLM calls (threads)
_active_llm_threads = threading.local()
_active_llm_threads.count = 0
_llm_thread_lock = threading.Lock()

# Request monitoring
_request_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "requests_by_endpoint": {},
    "requests_by_status": {},
    "request_times": [],  # Keep last 1000 request times
    "max_active": 0,
}
_request_stats_lock = threading.Lock()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and concurrency info."""
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    # Update active requests
    with _request_stats_lock:
        _request_stats["active_requests"] += 1
        _request_stats["total_requests"] += 1
        _request_stats["max_active"] = max(_request_stats["max_active"], _request_stats["active_requests"])
        
        # Track by endpoint
        endpoint_key = f"{method} {path}"
        _request_stats["requests_by_endpoint"][endpoint_key] = _request_stats["requests_by_endpoint"].get(endpoint_key, 0) + 1
    
    # Log request arrival
    logger.info(f"[REQ] {method} {path} | Active: {_request_stats['active_requests']} | Total: {_request_stats['total_requests']}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Update stats
        with _request_stats_lock:
            _request_stats["active_requests"] -= 1
            status_code = response.status_code
            _request_stats["requests_by_status"][status_code] = _request_stats["requests_by_status"].get(status_code, 0) + 1
            
            # Keep last 1000 request times
            _request_stats["request_times"].append(process_time)
            if len(_request_stats["request_times"]) > 1000:
                _request_stats["request_times"].pop(0)
        
        # Log response
        logger.info(
            f"[RES] {method} {path} | Status: {status_code} | "
            f"Time: {process_time*1000:.1f}ms | Active: {_request_stats['active_requests']}"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Active-Requests"] = str(_request_stats["active_requests"])
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        with _request_stats_lock:
            _request_stats["active_requests"] -= 1
        logger.error(f"[ERR] {method} {path} | Error: {e} | Time: {process_time*1000:.1f}ms")
        raise


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def cleanup_expired_runs() -> int:
    """Remove runs older than TTL. Returns number of cleaned runs."""
    now = time.time()
    expired = [
        run_id for run_id, entry in RUN_STORE.items()
        if now - entry.created_at > RUN_STORE_TTL
    ]
    for run_id in expired:
        del RUN_STORE[run_id]
    return len(expired)


@app.on_event("startup")
async def start_cleanup_task():
    """Start background task to clean up expired runs."""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)  # Run cleanup every 5 minutes
            cleaned = cleanup_expired_runs()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired runs (TTL: {RUN_STORE_TTL/3600:.1f}h)")
    
    asyncio.create_task(cleanup_loop())
    logger.info(f"Started RUN_STORE cleanup task (TTL: {RUN_STORE_TTL/3600:.1f}h, interval: 5min)")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        if psutil is None:
            return 0.0
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_active_thread_count() -> int:
    """Get count of active threads (approximate for LLM calls)."""
    try:
        return threading.active_count()
    except Exception:
        return 0


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
    created_at: float = field(default_factory=time.time)  # Timestamp for TTL cleanup


RUN_STORE: Dict[str, RunEntry] = {}
RUN_STORE_TTL = 3600  # 1 hour in seconds
MAX_ACTIVE_SESSIONS = 1000


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
    # Graceful degradation: check if server is at capacity
    if len(RUN_STORE) >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server at capacity ({len(RUN_STORE)}/{MAX_ACTIVE_SESSIONS} active sessions). Please try again later."
        )
    
    _ensure_mode_choice(payload.mode, auth)

    # Map instance number to folder name
    # Note: instance 1 is the swimwear instance (599580017)
    instance_folders = {
        0: "tutorial",
        1: "599580017",  # Swimwear instance
        2: "568601006",
        3: "706016001",
    }
    instance_folder = instance_folders[payload.instance]
    
    # Construct paths to H&M instance based on instance number
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / "biweely_H&M_instances" / instance_folder
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
    
    # Promised lead time by instance (periods)
    # Swimwear instance has promised lead time 0, others 1
    instance_promised_lead_times = {
        0: 1,
        1: 0,  # swimwear
        2: 1,
        3: 1,
    }
    promised_lead_time = instance_promised_lead_times.get(payload.instance, 1)

    config = SimulationConfig(
        mode=payload.mode,  # type: ignore[arg-type]
        demand_file=str(test_csv_path),
        train_file=str(train_csv_path),
        promised_lead_time=promised_lead_time,
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
        
        # Set up completion callback for modeC (to log game completion when async loop finishes)
        def completion_callback(state: Dict[str, Any]) -> None:
            try:
                _maybe_persist(run_id, session, state, auth, supabase_logger)
            except Exception as e:
                logging.warning(f"Failed to persist game completion in modeC: {e}", exc_info=True)
        
        session._completion_callback = completion_callback

    # For modeB, trigger LLM proposal generation in the background
    # This allows the frontend to show OR recommendation and "thinking" state immediately
    if payload.mode == "modeB" and hasattr(session, "_trigger_llm_proposal_async"):
        background_tasks.add_task(session._trigger_llm_proposal_async)
    
    # For modeC, trigger auto-play loop in the background (creates its own async task)
    # This allows the frontend to return immediately and poll for updates
    if payload.mode == "modeC" and hasattr(session, "_trigger_modeC_auto_play_async"):
        # Use BackgroundTasks to ensure proper async execution
        background_tasks.add_task(session._trigger_modeC_auto_play_async)

    state = session.serialize_state()
    state.update({"run_id": run_id})
    _maybe_persist(run_id, session, state, auth, supabase_logger)
    return state


@app.get("/runs/{run_id}")
def get_run(run_id: str, auth: AuthContext = Depends(get_auth_context)):
    entry = _get_entry(run_id)
    _ensure_user_access(entry, auth)
    return entry.session.serialize_state()


@app.get("/runs/{run_id}/llm-stream")
async def stream_llm_updates(
    run_id: str, 
    request: Request,
    auth: AuthContext = Depends(get_auth_context)
):
    """Stream LLM generation updates via Server-Sent Events.
    
    This endpoint streams streaming_text updates while LLM is generating a proposal.
    The connection closes automatically when the proposal is complete.
    
    Note: EventSource doesn't support custom headers, so user_id can be passed via
    query parameter ?user_id=... as fallback.
    """
    entry = _get_entry(run_id)
    
    # Check query param as fallback (EventSource doesn't support headers)
    # EventSource can't send custom headers, so we must use query parameter
    query_user_id = request.query_params.get("user_id")
    if query_user_id:
        # Use query parameter user_id for EventSource requests
        # Verify it matches the run owner
        if entry.owner_id != query_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Run owned by another user")
    else:
        # Fall back to header-based auth if no query param
        _ensure_user_access(entry, auth)
    session = entry.session
    
    async def event_generator():
        last_streaming_text = ""
        last_proposal_count = 0
        last_period = session.current_day
        last_proposal_periods = set()  # Track which periods have proposals
        
        while True:
            # Check if run still exists
            if run_id not in RUN_STORE:
                break
            
            # Get current streaming text
            streaming_text = session.get_streaming_text()
            
            # Get current period
            current_period = session.current_day
            
            # Track proposals by period
            proposal_periods = set()
            for evt in session.transcript.events:
                if evt.kind == "agent_proposal" and evt.payload:
                    day = evt.payload.get("day")
                    if day:
                        proposal_periods.add(day)
            
            # Count proposals for current period
            proposal_count = len([p for p in proposal_periods if p == current_period])
            
            # Send update if streaming text changed
            if streaming_text != last_streaming_text:
                yield f"data: {json.dumps({'streaming_text': streaming_text})}\n\n"
                last_streaming_text = streaming_text
            
            # Send period completion notification when a new period gets a proposal
            new_completed_periods = proposal_periods - last_proposal_periods
            if new_completed_periods:
                for period in sorted(new_completed_periods):
                    yield f"data: {json.dumps({'period_complete': period, 'current_period': current_period})}\n\n"
                last_proposal_periods = proposal_periods
            
            # Send period change notification (period advanced after completion)
            if current_period != last_period:
                # Period has advanced - this means the previous period completed
                completed_period = last_period
                # Send explicit period completion notification
                yield f"data: {json.dumps({'period_complete': completed_period, 'current_period': current_period})}\n\n"
                yield f"data: {json.dumps({'period_changed': current_period, 'previous_period': last_period, 'completed_period': completed_period})}\n\n"
                last_period = current_period
            
            # Send notification when LLM proposal completes for a period
            # This happens right after LLM finishes, before period advances
            if proposal_count > last_proposal_count:
                # LLM just finished for current period
                yield f"data: {json.dumps({'llm_complete': True, 'period': current_period})}\n\n"
                # Don't break - keep connection open for Mode C to track multiple periods
                last_proposal_count = proposal_count
            
            # Close if game completed
            if session.transcript.completed:
                yield f"data: {json.dumps({'game_complete': True})}\n\n"
                break
            
            # Close if streaming text is empty and we've been waiting (LLM finished without proposal)
            if not streaming_text and last_streaming_text:
                # Wait a bit to see if proposal appears
                await asyncio.sleep(0.1)
                if proposal_count == last_proposal_count:
                    break
            
            await asyncio.sleep(0.05)  # Check every 50ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        }
    )


@app.get("/instances/{instance_num}/image")
def get_instance_image(instance_num: int):
    """Get the product image for a specific instance."""
    instance_folders = {
        0: "tutorial",
        1: "599580017",
        2: "568601006",
        3: "706016001",
    }
    
    if instance_num not in instance_folders:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid instance number")
    
    instance_folder = instance_folders[instance_num]
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / "biweely_H&M_instances" / instance_folder
    image_path = instance_dir / "image.jpg"
    
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/instances/{instance_num}/description")
def get_instance_description(instance_num: int):
    """Get the product description for a specific instance."""
    instance_folders = {
        0: "tutorial",
        1: "599580017",
        2: "568601006",
        3: "706016001",
    }
    
    if instance_num not in instance_folders:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid instance number")
    
    instance_folder = instance_folders[instance_num]
    backend_dir = Path(__file__).resolve().parent
    examples_dir = backend_dir.parent.parent
    instance_dir = examples_dir / "H&M_instances" / "biweely_H&M_instances" / instance_folder
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
    
    # Get current period BEFORE submitting (period doesn't change until after action)
    current_period = session.current_day
    
    # Get OR recommendation BEFORE submitting (it might change after action)
    or_recommendation = session._or_recommendations if hasattr(session, "_or_recommendations") else None
    
    # Extract prompts BEFORE submitting (from current period's LLM proposal)
    prompts = _extract_prompts_from_transcript(session, current_period)
    
    # Submit the final decision (this executes the action and advances the game)
    result = session.submit_final_decision(payload.action_json)
    
    # Log the step AFTER action is executed and reward is updated
    # This ensures we log the reward AFTER the action, matching the final reward
    if entry.user_index is not None and entry.user_uuid and entry.instance:
        try:
            # Get reward AFTER action is executed using the same calculation as final reward
            if hasattr(session, "_get_current_total_reward"):
                current_reward = session._get_current_total_reward()
            else:
                # Fallback if method doesn't exist
                current_reward = session._running_reward if hasattr(session, "_running_reward") else 0.0
            
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
    background_tasks: BackgroundTasks,
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

    result = session.submit_guidance(payload.message, background_tasks)
    
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
    else:
        # Debug: log why completion wasn't logged
        missing_fields = []
        if not entry:
            missing_fields.append("entry")
        elif entry.user_index is None:
            missing_fields.append("user_index")
        elif not entry.user_uuid:
            missing_fields.append("user_uuid")
        elif not entry.instance:
            missing_fields.append("instance")
        logging.warning(
            f"Game completion not logged for run {run_id} (mode={state.get('mode')}): "
            f"missing fields: {', '.join(missing_fields)}"
        )


@app.get("/metrics")
def get_metrics():
    """Get server metrics for monitoring."""
    cleaned = cleanup_expired_runs()  # Clean up expired runs when metrics are requested
    
    # Calculate request statistics
    with _request_stats_lock:
        request_times = _request_stats["request_times"].copy()
        avg_time = sum(request_times) / len(request_times) if request_times else 0
        max_time = max(request_times) if request_times else 0
        min_time = min(request_times) if request_times else 0
        
        # Get top endpoints
        top_endpoints = sorted(
            _request_stats["requests_by_endpoint"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    
    return {
        "active_sessions": len(RUN_STORE),
        "max_active_sessions": MAX_ACTIVE_SESSIONS,
        "memory_mb": round(get_memory_usage_mb(), 2),
        "active_threads": get_active_thread_count(),
        "run_store_ttl_hours": RUN_STORE_TTL / 3600,
        "expired_runs_cleaned": cleaned,
        "requests": {
            "total": _request_stats["total_requests"],
            "active": _request_stats["active_requests"],
            "max_concurrent": _request_stats["max_active"],
            "avg_response_time_ms": round(avg_time * 1000, 2),
            "max_response_time_ms": round(max_time * 1000, 2),
            "min_response_time_ms": round(min_time * 1000, 2),
            "by_status": dict(_request_stats["requests_by_status"]),
            "top_endpoints": [{"endpoint": k, "count": v} for k, v in top_endpoints],
        },
        "connection_limits": {
            "max_active_sessions": MAX_ACTIVE_SESSIONS,
            "note": "Browser limit: ~6 concurrent connections per domain. With 10 browsers: ~60 connections.",
        },
    }


@app.get("/metrics/reset")
def reset_metrics():
    """Reset request statistics (for testing)."""
    with _request_stats_lock:
        _request_stats["total_requests"] = 0
        _request_stats["max_active"] = 0
        _request_stats["requests_by_endpoint"].clear()
        _request_stats["requests_by_status"].clear()
        _request_stats["request_times"].clear()
    return {"message": "Metrics reset"}


# Mount static files after all API routes to avoid conflicts
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
