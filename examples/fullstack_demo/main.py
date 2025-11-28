"""Unified entry point for the fullstack vending machine demo."""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path
import sys

import uvicorn
from dotenv import load_dotenv
from uvicorn import Config, Server

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.fullstack_demo.backend.app import app as fastapi_app


def _should_open_browser(reload_enabled: bool) -> bool:
    if not reload_enabled:
        return True
    return os.environ.get("UVICORN_RUN_MAIN") == "true"


def launch_uvicorn(*, reload_enabled: bool) -> None:
    # Render provides PORT environment variable, fallback to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    
    if reload_enabled:
        backend_dir = Path(__file__).resolve().parent / "backend"
        frontend_dir = Path(__file__).resolve().parent / "frontend"
        uvicorn.run(
            "examples.fullstack_demo.backend.app:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info",
            reload_dirs=[
                str(backend_dir),
                str(frontend_dir),
            ],
        )
        return

    config = Config(
        app=fastapi_app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        limit_concurrency=1000,  # Maximum concurrent connections per worker
        limit_max_requests=10000,  # Maximum requests before worker restart (prevents memory leaks)
        timeout_keep_alive=30,  # Keep-alive timeout in seconds
    )
    server = Server(config=config)
    server.run()


def main() -> None:
    # Try loading .env from multiple locations
    # 1. From fullstack_demo directory
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    # 2. From project root (OR_Agent directory)
    project_root_env = PROJECT_ROOT / ".env"
    if project_root_env.exists():
        load_dotenv(project_root_env)
    # 3. From current working directory
    load_dotenv()

    # Disable reload in production (Render sets FULLSTACK_DEMO_RELOAD=0)
    reload_flag = os.environ.get("FULLSTACK_DEMO_RELOAD", "1")
    reload_enabled = reload_flag not in {"0", "false", "False"}

    if _should_open_browser(reload_enabled):
        port = int(os.environ.get("PORT", 8000))
        frontend_url = f"http://localhost:{port}/"
        try:
            webbrowser.open(frontend_url)
        except Exception:
            pass

    launch_uvicorn(reload_enabled=reload_enabled)


if __name__ == "__main__":
    main()

