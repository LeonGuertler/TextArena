"""Unified entry point for the fullstack vending machine demo."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
import sys

from dotenv import load_dotenv
from uvicorn import Config, Server

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.fullstack_demo.backend.app import app as fastapi_app


def launch_uvicorn() -> None:
    config = Config(
        app=fastapi_app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
    server = Server(config=config)
    server.run()


def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv()

    # Open browser first
    frontend_url = "http://localhost:8000/"
    try:
        webbrowser.open(frontend_url)
    except Exception:
        pass

    # Run server in foreground so Ctrl+C works
    launch_uvicorn()


if __name__ == "__main__":
    main()

