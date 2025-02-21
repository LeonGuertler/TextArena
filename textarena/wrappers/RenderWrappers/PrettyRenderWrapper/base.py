from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import threading
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Optional
import shutil
from datetime import datetime
import re


class BaseRenderer(ABC):
    """
    Base class to create a real-time game server using FastAPI and WebSocket.
    """
    def __init__(self, env: Any, player_names: Optional[Dict[int, str]] = None, port: int = 8000, host: str = "127.0.0.1"):
        """
        Initialize the server with environment, player names, and server details.
        """
        self.env = env  # Game environment
        self.player_names = player_names or {i: f"Player {i}" for i in range(env.state.num_players)}
        self.port = port
        self.host = host
        self.active_connections: Set[WebSocket] = set()  # Store connected clients
        self.chat_history = []  # Store messages between players
        self.end_game_state = None  # Store the final game state
        
        # Set up directories for static files
        self.base_dir = Path(__file__).parent
        self.template_dir = self.base_dir / "templates"
        self.static_dir = self.base_dir / "static_temp"
        self._setup_static_files()
        
        # Create and start the server
        self.app = self._create_app()
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)  # Give server time to start

    def _create_app(self) -> FastAPI:
        """Create FastAPI app with endpoints."""
        app = FastAPI()
        app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the main HTML page."""
            with open(self.template_dir / "index.html") as f:
                return HTMLResponse(f.read())

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections for real-time updates."""
            await websocket.accept()
            self.active_connections.add(websocket)
            try:
                await self.broadcast_state()  # Send initial state
                while True:
                    await websocket.receive_text()  # Wait for messages (not used yet)
                    await websocket.send_json({"status": "ok"})
            except Exception:
                pass
            finally:
                self.active_connections.remove(websocket)  # Clean up disconnected clients
        return app

    def _run_server(self):
        """Run the FastAPI server."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        server = uvicorn.Server(config)
        server.run()

    def _setup_static_files(self):
        """Set up static CSS, JS, and additional assets for the user interface."""
        # Ensure the static directory exists and is empty
        if self.static_dir.exists():
            for file in self.static_dir.glob("*"):
                if file.is_file():
                    file.unlink()  # Delete files
                elif file.is_dir():
                    shutil.rmtree(file)  # Delete directories
        else:
            self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the assets directory
        assets_dir = self.static_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Copy the GitHub logo into the static_temp/assets folder
        for image in self.base_dir.glob("static/*.png"):
            shutil.copy(image, assets_dir)
            # source_logo_path = self.base_dir / "static" / "github-mark-white.png"
            # target_logo_path = assets_dir / "github-mark-white.png"
            # if source_logo_path.exists():
            #     shutil.copy(source_logo_path, target_logo_path)

        # Combine base and custom JavaScript
        base_static = self.base_dir / "static"
        base_js = (base_static / "app.js").read_text() if (base_static / "app.js").exists() else ""
        custom_js = self.get_custom_js()
        with open(self.static_dir / "app.js", "w") as f:
            f.write(base_js)
            if custom_js:
                f.write("\n// Custom Game-Specific JS\n" + custom_js)

        # Combine base and custom CSS
        base_css = (base_static / "style.css").read_text() if (base_static / "style.css").exists() else ""
        custom_css = self.get_custom_css()
        with open(self.static_dir / "style.css", "w") as f:
            f.write(base_css)
            if custom_css:
                f.write("\n/* Custom Game-Specific CSS */\n" + custom_css)



    def get_custom_js(self) -> str:
        """Override to add custom JavaScript logic for the game."""
        return ""

    def get_custom_css(self) -> str:
        """Override to add custom CSS styles for the game."""
        return ""

    def draw(self):
        """Trigger a state update for all connected clients."""
        if self.active_connections:
            asyncio.run(self.broadcast_state())

    async def broadcast_state(self):
        """Send the current game state to all clients."""
        if not self.active_connections:
            return
        
        try:
            state = self.get_state()
            state["num_player_class"] = self._get_num_player_class()
            state["env_id"] = self.env.env_id
            state["github_link"] = self._entry_point_to_github_url(entry_point=self.env.entry_point)
            state["gameplay_instructions"] = self._get_gameplay_instructions()
            state["player_names"] = self.player_names
            state["chat_history"] = self.get_chat_with_colors(self.chat_history, self.player_names)
            state["end_game_state"] = self.end_game_state

            await asyncio.gather(*[conn.send_json(state) for conn in self.active_connections])
        except Exception as e:
            print(f"Error broadcasting state: {e}")


    @abstractmethod
    def get_state(self) -> dict:
        """Override this to provide the game-specific state."""
        pass

    def set_end_game_state(self, rewards: Dict[int, float], info: Dict[str, Any]):
        """Calculate and set the final game state."""
        ## determine if single player or multiplayer
        if len(self.player_names) == 1:
            ## single player
            if rewards:
                winner_text = "You Win!" if list(rewards.values())[0] > 0 else "You Lose!"
        else:
            ## two or more players
            if rewards:
                max_reward = max(rewards.values())
                winners = [pid for pid, r in rewards.items() if r == max_reward]
                if len(winners) > 1:
                    winner_text = "Game ended in a draw"
                else:
                    winner_text = f"Winner: {self.player_names[winners[0]]}"
            else:
                winner_text = "Game Over"

        reason = info.get('reason', 'Game Over')
        self.end_game_state = {"winner_text": winner_text, "reason": reason}
        self.draw()

    def get_chat_with_colors(self, chat_history: list, player_names: dict) -> list:
        """
        Adds a color key to chat messages based on player IDs.
        Supports any number of players and highlights the GAME referee.
        """
        color_map = self._generate_player_colors(player_names)

        # Append color information to each chat message
        chat_with_colors = [
            {
                "player_id": msg["player_id"],
                "player_name": (
                    "GAME" if msg["player_id"] == -1 
                    else player_names.get(msg["player_id"])
                ),
                "message": msg["message"],
                "color": color_map.get(msg["player_id"], "#000000"),  # Default to black
                "timestamp": self._format_timestamp(msg["timestamp"])
            }
            for msg in chat_history
        ]
        return chat_with_colors

    def _generate_player_colors(self, player_names: dict) -> dict:
        """
        Generates a color map for players dynamically based on their IDs.
        Supports any number of players and assigns a standout color to the GAME referee.
        """
        base_colors = [
            "#F4A261",  # Muted Orange (Soft Coral)
            "#2A9D8F",  # Modern Teal (Deep Aqua)
            "#264653",  # Dark Slate Blue (Charcoal Blue)
            "#E76F51",  # Terracotta Red (Warm Rust)
            "#A8DADC",  # Soft Cyan (Pastel Aqua)
            "#457B9D",  # Muted Blue (Steel Blue)
            "#F1FAEE",  # Off-White (Soft Cream)
            "#8D99AE",  # Muted Blue-Gray (Slate Gray)
            "#D4A5A5",  # Pastel Rose (Dusty Pink)
            "#BDE0FE",  # Soft Sky Blue (Light Azure)
            "#FFB4A2",  # Peachy Coral (Warm Pastel)
            "#6D6875"   # Warm Purple-Gray (Mauve)
        ]

        referee_color = "#0074D9"  # Blue (Royal Blue)
        color_map = {}

        # Assign colors for player IDs
        for i, player_id in enumerate(player_names.keys()):
            color_map[player_id] = base_colors[i % len(base_colors)]

        # Explicitly set color for "GAME" judge (-1)
        color_map[-1] = referee_color

        return color_map

    def _format_timestamp(self, unix_timestamp: float) -> str:
        """
        Converts a Unix timestamp to a human-readable format (HH:MM:SS AM/PM).
        """
        return datetime.fromtimestamp(unix_timestamp).strftime("%I:%M:%S %p")

    def _entry_point_to_github_url(self, entry_point: str, base_url = "https://github.com/LeonGuertler/TextArena") -> str:
        """
        Converts an entry_point string to a GitHub directory URL up to the module's directory.

        Args:
            entry_point (str): Entry point string in the format 'module.path:ClassName'.
            base_url (str): Base GitHub URL to prepend (e.g., 'https://github.com/LeonGuertler/TextArena').

        Returns:
            str: GitHub URL pointing to the module's directory.
        """
        # Extract module path from entry_point
        module_path, _ = entry_point.split(":")  # Split at colon to remove class name
        
        # Convert module path to a directory path
        dir_path = module_path.replace(".", "/")  # Replace dots with slashes
        
        # Remove the last segment (filename) to get the directory
        dir_path = "/".join(dir_path.split("/")[:-1])
        
        # Construct the full GitHub URL
        github_url = f"{base_url}/tree/main/{dir_path}"
        
        return github_url

    def _get_num_player_class(self):
        """
        Get the class type for the number of players in the game.
        """
        if len(self.player_names) == 1:
            return "Single Player"
        elif len(self.player_names) == 2:
            return "Two Player"
        else:
            return "Multiplayer"
        
    def _get_gameplay_instructions(self):
        """
        Get the gameplay instructions for the game.
        """
        ## use the entry point to get the readme file
        entry_point = self.env.entry_point
        module_path, _ = entry_point.split(":") 
        module_path = "/".join(module_path.split(".")[:-1])
        readme_path = Path(module_path) / "README.md"
        
        # Initialize content as None
        content = None
        
        if readme_path.exists():
            try:
                with open(readme_path, "r") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading README: {e}")
                return "No gameplay instructions available."

        if not content:
            return "No gameplay instructions available."

        ## extract the instructions from the readme file
        pattern = r'## Gameplay\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            text = match.group(1).strip()
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'^\s*-\s(.*)', r'<li>\1</li>', text, flags=re.MULTILINE)
            text = f"<ul>{text.strip()}</ul>"
            return text
        else:
            return "No gameplay instructions available."
            