import re
from typing import Dict, Optional, Tuple
from io import StringIO
import mss
import cv2
import numpy as np
import time

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textarena.core import Env, Message, Info, Rewards, RenderWrapper, State

__all__ = ["SimpleRenderWrapper"]


class SimpleRenderWrapper(RenderWrapper):
    """A general-purpose render wrapper that provides a formatted and enhanced rendering of any environment.

    This wrapper uses the 'rich' library to render the game state and logs in a more readable and visually appealing way.
    It is designed to be flexible and work with any game environment that provides a 'state' object with 'game_state' and 'logs'.
    """

    # Define a list of colors to assign to players
    PLAYER_COLORS = [
        "red",
        "green",
        "blue",
        "magenta",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
    ]

    # Define the color for [GAME] messages
    GAME_MESSAGE_COLOR = "yellow"
    GAME_KEYWORD_COLOR = "cyan"

    def __init__(
        self,
        env: Env,
        player_names: Optional[Dict[int, str]] = None,
        record_video: bool = False,
        video_path: str = "game_recording.mp4",
    ):
        """
        Initialize the SimpleRenderWrapper.

        Args:
            env (Env): The environment to wrap.
            player_names (Optional[Dict[int, str]]): Mapping from player IDs to agent names.
        """
        super().__init__(env)
        self.player_names = player_names
        self.record_video = record_video
        self.video_path = video_path

        self.console = Console()

        if record_video:
            self.frames = []
            self._capture_and_store_frame()

        # Assign colors to each player
        self.player_color_map = self._assign_player_colors()

    def _assign_player_colors(self) -> Dict[int, str]:
        """
        Assign a unique color to each player based on their player ID.

        Returns:
            Dict[int, str]: Mapping from player IDs to colors.
        """
        player_ids = list(self.player_names.keys())
        player_ids.sort()  # Ensure consistent color assignment

        color_map = {}
        num_colors = len(self.PLAYER_COLORS)
        for idx, player_id in enumerate(player_ids):
            color = self.PLAYER_COLORS[idx % num_colors]
            color_map[player_id] = color  # Map player_id to color

        return color_map

    def _process_logs(self, logs: list[Message], max_lines: int = None) -> Text:
        """
        Process logs by replacing player IDs with agent names, highlighting them, and color-coding [GAME] messages.

        Args:
            logs (list): List of log tuples (role, message).
            max_lines (int): Maximum number of lines to display in the console window.

        Returns:
            Text: Processed and colorized log Text object.
        """
        processed_lines = []
        for role, message in logs:
            message = str(message) # Ensure message is a string
            str_message = message.replace("[", "\[")
            if role != -1:
                # Player message
                player_name = self.player_names.get(role, f"Player {role}")
                color = self.player_color_map.get(role, "white")
                player_name_colored = f"[{color}]{player_name}[/{color}]"
                log = f"{player_name_colored}: {str_message}"
            else:
                # Game message
                log = f"[{self.GAME_MESSAGE_COLOR}][GAME]: {str_message}[/{self.GAME_MESSAGE_COLOR}]"

            # Create Text object from log
            log_text = Text.from_markup(log)
            # Wrap the log_text to the console width
            wrapped_lines = log_text.wrap(self.console, width=self.console.size.width)
            # Append wrapped lines to processed_lines
            processed_lines.extend(wrapped_lines)

            # Trim processed_lines if it exceeds max_lines
            if max_lines is not None and len(processed_lines) > max_lines:
                processed_lines = processed_lines[-max_lines:]

        # Create a single Text object with all processed lines
        log_text = Text('\n').join(processed_lines)

        return log_text

    def _capture_frame_with_mss(self):
        """Capture a frame from the specified monitor using mss."""
        with mss.mss() as sct:
            monitor = sct.monitors[-1]
            region = {
                "top": monitor["top"],
                "left": monitor["left"],
                "width": monitor["width"],
                "height": monitor["height"]
            }

            # Wait for the page to load
            time.sleep(0.5)

            try:
                screenshot = sct.grab(region)
                # Wait for the page to load
                time.sleep(0.2)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert to OpenCV format
                return frame
            except Exception as e:
                print(f"Error capturing screenshot: {e}")
                return None

    def _capture_and_store_frame(self):
        """Capture and store a frame in the buffer."""
        frame = self._capture_frame_with_mss()
        if frame is not None:
            self.frames.append(frame)

    def _save_video(self):
        """Save the recorded frames as a video file using OpenCV only."""
        if not self.frames:
            print("No frames captured, skipping video save.")
            return

        try:
            height, width = self.frames[0].shape[:2]
            
            # Use XVID codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.video_path, fourcc, 1, (width, height), True)

            for frame in self.frames:
                out.write(frame)

            out.release()
            print(f"\nVideo saved to {self.video_path} (XVID format)")

        except Exception as e:
            print(f"Error saving video: {e}")


    def _render_game_state(self, state: State) -> Panel:
        """
        Render the game state dynamically into multiple tables, only rendering keys specified in state.render_keys.
        Always include current turn count and max number of turns.

        Args:
            state (State): The game state object.

        Returns:
            Panel: A Rich Panel containing the formatted game state.
        """
        render_keys = self.env.terminal_render_keys
        if not isinstance(render_keys, list):
            self.console.print(
                "[bold red]'render_keys' in state should be a list. Skipping game state rendering.[/bold red]"
            )
            return Panel(
                "Invalid 'render_keys' configuration.",
                title="Game State",
                border_style="blue",
            )

        game_state = state.game_state

        # Filter game_state to include only the keys specified in render_keys
        filtered_game_state = {}
        for key in render_keys:
            if isinstance(key, str):
                filtered_game_state[key] = game_state.get(key)
            elif isinstance(key, list):  # flatten keys
                sub_dict = game_state
                try:
                    for sub_key in key:
                        sub_dict = sub_dict[sub_key]
                    filtered_game_state[".".join([str(k) for k in key])] = sub_dict
                except KeyError:
                    continue  # Skip if the key path doesn't exist

        # Always include current turn count and max number of turns
        filtered_game_state['current_turn'] = state.turn
        filtered_game_state['max_turns'] = state.max_turns if state.max_turns is not None else '∞'

        # Categorize filtered_game_state entries
        basic_entries = {}
        player_attributes = {}
        nested_player_attributes = {}

        for key, value in filtered_game_state.items():
            if isinstance(value, (str, int, float, bool)):
                basic_entries[key] = value
            elif isinstance(value, dict):
                # Check if the dict has int keys and basic type values
                if all(isinstance(k, int) for k in value.keys()) and all(
                    isinstance(v, (str, int, float, bool)) for v in value.values()
                ):
                    player_attributes[key] = value
                else:
                    # Assume it's a nested dict for player attributes
                    nested_player_attributes[key] = value
            elif isinstance(value, list):
                # Handle lists separately if needed
                basic_entries[key] = value
            else:
                basic_entries[key] = value  # Fallback to basic entries

        renderables = []

        # 1. Basic Key-Value Pairs Table
        if basic_entries:
            table_basic = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.MINIMAL_DOUBLE_HEAD,
            )
            table_basic.add_column("Key", style="cyan", no_wrap=True)
            table_basic.add_column("Value", style="green")

            for key, value in basic_entries.items():
                table_basic.add_row(f"[bold]{key}[/bold]", str(value))

            panel_basic = Panel(
                table_basic,
                title="Basic Information",
                border_style="blue",
                padding=(1, 1),
            )
            renderables.append(panel_basic)

        # 2. Player Attributes Table
        if player_attributes:
            table_players = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.MINIMAL_DOUBLE_HEAD,
            )
            table_players.add_column("Player", style="cyan", no_wrap=True)
            columns = list(player_attributes.keys())
            for attribute in columns:
                table_players.add_column(str(attribute), style="green")

            player_ids = set()
            for attribute_dict in player_attributes.values():
                player_ids.update(attribute_dict.keys())

            player_rows = {}
            for player_id in player_ids:
                player_name = self.player_names.get(
                    player_id, f"Player {player_id}"
                )
                color = self.player_color_map.get(player_id, "white")
                player_rows[player_id] = [f"[bold {color}]{player_name}[/bold {color}]"]

            for attribute in columns:
                for player_id in player_ids:
                    value = player_attributes[attribute].get(player_id, "")
                    player_rows[player_id].append(str(value))

            for player_id in player_ids:
                table_players.add_row(*player_rows[player_id])

            panel_players = Panel(
                table_players,
                title="Player Attributes",
                border_style="blue",
                padding=(1, 1),
            )
            renderables.append(panel_players)

        # 3. Nested Player Attributes Tables
        for key, value in nested_player_attributes.items():
            if isinstance(value, dict):
                table_nested = Table(
                    show_header=True,
                    header_style="bold magenta",
                    box=box.MINIMAL_DOUBLE_HEAD,
                )
                # Assuming nested dict has int keys (player IDs) and dict values
                table_nested.add_column("Player", style="cyan", no_wrap=True)
                # Dynamically add columns based on the nested keys
                nested_keys = set()
                for player_dict in value.values():
                    if isinstance(player_dict, dict):
                        nested_keys.update(player_dict.keys())
                nested_keys = sorted(nested_keys)
                for nk in nested_keys:
                    table_nested.add_column(str(nk), style="green")

                for player_id, attributes in value.items():
                    player_name = self.player_names.get(
                        player_id, f"Player {player_id}"
                    )
                    color = self.player_color_map.get(player_id, "white")
                    row = [f"[bold {color}]{player_name}[/bold {color}]"]
                    for nk in nested_keys:
                        row.append(str(attributes.get(nk, "")))
                    table_nested.add_row(*row)

                panel_nested = Panel(
                    table_nested, title=key, border_style="blue", padding=(1, 1)
                )
                renderables.append(panel_nested)
            elif isinstance(value, list):
                # Handle lists if needed
                list_content = "\n".join([str(item) for item in value])
                panel_list = Panel(
                    Text(list_content),
                    title=key,
                    border_style="blue",
                    padding=(1, 1),
                )
                renderables.append(panel_list)
            else:
                # Handle other types if necessary
                panel_other = Panel(
                    str(value), title=key, border_style="blue", padding=(1, 1)
                )
                renderables.append(panel_other)

        # Combine all renderables into Columns
        if renderables:
            # Arrange tables in Columns, wrapping as needed
            columns = Columns(renderables, expand=True)
            game_state_content = columns
            return Panel(
                game_state_content,
                title="Game State",
                border_style="blue",
                padding=(1, 1),
            )
        else:
            return Panel(
                "No game state information available.",
                title="Game State",
                border_style="blue",
            )

    def _get_rendered_height(self, renderable) -> int:
        """
        Helper function to get the height of a renderable.

        Args:
            renderable: The renderable object to measure.

        Returns:
            int: The number of lines the renderable will occupy.
        """
        temp_console = Console(
            file=StringIO(),
            width=self.console.size.width,
            record=True,
            legacy_windows=False,
            _environ={},
        )
        temp_console.print(renderable)
        rendered_output = temp_console.export_text()
        lines = rendered_output.split('\n')
        return len(lines)

    def _render(self):
        """
        Renders the current game state and logs using the 'rich' library.

        This method displays the game state and logs in a formatted layout using the 'rich' library.
        """
        env = self.env  # The wrapped environment
        # Access state
        if hasattr(env, "state") and isinstance(env.state, State):
            state = env.state
        else:
            # If the environment does not have a valid state, use a minimal render
            self.console.print(
                "[bold red]Environment does not have a valid 'state'. Using minimal render.[/bold red]"
            )
            if hasattr(env, "render") and callable(getattr(env, "render")):
                env.render()
            else:
                self.console.print(
                    "[bold yellow]No render method available in the environment.[/bold yellow]"
                )
            return

        # Render game state panel
        game_state_panel = self._render_game_state(state)
        game_state_height = self._get_rendered_height(game_state_panel)

        # Create an empty spacer panel to push everything down
        spacer_panel = Panel("", border_style="black", padding=(1, 1))  # Invisible panel

        # Create layout
        layout = Layout()
        # layout.split_column(
        #     Layout(spacer_panel, size=1),  # This pushes everything down by 1 line
        #     Layout(name='game_state', size=game_state_height),
        #     Layout(name='game_log')
        # )

        layout.split_column(
            Layout(spacer_panel, size=1),  # Keeps the spacer panel at the top
            Layout(name="main_row", ratio=11)  # Sum of 6 + 5 for game_state and game_log
        )

        layout["main_row"].split_row(
            Layout(name='game_state', ratio=6),
            Layout(name='game_log', ratio=5)
        )

        # Adjust for borders and padding in game_state panel
        available_log_height = self.console.size.height - (15) ## 5 is the padding

        if available_log_height < 5:
            available_log_height = 5  # Set a minimum height for logs

        # Process logs without truncation
        log_text = self._process_logs(state.logs, max_lines=available_log_height)

        log_panel = Panel(
            log_text, title="Game Log", border_style="green", padding=(1, 1)
        )

        # Update the layout sections
        layout['game_state'].update(game_state_panel)
        layout['game_log'].update(log_panel)

        # Render the layout
        self.console.print(layout)

    def reset(self, num_players: int, seed: Optional[int] = None) -> None:
        """Reset the environment and return the initial observation."""
        result = self.env.reset(num_players=num_players, seed=seed)
        self.state = self.env.state
        if self.player_names is None:
            self.player_names = {pid: f"Player {pid}" for pid in range(self.state.num_players)}
        self.player_names.update(self.state.role_mapping)
        self.game_over = False
        return result

    def step(self, action: str) -> Tuple[Optional[Rewards], bool, bool, Optional[Info]]:
        step_results = self.env.step(action=action)
        time.sleep(0.2)
        self._render()
        if self.record_video:
            self._capture_and_store_frame()
        else:
            time.sleep(0.2)
        return step_results
    
    def close(self):
        """Clean up resources"""
        if self.record_video:
            try:
                print("Saving video...")
                self._save_video()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        return self.env.close()