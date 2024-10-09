import re
from typing import Dict, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textarena.core import Env, Message, RenderWrapper, State

__all__ = ["PrettyRenderWrapper"]


class PrettyRenderWrapper(RenderWrapper):
    """A general-purpose render wrapper that provides a formatted and enhanced rendering of any environment.

    This wrapper uses the 'rich' library to render the game state and logs in a more readable and visually appealing way.
    It is designed to be flexible and work with any game environment that provides a 'game_state' dictionary and a 'logs' list.
    """

    # Define a list of colors to assign to players
    PLAYER_COLORS = [
        "red",
        "green",
        # "yellow",
        "blue",
        "magenta",
        # "cyan",
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
        agent_identifiers: Optional[Dict[int, str]] = None,
        max_log_lines: int = 8,  # TODO make this dynamic
    ):
        """
        Initialize the PrettyRenderWrapper.

        Args:
            env (Env): The environment to wrap.
            agent_identifiers (Optional[Dict[int, str]]): Mapping from player IDs to agent names.
            max_log_lines (int): Maximum number of log lines to display in the Game Log panel.
        """
        super().__init__(env)
        # Default agent identifiers if none provided
        if agent_identifiers is None:
            agent_identifiers = {}
        self.agent_identifiers = agent_identifiers

        self.console = Console()
        self.layout = Layout()

        # Initialize layout with two sections: game state and logs
        self.layout.split(Layout(name="upper", ratio=1), Layout(name="lower", ratio=1))

        self.max_log_lines = max_log_lines  # Maximum number of log lines to display

        # Assign colors to each player
        self.player_color_map = self._assign_player_colors()

    def _assign_player_colors(self) -> Dict[str, str]:
        """
        Assign a unique color to each player.

        Returns:
            Dict[str, str]: Mapping from player names to colors.
        """
        player_names = list(self.agent_identifiers.values())
        player_names.sort()  # Ensure consistent color assignment

        color_map = {}
        num_colors = len(self.PLAYER_COLORS)
        for idx, player in enumerate(player_names):
            color = self.PLAYER_COLORS[idx % num_colors]
            color_map[player] = color

        # If there are players without agent identifiers, assign default names and colors
        # This part depends on how players are represented in the game_state
        # For this example, we'll assume that all players have agent identifiers

        return color_map

    def _process_logs(self, logs: list[Message]) -> str:
        """
        Process logs by replacing player IDs with agent names, highlighting them, and color-coding [GAME] messages.
        Additionally, extract and process valid chess moves.

        Args:
            logs (list): List of log strings.

        Returns:
            str: Processed and colorized log string.
        """
        processed_logs = []

        # Create a pattern to exclude player names
        exclude_pattern = "|".join(
            [re.escape(name) for name in self.player_color_map.keys()] + ["GAME"]
        )
        move_pattern = re.compile(rf"\[(?!{exclude_pattern})(.*?)\]", re.IGNORECASE)

        for role, message in logs:
            if role in self.agent_identifiers:
                player_name = self.agent_identifiers[role]
                if player_name in self.player_color_map:
                    colour = self.player_color_map[player_name]
                    player_name = f"[{colour}]{player_name}[/{colour}]"
                log = f"{player_name}: {message}"
            else:  # role should be -1, i.e. a game message
                log = f"[{self.GAME_MESSAGE_COLOR}[GAME]: {message}[/{self.GAME_MESSAGE_COLOR}]"

            # Find all matches in square brackets except the player names
            matches = move_pattern.findall(log)
            for match in matches:
                log = log.replace(
                    f"[{match}]",
                    f"[[{self.GAME_KEYWORD_COLOR}]{match}[/{self.GAME_KEYWORD_COLOR}]]",
                )
            processed_logs.append(log)

        if len(processed_logs) > self.max_log_lines:
            truncated_count = len(processed_logs) - self.max_log_lines
            truncated_message = f"\n...[Truncated {truncated_count} more logs]"
            return "\n".join(
                [truncated_message] + processed_logs[-self.max_log_lines :]
            )
        else:
            return "\n".join(processed_logs)

    def _render_game_state(self, game_state: State) -> Panel:
        """
        Render the game state dynamically into multiple tables, only rendering keys specified in game_state["render"].

        Args:
            game_state (Dict[str, Any]): The game state dictionary.

        Returns:
            Panel: A Rich Panel containing the formatted game state.
        """
        render_keys = game_state.get("render", [])
        if not isinstance(render_keys, list):
            self.console.print(
                "[bold red]'render' key in game_state should be a list. Skipping game state rendering.[/bold red]"
            )
            return Panel(
                "Invalid 'render' configuration.",
                title="Game State",
                border_style="blue",
            )

        # Filter game_state to include only the keys specified in render_keys
        filtered_game_state = {}
        for key in render_keys:
            if isinstance(key, str):
                filtered_game_state[key] = game_state[key]
            elif isinstance(key, list):  # flatten keys
                sub_dict = game_state
                for sub_key in key:
                    sub_dict = sub_dict[sub_key]
                filtered_game_state[".".join([str(k) for k in key])] = sub_dict

        # Categorize filtered_game_state entries
        basic_entries = {}
        player_attributes = {}
        nested_player_attributes = {}

        for key, value in filtered_game_state.items():
            if isinstance(value, (str, int, float, bool)):
                basic_entries[key] = value
            elif isinstance(
                value, dict
            ):  # TODO: clean this code up... we should not be trying to infer this
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
            player_ids = set()
            columns = set()
            for attribute, assignment in player_attributes.items():
                table_players.add_column(str(attribute), style="green")
                for player_id in assignment:
                    player_ids.add(player_id)
                columns.add(attribute)

            player_rows = {}
            for player_id in player_ids:
                player_name = self.agent_identifiers.get(
                    player_id, f"Player {player_id}"
                )
                player_rows[player_id] = [f"[bold]{player_name}[/bold]"]

            for column in columns:
                for player_id, value in player_attributes[column].items():
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
                    player_name = self.agent_identifiers.get(
                        player_id, f"Player {player_id}"
                    )
                    row = [f"[bold]{player_name}[/bold]"]
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
                    Text(list_content), title=key, border_style="blue", padding=(1, 1)
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

    def _render_logs(self, logs: list) -> Panel:
        """
        Render the logs into a truncated and colorized text panel, with [GAME] messages color-coded.

        Args:
            logs (list): List of log strings.

        Returns:
            Panel: A Rich Panel containing the formatted and truncated logs.
        """
        processed_logs = self._process_logs(logs)
        # Use Text.from_markup to parse Rich markup if present
        log_text = Text.from_markup(processed_logs)

        return Panel(log_text, title="Game Log", border_style="green", padding=(1, 1))

    def render(self):
        """
        Renders the current game state and logs using the 'rich' library.

        This method displays the game state and logs in a formatted layout using the 'rich' library.
        """
        env = self.env  # The wrapped environment

        # Access game state
        if hasattr(env, "game_state") and isinstance(env.game_state, State):
            game_state = env.game_state
        else:
            # If the environment does not have a valid game_state, use a minimal render
            self.console.print(
                "[bold red]Environment does not have a valid 'game_state' dictionary. Using minimal render.[/bold red]"
            )
            if hasattr(env, "render") and callable(getattr(env, "render")):
                env.render()
            else:
                self.console.print(
                    "[bold yellow]No render method available in the environment.[/bold yellow]"
                )
            return

        # Access logs
        logs = game_state.get("logs", [])
        if not isinstance(logs, list):
            logs = []

        # Render game state and logs
        game_state_panel = self._render_game_state(game_state)
        log_panel = self._render_logs(logs)

        # Update the layout sections
        self.layout["upper"].update(game_state_panel)
        self.layout["lower"].update(log_panel)

        # Render the layout
        self.console.print(self.layout)
