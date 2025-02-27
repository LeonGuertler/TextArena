import curses
import threading
import time
import logging
from typing import Dict, Optional, Tuple, List
from textarena.core import Env, Message, RenderWrapper
import re


class CursesRenderWrapper(RenderWrapper):
    """A curses-based render wrapper with threaded rendering, crash handling, and toggleable logging."""

    # Color pair constants (unchanged)
    HEADER1_COLOR = 1
    KEY_COLOR = 2
    VALUE_COLOR = 3
    BORDER1_COLOR = 4
    GAME_MSG_COLOR = 5
    PLAYER1_COLOR = 6
    PLAYER2_COLOR = 7
    PLAYER3_COLOR = 8
    PLAYER4_COLOR = 9
    PLAYER5_COLOR = 10
    PLAYER6_COLOR = 11
    PLAYER7_COLOR = 12
    PLAYER8_COLOR = 13
    PLAYER9_COLOR = 14
    PLAYER10_COLOR = 15
    PLAYER11_COLOR = 16
    PLAYER12_COLOR = 17
    PLAYER13_COLOR = 18
    PLAYER14_COLOR = 19
    PLAYER15_COLOR = 20
    PLAYER16_COLOR = 21
    PLAYER17_COLOR = 22
    PLAYER18_COLOR = 23
    PLAYER19_COLOR = 24
    PLAYER20_COLOR = 25
    COLUMN_HEADER_COLOR = 26
    ROW_NAME_COLOR = 27
    HEADER2_COLOR = 28
    BORDER2_COLOR = 29

    # Custom color definitions (unchanged)
    _COLORS = {
        "Dark Slate Green (TextArena Background)": (10, 16, 122, 122),
        "Deep Indigo": (11, 51, 443, 412),
        "Teal Gray": (12, 74, 231, 184),
        "Dark Slate": (13, 12, 55, 39),
        "Vivid Plum": (14, 655, 258, 902),
        "Sky Cyan": (15, 149, 451, 902),
        "Vivid Orange Red": (16, 902, 525, 94),
        "Pure Red": (17, 902, 0, 0),
        "Lime Green": (18, 0, 784, 0),
        "Magenta": (19, 784, 0, 784),
        "Cyan": (20, 0, 784, 784),
        "Soft Pink": (21, 902, 588, 706),
        "Yellow Green": (22, 588, 902, 0),
        "Violet": (23, 588, 0, 902),
        "Amber Orange": (24, 902, 392, 0),
        "Blue Lilac": (25, 392, 627, 902),
        "Mint Green": (26, 392, 902, 392),
        "Salmon Pink": (27, 902, 392, 392),
        "Teal": (28, 0, 588, 588),
        "Golden Yellow": (29, 784, 627, 0),
        "Plum Purple": (30, 627, 0, 902),
        "Olive Green": (31, 627, 627, 0),
        "Peach": (32, 902, 510, 392),
        "Slate Blue": (33, 392, 0, 627),
        "Turquoise": (34, 0, 902, 627),
        "Ivory": (35, 863, 847, 824),
        "Steel Blue": (36, 443, 551, 588),
        "Gray": (37, 384, 384, 384),
        "White": (38, 902, 902, 902),
        "Black": (39, 0, 0, 0),
        "Green": (40, 0, 902, 0),
        "Blue": (41, 0, 0, 902),
        "Yellow": (42, 902, 902, 0),
        "Cyan Standard": (43, 0, 902, 902),
        "Magenta Standard": (44, 902, 0, 902),
        "Orange": (45, 902, 583, 0),         
        "Purple": (46, 452, 0, 452),         
        "Pink": (47, 902, 679, 717),         
        "Brown": (48, 583, 149, 149),       
        "Lime": (49, 451, 902, 0),          
        "Indigo": (50, 265, 0, 459),        
        "Violet Standard": (51, 842, 460, 842),
        "Light Cream": (52, 952, 952, 840),  
        "Forest Green": (53, 472, 600, 344) 
    }

    def __init__(self, env: Env, player_names: Optional[Dict[int, str]] = None, enable_logging: bool = False):
        """Initialize the render wrapper with environment, player names, and logging toggle."""
        super().__init__(env)
        self.player_names = player_names  # Defer default setup to reset
        self.player_color_map = {
            0: self.PLAYER1_COLOR, 1: self.PLAYER2_COLOR, 2: self.PLAYER3_COLOR, 3: self.PLAYER4_COLOR,
            4: self.PLAYER5_COLOR, 5: self.PLAYER6_COLOR, 6: self.PLAYER7_COLOR, 7: self.PLAYER8_COLOR,
            8: self.PLAYER9_COLOR, 9: self.PLAYER10_COLOR, 10: self.PLAYER11_COLOR, 11: self.PLAYER12_COLOR,
            12: self.PLAYER13_COLOR, 13: self.PLAYER14_COLOR, 14: self.PLAYER15_COLOR, 15: self.PLAYER16_COLOR,
            16: self.PLAYER17_COLOR, 17: self.PLAYER18_COLOR, 18: self.PLAYER19_COLOR, 19: self.PLAYER20_COLOR,
        }

        # Initialize next_pair_id after predefined color pairs
        self._next_pair_id = 30  # Start after predefined pairs (1-27 from _setup_colors)
        self._pair_map = {}  # (fg_id, bg_id) -> pair_id

        # Logging setup
        self.enable_logging = enable_logging
        if self.enable_logging:
            logging.basicConfig(filename='render_crash.log', level=logging.DEBUG, 
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.disable(logging.CRITICAL)
        try:
            if hasattr(self.env, "set_player_names") and self.player_names is not None:
                self.env.set_player_names(self.player_names)
        except AttributeError:
            pass
        
        self.stdscr = None
        self.game_win = None
        self.log_win = None
        self.game_over = False
        self.initialized = False
        self._stop_render = threading.Event()
        self._render_thread = None
        self._render_error = None
        self._render_lock = threading.Lock()

    def toggle_logging(self, enable: bool) -> None:
        """Toggle logging on or off during runtime."""
        self.enable_logging = enable
        if enable:
            logging.basicConfig(filename='render_crash.log', level=logging.DEBUG, 
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.disable(logging.CRITICAL)

    def _initialize_curses(self, stdscr) -> None:
        """Set up curses environment and colors within the render thread."""
        if self.enable_logging:
            logging.debug("Initializing curses in render thread")
        with self._render_lock:
            self.stdscr = stdscr
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            curses.curs_set(0)
            self._setup_colors()
            try:
                self.stdscr.bkgd(' ', curses.color_pair(self.KEY_COLOR))
            except curses.error:
                if self.enable_logging:
                    logging.warning("Failed to set background in _initialize_curses")
            self.stdscr.clear()
            self.stdscr.refresh()
            self.initialized = True
        if self.enable_logging:
            logging.debug("Curses initialized successfully")

    def _setup_colors(self):
        """Configure color pairs for rendering, including support for dynamic fg/bg pairs."""
        try:
            curses.start_color()
            if curses.can_change_color():
                # Initialize all custom colors from _COLORS dictionary
                for name, (id_, r, g, b) in self._COLORS.items():
                    curses.init_color(id_, r, g, b)

                # Step 1: Initialize player color pairs (PLAYER1_COLOR to PLAYER20_COLOR, pairs 6-25)
                for i in range(20):
                    player_color_id = 15 + i  # player0=15 to player19=34
                    pair_id = self.PLAYER1_COLOR + i  # 6 to 25
                    curses.init_pair(pair_id, player_color_id, 10)

                # Step 2: Initialize non-player color pairs (Entity id, fg_id, bg_id)
                curses.init_pair(self.HEADER1_COLOR, 53, 10)
                curses.init_pair(self.BORDER1_COLOR, 53, 10)  
                curses.init_pair(self.HEADER2_COLOR, curses.COLOR_WHITE, 10)
                curses.init_pair(self.BORDER2_COLOR, curses.COLOR_WHITE, 10)
                curses.init_pair(self.KEY_COLOR, curses.COLOR_WHITE, 10)
                curses.init_pair(self.VALUE_COLOR, 35, 10)  
                curses.init_pair(self.GAME_MSG_COLOR, 14, 10)  
                curses.init_pair(self.COLUMN_HEADER_COLOR, 44, 10)  
                curses.init_pair(self.ROW_NAME_COLOR, 43, 10) 

            else:
                # Fallback for non-custom terminals
                # Step 1: Initialize player color pairs
                for i in range(20):
                    pair_id = self.PLAYER1_COLOR + i
                    curses.init_pair(pair_id, [curses.COLOR_BLUE, curses.COLOR_YELLOW, curses.COLOR_RED,
                                            curses.COLOR_GREEN, curses.COLOR_MAGENTA, curses.COLOR_CYAN,
                                            curses.COLOR_WHITE, curses.COLOR_GREEN, curses.COLOR_MAGENTA,
                                            curses.COLOR_YELLOW, curses.COLOR_CYAN, curses.COLOR_GREEN,
                                            curses.COLOR_RED, curses.COLOR_CYAN, curses.COLOR_YELLOW,
                                            curses.COLOR_MAGENTA, curses.COLOR_GREEN, curses.COLOR_RED,
                                            curses.COLOR_BLUE, curses.COLOR_CYAN][i], curses.COLOR_BLACK)

                # Step 2: Initialize non-player color pairs
                curses.init_pair(self.HEADER1_COLOR, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(self.KEY_COLOR, curses.COLOR_WHITE, curses.COLOR_BLACK)
                curses.init_pair(self.VALUE_COLOR, curses.COLOR_WHITE, curses.COLOR_BLACK)
                curses.init_pair(self.BORDER1_COLOR, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(self.GAME_MSG_COLOR, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
                curses.init_pair(self.COLUMN_HEADER_COLOR, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
                curses.init_pair(self.ROW_NAME_COLOR, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(self.HEADER2_COLOR, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(self.BORDER2_COLOR, curses.COLOR_RED, curses.COLOR_BLACK)

        except Exception as e:
            if self.enable_logging:
                logging.error(f"Failed to set up colors: {str(e)}")
            raise

    def _get_or_create_pair(self, fg_id, bg_id):
        """Get or create a color pair for the given foreground and background IDs."""
        pair_key = (fg_id, bg_id)
        if pair_key not in self._pair_map:
            if self._next_pair_id >= curses.COLOR_PAIRS:
                if self.enable_logging:
                    logging.warning("Max color pairs reached, reusing VALUE_COLOR")
                return self.VALUE_COLOR
            # Validate colors against _COLORS
            valid_colors = {id_ for name, (id_, _, _, _) in self._COLORS.items()}
            fg_id = fg_id if fg_id in valid_colors else 35  # Default to VALUE_COLOR fg
            bg_id = bg_id if bg_id in valid_colors else 10  # Default to background
            curses.init_pair(self._next_pair_id, fg_id, bg_id)
            self._pair_map[pair_key] = self._next_pair_id
            self._next_pair_id += 1
            if self.enable_logging:
                logging.debug(f"Created new pair {self._next_pair_id-1} for fg={fg_id}, bg={bg_id}")
        return self._pair_map[pair_key]

    def _cleanup_curses(self) -> None:
        """Safely terminate curses environment."""
        if self.enable_logging:
            logging.debug("Cleaning up curses")
        if self.initialized:
            try:
                with self._render_lock:
                    curses.nocbreak()
                    if self.stdscr:
                        self.stdscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                if self.enable_logging:
                    logging.debug("Curses cleaned up successfully")
            except Exception as e:
                if self.enable_logging:
                    logging.error(f"Failed to clean up curses: {str(e)}")
            finally:
                self.initialized = False

    def _start_render_thread(self):
        """Start the rendering thread."""
        if self.enable_logging:
            logging.debug("Starting render thread")
        if not self._render_thread:
            self._stop_render.clear()
            self._render_error = None
            self._render_thread = threading.Thread(target=self._run_render_loop)
            self._render_thread.daemon = True
            self._render_thread.start()
            time.sleep(0.1)
            if self.enable_logging:
                logging.debug("Render thread started")

    def _run_render_loop(self):
        """Run the curses rendering loop in a separate thread."""
        if self.enable_logging:
            logging.debug("Entering render loop")
        try:
            curses.wrapper(self._render_main_loop)
        except Exception as e:
            self._render_error = str(e)
            if self.enable_logging:
                logging.error(f"Render thread crashed: {str(e)}")
            print(f"Render thread crashed: {str(e)}")

    def _render_main_loop(self, stdscr):
        """Main rendering loop for curses with keypress prompt on game over."""
        self._initialize_curses(stdscr)
        if self.enable_logging:
            logging.debug("Render main loop started")
        while not self._stop_render.is_set():
            with self._render_lock:
                if not self.initialized:
                    if self.enable_logging:
                        logging.warning("Render loop found uninitialized curses")
                    break
                try:
                    if self.enable_logging:
                        logging.debug("Updating display in render loop")
                    self.update_display()
                    if self.game_over:
                        self.stdscr.timeout(-1)
                        if self.enable_logging:
                            logging.debug("Game over, waiting for keypress")
                        self.stdscr.getch()
                        self._stop_render.set()
                        if self.enable_logging:
                            logging.debug("Key pressed, stopping render loop")
                    else:
                        self.stdscr.timeout(100)
                        key = self.stdscr.getch()
                        if key in (ord("q"), 27):
                            if self.enable_logging:
                                logging.debug("Exit key pressed, stopping render loop")
                            self._stop_render.set()
                except Exception as e:
                    height, width = self.stdscr.getmaxyx()
                    error_msg = f"Render error: {str(e)[:width-20]}"
                    if self.enable_logging:
                        logging.error(f"Render loop exception: {str(e)}")
                    if height > 1 and width > len(error_msg):
                        try:
                            self.stdscr.addstr(height-1, 0, error_msg, curses.color_pair(self.PLAYER2_COLOR) | curses.A_BOLD)
                            self.stdscr.refresh()
                        except curses.error:
                            if self.enable_logging:
                                logging.warning("Failed to display error message on screen")
                    time.sleep(1)
            if not self.game_over:
                time.sleep(0.1)
        if self.enable_logging:
            logging.debug("Render main loop exited")

    def _draw_box(self, win, header_color: str, border_color: str, y: int, x: int, height: int, width: int, title: str = None) -> None:
        """Draw a box with optional title."""
        if self.enable_logging:
            logging.debug(f"Drawing box at {y},{x} with height={height}, width={width}, title={title}")
        try:
            win_height, win_width = win.getmaxyx()
            for i in range(max(0, y), min(y + height, win_height)):
                for j in range(max(0, x), min(x + width, win_width)):
                    try:
                        win.addstr(i, j, " ", curses.color_pair(border_color))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw background at {i},{j}")
            
            win.attrset(curses.color_pair(border_color))
            if y < win_height and x < win_width and y + height - 1 < win_height and x + width - 1 < win_width:
                if y < win_height and x < win_width:
                    try:
                        win.addch(y, x, curses.ACS_ULCORNER)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw UL corner at {y},{x}")
                if y < win_height and x + width - 1 < win_width:
                    try:
                        win.addch(y, x + width - 1, curses.ACS_URCORNER)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw UR corner at {y},{x + width - 1}")
                if y + height - 1 < win_height and x < win_width:
                    try:
                        win.addch(y + height - 1, x, curses.ACS_LLCORNER)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw LL corner at {y + height - 1},{x}")
                if y + height - 1 < win_height and x + width - 1 < win_width:
                    try:
                        win.addch(y + height - 1, x + width - 1, curses.ACS_LRCORNER)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw LR corner at {y + height - 1},{x + width - 1}")
                
                for i in range(x + 1, min(x + width - 1, win_width)):
                    if y < win_height:
                        try:
                            win.addch(y, i, curses.ACS_HLINE)
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to draw top HLINE at {y},{i}")
                    if y + height - 1 < win_height:
                        try:
                            win.addch(y + height - 1, i, curses.ACS_HLINE)
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to draw bottom HLINE at {y + height - 1},{i}")
                
                for i in range(y + 1, min(y + height - 1, win_height)):
                    if x < win_width:
                        try:
                            win.addch(i, x, curses.ACS_VLINE)
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to draw left VLINE at {i},{x}")
                    if x + width - 1 < win_width:
                        try:
                            win.addch(i, x + width - 1, curses.ACS_VLINE)
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to draw right VLINE at {i},{x + width - 1}")

                if title and y < win_height and x + 2 < win_width and len(f" {title} ") < win_width - x - 4:
                    try:
                        win.addstr(y, x + 2, f" {title} ", curses.color_pair(header_color))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw title '{title}' at {y},{x + 2}")
        except curses.error as e:
            if self.enable_logging:
                logging.error(f"Error drawing box: {str(e)}")

    def _categorize_game_state(self, render_keys: List) -> Dict[str, any]:
        """Organize game state."""
        if self.enable_logging:
            logging.debug("Categorizing game state")
        try:
            game_state = self.state.game_state
            if self.enable_logging:
                logging.debug(f"Game state contents: {game_state}")
            filtered_state = {
                key if isinstance(key, str) else ".".join(map(str, key)): (
                    game_state.get(key) if isinstance(key, str) else self._get_nested_value(game_state, key)
                )
                for key in render_keys
            }
            if self.enable_logging:
                logging.debug(f"Terminal render keys requested: {render_keys}")
                logging.debug(f"Filtered state from render keys: {filtered_state}")
            filtered_state.update({"current_turn": self.state.turn, "max_turns": self.state.max_turns or "∞"})
            if self.enable_logging:
                logging.debug(f"Filtered state after adding turn info: {filtered_state}")

            categories = {"board": None, "basic": {}, "players": {}, "lists": {}, "nested_tables": {}}
            
            board_keys = [key for key in filtered_state if isinstance(key, str) and 
                        any(term in key.lower() for term in ["board", "main", "rendered"])]
            if self.enable_logging:
                logging.debug(f"Identified board keys: {board_keys}")
            if board_keys:
                board_key = "board" if "board" in board_keys else board_keys[0]
                categories["board"] = filtered_state.pop(board_key, None)
                if self.enable_logging:
                    logging.debug(f"Assigned to board category: {board_key} = {categories['board']}")
            
            for key, value in filtered_state.items():
                if self.enable_logging:
                    logging.debug(f"Categorizing key: {key}, value: {value}, type: {type(value)}")
                if isinstance(value, (str, int, float, bool)):
                    categories["basic"][key] = value
                elif isinstance(value, dict) and all(isinstance(k, int) for k in value):
                    if value and all(isinstance(v, dict) for v in value.values()):
                        categories["nested_tables"][key] = value
                    else:
                        categories["players"][key] = value
                else:
                    categories["basic"][key] = str(value)
            
            if self.enable_logging:
                logging.debug(f"Final categories: {categories}")
                logging.debug("Game state categorized successfully")
            return categories
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error categorizing game state: {str(e)}")
            return {"board": None, "basic": {}, "players": {}, "lists": {}, "nested_tables": {}}

    def _get_nested_value(self, data: Dict, keys: List) -> any:
        """Retrieve a nested value."""
        try:
            for key in keys:
                data = data[key]
            return data
        except (KeyError, TypeError) as e:
            if self.enable_logging:
                logging.warning(f"Failed to get nested value: {str(e)}")
            return None

    def _render_board(self, win, board, y, x, width):
        """Render the game board with [fg=X,bg=Y]text[/color] tags, centered as a block while preserving original whitespace."""
        if not board:
            return y
        lines = board.split("\n")
        win_height, win_width = win.getmaxyx()
        height = min(len(lines) + 4, win_height - y - 1)
        width = min(width, win_width - x - 1)
        if height <= 4 or width <= 2:
            if self.enable_logging:
                logging.debug("Board skipped: insufficient space")
            return y
        self._draw_box(win, self.HEADER2_COLOR, self.BORDER2_COLOR, y, x, height, width, "Game Board")

        color_pattern = re.compile(r'\[fg=(\d+)(?:,bg=(\d+))?\](.*?)\[/color\]')
        valid_colors = {id_ for name, (id_, _, _, _) in self._COLORS.items()}

        # Find the longest line length (including original whitespace) for centering
        max_line_length = 0
        for line in lines[:height - 4]:
            if line:  # Check if line is not empty
                segments = []
                last_end = 0
                for match in color_pattern.finditer(line):
                    if match.start() > last_end:
                        segments.append(line[last_end:match.start()])
                    segments.append(match.group(3))  # Extract just the text inside color tags
                    last_end = match.end()
                if last_end < len(line):
                    segments.append(line[last_end:])
                total_len = sum(len(text) for text in segments)
                max_line_length = max(max_line_length, total_len)

        for i, line in enumerate(lines[:height - 4]):
            if not line:  # Skip empty lines
                continue
            segments = []
            last_end = 0
            # Preserve the original line, including whitespace, without stripping
            for match in color_pattern.finditer(line):
                if match.start() > last_end:
                    segments.append((line[last_end:match.start()], self.VALUE_COLOR))
                fg_id = int(match.group(1))
                bg_id = int(match.group(2)) if match.group(2) else 10  # Default bg to 10
                text = match.group(3)
                fg_id = fg_id if fg_id in valid_colors else 35  # Default to VALUE_COLOR fg
                bg_id = bg_id if bg_id in valid_colors else 10  # Default to background
                color_pair = self._get_or_create_pair(fg_id, bg_id)
                segments.append((text, color_pair))
                last_end = match.end()
            if last_end < len(line):
                segments.append((line[last_end:], self.VALUE_COLOR))

            # Calculate total length of the line (including original whitespace)
            total_len = sum(len(text) for text, _ in segments)
            # Center the block as a whole, using the longest line length
            padding = (width - max_line_length - 2) // 2 if max_line_length < width - 4 else 0
            render_x = max(x + 2, x + padding)

            for text, color in segments:
                if y + 2 + i < win_height and render_x + len(text) <= win_width:
                    try:
                        win.addstr(y + 2 + i, render_x, text, curses.color_pair(color))
                        render_x += len(text)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render segment '{text}' at {y + 2 + i},{render_x}")
                else:
                    break

        return y + height
    
    def _render_basic(self, win, data: Dict, y: int, x: int, width: int) -> int:
        """Render simple key-value pairs with improved spacing around the vertical line."""
        if self.enable_logging:
            logging.debug(f"Rendering basic info at {y},{x} with width={width}")
        try:
            if not data:
                return y
            win_height, win_width = win.getmaxyx()
            # Height includes 5 for box + data rows
            height = min(len(data) + 5, win_height - y - 1)
            width = min(width, win_width - x - 1)
            if height <= 5 or width <= 2:
                if self.enable_logging:
                    logging.debug("Basic info skipped: insufficient space")
                return y
            self._draw_box(win, self.HEADER2_COLOR, self.BORDER2_COLOR, y, x, height, width, "Basic Information")
            max_key_len = min(max(map(len, map(str, list(data.keys())[:height-5]))), width // 2 - 4)
            if max_key_len < 1:
                return y
            
            # Calculate position for the vertical line with better spacing
            vertical_line_pos = x + max_key_len + 3  # Position with extra space after key column
            
            # Draw column headers with proper padding
            if y + 2 < win_height and x + 2 < win_width:
                try:
                    win.addstr(y + 2, x + 2, "Key".ljust(max_key_len), curses.color_pair(self.COLUMN_HEADER_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to render 'Key' header at {y + 2},{x + 2}")
            if y + 2 < win_height and vertical_line_pos + 2 < win_width:
                try:
                    win.addstr(y + 2, vertical_line_pos + 2, "Value", curses.color_pair(self.COLUMN_HEADER_COLOR))  # Extra space after vertical line
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to render 'Value' header at {y + 2},{vertical_line_pos + 2}")
            
            # Draw horizontal line below headers
            if y + 3 < win_height and x + 2 < win_width and x + width - 2 < win_width:
                try:
                    win.hline(y + 3, x + 2, curses.ACS_HLINE, width - 4, curses.color_pair(self.VALUE_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to draw horizontal line at {y + 3},{x + 2}")
            
            # Draw vertical line between columns (extend to column headers)
            num_data_rows = len(data) if len(data) < height - 5 else height - 5
            total_rows = num_data_rows + 2  # Include header row and data rows
            if total_rows > 0 and vertical_line_pos < win_width:
                for i in range(y + 2, y + 2 + total_rows):  # Start at headers, end before bottom border
                    if i < win_height:
                        try:
                            win.vline(i, vertical_line_pos, curses.ACS_VLINE, 1, curses.color_pair(self.VALUE_COLOR))
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to draw vertical line at {i},{vertical_line_pos}")
            
            # Render rows (key-value pairs) with proper spacing
            for i, (k, v) in enumerate(list(data.items())[:height-5], start=y + 4):
                truncated_k = str(k)[:max_key_len].ljust(max_key_len)
                truncated_v = str(v)[:width - max_key_len - 8]  # Adjusted for extra spacing
                if i < win_height and x + 2 < win_width:
                    try:
                        win.addstr(i, x + 2, truncated_k, curses.color_pair(self.ROW_NAME_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render key '{truncated_k}' at {i},{x + 2}")
                if i < win_height and vertical_line_pos + 2 + len(truncated_v) <= win_width:  # Extra space after vertical line
                    try:
                        win.addstr(i, vertical_line_pos + 2, truncated_v, curses.color_pair(self.VALUE_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render value '{truncated_v}' at {i},{vertical_line_pos + 2}")
            return y + height
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error rendering basic info: {str(e)}")
            return y
        
    def _render_players(self, win, data: Dict, y: int, x: int, width: int) -> int:
        """Render player-specific attributes with multiple columns and proper vertical lines."""
        if self.enable_logging:
            logging.debug(f"Rendering players at {y},{x} with width={width}")
        try:
            if not data:
                return y
            win_height, win_width = win.getmaxyx()
            player_ids = []
            seen = set()
            for attr_dict in data.values():
                for pid in attr_dict.keys():
                    if pid not in seen:
                        player_ids.append(pid)
                        seen.add(pid)
            attrs = list(data.keys())
            
            # Calculate column widths
            max_name_width = min(max(len(self.player_names.get(pid, f"Player {pid}")) for pid in player_ids), width // (len(attrs) + 1) - 2)
            if max_name_width < 1:
                return y
                
            # Calculate width for each attribute column
            attr_widths = {}
            for a in attrs:
                max_attr_value_width = max(len(str(a)), max(len(str(data[a].get(pid, ""))) for pid in player_ids))
                attr_widths[a] = min(max_attr_value_width, (width - max_name_width - 4) // len(attrs) - 2)
                
            # Height includes 5 for box + data rows
            height = min(len(player_ids) + 5, win_height - y - 1)
            width = min(width, win_width - x - 1)
            if height <= 5 or width <= 2:
                if self.enable_logging:
                    logging.debug("Players skipped: insufficient space")
                return y
                
            self._draw_box(win, self.HEADER2_COLOR, self.BORDER2_COLOR, y, x, height, width, "Player Attributes")
            
            # Calculate starting positions for each column
            column_positions = [x + 2]  # Start of "Player" column
            current_x = x + 2 + max_name_width + 3  # Position after "Player" column with extra spacing
            
            for attr in attrs:
                column_positions.append(current_x)
                current_x += attr_widths[attr] + 3  # Add extra spacing for next column
            
            # Draw column headers
            if y + 2 < win_height and column_positions[0] < win_width:
                try:
                    win.addstr(y + 2, column_positions[0], "Player".ljust(max_name_width), 
                            curses.color_pair(self.COLUMN_HEADER_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to render 'Player' header at {y + 2},{column_positions[0]}")
            
            for i, attr in enumerate(attrs):
                if i < len(column_positions) - 1 and y + 2 < win_height and column_positions[i + 1] < win_width:
                    try:
                        win.addstr(y + 2, column_positions[i + 1] + 1,  # Add extra space after vertical line
                                attr.ljust(attr_widths[attr]), 
                                curses.color_pair(self.COLUMN_HEADER_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render column header '{attr}' at {y + 2},{column_positions[i + 1] + 1}")
            
            # Draw horizontal line below headers
            if y + 3 < win_height and x + 2 < win_width and x + width - 2 < win_width:
                try:
                    win.hline(y + 3, x + 2, curses.ACS_HLINE, width - 4, curses.color_pair(self.VALUE_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to draw horizontal line at {y + 3},{x + 2}")
            
            # Draw vertical lines between columns
            vertical_positions = []
            for i in range(1, len(column_positions)):
                vertical_positions.append(column_positions[i] - 1)  # Position line 1 space before next column
            
            num_data_rows = min(len(player_ids), height - 5)
            total_rows = num_data_rows + 2  # Include header row and data rows
            
            if total_rows > 0 and vertical_positions:
                for pos in vertical_positions:
                    if pos < win_width:
                        for i in range(y + 2, y + 2 + total_rows):  # Start at headers, end before bottom border
                            if i < win_height:
                                try:
                                    win.vline(i, pos, curses.ACS_VLINE, 1, curses.color_pair(self.VALUE_COLOR))
                                except curses.error:
                                    if self.enable_logging:
                                        logging.warning(f"Failed to draw vertical line at {i},{pos}")
            
            # Render player data rows
            for i, pid in enumerate(player_ids[:height-5], start=y + 4):
                name = self.player_names.get(pid, f"Player {pid}")[:max_name_width].ljust(max_name_width)
                if i < win_height and column_positions[0] + len(name) <= win_width:
                    try:
                        win.addstr(i, column_positions[0], name, curses.color_pair(self.player_color_map.get(pid, 0)))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render player name '{name}' at {i},{column_positions[0]}")
                
                for j, attr in enumerate(attrs):
                    if j < len(column_positions) - 1:
                        value = str(data[attr].get(pid, ""))[:attr_widths[attr]].ljust(attr_widths[attr])
                        if i < win_height and column_positions[j + 1] + 1 + len(value) <= win_width:
                            try:
                                win.addstr(i, column_positions[j + 1] + 1, value, curses.color_pair(self.VALUE_COLOR))  # Add extra space after vertical line
                            except curses.error:
                                if self.enable_logging:
                                    logging.warning(f"Failed to render value '{value}' at {i},{column_positions[j + 1] + 1}")
            
            return y + height
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error rendering players: {str(e)}")
            return y

    def _render_nested_tables(self, win, data: Dict, y: int, x: int, width: int) -> int:
        """Render nested dictionary data with dynamic column alignment and extra spacing."""
        if self.enable_logging:
            logging.debug(f"Rendering nested tables at {y},{x} with width={width}")
        try:
            if not data:
                return y
            win_height, win_width = win.getmaxyx()
            next_y = y
            for key, nested_dict in data.items():
                outer_ids = list(nested_dict.keys())
                inner_keys = []
                seen = set()
                for inner_dict in nested_dict.values():
                    for k in inner_dict.keys():
                        if k not in seen:
                            inner_keys.append(k)
                            seen.add(k)
                if not inner_keys or not outer_ids:
                    continue
                
                # Calculate maximum widths for each column (ID and inner keys)
                max_id_len = min(max(len(str(pid)) for pid in outer_ids), width // (len(inner_keys) + 1) - 2)
                if max_id_len < 1:
                    continue
                key_widths = {}
                for inner_key in inner_keys:
                    max_width = max(len(str(inner_key)), max(len(str(nested_dict.get(pid, {}).get(inner_key, ""))) 
                                    for pid in outer_ids if inner_key in nested_dict.get(pid, {})))
                    key_widths[inner_key] = min(max_width, width // (len(inner_keys) + 1) - 2)
                
                # Total number of columns (ID + inner keys)
                total_columns = 1 + len(inner_keys)
                # Height includes 5 for box + data rows
                height = min(len(outer_ids) + 5, win_height - next_y - 1)
                width = min(width, win_width - x - 1)
                if height <= 5 or width <= 2:
                    if self.enable_logging:
                        logging.debug(f"Nested table '{key}' skipped: insufficient space")
                    continue
                
                self._draw_box(win, self.HEADER2_COLOR, self.BORDER2_COLOR, next_y, x, height, width, key)
                
                # Calculate starting positions for each column with dynamic spacing and extra space after vertical lines
                column_positions = [x + 2]  # Start of "ID" column
                current_x = x + 2 + max_id_len + 2  # Start after "ID" with initial spacing (2 for gap)
                for inner_key in inner_keys[:width // (max_id_len + 1) - 1]:  # Limit based on available width
                    # Add extra space (4) for gap, ensuring 1 space between column and left vertical line
                    current_x += key_widths[inner_key] + 4  # Increase from +3 to +4 for extra space between column and line
                    column_positions.append(current_x)
                
                # Draw column headers with dynamic positioning and extra space
                if next_y + 2 < win_height:
                    try:
                        win.addstr(next_y + 2, column_positions[0], "ID".ljust(max_id_len), 
                                curses.color_pair(self.COLUMN_HEADER_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render 'ID' header at {next_y + 2},{column_positions[0]}")
                    for i, inner_key in enumerate(inner_keys[:len(column_positions) - 1]):
                        if column_positions[i + 1] < win_width:
                            try:
                                win.addstr(next_y + 2, column_positions[i + 1], 
                                        inner_key.ljust(key_widths[inner_key]), 
                                        curses.color_pair(self.COLUMN_HEADER_COLOR))
                            except curses.error:
                                if self.enable_logging:
                                    logging.warning(f"Failed to render inner key '{inner_key}' at {next_y + 2},{column_positions[i + 1]}")
                
                # Draw horizontal line below headers
                if next_y + 3 < win_height and x + 2 < win_width and x + width - 2 < win_width:
                    try:
                        win.hline(next_y + 3, x + 2, curses.ACS_HLINE, width - 4, curses.color_pair(self.VALUE_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to draw horizontal line at {next_y + 3},{x + 2}")
                
                # Draw N-1 vertical lines between N columns, using column positions
                vertical_positions = []
                for i in range(1, len(column_positions)):  # Draw N-1 lines for N columns
                    vertical_positions.append(column_positions[i] - 2)  # Position line 2 spaces before next column (for 1-space gap)
                
                num_data_rows = len(outer_ids) if len(outer_ids) < height - 5 else height - 5
                total_rows = num_data_rows + 2  # Include header row and data rows
                if total_rows > 0 and vertical_positions:
                    for pos in vertical_positions:
                        if pos < win_width:
                            for i in range(next_y + 2, next_y + 2 + total_rows):  # Start at headers, end before bottom border
                                if i < win_height:
                                    try:
                                        win.vline(i, pos, curses.ACS_VLINE, 1, curses.color_pair(self.VALUE_COLOR))
                                    except curses.error:
                                        if self.enable_logging:
                                            logging.warning(f"Failed to draw vertical line at {i},{pos}")
                
                # Render rows (ID and inner key values) with dynamic positioning and extra space
                for i, pid in enumerate(outer_ids[:height-5], start=next_y + 4):
                    current_x = column_positions[0]
                    id_str = str(pid)[:max_id_len].ljust(max_id_len)
                    if i < win_height and current_x + len(id_str) <= win_width:
                        try:
                            win.addstr(i, current_x, id_str, curses.color_pair(self.ROW_NAME_COLOR))
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to render ID '{id_str}' at {i},{current_x}")
                    for j, inner_key in enumerate(inner_keys[:len(column_positions) - 1]):
                        value = str(nested_dict.get(pid, {}).get(inner_key, ""))[:key_widths[inner_key]].ljust(key_widths[inner_key])
                        if i < win_height and column_positions[j + 1] + len(value) <= win_width:
                            try:
                                win.addstr(i, column_positions[j + 1], value, curses.color_pair(self.VALUE_COLOR))
                            except curses.error:
                                if self.enable_logging:
                                    logging.warning(f"Failed to render value '{value}' at {i},{column_positions[j + 1]}")
                next_y += height + 1
            return next_y
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error rendering nested tables: {str(e)}")
            return y

    def _render_game_state(self, win, y: int, width: int) -> int:
        if self.enable_logging:
            logging.debug(f"Rendering game state at {y} with width={width}")
            logging.debug(f"self.env type: {type(self.env)}")
            logging.debug(f"self.env.terminal_render_keys: {self.env.terminal_render_keys}")
        try:
            if not hasattr(self.env, "terminal_render_keys"):
                if self.enable_logging:
                    logging.debug("No terminal_render_keys found")
                return y
            categories = self._categorize_game_state(self.env.terminal_render_keys)
            win_height, win_width = win.getmaxyx()

            ########################### DEBUGGING SECTION ###########################
            # colored_text = "[fg=43,bg=51] [/color]\n[fg=50,bg=51]No you tell me a song.[/color]"
            # colored_text = (
            #     "[fg=35,bg=12] ♜ [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color]\n"  # Row 1
            #     "[fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color]\n"  # Row 2
            #     "[fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color]\n"  # Row 3
            #     "[fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color]\n"  # Row 4
            #     "[fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color]\n"  # Row 5
            #     "[fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color]\n"  # Row 6
            #     "[fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color]\n"  # Row 7
            #     "[fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color][fg=35,bg=13]  [/color][fg=35,bg=12]  [/color]"   # Row 8
            # )
            # next_y = self._render_board(win, categories["board"], y, 1, min(width - 2, win_width - 2)) + 1
            # next_y = self._render_board(win, self.env.create_board_str(), y, 1, min(width - 2, win_width - 2)) + 1
            # next_y = self._render_board(win, "Hello [fg=29,bg=0]No you tell me a song.[/color]", y, 1, min(width - 2, win_width - 2)) + 1

            # Check if the environment has a create_board_str method and use it
            board_to_render = categories["board"]
            if hasattr(self.env, "create_board_str") and callable(self.env.create_board_str):
                if self.enable_logging:
                    logging.debug("Using env.create_board_str() for board rendering")
                board_to_render = self.env.create_board_str()
            else:
                if self.enable_logging:
                    logging.debug("Falling back to static board string from game state")

            next_y = self._render_board(win, board_to_render, y, 1, min(width - 2, win_width - 2)) + 1
            if win_width >= 100 and categories["basic"] and categories["players"]:
                left_width = (win_width // 2) - 2
                right_width = win_width - left_width - 3
                basic_y = self._render_basic(win, categories["basic"], next_y, 1, left_width)
                player_y = self._render_players(win, categories["players"], next_y, left_width + 3, right_width)
                next_y = max(basic_y, player_y) + 1
            else:
                next_y = self._render_basic(win, categories["basic"], next_y, 1, min(width - 2, win_width - 2)) + 1
                next_y = self._render_players(win, categories["players"], next_y, 1, min(width - 2, win_width - 2)) + 1
            next_y = self._render_nested_tables(win, categories["nested_tables"], next_y, 1, min(width - 2, win_width - 2)) + 1
            return next_y
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error rendering game state: {str(e)}")
            return y

    def _process_logs(self, max_lines: int, width: int) -> List[Tuple[str, int]]:
        """Process game logs."""
        if self.enable_logging:
            logging.debug(f"Processing logs with max_lines={max_lines}, width={width}")
        try:
            lines = []
            win_width = self.stdscr.getmaxyx()[1] if self.stdscr else width
            effective_width = min(width - 4, win_width - 4)
            for role, msg in self.state.logs:
                prefix = f"{self.player_names.get(role, f'Player {role}')}: " if role != -1 else "[GAME]: "
                color = self.player_color_map.get(role, self.GAME_MSG_COLOR if role == -1 else curses.COLOR_WHITE)
                paragraphs = str(msg).split("\n")
                for i, para in enumerate(paragraphs):
                    if not para:
                        lines.append(("", color))
                        continue
                    current = prefix if i == 0 else " " * len(prefix)
                    for char in para:
                        if len(current) + 1 > effective_width:
                            lines.append((current, color))
                            current = " " * len(prefix) + char
                        else:
                            current += char
                    if current:
                        lines.append((current, color))
            return lines[-max_lines:] if len(lines) > max_lines else lines
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error processing logs: {str(e)}")
            return []

    def _display_error(self, message: str) -> None:
        """Show an error message."""
        if self.enable_logging:
            logging.error(f"Displaying error: {message}")
        try:
            if not self.initialized:
                if self.enable_logging:
                    logging.warning("Cannot display error: not initialized")
                return
            with self._render_lock:
                height, width = self.stdscr.getmaxyx()
                lines = message.split("\n")
                err_width = min(max(len(line) for line in lines) + 6, width - 4)
                err_height = min(len(lines) + 6, height - 2)
                y, x = max(0, (height - err_height) // 2), max(0, (width - err_width) // 2)
                
                err_win = curses.newwin(err_height, err_width, y, x)
                try:
                    err_win.bkgd(" ", curses.color_pair(self.PLAYER1_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning("Failed to set error window background")
                err_win.box()
                if 0 < err_height and 2 < err_width:
                    try:
                        err_win.addstr(0, 2, " ERROR ", curses.A_BOLD)
                    except curses.error:
                        if self.enable_logging:
                            logging.warning("Failed to render ERROR title")
                for i, line in enumerate(lines[:err_height-4], start=2):
                    if i < err_height and 3 + len(line[:err_width-6]) <= err_width:
                        try:
                            err_win.addstr(i, 3, line[:err_width-6])
                        except curses.error:
                            if self.enable_logging:
                                logging.warning(f"Failed to render error line at {i},{3}")
                if err_height - 2 < err_height and 3 + len("Press any key to exit...") <= err_width:
                    try:
                        err_win.addstr(err_height - 2, 3, "Press any key to exit...")
                    except curses.error:
                        if self.enable_logging:
                            logging.warning("Failed to render exit prompt")
                err_win.refresh()
                self.stdscr.timeout(-1)
                self.stdscr.getch()
                self.game_over = True
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Critical error displaying error message: {str(e)}")
            print(f"Critical error displaying error message: {str(e)}")
        finally:
            self._cleanup_curses()

    def update_display(self) -> None:
        """Refresh the terminal display."""
        if not self.initialized or not self.stdscr:
            if self.enable_logging:
                logging.warning("Update display skipped: not initialized or no stdscr")
            return
        if self.enable_logging:
            logging.debug("Updating display")
        try:
            height, width = self.stdscr.getmaxyx()
            if height < 5 or width < 20:
                self.stdscr.clear()
                try:
                    self.stdscr.bkgd(" ", curses.color_pair(self.KEY_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning("Failed to set background for small terminal")
                if 0 < height and len("Terminal too small! Resize and press any key to continue...") <= width:
                    try:
                        self.stdscr.addstr(0, 0, "Terminal too small! Resize and press any key to continue...",
                                          curses.color_pair(self.VALUE_COLOR))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning("Failed to render small terminal message")
                self.stdscr.refresh()
                return
            
            self.stdscr.clear()
            try:
                self.stdscr.bkgd(" ", curses.color_pair(self.KEY_COLOR))
            except curses.error:
                if self.enable_logging:
                    logging.warning("Failed to set main background")
            game_width = max(20, width // 2)
            log_width = max(20, width - game_width - 1)
            
            self.game_win = curses.newwin(height - 2, game_width, 1, 0)
            self.log_win = curses.newwin(height - 2, log_width, 1, game_width + 1)
            for win in (self.game_win, self.log_win):
                try:
                    win.bkgd(" ", curses.color_pair(self.BORDER1_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning(f"Failed to set background for {win}")
                win.box()
            if 0 < height - 2 and 2 + len(" Game State ") <= game_width:
                try:
                    self.game_win.addstr(0, 2, " Game State ", curses.color_pair(self.HEADER1_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning("Failed to render 'Game State' title")
            if 0 < height - 2 and 2 + len(" Game Log ") <= log_width:
                try:
                    self.log_win.addstr(0, 2, " Game Log ", curses.color_pair(self.HEADER1_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning("Failed to render 'Game Log' title")
            
            self._render_game_state(self.game_win, 2, game_width - 2)
            max_log_lines = max(1, height - 6)
            for i, (line, color) in enumerate(self._process_logs(max_log_lines, log_width - 4)):
                if i + 2 < height - 2 and 2 + len(line) <= log_width:
                    try:
                        self.log_win.addstr(i + 2, 2, line, curses.color_pair(color))
                    except curses.error:
                        if self.enable_logging:
                            logging.warning(f"Failed to render log line '{line}' at {i + 2},{2}")
            
            game_over_msg = "Game over! Press any key to exit..." if self.game_over else " " * (width - 1)
            if height - 1 < height and len(game_over_msg) <= width:
                try:
                    self.stdscr.addstr(height - 1, 0, game_over_msg, curses.color_pair(self.KEY_COLOR))
                except curses.error:
                    if self.enable_logging:
                        logging.warning("Failed to render game over message")
            self.stdscr.refresh()
            self.game_win.refresh()
            self.log_win.refresh()
            if self.enable_logging:
                logging.debug("Display updated successfully")
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error updating display: {str(e)}")

    def reset(self, num_players: int, seed: Optional[int] = None) -> any:
        """Reset the environment and start rendering."""
        if self.enable_logging:
            logging.debug("Resetting environment")
        try:
            result = self.env.reset(num_players=num_players, seed=seed)
            self.state = self.env.state
            if self.player_names is None:
                self.player_names = {pid: f"Player {pid}" for pid in range(self.state.num_players)}
            self.player_names.update(self.state.role_mapping)
            self.game_over = False
            self._start_render_thread()
            return result
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Reset failed: {str(e)}")
            self._display_error(f"Reset failed: {str(e)}")
            return None

    def step(self, action: str) -> Tuple[bool, Dict[str, any]]:
        """Execute a game step."""
        if self.enable_logging:
            logging.debug(f"Executing step with action: {action}")
        try:
            done, info = self.env.step(action)
            if done:
                self.game_over = True
                if self.enable_logging:
                    logging.debug("Game over set")
            if self._render_error:
                if self.enable_logging:
                    logging.error(f"Render thread error detected: {self._render_error}")
                raise RuntimeError(f"Rendering thread failed: {self._render_error}")
            return done, info
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Step failed: {str(e)}")
            self._display_error(f"Step failed: {str(e)}")
            return True, {"error": str(e)}

    def close(self) -> Dict:
        """Close the environment and wait for keypress if game over."""
        if self.enable_logging:
            logging.debug("Closing environment")
        try:
            rewards = self.env.close()
            if self._render_thread:
                if self.game_over:
                    if self.enable_logging:
                        logging.debug("Game over, waiting for render thread to finish after keypress")
                    self._render_thread.join(timeout=5)
                    if self._render_thread.is_alive():
                        if self.enable_logging:
                            logging.warning("Render thread did not stop after 5 seconds, forcing stop")
                        self._stop_render.set()
                        self._render_thread.join(timeout=1)
                else:
                    if self.enable_logging:
                        logging.debug("Game not over, stopping render thread immediately")
                    self._stop_render.set()
                    self._render_thread.join(timeout=1)
                    if self._render_thread.is_alive():
                        if self.enable_logging:
                            logging.warning("Render thread did not stop after 1 second")
                self._render_thread = None
            self._cleanup_curses()
            if self._render_error:
                if self.enable_logging:
                    logging.error(f"Rendering thread had an error: {self._render_error}")
                print(f"Rendering thread had an error: {self._render_error}")
            if self.enable_logging:
                logging.debug("Environment closed successfully")
            return rewards
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Close failed: {str(e)}")
            self._display_error(f"Close failed: {str(e)}")
            self._cleanup_curses()
            return {}