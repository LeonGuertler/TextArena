# Inspired by the ARC Snakebench (https://github.com/gkamradt/SnakeBench)

import re, random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta


class Snake:
    """Represents a snake in the game with position and alive status."""
    def __init__(self, positions: List[Tuple[int, int]]):
        self.positions = deque(positions)
        self.alive = True
        self.death_reason = None

    @property
    def head(self) -> Tuple[int, int]:
        return self.positions[0]


class TwoPlayerSnakeEnv(ta.Env):
    """
    Two-player Snake environment that uses a single step(action: str) per call,
    but accumulates each player's action and applies them *simultaneously* once
    both players (or all living snakes) have submitted their move.
    """
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        num_apples: int = 3,
        max_turns: int = 100
    ):
    
        if width * height < (num_apples + 2):  # +2 for the two snakes
            raise ValueError(f"Board size ({width}x{height}) too small for {num_apples} apples and 2 snakes")


        self.width = width
        self.height = height
        self.num_apples = num_apples

        # We'll set check_truncated=False so we can manage the turn limit ourselves.
        self.state = ta.State(
            num_players=2,
            max_turns=max_turns,
            check_truncated=False,
        )

        # Move validation patterns
        self.up_pattern = re.compile(r"\[(up|w)\]", re.IGNORECASE)
        self.down_pattern = re.compile(r"\[(down|s)\]", re.IGNORECASE)
        self.left_pattern = re.compile(r"\[(left|a)\]", re.IGNORECASE)
        self.right_pattern = re.compile(r"\[(right|d)\]", re.IGNORECASE)

        # Store each player's pending move (None if no move yet this round)
        self.pending_actions: Dict[int, Optional[str]] = {0: None, 1: None}

    @property
    def offline_renderer(self):
        """Not implemented here."""
        raise NotImplementedError

    @property
    def terminal_render_keys(self):
        """Which keys to show in a terminal rendering if used outside."""
        return ["board_state", "scores"]

    def _random_free_cell(
        self,
        current_snakes: Dict[int, Snake] = None,
        current_apples: List[Tuple[int, int]] = None,
        max_attempts: int = 1000
    ) -> Optional[Tuple[int, int]]:
        """
        Return a random (x, y) that is not occupied by a snake body
        or an existing apple. Returns None if no free cell found after max_attempts.
        """
        attempts = 0
        while attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            occupied_by_snake = False
            if current_snakes:
                for s in current_snakes.values():
                    if s.alive and (x, y) in s.positions:
                        occupied_by_snake = True
                        break
                        
            occupied_by_apple = False
            if current_apples and (x, y) in current_apples:
                occupied_by_apple = True
                
            if not occupied_by_snake and not occupied_by_apple:
                return (x, y)
                
            attempts += 1
        return None

    def reset(self, seed: Optional[int] = None):
        """Initialize the environment with 2 snakes & N apples."""
        if seed is not None:
            random.seed(seed)

        # Initialize snakes with distinct positions
        snakes = {}

        snake0_pos = self._random_free_cell()
        snakes[0] = Snake([snake0_pos])

        snake1_pos = self._random_free_cell(current_snakes=snakes)
        snakes[1] = Snake([snake1_pos])

        # Initialize apples on free cells
        apples = []
        for _ in range(self.num_apples):
            apple_pos = self._random_free_cell(current_snakes=snakes, current_apples=apples)
            apples.append(apple_pos)

        # Build the game_state dict
        game_state = {
            "snakes": snakes,
            "apples": apples,
            "scores": {0: 0, 1: 0},
            "board_state": self._get_board_string(snakes, apples)
        }

        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

        # Clear pending actions for this new game
        self.pending_actions = {0: None, 1: None}

        return self.state.done, self.state.info

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"Two-Player Snake on a {self.width}x{self.height} grid.\n"
            f"You control snake {player_id}.\n"
            f"Valid moves: [up]/[w], [down]/[s], [left]/[a], [right]/[d].\n"
            f"Current board:\n{game_state['board_state']}\n"
        )

    def _get_board_string(self, snakes: Dict[int, Snake], apples: List[Tuple[int, int]]) -> str:
        """
        Create an ASCII board representation. Top row is printed last.
        """
        board = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Place apples
        for (ax, ay) in apples:
            board[ay][ax] = 'A'

        # Place snake body and head
        for pid, snake in snakes.items():
            if not snake.alive:
                continue
            for idx, (x, y) in enumerate(snake.positions):
                if idx == 0:
                    board[y][x] = str(pid)  # Snake's head
                else:
                    board[y][x] = '#'

        lines = []
        content_width = self.width * 2 - 1
        lines.append("+" + "-"*(content_width + 2) + "+")  # top border
        # Row from top (height-1) down to 0
        for row_idx in range(self.height-1, -1, -1):
            row_str = " ".join(board[row_idx])
            lines.append(f"| {row_str} |")
        lines.append("+" + "-"*(content_width + 2) + "+")  # bottom border
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Called once per player's turn (the library usage).
        We store this move in pending_actions. Once we have
        all living snakes' moves, we do a simultaneous update.
        """
        if self.state.done:
            return self.state.done, self.state.info

        snakes = self.state.game_state["snakes"]
        current_player = self.state.current_player_id
        current_snake = snakes[current_player]

        # If this snake is already dead, it can't move. We just rotate to the next
        # player. There's no immediate update until all living snakes submit.
        if not current_snake.alive:
            done, info = self.state.step(rotate_player=True)
            return done, info

        # Validate action
        move_lower = action.lower()
        if not (
            self.up_pattern.search(move_lower) or
            self.down_pattern.search(move_lower) or
            self.left_pattern.search(move_lower) or
            self.right_pattern.search(move_lower)
        ):
            # Invalid, do not rotate player. Let them retry.
            self.state.set_invalid_move(current_player, "Invalid move format.")
            done, info = self.state.step(rotate_player=False)
            return done, info

        # Action is valid -> store it
        self.pending_actions[current_player] = action

        # Rotate to next player immediately
        done, info = self.state.step(rotate_player=True)
        if done:
            return done, info

        # Now see if we've collected all moves from the *living* snakes
        living_snakes = [pid for pid, s in snakes.items() if s.alive]
        moves_needed = len(living_snakes)
        moves_received = sum(
            1 for pid in living_snakes if self.pending_actions[pid] is not None
        )

        if moves_received < moves_needed:
            # We still need the other player's move -> no update yet
            return self.state.done, self.state.info

        # Otherwise, do the simultaneous update
        self._apply_simultaneous_moves()

        # Clear pending actions for next round
        for pid in living_snakes:
            self.pending_actions[pid] = None

        # We've completed a full round
        self.state.turn += 1

        # Check turn limit if game is still ongoing
        if not self.state.done and self.state.turn >= self.state.max_turns:
            self._handle_turn_limit()

        return self.state.done, self.state.info

    def _apply_simultaneous_moves(self):
        """Perform a single simultaneous update from self.pending_actions."""
        snakes = self.state.game_state["snakes"]
        apples = self.state.game_state["apples"]
        scores = self.state.game_state["scores"]

        # Step 1: collect each snake's new head position from the pending move
        desired_moves: Dict[int, Tuple[int,int]] = {}
        for pid, snake in snakes.items():
            if not snake.alive:
                continue
            hx, hy = snake.head
            move_str = self.pending_actions[pid].lower()

            if self.up_pattern.search(move_str):
                hy += 1
            elif self.down_pattern.search(move_str):
                hy -= 1
            elif self.left_pattern.search(move_str):
                hx -= 1
            elif self.right_pattern.search(move_str):
                hx += 1

            desired_moves[pid] = (hx, hy)

        # Step 2: kill snakes that go out-of-bounds or that do head-on collisions
        for pid, (nx, ny) in desired_moves.items():
            s = snakes[pid]
            # Wall check
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                s.alive = False
                s.death_reason = "wall"

        alive_pids = [p for p in snakes if snakes[p].alive]
        if len(alive_pids) == 2:
            # If 2 snakes are alive, check if they moved to the same cell
            p0, p1 = alive_pids
            if desired_moves[p0] == desired_moves[p1]:
                snakes[p0].alive = False
                snakes[p1].alive = False
                snakes[p0].death_reason = "head-on collision"
                snakes[p1].death_reason = "head-on collision"

        # Step 3: figure out who is growing + track old tails
        old_tails = {}
        grows = {}
        for pid, s in snakes.items():
            if not s.alive:
                grows[pid] = False
                continue
            old_tails[pid] = s.positions[-1]  # last segment
            new_head = desired_moves[pid]
            grows[pid] = (new_head in apples)

        # Step 4: allow tail-collision (exclude old tail from occupied set if not growing)
        occupied_cells = set()
        for pid, s in snakes.items():
            if not s.alive:
                continue
            for idx, cell in enumerate(s.positions):
                # skip the old tail if not growing
                if idx == len(s.positions) - 1 and not grows[pid]:
                    continue
                occupied_cells.add(cell)

        # Check if new head hits a body
        for pid in [p for p in snakes if snakes[p].alive]:
            new_head = desired_moves[pid]
            if new_head in occupied_cells:
                snakes[pid].alive = False
                snakes[pid].death_reason = "body collision"

        # Step 5: move each surviving snake, handle apple consumption
        for pid, s in snakes.items():
            if not s.alive:
                continue
            nx, ny = desired_moves[pid]
            s.positions.appendleft((nx, ny))
            if grows[pid]:
                # eat apple
                apples.remove((nx, ny))
                scores[pid] += 1
                # spawn a new apple
                apples.append(self._random_free_cell(snakes, apples))
            else:
                # pop tail
                s.positions.pop()

        # Step 6: if only one snake (or none) survived => end
        still_alive = [pid for pid, s in snakes.items() if s.alive]
        if len(still_alive) <= 1:
            if len(still_alive) == 1:
                winner = still_alive[0]
                self.state.set_winners(
                    [winner],
                    f"Player {winner} survived; other snake died."
                )
            else:
                self.state.set_draw("Both snakes died simultaneously.")

        # Step 7: update board_state & broadcast the new board
        self.state.game_state["board_state"] = self._get_board_string(snakes, apples)
        self.state.add_observation(
            from_id=ta.GAME_ID,
            to_id=-1,
            message="Board after simultaneous moves:\n" 
                    f"{self.state.game_state['board_state']}",
            for_logging=False
        )

    def _handle_turn_limit(self):
        """Compare scores if the turn limit is reached and nobody has won yet."""
        s0 = self.state.game_state["scores"][0]
        s1 = self.state.game_state["scores"][1]
        if s0 > s1:
            self.state.set_winners([0], "Turn limit reached, Player 0 has higher score.")
        elif s1 > s0:
            self.state.set_winners([1], "Turn limit reached, Player 1 has higher score.")
        else:
            self.state.set_draw("Turn limit reached, tie in scores.")
