import re, random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Snake.renderer import create_board_str

class Snake:
    """Represents a snake in the game with position and alive status."""
    def __init__(self, positions: List[Tuple[int, int]]):
        self.positions = deque(positions)
        self.alive = True
        self.death_reason = None

    @property
    def head(self) -> Tuple[int, int]:
        return self.positions[0]

class SnakeEnv(ta.Env):
    """
    N-player Snake environment that uses a single step(action: str) per call,
    but accumulates each player's action and applies them *simultaneously* once
    all living snakes have submitted their move.
    """
    # Move validation patterns
    up_pattern = re.compile(r"\[(up|w)\]", re.IGNORECASE)
    down_pattern = re.compile(r"\[(down|s)\]", re.IGNORECASE)
    left_pattern = re.compile(r"\[(left|a)\]", re.IGNORECASE)
    right_pattern = re.compile(r"\[(right|d)\]", re.IGNORECASE)

    def __init__(self, width: int=10, height: int=10, num_apples: int=3, max_turns: int=100):
        if width * height < (num_apples + 15):  # Ensure space for apples and max players
            raise ValueError(f"Board size ({width}x{height}) too small for {num_apples} apples and up to 15 snakes")
        
        self.width = width
        self.height = height
        self.num_apples = num_apples
        self.max_turns = max_turns
        
        # Store each player's pending move (None if no move yet this round)
        self.pending_actions = {}

    def get_board_str(self):
        return create_board_str(
            width=self.width,
            height=self.height,
            snakes=self.state.game_state["snakes"],
            apples=self.state.game_state["apples"],
        )

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

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Initialize the environment with N snakes & apples."""
        self.state = ta.State(num_players=num_players, min_players=2, max_players=15, max_turns=self.max_turns, check_truncated=False)

        # Initialize snakes with distinct positions
        snakes = {}
        
        # Initialize all player snakes
        for player_id in range(num_players):
            snake_pos = self._random_free_cell(current_snakes=snakes)
            if snake_pos is None:
                raise RuntimeError(f"Could not find free cell for snake {player_id}")
            snakes[player_id] = Snake([snake_pos])

        # Initialize apples on free cells
        apples = []
        for _ in range(self.num_apples):
            apple_pos = self._random_free_cell(current_snakes=snakes, current_apples=apples)
            if apple_pos:
                apples.append(apple_pos)
            else:
                break  # If we can't place all apples, just use what we have

        # Build the game_state dict
        game_state = {
            "snakes": snakes,
            "apples": apples,
            "scores": {player_id: 0 for player_id in range(num_players)},
            "board_state": self._get_board_string(snakes, apples),
            "death_turn": {} # keep track of when each snake died to determine winners
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

        # Clear pending actions for this new game
        self.pending_actions = {player_id: None for player_id in range(num_players)}

        self.state.game_state["board_state"] = self._get_board_string(self.state.game_state["snakes"], self.state.game_state["apples"])
        message = (
            f"Board after simultaneous moves:\n"
            f"{self.state.game_state['board_state']}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        prompt = (
            f"{self.state.num_players}-Player Snake on a {self.width}x{self.height} grid.\n"
            f"You control snake {player_id}.\n"
            f"Valid moves: '[up]'/'[w]', '[down]'/'[s]', '[left]'/'[a]', '[right]'/'[d]'.\n"
            f"Your objective is to outlast all other snakes, or be longest snake at the end of {self.max_turns} turns."
            # f"Current board:\n{game_state['board_state']}\n"
            # f"Scores: {game_state['scores']}\n"
        )
        return prompt
       
    def _get_board_string(self, snakes: Dict[int, Snake], apples: List[Tuple[int, int]]) -> str:
        """ Create an ASCII board representation. Top row is printed last """
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
                    # Snake's head - use a hex digit for player IDs 0-15
                    board[y][x] = format(pid, 'X')  # Ensures compatibility with up to 15 players
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
        snakes = self.state.game_state["snakes"]
        current_player = self.state.current_player_id
        current_snake = snakes[current_player]

        # Validate action
        move_lower = action.lower()
        if not (
            self.up_pattern.search(move_lower) or
            self.down_pattern.search(move_lower) or
            self.left_pattern.search(move_lower) or
            self.right_pattern.search(move_lower)
        ):
            # Invalid, do not rotate player. Let them retry.
            reason=f"Invalid move format. Valid moves: '[up]'/'[w]', '[down]'/'[s]', '[left]'/'[a]', '[right]'/'[d]'."
            self.state.set_invalid_move(player_id=current_player, reason=reason)
            return self.state.step(rotate_player=False)

        # Action is valid -> store it
        self.pending_actions[current_player] = action

        # Now see if we've collected all moves from the *living* snakes
        living_snakes = [pid for pid, s in snakes.items() if s.alive]
        moves_needed = len(living_snakes)
        moves_received = sum(
            1 for pid in living_snakes if self.pending_actions[pid] is not None
        )

        if moves_received == moves_needed:
            # execute moves
            self._apply_simultaneous_moves()

            # Clear pending actions for next round
            for pid in living_snakes:
                self.pending_actions[pid] = None

        # rotate to the next alive snake
        self._rotate_players()
        self._check_turn_limit()
        return self.state.step(rotate_player=False)

    def _check_turn_limit(self):
        # check that not yet done
        if self.state.done:
            return

        # check if turn limit has been reached and determine a winner
        if self.state.turn >= self.state.max_turns:
            # determine winner based on score of alive
            still_alive = [pid for pid, s in self.state.game_state["snakes"].items() if s.alive]
            max_score = max(
                (self.state.game_state["scores"][pid] for pid in still_alive), 
                default=None
            )
            highest_scoring_pids = [pid for pid in still_alive if self.state.game_state["scores"][pid] == max_score]

            # set winner
            self.state.set_winners(player_ids=highest_scoring_pids, reason=f"Turn limit reached. Winner(s) determined by score.")



    def _rotate_players(self):
        # check that not yet done
        if self.state.done:
            return
        current_player_id = self.state.current_player_id 
        next_player_id = (current_player_id + 1) % self.state.num_players
        while current_player_id != next_player_id:
            # check if still alive
            if self.state.game_state["snakes"][next_player_id].alive:
                self.state.manually_update_current_player(new_player_id=next_player_id)
                break
            else:
                # rotate
                next_player_id = (next_player_id + 1) % self.state.num_players
        else:
            # check if current_player_id is still alive
            if self.state.game_state["snakes"][current_player_id].alive:
                # last snake standing wins
                self.state.set_winners(player_ids=[current_player_id], reason=f"Player outlived all other snakes.")
            else:
                # all snakes are dead, determine winner by longest living
                max_living_turn = max(self.state.game_state["death_turn"].values())
                winner_ids = [pid for pid,t in self.state.game_state["death_turn"].items() if t == max_living_turn]

                self.state.set_winners(player_ids=winner_ids, reason=f"Player(s) outlived all other snakes.")
            


    def _apply_simultaneous_moves(self):
        snakes = self.state.game_state["snakes"]
        apples = self.state.game_state["apples"]
        scores = self.state.game_state["scores"]

        # --- Step 1: Lock in each snake’s starting head position ---
        old_heads: Dict[int, Tuple[int, int]] = {}
        for pid, snake in snakes.items():
            if snake.alive:
                old_heads[pid] = snake.head

        # --- Step 2: Compute desired new head positions ---
        desired_moves: Dict[int, Tuple[int, int]] = {}
        for pid, snake in snakes.items():
            if not snake.alive:
                continue
            move_str = self.pending_actions[pid].lower()
            hx, hy = snake.head
            if self.up_pattern.search(move_str):
                hy += 1
            elif self.down_pattern.search(move_str):
                hy -= 1
            elif self.left_pattern.search(move_str):
                hx -= 1
            elif self.right_pattern.search(move_str):
                hx += 1
            desired_moves[pid] = (hx, hy)

        # --- Step 3: Out-of-Bounds Check ---
        for pid, (nx, ny) in list(desired_moves.items()):
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                snakes[pid].alive = False
                snakes[pid].death_reason = "wall"
                self.state.game_state["death_turn"][pid] = self.state.turn
                del desired_moves[pid]

        # --- Step 4: Standard Head-On Collisions ---
        head_positions: Dict[Tuple[int, int], List[int]] = {}
        for pid, pos in desired_moves.items():
            head_positions.setdefault(pos, []).append(pid)
        for pos, pids in head_positions.items():
            if len(pids) > 1:
                for pid in pids:
                    snakes[pid].alive = False
                    snakes[pid].death_reason = "head-on collision"
                    self.state.game_state["death_turn"][pid] = self.state.turn
                    if pid in desired_moves:
                        del desired_moves[pid]

        # --- Step 5: Swap Collision Check ---
        # If two snakes are swapping positions—that is, snake A’s desired move equals snake B’s old head
        # and snake B’s desired move equals snake A’s old head—then both should die.
        swap_collision = set()
        pids = list(desired_moves.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pid_i = pids[i]
                pid_j = pids[j]
                if (desired_moves.get(pid_i) == old_heads.get(pid_j) and
                    desired_moves.get(pid_j) == old_heads.get(pid_i)):
                    swap_collision.add(pid_i)
                    swap_collision.add(pid_j)
        for pid in swap_collision:
            if pid in desired_moves:
                snakes[pid].alive = False
                snakes[pid].death_reason = "head-on collision"
                self.state.game_state["death_turn"][pid] = self.state.turn
                del desired_moves[pid]

        # --- Step 6: Prevent Self-Flipping in Length-1 Snakes ---
        # For a snake of length 1, moving into its own cell (i.e. not moving at all) should be disallowed.
        for pid, snake in snakes.items():
            if not snake.alive:
                continue
            if len(snake.positions) == 1 and desired_moves.get(pid) == snake.head:
                snake.alive = False
                snake.death_reason = "body collision"
                self.state.game_state["death_turn"][pid] = self.state.turn
                if pid in desired_moves:
                    del desired_moves[pid]

        # --- Step 7: Determine Growth and Record Old Tails ---
        grows: Dict[int, bool] = {}
        old_tails: Dict[int, Tuple[int, int]] = {}
        for pid, snake in snakes.items():
            if not snake.alive:
                grows[pid] = False
                continue
            if pid in desired_moves:
                new_head = desired_moves[pid]
                grows[pid] = new_head in apples
            else:
                grows[pid] = False
            if snake.positions:
                old_tails[pid] = snake.positions[-1]

        # --- Step 8: Build Set of Occupied Cells After Movement ---
        # For each snake, all body segments will remain occupied except the tail if the snake is not growing.
        occupied_cells = set()
        for pid, snake in snakes.items():
            if not snake.alive:
                continue
            for idx, pos in enumerate(snake.positions):
                # If this is the tail and the snake is not growing, it will vacate.
                if idx == len(snake.positions) - 1 and not grows.get(pid, False):
                    continue
                occupied_cells.add(pos)

        # --- Step 9: Body Collision Check ---
        # If a snake’s new head lands in a cell that remains occupied, it dies—
        # except that moving into its own tail is allowed only if the snake is longer than 1 and not growing.
        for pid, new_head in list(desired_moves.items()):
            snake = snakes[pid]
            own_tail = snake.positions[-1] if snake.positions else None
            if new_head in occupied_cells:
                if new_head == own_tail and len(snake.positions) > 1 and not grows[pid]:
                    continue
                else:
                    snakes[pid].alive = False
                    snakes[pid].death_reason = "body collision"
                    self.state.game_state["death_turn"][pid] = self.state.turn
                    del desired_moves[pid]

        # --- Step 10: Execute Moves and Handle Apples ---
        for pid, new_head in desired_moves.items():
            snake = snakes[pid]
            snake.positions.appendleft(new_head)
            if new_head in apples:
                apples.remove(new_head)
                scores[pid] += 1
                new_apple = self._random_free_cell(current_snakes=snakes, current_apples=apples)
                if new_apple:
                    apples.append(new_apple)
            else:
                snake.positions.pop()

        # --- Step 11: End-of-Game Check ---
        still_alive = [pid for pid, s in snakes.items() if s.alive]
        if len(still_alive) <= 1:
            if len(still_alive) == 1:
                winner = still_alive[0]
                self.state.set_winners([winner], f"Player {winner} survived; all other snakes died.")
            else:
                # check if death turn is the same for all to determine draw
                # print(len(set(self.state.game_state["death_turn"].values())) != 1)
                # print(self.state.game_state["death_turn"].values())
                if self.state.num_players > 2 and len(set(self.state.game_state["death_turn"].values())) != 1:
                    max_turn = max(self.state.game_state["death_turn"].values(), default=self.state.turn)
                    winners = [pid for pid, turn in self.state.game_state["death_turn"].items() if turn == max_turn]
                    self.state.set_winners(player_ids=winners, reason="Player(s) outlived all other snakes.")
                else:
                    # print(f"setting draw")
                    self.state.set_draw("All snakes died simultaneously.")

                return self.state.step()

        # --- Step 12: Update Board and Broadcast New State ---
        self.state.game_state["board_state"] = self._get_board_string(snakes, apples)
        message = (
            f"Board after simultaneous moves:\n"
            f"{self.state.game_state['board_state']}" #\nScores: {scores}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)
