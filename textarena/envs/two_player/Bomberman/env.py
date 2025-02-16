import re, random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set
import textarena as ta

class Enemy:
    """Represents a moving enemy that follows a pattern."""
    def __init__(self, position: Tuple[int, int], pattern_type: str = "horizontal"):
        self.position = position
        self.pattern_type = pattern_type
        self.direction = 1  # 1 or -1
        self.steps_in_direction = 0
        self.max_steps = 3  # How far it moves before turning around

    def get_next_position(self, game_state: Dict) -> Tuple[int, int]:
        """Calculate next position based on pattern."""
        x, y = self.position
        
        if self.pattern_type == "horizontal":
            next_pos = (x + self.direction, y)
        elif self.pattern_type == "vertical":
            next_pos = (x, y + self.direction)
        else:  # "random"
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dx, dy = random.choice(directions)
            next_pos = (x + dx, y + dy)
            
        # Check if move is valid
        if self._is_valid_move(next_pos, game_state):
            self.steps_in_direction += 1
            if self.steps_in_direction >= self.max_steps:
                self.direction *= -1
                self.steps_in_direction = 0
            return next_pos
            
        # If move is invalid, reverse direction
        self.direction *= -1
        self.steps_in_direction = 0
        return self.position

    def _is_valid_move(self, pos: Tuple[int, int], game_state: Dict) -> bool:
        """Check if the enemy can move to the given position."""
        if pos in game_state["indestructible_walls"] or pos in game_state["walls"]:
            return False
        if pos in [b.position for b in game_state["bombs"]]:
            return False
        x, y = pos
        if x < 0 or x >= game_state["width"] or y < 0 or y >= game_state["height"]:
            return False
        return True

class Bomb:
    """Represents a bomb on the board with position and timer."""
    def __init__(self, position: Tuple[int, int], timer: int = 3, radius: int = 2):
        self.position = position
        self.timer = timer
        self.radius = radius
        self.owner = None

class Player:
    """Represents a player in the game with position and alive status."""
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.alive = True
        self.death_reason = None
        self.bomb_limit = 1
        self.active_bombs = 0

class TwoPlayerBombermanEnv(ta.Env):
    """
    Two-player Bomberman environment with indestructible walls and moving enemies.
    """
    def __init__(
        self,
        width: int = 13,
        height: int = 11,
        wall_density: float = 0.3,
        num_enemies: int = 3,
        max_turns: int = 100
    ):
        if width < 7 or height < 7:
            raise ValueError("Board must be at least 7x7")
        
        self.width = width
        self.height = height
        self.wall_density = wall_density
        self.num_enemies = num_enemies

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
        self.bomb_pattern = re.compile(r"\[bomb|b\]", re.IGNORECASE)
        self.wait_pattern = re.compile(r"\[wait\]", re.IGNORECASE)

        self.pending_actions: Dict[int, Optional[str]] = {0: None, 1: None}


    @property
    def offline_renderer(self):
        raise NotImplementedError

    @property
    def terminal_render_keys(self):
        return ["board_state"]

    def _is_valid_position(self, pos: Tuple[int, int], game_state: Dict) -> bool:
        """Check if a position is valid and unoccupied."""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        if (pos in game_state["walls"] or 
            pos in game_state["indestructible_walls"] or 
            pos in [b.position for b in game_state["bombs"]]):
            return False
            
        return True

    def _generate_map(self) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[Enemy]]:
        """Generate initial map layout with both wall types and enemies."""
        indestructible_walls = set()
        breakable_walls = set()
        enemies = []
        
        # Add border walls (indestructible)
        for x in range(self.width):
            indestructible_walls.add((x, 0))
            indestructible_walls.add((x, self.height - 1))
        for y in range(self.height):
            indestructible_walls.add((0, y))
            indestructible_walls.add((self.width - 1, y))
            
        # Add indestructible wall pattern (classic Bomberman style)
        for x in range(2, self.width - 2, 2):
            for y in range(2, self.height - 2, 2):
                indestructible_walls.add((x, y))
                
        # Add random breakable walls
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if (x, y) not in indestructible_walls:
                    if random.random() < self.wall_density:
                        # Don't place walls in player starting areas
                        if not ((x < 3 and y < 3) or 
                               (x > self.width - 4 and y > self.height - 4)):
                            breakable_walls.add((x, y))
        
        # Add enemies
        available_spots = []
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                pos = (x, y)
                if (pos not in indestructible_walls and 
                    pos not in breakable_walls and
                    not ((x < 3 and y < 3) or 
                         (x > self.width - 4 and y > self.height - 4))):
                    available_spots.append(pos)
                    
        if available_spots:
            for _ in range(min(self.num_enemies, len(available_spots))):
                pos = random.choice(available_spots)
                available_spots.remove(pos)
                pattern = random.choice(["horizontal", "vertical", "random"])
                enemies.append(Enemy(pos, pattern))
                
        return indestructible_walls, breakable_walls, enemies

    def reset(self, seed: Optional[int] = None):
        """Initialize the environment."""
        if seed is not None:
            random.seed(seed)

        # Initialize players in opposite corners
        players = {
            0: Player((1, 1)),
            1: Player((self.width - 2, self.height - 2))
        }

        # Generate map elements
        indestructible_walls, walls, enemies = self._generate_map()

        game_state = {
            "width": self.width,
            "height": self.height,
            "players": players,
            "indestructible_walls": indestructible_walls,
            "walls": walls,
            "enemies": enemies,
            "bombs": [],
            "board_state": self._get_board_string(
                players, indestructible_walls, walls, enemies, []
            )
        }

        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

        self.pending_actions = {0: None, 1: None}

        return self.state.done, self.state.info

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"Two-Player Bomberman on a {self.width}x{self.height} grid.\n"
            f"You are player {player_id}. Watch out for enemies (E)!\n"
            f"Valid moves: [up]/[w], [down]/[s], [left]/[a], [right]/[d], [bomb]/[b], [wait]\n"
            f"Legend: # = indestructible wall, * = breakable wall, E = enemy\n"
            f"Current board:\n{game_state['board_state']}\n"
        )

    def _get_board_string(
        self, 
        players: Dict[int, Player],
        indestructible_walls: Set[Tuple[int, int]],
        walls: Set[Tuple[int, int]],
        enemies: List[Enemy],
        bombs: List[Bomb]
    ) -> str:
        """Create ASCII board representation."""
        board = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Place indestructible walls
        for (wx, wy) in indestructible_walls:
            board[wy][wx] = '#'

        # Place breakable walls
        for (wx, wy) in walls:
            board[wy][wx] = '*'

        # Place bombs with their timers
        for bomb in bombs:
            bx, by = bomb.position
            board[by][bx] = str(bomb.timer)

        # Place enemies
        for enemy in enemies:
            ex, ey = enemy.position
            board[ey][ex] = 'E'

        # Place players
        for pid, player in players.items():
            if player.alive:
                px, py = player.position
                board[py][px] = str(pid)

        # Create board string with borders
        lines = []
        content_width = self.width * 2 - 1
        lines.append("+" + "-" * (content_width + 2) + "+")
        for row in board:
            row_str = " ".join(row)
            lines.append(f"| {row_str} |")
        lines.append("+" + "-" * (content_width + 2) + "+")
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process a single player's action."""
        if self.state.done:
            return self.state.done, self.state.info

        players = self.state.game_state["players"]
        current_player = self.state.current_player_id
        current_player_obj = players[current_player]

        if not current_player_obj.alive:
            done, info = self.state.step(rotate_player=True)
            return done, info

        # Validate action
        move_lower = action.lower()
        if not (
            self.up_pattern.search(move_lower) or
            self.down_pattern.search(move_lower) or
            self.left_pattern.search(move_lower) or
            self.right_pattern.search(move_lower) or
            self.bomb_pattern.search(move_lower) or
            self.wait_pattern.search(move_lower)
        ):
            self.state.set_invalid_move(current_player, "Invalid move format.")
            done, info = self.state.step(rotate_player=False)
            return done, info

        self.pending_actions[current_player] = action

        done, info = self.state.step(rotate_player=True)
        if done:
            return done, info

        living_players = [pid for pid, p in players.items() if p.alive]
        moves_needed = len(living_players)
        moves_received = sum(
            1 for pid in living_players if self.pending_actions[pid] is not None
        )

        if moves_received < moves_needed:
            return self.state.done, self.state.info

        self._apply_simultaneous_moves()

        for pid in living_players:
            self.pending_actions[pid] = None

        self.state.turn += 1

        if not self.state.done and self.state.turn >= self.state.max_turns:
            self.state.set_draw("Turn limit reached.")

        return self.state.done, self.state.info

    def _apply_simultaneous_moves(self):
        """Apply all pending moves and update game state."""
        game_state = self.state.game_state
        players = game_state["players"]
        walls = game_state["walls"]
        indestructible_walls = game_state["indestructible_walls"]
        enemies = game_state["enemies"]
        bombs = game_state["bombs"]

        # Step 1: Process bomb timers and explosions
        exploded_bombs = []
        affected_positions = set()
        
        for bomb in bombs[:]:
            bomb.timer -= 1
            if bomb.timer <= 0:
                exploded_bombs.append(bomb)
                bombs.remove(bomb)
                
                # Calculate explosion area
                bx, by = bomb.position
                for dx in range(-bomb.radius, bomb.radius + 1):
                    for dy in range(-bomb.radius, bomb.radius + 1):
                        if abs(dx) + abs(dy) <= bomb.radius:  # Diamond shape
                            pos = (bx + dx, by + dy)
                            if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
                                if pos not in indestructible_walls:
                                    affected_positions.add(pos)
                                    if pos in walls:
                                        walls.remove(pos)  # Destroy walls
                
                if bomb.owner is not None:
                    players[bomb.owner].active_bombs -= 1

        # Step 2: Move enemies
        new_enemy_positions = {}
        for enemy in enemies[:]:  # Use copy since we might remove enemies
            if enemy.position in affected_positions:
                enemies.remove(enemy)
                continue
            new_pos = enemy.get_next_position(game_state)
            new_enemy_positions[enemy] = new_pos

        # Step 3: Process player moves and bomb placement
        new_positions = {}
        new_bombs = []
        
        for pid, player in players.items():
            if not player.alive:
                continue
                
            current_pos = player.position
            move_str = self.pending_actions[pid].lower()
            new_pos = current_pos
            
            # Handle movement
            if self.up_pattern.search(move_str):
                new_pos = (current_pos[0], current_pos[1] - 1)
            elif self.down_pattern.search(move_str):
                new_pos = (current_pos[0], current_pos[1])