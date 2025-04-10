import re, random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Surround.renderer import create_board_str

class SurroundEnv(ta.Env):
    """
    N-player Surround environment that uses a single step(action: str) per call,
    but accumulates each player's action and applies them *simultaneously* once
    all living players have submitted their move.
    """
    # Move validation patterns
    up_pattern = re.compile(r"\[(up|w)\]", re.IGNORECASE)
    down_pattern = re.compile(r"\[(down|s)\]", re.IGNORECASE)
    left_pattern = re.compile(r"\[(left|a)\]", re.IGNORECASE)
    right_pattern = re.compile(r"\[(right|d)\]", re.IGNORECASE)

    def __init__(self, width: int = 10, height: int = 10, max_turns: int = 100):
        """
        Initialize the Surround environment.

        Args:
            width (int): Width of the game board.
            height (int): Height of the game board.
            max_turns (int): Maximum number of turns before the game is considered a draw.
        """
        self.width = width
        self.height = height
        self.max_turns = max_turns
        self.pending_actions = {}

    def get_board_str(self):
        return create_board_str(width=self.width, height=self.height, game_state=self.state.game_state)


    def reset(self, num_players: int, seed: Optional[int] = None):
        """
        Initialize the environment with N players at random positions.

        Args:
            num_players (int): Number of players (2 to 15).
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=15,
            max_turns=self.max_turns,
            check_truncated=False
        )

        players = {}
        for player_id in range(num_players):
            pos = self._random_free_cell(players)
            if pos is None:
                raise RuntimeError(f"Could not find free cell for player {player_id}")
            players[player_id] = {"position": pos, "alive": True}

        board = [[None for _ in range(self.width)] for _ in range(self.height)]
        game_state = {
            "board": board,
            "players": players,
            "death_turn": {},
            "board_state": self._get_board_string(board, players),
        }
        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )
        self.pending_actions = {player_id: None for player_id in range(num_players)}

    def _random_free_cell(self, current_players: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """
        Return a random (x, y) that is not occupied by another player.

        Args:
            current_players (Dict[int, Dict[str, Any]]): Current players and their positions.

        Returns:
            Optional[Tuple[int, int]]: A free cell position or None if not found.
        """
        attempts = 0
        max_attempts = 1000
        occupied = set(player["position"] for player in current_players.values() if player["alive"])

        while attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in occupied:
                return (x, y)
            attempts += 1
        return None

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the prompt for a player, showing the current board state.

        Args:
            player_id (int): ID of the player.
            game_state (Dict[str, Any]): Current game state.

        Returns:
            str: Prompt string for the player.
        """
        prompt = (
            f"{self.state.num_players}-Player Surround on a {self.width}x{self.height} grid.\n"
            f"You control player {player_id}.\n"
            f"Valid moves: '[up]'/'[w]', '[down]'/'[s]', '[left]'/'[a]', '[right]'/'[d]'.\n"
            f"Current board:\n{game_state['board_state']}\n"
        )
        return prompt

    def _get_board_string(self, board: List[List[Optional[int]]], players: Dict[int, Dict[str, Any]]) -> str:
        """
        Create an ASCII board representation.

        - Trails are marked with '#'.
        - Current positions of alive players are marked with their player ID (0-9, A-F).

        Args:
            board (List[List[Optional[int]]]): Board with trails marked by player IDs.
            players (Dict[int, Dict[str, Any]]): Player positions and statuses.

        Returns:
            str: ASCII representation of the board.
        """
        board_display = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if board[y][x] is not None:
                    board_display[y][x] = '#'
        for pid, player in players.items():
            if player["alive"]:
                px, py = player["position"]
                board_display[py][px] = format(pid, 'X')  # '0'-'9', 'A'-'F'

        lines = []
        content_width = self.width * 2 - 1
        lines.append("+" + "-" * (content_width + 2) + "+")
        for row_idx in range(self.height - 1, -1, -1):
            row_str = " ".join(board_display[row_idx])
            lines.append(f"| {row_str} |")
        lines.append("+" + "-" * (content_width + 2) + "+")
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a player's action. Accumulate actions and apply them simultaneously once
        all living players have submitted.

        Args:
            action (str): Player's move ('[up]'/'[w]', '[down]'/'[s]', '[left]'/'[a]', '[right]'/'[d]').

        Returns:
            Tuple[bool, ta.Info]: State step result.
        """
        current_player = self.state.current_player_id

        # Validate action
        move_lower = action.lower()
        if not (
            self.up_pattern.search(move_lower) or
            self.down_pattern.search(move_lower) or
            self.left_pattern.search(move_lower) or
            self.right_pattern.search(move_lower)
        ):
            reason = "Invalid move format. Valid moves: '[up]'/'[w]', '[down]'/'[s]', '[left]'/'[a]', '[right]'/'[d]'."
            self.state.set_invalid_move(player_id=current_player, reason=reason)
            return self.state.step(rotate_player=False)

        # Store action
        self.pending_actions[current_player] = action

        # Check if all living players have submitted
        living_players = [pid for pid, p in self.state.game_state["players"].items() if p["alive"]]
        if all(self.pending_actions[pid] is not None for pid in living_players):
            self._apply_simultaneous_moves()
            for pid in living_players:
                self.pending_actions[pid] = None

        # Rotate to next player
        self._rotate_players()
        self._check_turn_limit()
        return self.state.step(rotate_player=False)

    # def _apply_simultaneous_moves(self):
    #     """
    #     Apply all pending moves simultaneously, handling collisions and trails.
    #     """
    #     players = self.state.game_state["players"]
    #     board = self.state.game_state["board"]
    #     living_players = [pid for pid, p in players.items() if p["alive"]]

    #     # Record old positions
    #     old_positions = {pid: players[pid]["position"] for pid in living_players}
    #     desired_moves = {}

    #     # Compute desired new positions
    #     for pid in living_players:
    #         move_str = self.pending_actions[pid].lower()
    #         x, y = old_positions[pid]
    #         if self.up_pattern.search(move_str):
    #             new_pos = (x, y + 1)
    #         elif self.down_pattern.search(move_str):
    #             new_pos = (x, y - 1)
    #         elif self.left_pattern.search(move_str):
    #             new_pos = (x - 1, y)
    #         elif self.right_pattern.search(move_str):
    #             new_pos = (x + 1, y)
    #         desired_moves[pid] = new_pos

    #     # Identify crashing players
    #     crashing_players = set()

    #     # Out-of-bounds
    #     for pid, (nx, ny) in desired_moves.items():
    #         if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
    #             crashing_players.add(pid)

    #     # Trail collisions
    #     for pid, (nx, ny) in desired_moves.items():
    #         if (nx, ny) not in crashing_players and board[ny][nx] is not None:
    #             crashing_players.add(pid)

    #     # Head-on collisions
    #     position_to_players = defaultdict(list)
    #     for pid, pos in desired_moves.items():
    #         if pid not in crashing_players:
    #             position_to_players[pos].append(pid)
    #     for pos, pids in position_to_players.items():
    #         if len(pids) > 1:
    #             for pid in pids:
    #                 crashing_players.add(pid)

    #     # Swap collisions
    #     for pid_i in living_players:
    #         for pid_j in living_players:
    #             if pid_i < pid_j:
    #                 if (desired_moves.get(pid_i) == old_positions.get(pid_j) and
    #                     desired_moves.get(pid_j) == old_positions.get(pid_i)):
    #                     if pid_i not in crashing_players and pid_j not in crashing_players:
    #                         crashing_players.add(pid_i)
    #                         crashing_players.add(pid_j)

    #     # Move non-crashing players and leave trails
    #     for pid in living_players:
    #         if pid not in crashing_players:
    #             new_pos = desired_moves[pid]
    #             players[pid]["position"] = new_pos
    #             old_x, old_y = old_positions[pid]
    #             board[old_y][old_x] = pid

    #     # Set crashing players as not alive
    #     for pid in crashing_players:
    #         players[pid]["alive"] = False
    #         self.state.game_state["death_turn"][pid] = self.state.turn

    #     # Check if game is over
    #     still_alive = [pid for pid in players if players[pid]["alive"]]
    #     if len(still_alive) == 0:
    #         if self.state.num_players > 1:
    #             max_death_turn = max(self.state.game_state["death_turn"].values())
    #             winners = [pid for pid, turn in self.state.game_state["death_turn"].items() if turn == max_death_turn]
    #             if len(winners) == 1:
    #                 self.state.set_winners(winners, f"Player {winners[0]} outlived all other players.")
    #             else:
    #                 self.state.set_draw("Multiple players died simultaneously on the last turn.")
    #         else:
    #             self.state.set_draw("All players died simultaneously.")
    #     elif len(still_alive) == 1:
    #         winner = still_alive[0]
    #         self.state.set_winners([winner], f"Player {winner} survived; all other players crashed.")

    #     # Update board_state
    #     self.state.game_state["board_state"] = self._get_board_string(board, players)
    #     message = f"Board after simultaneous moves:\n{self.state.game_state['board_state']}"
    #     self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _apply_simultaneous_moves(self):
        players = self.state.game_state["players"]
        board = self.state.game_state["board"]
        living_players = [pid for pid, p in players.items() if p["alive"]]

        # Record old positions
        old_positions = {pid: players[pid]["position"] for pid in living_players}
        desired_moves = {}

        # Compute desired new positions based on player actions
        for pid in living_players:
            move_str = self.pending_actions[pid].lower()
            x, y = old_positions[pid]
            if self.up_pattern.search(move_str):
                new_pos = (x, y + 1)
            elif self.down_pattern.search(move_str):
                new_pos = (x, y - 1)
            elif self.left_pattern.search(move_str):
                new_pos = (x - 1, y)
            elif self.right_pattern.search(move_str):
                new_pos = (x + 1, y)
            desired_moves[pid] = new_pos

        # Set of current positions that will become trails
        current_positions = {old_positions[pid] for pid in living_players}

        # Identify crashing players
        crashing_players = set()

        # 1. Out-of-bounds check
        for pid, (nx, ny) in desired_moves.items():
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                crashing_players.add(pid)

        # 2. Existing trail collisions
        for pid, (nx, ny) in desired_moves.items():
            if pid not in crashing_players and board[ny][nx] is not None:
                crashing_players.add(pid)

        # 3. New trail collisions: moving to a current player position
        for pid, pos in desired_moves.items():
            if pid not in crashing_players and pos in current_positions:
                crashing_players.add(pid)

        # 4. Head-on collisions
        position_to_players = defaultdict(list)
        for pid, pos in desired_moves.items():
            if pid not in crashing_players:
                position_to_players[pos].append(pid)
        for pos, pids in position_to_players.items():
            if len(pids) > 1:
                for pid in pids:
                    crashing_players.add(pid)

        # Move non-crashing players and leave trails
        for pid in living_players:
            if pid not in crashing_players:
                new_pos = desired_moves[pid]
                players[pid]["position"] = new_pos
                old_x, old_y = old_positions[pid]
                board[old_y][old_x] = pid  # Leave trail

        # Update status of crashing players
        for pid in crashing_players:
            players[pid]["alive"] = False
            self.state.game_state["death_turn"][pid] = self.state.turn

        # Check game end conditions
        still_alive = [pid for pid in players if players[pid]["alive"]]
        if len(still_alive) == 0:
            if self.state.num_players > 1:
                max_death_turn = max(self.state.game_state["death_turn"].values())
                winners = [pid for pid, turn in self.state.game_state["death_turn"].items() if turn == max_death_turn]
                if len(winners) == 1:
                    self.state.set_winners(winners, f"Player {winners[0]} outlived all other players.")
                else:
                    self.state.set_draw("Multiple players died simultaneously on the last turn.")
            else:
                self.state.set_draw("All players died simultaneously.")
        elif len(still_alive) == 1:
            winner = still_alive[0]
            self.state.set_winners([winner], f"Player {winner} survived; all other players crashed.")

        # Update board state and log observation
        self.state.game_state["board_state"] = self._get_board_string(board, players)
        message = f"Board after simultaneous moves:\n{self.state.game_state['board_state']}"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)



    def _rotate_players(self):
        """
        Rotate to the next alive player. If the game is done, do nothing.
        """
        if self.state.done:
            return
        current_player_id = self.state.current_player_id
        next_player_id = (current_player_id + 1) % self.state.num_players
        while current_player_id != next_player_id:
            if self.state.game_state["players"][next_player_id]["alive"]:
                self.state.manually_update_current_player(new_player_id=next_player_id)
                break
            else:
                next_player_id = (next_player_id + 1) % self.state.num_players
        else:
            if self.state.game_state["players"][current_player_id]["alive"]:
                self.state.set_winners([current_player_id], f"Player {current_player_id} outlived all other players.")

    def _check_turn_limit(self):
        """
        Check if the turn limit has been reached. If so, set draw or winner.
        """
        if self.state.done:
            return
        if self.state.turn >= self.state.max_turns:
            still_alive = [pid for pid, p in self.state.game_state["players"].items() if p["alive"]]
            if len(still_alive) > 1:
                self.state.set_draw("Turn limit reached with multiple players still alive.")
            elif len(still_alive) == 1:
                self.state.set_winners(still_alive, f"Player {still_alive[0]} survived until turn limit.")