import re
import random
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.LightsOut.renderer import create_board_str

class LightsOutEnv(ta.Env):
    """Lights Out game environment"""

    def __init__(self, grid_size: int = 5, max_turns: int = 100):
        """
        Initialize the Lights Out environment

        Args:
            grid_size (int): The size of the grid (e.g., 5 for a 5x5 grid)
            max_turns (int): The max number of turns
        """
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.action_pattern = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")

    def get_board_str(self):
        return create_board_str(self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the environment to its initial state"""
        if num_players != 1:
            raise ValueError("Lights Out is a single-player game.")

        self.state = ta.State(
            num_players=1,
            min_players=1,
            max_players=1,
            max_turns=self.max_turns,
        )

        # generate a solvable puzzle by starting with an empty board,
        board = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # and pressing a number of random cells. if the board is 
        # solved (all zeros), try again
        while self._check_win(board):
            num_presses = random.randint(self.grid_size, self.grid_size * 2)
            for _ in range(num_presses):
                r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                self._press_cell(board, r, c)

        game_state = {
            "board": board,
            "rendered_board": self._render_board(board)
        }

        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the initial prompt for the player"""
        prompt = (
            f"You are Player {player_id}, playing Lights Out.\n"
            f"The board is a {self.grid_size}x{self.grid_size} grid of lights. '1' means ON, '0' means OFF.\n"
            "The goal is to turn all the lights OFF.\n"
            "On your turn, choose a cell to press. Pressing a cell toggles its state and the state of its "
            "adjacent (up, down, left, right) neighbors.\n"
            "Submit your move as [row, col]. For example, [2, 3] to press the light at row 2, column 3.\n\n"
            "Initial board state:\n"
            f"{game_state['rendered_board']}"
        )
        return prompt

    def _render_board(self, board: List[List[int]]) -> str:
        """Render the current state of the board with aligned column headers"""
        # 4 spaces before the header so column numbers align with cell positions (every 4 characters)
        header = " " * 4 + "   ".join(str(i) for i in range(self.grid_size))
        # horizontal line separator
        separator = "  +" + "---+" * self.grid_size

        lines = [header, separator]
        for r, row in enumerate(board):
            row_str = " | ".join(str(cell) for cell in row)
            lines.append(f"{r} | {row_str} |")
            lines.append(separator)
        
        return "\n".join(lines)

    def _press_cell(self, board: List[List[int]], r: int, c: int):
        """Toggles the light at cell (r, c) and its von Neumann neighbors"""
        for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                board[nr][nc] = 1 - board[nr][nc] # toggle 0 to 1 or 1 to 0

    def _check_win(self, board: List[List[int]]) -> bool:
        """Check if all lights are off"""
        return all(cell == 0 for row in board for cell in row)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process a player's move"""
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        match = self.action_pattern.search(action.strip())
        if not match:
            reason = "Invalid move format. Use [row, col]."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            try:
                row, col = map(int, match.groups())
                if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                    reason = f"Coordinates ({row}, {col}) are out of bounds."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                else:
                    board = self.state.game_state["board"]
                    self._press_cell(board, row, col)
                    self.state.game_state["rendered_board"] = self._render_board(board)

                    message = f"Player {player_id} pressed cell [{row}, {col}].\nNew board:\n{self.state.game_state['rendered_board']}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

                    if self._check_win(board):
                        reason = f"Congratulations! Player {player_id} solved the puzzle!"
                        self.state.set_winners(player_ids=[player_id], reason=reason)

            except ValueError:
                reason = "Invalid coordinates provided."
                self.state.set_invalid_move(player_id=player_id, reason=reason)

        return self.state.step()