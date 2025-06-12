import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.games.ThreePlayerTicTacToe.renderer import create_board_str

class ThreePlayerTicTacToeEnv(ta.Env):
    """ A Tic Tac Toe variant for three players on a 5x5 board. """
    def __init__(self):
        super().__init__()
        self.board_size = 5
        self.cell_mapping = {i * self.board_size + j: (i, j) for i in range(self.board_size) for j in range(self.board_size)}
        self.symbols = {0: 'A', 1: 'B', 2: 'C'}
        self.action_space = {i: re.compile(r"\[(\d+)\]") for i in range(3)}

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
        
    def reset(self, num_players: int, seed: Optional[int] = None):
        assert num_players==3, f"ThreePlayerTicTacToe only works with exactly three players. {num_players} players were provided."
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"board": [['' for _ in range(self.board_size)] for _ in range(self.board_size)]}, player_prompt_function=self._prompt)
        self._observer_current_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Three-Player Tic Tac Toe.\nYour symbol is '{self.symbols[player_id]}'.\n"
            "You take turns placing your symbol on a 5x5 board to form a line of four.\nLines can be horizontal, vertical, or diagonal.\n"
            "Submit your move using the format '[4]' to mark cell 4.\n"
        )
    
    def _render_board(self) -> str:
        """ Render the board with uniform cell widths, including framing lines. Cells show either the symbol ('A', 'B', 'C') or the cell index """
        cell_width = max(len(str(self.board_size * self.board_size - 1)), 2)  # ensures symbols like 'A', 'B', 'C' are well-centered
        def cell_str(r: int, c: int) -> str:
            return self.state.game_state["board"][r][c] if self.state.game_state["board"][r][c] != '' else str(r * self.board_size + c)

        def build_hline() -> str:
            return "+" + "+".join("-" * (cell_width + 2) for _ in range(self.board_size)) + "+"

        lines = [build_hline()]
        for r in range(self.board_size):
            row_cells = [f" {cell_str(r, c):^{cell_width}} " for c in range(self.board_size)]
            row_line = "|" + "|".join(row_cells) + "|"
            lines.append(row_line)
            lines.append(build_hline())

        return "\n".join(lines)

    def _observer_current_state(self) -> None:
        """ Broadcast the current state of the board to all players, listing empty cells as possible moves """
        available_moves = []
        for i in range(self.board_size * self.board_size):
            r, c = self.cell_mapping[i]
            if self.state.game_state["board"][r][c] == '':
                available_moves.append(f"[{i}]")
        self.state.add_observation(message=f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: " + ", ".join(available_moves), observation_type=ta.ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        match = self.action_space[player_id].search(action)
        if not match:
            if self.state.set_invalid_move(reason=f"Invalid move format. Use '[cell]' where cell is 0-{self.board_size ** 2 - 1}."):
                self.state.set_winners(player_ids=[pid for pid in range(3) if pid!=player_id], reason=f"Player {player_id} made an invalid move")
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping:
                if self.state.set_invalid_move(reason=f"Invalid cell number: {cell}. Must be between 0 and {self.board_size ** 2 - 1}."):
                    self.state.set_winners(player_ids=[pid for pid in range(3) if pid!=player_id], reason=f"Player {player_id} made an invalid move")
            else:
                row, col = self.cell_mapping[cell]
                if self.state.game_state["board"][row][col] == '':
                    self.state.game_state["board"][row][col] = self.symbols[player_id]
                    self.state.add_observation(message=f"Player {player_id} places their symbol ({self.symbols[player_id]}) in field ({row}, {col}).", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
                    if self._check_winner(self.symbols[player_id]): 
                        self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} ({self.symbols[player_id]}) wins!")
                    elif all(cell != '' for row in self.state.game_state["board"] for cell in row):
                        self.state.set_draw(reason="The game is a draw!")
                else:
                    if self.state.set_invalid_move(reason=f"Cell {cell} is already occupied."):
                        self.state.set_winners(player_ids=[pid for pid in range(3) if pid!=player_id], reason=f"Player {player_id} made an invalid move")

        self._observer_current_state()
        return self.state.step()

    def _check_winner(self, symbol: str) -> bool:
        win_length = 4
        # Horizontal & Vertical
        for r in range(self.board_size):
            for c in range(self.board_size - win_length + 1):
                if all(self.state.game_state["board"][r][c + i] == symbol for i in range(win_length)): return True
        for c in range(self.board_size):
            for r in range(self.board_size - win_length + 1):
                if all(self.state.game_state["board"][r + i][c] == symbol for i in range(win_length)): return True
        # Diagonal \ and /
        for r in range(self.board_size - win_length + 1):
            for c in range(self.board_size - win_length + 1):
                if all(self.state.game_state["board"][r + i][c + i] == symbol for i in range(win_length)): return True
                if all(self.state.game_state["board"][r + i][c + win_length - 1 - i] == symbol for i in range(win_length)): return True
        return False