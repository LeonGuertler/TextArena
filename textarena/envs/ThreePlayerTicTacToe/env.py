import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.ThreePlayerTicTacToe.renderer import create_board_str

class ThreePlayerTicTacToeEnv(ta.Env):
    """ A Tic Tac Toe variant for three players on a 5x5 board. """
    def __init__(self):
        super().__init__()
        self.board_size = 5
        self.cell_mapping = {i * self.board_size + j: (i, j) for i in range(self.board_size) for j in range(self.board_size)}
        self.symbols = {0: 'A', 1: 'B', 2: 'C'}

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
        
    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 3:
            raise ValueError("ThreePlayerTicTacToeEnv requires exactly 3 players.")
        self.state = ta.State(num_players=3, min_players=3, max_players=3)
        self.state.reset(
            game_state={"board": [['' for _ in range(self.board_size)] for _ in range(self.board_size)]},
            player_prompt_function=self._generate_player_prompt
        )
        self._observer_current_state()

    def _render_board(self) -> str:
        """
        Render the board with uniform cell widths, including framing lines.
        Cells show either the symbol ('A', 'B', 'C') or the cell index.
        """

        board = self.state.game_state["board"]
        max_cell_num = self.board_size * self.board_size - 1
        digit_count = len(str(max_cell_num))
        cell_width = max(digit_count, 2)  # ensures symbols like 'A', 'B', 'C' are well-centered

        def cell_str(r: int, c: int) -> str:
            return board[r][c] if board[r][c] != '' else str(r * self.board_size + c)

        def build_hline() -> str:
            return "+" + "+".join("-" * (cell_width + 2) for _ in range(self.board_size)) + "+"

        lines = [build_hline()]
        for r in range(self.board_size):
            row_cells = [
                f" {cell_str(r, c):^{cell_width}} " for c in range(self.board_size)
            ]
            row_line = "|" + "|".join(row_cells) + "|"
            lines.append(row_line)
            lines.append(build_hline())

        return "\n".join(lines)

    def _observer_current_state(self) -> None:
        """
        Broadcast the current state of the board to all players,
        listing empty cells as possible moves.
        """
        board = self.state.game_state["board"]
        board_str = self._render_board()

        available_moves = []
        for i in range(self.board_size * self.board_size):
            r, c = self.cell_mapping[i]
            if board[r][c] == '':
                available_moves.append(f"[{i}]")

        message = (
            "Current Board:\n\n"
            f"{board_str}\n"
            "\nAvailable Moves: " + ", ".join(available_moves)
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        symbol = self.symbols[player_id]
        prompt = (
            f"You are Player {player_id} in Three-Player Tic Tac Toe.\n"
            f"Your symbol is '{symbol}'.\n"
            "You take turns placing your symbol on a 5x5 board to form a line of four.\n"
            "Lines can be horizontal, vertical, or diagonal.\n"
            "Submit your move using the format '[4]' to mark cell 4.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        symbol = self.symbols[player_id]

        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        match = re.search(r"\[(\d+)\]", action)
        if not match:
            reason = f"Invalid move format. Use '[cell]' where cell is 0-{self.board_size ** 2 - 1}."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping:
                reason = f"Invalid cell number: {cell}. Must be between 0 and {self.board_size ** 2 - 1}."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                row, col = self.cell_mapping[cell]
                board = self.state.game_state["board"]

                if board[row][col] == '':
                    board[row][col] = symbol
                    if self._check_winner(symbol):
                        self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} ({symbol}) wins!")
                    elif all(cell != '' for row in board for cell in row):
                        self.state.set_draw(reason="The game is a draw!")
                else:
                    reason = f"Cell {cell} is already occupied."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

        self._observer_current_state()
        return self.state.step()

    def _check_winner(self, symbol: str) -> bool:
        board = self.state.game_state["board"]
        size = self.board_size
        win_length = 4

        # Horizontal & Vertical
        for r in range(size):
            for c in range(size - win_length + 1):
                if all(board[r][c + i] == symbol for i in range(win_length)):
                    return True
        for c in range(size):
            for r in range(size - win_length + 1):
                if all(board[r + i][c] == symbol for i in range(win_length)):
                    return True

        # Diagonal \ and /
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                if all(board[r + i][c + i] == symbol for i in range(win_length)):
                    return True
                if all(board[r + i][c + win_length - 1 - i] == symbol for i in range(win_length)):
                    return True

        return False