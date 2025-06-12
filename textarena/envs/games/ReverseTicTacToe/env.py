import re
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta
from textarena.envs.games.ReverseTicTacToe.renderer import create_board_str

class ReverseTicTacToeEnv(ta.Env):
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.action_space = {i: re.compile(r"\[(\d+)\]") for i in range(2)}

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"board": [['' for _ in range(3)] for _ in range(3)]}, player_prompt_function=self._prompt)
        self._observer_current_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Reverse Tic Tac Toe.\nYour symbol is '{'X' if player_id == 1 else 'O'}'.\n"
            "The goal is to avoid getting three in a row (horizontally, vertically, or diagonally).\n"
            "If you make three in a row, you LOSE.\nSubmit your move using the format '[4]' to place your symbol in cell 4.\n"
            f"As Player {player_id}, you are '{'X' if player_id == 1 else 'O'}' and your opponent is '{'X' if player_id == 0 else 'O'}'.\n"
        )

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def _render_board(self):
        return "\n---+---+---\n".join(
            "|".join(f" {self.state.game_state["board"][r][c]} " if self.state.game_state["board"][r][c] else f" {str(r * 3 + c)} " for c in range(3))
            for r in range(3)
        )

    def _observer_current_state(self):
        available_moves = [f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3) if self.state.game_state["board"][r][c] == '']
        self.state.add_observation(message=f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}")

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        current_symbol = 'X' if self.state.current_player_id == 1 else 'O'
        self.state.add_observation(from_id=self.state.current_player_id, message=action)

        match = self.action_space[self.state.current_player_id].search(action)
        if not match:
            self.state.set_invalid_move(reason="Invalid move format. Use '[cell]'.")
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping: self.state.set_invalid_move(reason="Invalid cell index.")
            else:
                row, col = self.cell_mapping[cell]
                board = self.state.game_state["board"]
                if board[row][col] == '':
                    board[row][col] = current_symbol
                    if self._check_loss(current_symbol): # The current player made 3 in a row => they LOSE => opponent wins
                        self.state.set_winner(player_id=1-self.state.current_player_id, reason=f"Player {self.state.current_player_id} loses by completing a line!")
                    elif all(cell != '' for row in board for cell in row):
                        self.state.set_draw(reason="It's a draw! No one lost.")
                else:
                    self.state.set_invalid_move(reason=f"Cell {cell} is already occupied.")

        self._observer_current_state()
        return self.state.step()

    def _check_loss(self, symbol: str) -> bool:
        board = self.state.game_state["board"]
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == symbol:
                return True
            if board[0][i] == board[1][i] == board[2][i] == symbol:
                return True
        if board[0][0] == board[1][1] == board[2][2] == symbol:
            return True
        if board[0][2] == board[1][1] == board[2][0] == symbol:
            return True
        return False
