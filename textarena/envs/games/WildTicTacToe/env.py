import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.games.WildTicTacToe.renderer import create_board_str

class WildTicTacToeEnv(ta.Env):
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.action_space = lambda player_id: re.compile(r"\[\s*([XO])\s+(\d+)\s*\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"board": [['' for _ in range(3)] for _ in range(3)]}, player_prompt_function=self._prompt)
        self._observer_current_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Wild Tic Tac Toe.\nOn your turn, you can place either an 'X' or an 'O' in an empty square.\n"
            "You win by aligning three of the same mark (X or O) in a row.\nYou can win with either symbol.\n"
            "Choose your move using the format '[X 4]' to place X in the center.\n"
        )
        
    def _render_board(self):
        board = self.state.game_state["board"]
        return "\n---+---+---\n".join("|".join(f" {board[r][c]} " if board[r][c] else f" {str(r * 3 + c)} " for c in range(3)) for r in range(3))

    def _observer_current_state(self):
        available_moves = [f"[{mark} {str(r*3+c)}]" for r in range(3) for c in range(3) if self.state.game_state["board"][r][c] == '' for mark in ['X', 'O']]
        self.state.add_observation(message=f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}")

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action)
        match = self.action_space(self.state.current_player_id).search(action)

        if match is None:
            self.state.set_invalid_move(reason=f"Invalid move format. Use '[X 4]' or '[O 2]' format.")
        else:
            mark = match.group(1).upper()
            cell = int(match.group(2))
            if cell not in self.cell_mapping:
                self.state.set_invalid_move(reason=f"Invalid cell number: {cell}. Must be between 0 and 8.")
            else:
                row, col = self.cell_mapping[cell]
                if self.state.game_state["board"][row][col] == '':
                    self.state.game_state["board"][row][col] = mark
                    if self._check_winner(mark): self.state.set_winner(player_id=self.state.current_player_id, reason=f"Player {self.state.current_player_id} wins with {mark}s!")
                    elif all(cell != '' for row in self.state.game_state["board"] for cell in row): self.state.set_draw(reason="The game is a draw!")
                else: self.state.set_invalid_move(reason=f"Invalid move. Cell {cell} is already occupied.")
        self._observer_current_state()
        return self.state.step()

    def _check_winner(self, mark: str) -> bool:
        board = self.state.game_state["board"]
        for i in range(3):
            if all(board[i][j] == mark for j in range(3)) or all(board[j][i] == mark for j in range(3)):
                return True
        if all(board[i][i] == mark for i in range(3)) or all(board[i][2 - i] == mark for i in range(3)):
            return True
        return False
