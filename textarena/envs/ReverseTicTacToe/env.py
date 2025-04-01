import re
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta

class ReverseTicTacToeEnv(ta.Env):
    """ Reverse Tic Tac Toe: Lose if you get 3 in a row. """
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.State(num_players=2, min_players=2, max_players=2)
        self.state.reset(
            game_state={"board": [['' for _ in range(3)] for _ in range(3)]},
            player_prompt_function=self._generate_player_prompt
        )
        self._observer_current_state()

    def _render_board(self):
        board = self.state.game_state["board"]
        return "\n---+---+---\n".join(
            "|".join(
                f" {board[r][c]} " if board[r][c] else f" {str(r * 3 + c)} "
                for c in range(3)
            )
            for r in range(3)
        )

    def _observer_current_state(self):
        available_moves = [
            f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3)
            if self.state.game_state["board"][r][c] == ''
        ]
        message = f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        symbol = 'X' if player_id == 1 else 'O'
        opp_symbol = 'O' if symbol == 'X' else 'X'
        prompt = (
            f"You are Player {player_id} in Reverse Tic Tac Toe.\n"
            f"Your symbol is '{symbol}'.\n"
            "The goal is to avoid getting three in a row (horizontally, vertically, or diagonally).\n"
            "If you make three in a row, you LOSE.\n"
            "Submit your move using the format '[4]' to place your symbol in cell 4.\n"
            f"As Player {player_id}, you are '{symbol}' and your opponent is '{opp_symbol}'.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        current_symbol = 'X' if player_id == 1 else 'O'
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        match = re.search(r"\[(\d+)\]", action)
        if not match:
            self.state.set_invalid_move(player_id=player_id, reason="Invalid move format. Use '[cell]'.")
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping:
                self.state.set_invalid_move(player_id=player_id, reason="Invalid cell index.")
            else:
                row, col = self.cell_mapping[cell]
                board = self.state.game_state["board"]
                if board[row][col] == '':
                    board[row][col] = current_symbol

                    if self._check_loss(current_symbol):
                        # The current player made 3 in a row => they LOSE => opponent wins
                        opponent_id = 1 - player_id
                        self.state.set_winners(player_ids=[opponent_id], reason=f"Player {player_id} loses by completing a line!")
                    elif all(cell != '' for row in board for cell in row):
                        self.state.set_draw(reason="It's a draw! No one lost.")
                else:
                    self.state.set_invalid_move(player_id=player_id, reason=f"Cell {cell} is already occupied.")

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
