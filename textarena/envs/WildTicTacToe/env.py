import re
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.WildTicTacToe.renderer import create_board_str

class WildTicTacToeEnv(ta.Env):
    """ Wild Tic Tac Toe: Players can choose to place either X or O on any turn """
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

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

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
            f"[{mark} {str(r*3+c)}]"
            for r in range(3) for c in range(3)
            if self.state.game_state["board"][r][c] == ''
            for mark in ['X', 'O']
        ]
        message = f"Current Board:\n\n{self._render_board()}\n\nAvailable Moves: {', '.join(available_moves)}"
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        prompt = (
            f"You are Player {player_id} in Wild Tic Tac Toe.\n"
            "On your turn, you can place either an 'X' or an 'O' in an empty square.\n"
            "You win by aligning three of the same mark (X or O) in a row.\n"
            "You can win with either symbol.\n"
            "Choose your move using the format '[X 4]' to place X in the center.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        action_search_pattern = re.compile(r"\[\s*([XO])\s+(\d+)\s*\]", re.IGNORECASE)
        match = action_search_pattern.search(action)

        if match is None:
            reason = f"Invalid move format. Use '[X 4]' or '[O 2]' format."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
        else:
            mark = match.group(1).upper()
            cell = int(match.group(2))

            if cell not in self.cell_mapping:
                reason = f"Invalid cell number: {cell}. Must be between 0 and 8."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            else:
                row, col = self.cell_mapping[cell]
                board = self.state.game_state["board"]

                if board[row][col] == '':
                    board[row][col] = mark

                    if self._check_winner(mark):
                        self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} wins with {mark}s!")
                    elif all(cell != '' for row in board for cell in row):
                        self.state.set_draw(reason="The game is a draw!")
                else:
                    reason = f"Invalid move. Cell {cell} is already occupied."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)

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
