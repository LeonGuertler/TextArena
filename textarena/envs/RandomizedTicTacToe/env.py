import re, random
from typing import Optional, Dict, Tuple, Any

import textarena as ta

class RandomizedTicTacToeEnv(ta.Env):
    """ Tic Tac Toe with random effects each turn. """
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.effects = ["swap", "block", "double", "wild"]

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.State(num_players=2, min_players=2, max_players=2)
        self.state.reset(
            game_state={
                "board": [['' for _ in range(3)] for _ in range(3)],
                "current_effect": None
            },
            player_prompt_function=self._generate_player_prompt
        )
        self._assign_random_effect()
        self._observer_current_state()

    def _assign_random_effect(self):
        effect = random.choice(self.effects)
        self.state.game_state["current_effect"] = effect

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
        board = self.state.game_state["board"]
        available_moves = [
            f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3)
            if board[r][c] == ''
        ]
        effect = self.state.game_state["current_effect"]
        message = (
            f"Current Board:\n\n{self._render_board()}\n\n"
            f"Current Effect: {effect.upper()}\n"
            f"Available Moves: {', '.join(available_moves)}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        symbol = 'X' if player_id == 1 else 'O'
        opp_symbol = 'O' if symbol == 'X' else 'X'
        return (
            f"You are Player {player_id} in Randomized Tic Tac Toe.\n"
            f"Your symbol is '{symbol}'.\n"
            "Each turn, a random effect will apply to the game, modifying your strategy.\n"
            "The effects include:\n"
            "- SWAP: Two filled cells will be swapped at random.\n"
            "- BLOCK: A random empty cell will be permanently blocked with a '#'.\n"
            "- DOUBLE: You will get to play two turns in a row.\n"
            "- WILD: Your symbol will be randomly chosen as 'X' or 'O' for this move.\n"
            "Your goal is to get three in a row (horizontally, vertically, or diagonally).\n"
            "Submit your move using '[cell]'. For example, '[4]' places your symbol in the center.\n"
            f"Your opponent is '{opp_symbol}'.\n"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        rotate_player = True
        symbol = 'X' if player_id == 1 else 'O'
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        match = re.search(r"\[(\d+)\]", action)
        if not match:
            self.state.set_invalid_move(player_id=player_id, reason="Invalid format. Use '[cell]'.")
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping:
                self.state.set_invalid_move(player_id=player_id, reason="Invalid cell index.")
            else:
                row, col = self.cell_mapping[cell]
                board = self.state.game_state["board"]
                if board[row][col] == '':
                    # Apply effect before the move
                    effect = self.state.game_state["current_effect"]
                    if effect == "block":
                        # Randomly block a cell before move
                        self._block_random_cell()
                    elif effect == "swap":
                        # Randomly swap two filled cells
                        self._swap_random_cells()
                    elif effect == "wild":
                        # Change current player's mark randomly (still their move)
                        symbol = random.choice(['X', 'O'])
                    elif effect == "double":
                        # Let them play two moves (second move requested next round)
                        #block player rotation
                        rotate_player=False

                    board[row][col] = symbol

                    if self._check_winner(symbol):
                        self.state.set_winners(player_ids=[player_id], reason=f"Player {player_id} wins!")
                    elif all(cell != '' for row in board for cell in row):
                        self.state.set_draw(reason="It's a draw!")
                else:
                    self.state.set_invalid_move(player_id=player_id, reason=f"Cell {cell} already filled.")

        self._assign_random_effect()
        self._observer_current_state()
        return self.state.step(rotate_player=rotate_player)

    def _block_random_cell(self):
        board = self.state.game_state["board"]
        empties = [(r, c) for r in range(3) for c in range(3) if board[r][c] == '']
        if empties:
            r, c = random.choice(empties)
            board[r][c] = '#'

    def _swap_random_cells(self):
        board = self.state.game_state["board"]
        filled = [(r, c) for r in range(3) for c in range(3) if board[r][c] in ['X', 'O']]
        if len(filled) >= 2:
            (r1, c1), (r2, c2) = random.sample(filled, 2)
            board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]

    def _check_winner(self, symbol: str) -> bool:
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