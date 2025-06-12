import re, random
from typing import Optional, Dict, Tuple, Any

import textarena as ta
from textarena.envs.games.RandomizedTicTacToe.renderer import create_board_str

class RandomizedTicTacToeEnv(ta.Env):
    """ Tic Tac Toe with random effects each turn. """
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.effects = ["swap", "block", "double", "wild"]
        self.action_space = {i: re.compile(r"\[(\d+)\]") for i in range(2)}

    def get_board_str(self):
        return create_board_str(board=self.state.game_state["board"])

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        self.state.reset(game_state={"board": [['' for _ in range(3)] for _ in range(3)], "current_effect": None}, player_prompt_function=self._prompt)
        self._assign_random_effect(); self._observer_current_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Randomized Tic Tac Toe.\n"
            f"Your symbol is '{'X' if player_id == 1 else 'O'}'.\n"
            "Each turn, a random effect will apply to the game, modifying your strategy.\n"
            "The effects include:\n"
            "- SWAP: Two filled cells will be swapped at random.\n"
            "- BLOCK: A random empty cell will be permanently blocked with a '#'.\n"
            "- DOUBLE: You will get to play two turns in a row.\n"
            "- WILD: Your symbol will be randomly chosen as 'X' or 'O' for this move.\n"
            "Your goal is to get three in a row (horizontally, vertically, or diagonally).\n"
            "Submit your move using '[cell]'. For example, '[4]' places your symbol in the center.\n"
            f"Your opponent is '{'X' if player_id == 0 else 'O'}'.\n"
        )

    def _assign_random_effect(self):
        self.state.game_state["current_effect"] = random.choice(self.effects)

    def _render_board(self):
        return "\n---+---+---\n".join(
            "|".join(f" {self.state.game_state["board"][r][c]} " if self.state.game_state["board"][r][c] else f" {str(r * 3 + c)} " for c in range(3))
            for r in range(3)
        )

    def _observer_current_state(self):
        available_moves = [f"'[{str(r*3+c)}]'" for r in range(3) for c in range(3) if self.state.game_state["board"][r][c] == '']
        message = f"Current Board:\n\n{self._render_board()}\n\nCurrent Effect: {self.state.game_state['current_effect'].upper()}\nAvailable Moves: {', '.join(available_moves)}"
        self.state.add_observation(message=message)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        rotate_player = True
        symbol = 'X' if self.state.current_player_id == 1 else 'O'
        self.state.add_observation(from_id=self.state.current_player_id, message=action)

        match = self.action_space[self.state.current_player_id].search(action)
        if not match:
            self.state.set_invalid_move(reason="Invalid format. Use '[cell]'.")
        else:
            cell = int(match.group(1))
            if cell not in self.cell_mapping:
                self.state.set_invalid_move(reason="Invalid cell index.")
            else:
                row, col = self.cell_mapping[cell]
                if self.state.game_state["board"][row][col] == '':
                    effect = self.state.game_state["current_effect"] # Apply effect before the move
                    if effect == "block": self._block_random_cell() # Randomly block a cell before move
                    elif effect == "swap": self._swap_random_cells() # Randomly swap two filled cells
                    elif effect == "wild": symbol = random.choice(['X', 'O']) # Change current player's mark randomly (still their move)
                    elif effect == "double": rotate_player=False # Let them play two moves (second move requested next round) block player rotation
                    self.state.game_state["board"][row][col] = symbol
                    if self._check_winner(symbol):
                        self.state.set_winner(player_id=self.state.current_player_id, reason=f"Player {self.state.current_player_id} wins!")
                    elif all(cell != '' for row in self.state.game_state["board"] for cell in row):
                        self.state.set_draw(reason="It's a draw!")
                else:
                    self.state.set_invalid_move(reason=f"Cell {cell} already filled.")
        self._assign_random_effect(); self._observer_current_state()
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