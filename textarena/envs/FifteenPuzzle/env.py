import re, random
from typing import Any, Dict, List, Tuple, Optional, Union, Literal

import textarena as ta
from textarena.envs.FifteenPuzzle.renderer import create_board_str
from textarena.envs.FifteenPuzzle.player_prompts import player_prompts
from textarena.envs.FifteenPuzzle.utils import generate_puzzle

class FifteenPuzzleEnv(ta.Env):
    """ Fifteen Puzzle environment """
    def __init__(self, lang : str = 'en', difficulty: Literal['easy', 'medium', 'hard'] = 'easy'):
        """ Initialize the Fifteen Puzzle environment """
        super().__init__()
        if lang not in player_prompts.keys():
            raise "Language Not Supported"
        self.lang = lang
        self.difficulty = difficulty

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment to its initial state """
        self.board = self._generate_board() ## initialize the game state
        self.state = ta.SinglePlayerState(num_players=num_players, seed=seed, max_turns=self.max_turns) ## initialize the game state
        self.initial_board = [row[:] for row in self.board]  # Deep copy of the initial board
        game_state = {"board": self.board, "rendered_board": self._render_board(self.board)} ## reset the game state
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        self._observe_current_state()  # Observe the initial state of the game

    def _observe_current_state(self) -> None:
        """Send current board and legal moves as observation."""
        r, c = self._get_empty_position()
        moves = {"[up]": r < 3, "[down]": r > 0, "[left]": c < 3, "[right]": c > 0}
        legal_moves = [m for m, valid in moves.items() if valid]
        msg = f"Current Board:\n\n{self.state.game_state['rendered_board']}\nAvailable Moves: {', '.join(legal_moves)}"
        self.state.add_observation(message=msg, observation_type=ta.ObservationType.GAME_BOARD)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        return player_prompts[self.lang].format(player_id=player_id)
    
    def _generate_board(self):
        """ Generate a shuffled board configuration """
        n_swaps = None
        if self.difficulty == 'easy':
            n_swaps = random.choice([i for i in range(1, 6)])
        elif self.difficulty == 'medium':
            n_swaps = random.choice([i for i in range(6, 11)])
        elif self.difficulty == 'hard':
            n_swaps = random.choice([i for i in range(11, 21)])
        self.max_turns = n_swaps * 20
        return generate_puzzle(n_swaps)
    
    def _render_board(self, board):
        """ Render the current board layout """
        rendered_board = ""
        for row in board:
            rendered_board += ' '.join(['__' if x is None else f"{x:2}" for x in row]) + "\n"
        return rendered_board
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action and update the environment state """
        player_id = self.state.current_player_id
        self.state.add_observation(from_id=player_id, to_id=-1, message=action, observation_type=ta.ObservationType.PLAYER_ACTION) ## add the action to the game state
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]") # e.g. [up]
        match = action_search_pattern.search(action)

        if match is None:
            reason=f"Invalid move format. Player {player_id} did not respond with a valid direction in square brackets."
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=reason)

        else:
            direction = match.group(1)
            if not self._move(direction):
                reason=f"Invalid move. The tile cannot be moved in the specified direction."
                self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=reason)

            else:
                self.state.game_state["rendered_board"] = self._render_board(self.board) ## update the rendered board
                message=f"Game Board:\n{self._render_board(self.board)}"
                self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_BOARD)
            
        if self._is_solved(): ## check if the puzzle is solved
            reason=f"Congratulations! Player {player_id} have successfully solved the 15-Puzzle."
            self.state.set_winners(player_ids=[player_id], reason=reason)
        elif self.state.check_turn_limit():
            pct_completion = self._get_percentage_completion()
            reason=f"The turn limit has been reached. The model completed {pct_completion*100} percent of the puzzle"
            self.state.set_outcome(reward=pct_completion, reason=reason)
        self._observe_current_state()  # Observe the new state after the move
        return self.state.step()
    
    def _is_solved(self) -> bool:
        """ Check if the board is in a solved state """
        correct_tiles = list(range(1, 16)) + [None]
        current_tiles = [tile for row in self.board for tile in row]
        return current_tiles == correct_tiles

    def _move(self, direction: str) -> bool:
        """ Move a tile into the empty space if the direction is valid """
        empty_row, empty_col = self._get_empty_position()
        target_row, target_col = empty_row, empty_col

        if direction == 'up' and empty_row < 3:         target_row += 1
        elif direction == 'down' and empty_row > 0:     target_row -= 1
        elif direction == 'left' and empty_col < 3:     target_col += 1
        elif direction == 'right' and empty_col > 0:    target_col -= 1
        else:                                           return False ## invalid move

        ## swap the target tile with the empty tile
        self.board[empty_row][empty_col], self.board[target_row][target_col] = (self.board[target_row][target_col], self.board[empty_row][empty_col])
        return True
    
    def _get_empty_position(self):
        for r in range(4):
            for c in range(4):
                if self.board[r][c] is None:
                    return r, c

    def _get_percentage_completion(self) -> float:
        goal = list(range(1, 16)) + [None]
        correct = 0
        total = 0
        # Flatten all 3 boards for easier comparison
        flat_current = [tile for row in self.board for tile in row]
        flat_initial = [tile for row in self.initial_board for tile in row]
        for idx, goal_tile in enumerate(goal):
            if flat_initial[idx] == goal_tile: continue  # Skip tiles that were already in the right place initially
            total += 1
            if flat_current[idx] == goal_tile:
                correct += 1
        return correct / total if total > 0 else 0.0
