from typing import Optional, Dict, Tuple, List, Any
import random
import textarena as ta
import re

class BattleshipEnv(ta.Env):
    """
    Environment for Battleship game.
    """
    def __init__(
        self,
        difficulty: Optional[str] = "easy",
    ):
        """
        Initialize the Battleship environment.
        
        Args:
            difficulty (str): Difficulty level of the game. Possible values are "easy", "medium", and "hard".
        """
        self.difficulty = difficulty

        ## initalize the grid size based on the difficulty level
        ## TODO - should number of boats be based on difficulty level as well?
        if self.difficulty == "easy":
            self.grid_size = 10
        elif self.difficulty == "medium":
            self.grid_size = 12
        elif self.difficulty == "hard":
            self.grid_size = 14

        ## Initialize the game state
        self.state = ta.State(
            num_players=2,
            max_turns=None,
        )
        self.ships = {"Aircraft Carrier": 5, "Battleship": 4, "Submarine": 3, "Destroyer": 3, "Patrol Boat": 2}


        # add render object
        self.board_state_render = ta.envs.two_player.Battleship.render.GameStateRender
        self.render_keys = ["rendered_board"]

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment to start a new game.
        
        Args:
            seed (int): Seed for the random number generator.
        """
        
        ## Initialize the board
        self.ships = {"Aircraft Carrier": 5, "Battleship": 4, "Submarine": 3, "Destroyer": 3, "Patrol Boat": 2}
        self.board, self.tracking_board = self._generate_board()

        ## initialize the game state
        self.state.reset(
            seed=seed,
            game_state={
                "board": self.board,
                "rendered_board": self._render_board(),
            },
            player_prompt_function=self._generate_player_prompt
        )
    
    def _generate_board(self) -> List[List[str]]:
        """
        Generate a new grid, tracking grid, and place ships on the grid for both players,
        where each entity is a dictionary with the player_ids as the keys.
        """
        board = {
            0: [['~'] * self.grid_size for _ in range(self.grid_size)],
            1: [['~'] * self.grid_size for _ in range(self.grid_size)],
        }
        tracking_board = {
            0: [['~'] * self.grid_size for _ in range(self.grid_size)],
            1: [['~'] * self.grid_size for _ in range(self.grid_size)],
        }

        ## place ships on the board for both players
        for player_id in range(2):
            for ship_name, length in self.ships.items():
                self._place_ship_on_board(board[player_id], ship_name, length)

        return board, tracking_board
    
    def _place_ship_on_board(self, grid: List[List[str]], ship_name: str, length: int) -> None:
        """
        Place a ship on the board.
        
        Args:
            grid (List[List[str]]): The grid to place the ship on.
            ship_name (str): The name of the ship.
            length (int): The length of the ship.
        """
        placed = False
        while not placed:
            orientation = random.choice(["horizontal", "vertical"])
            if orientation == "horizontal":
                row, col = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - length)
                if all(grid[row][col + i] == '~' for i in range(length)):
                    for i in range(length):
                        grid[row][col + i] = ship_name[0]
                    placed = True
            else:  ## vertical
                row, col = random.randint(0, self.grid_size - length), random.randint(0, self.grid_size - 1)
                if all(grid[row + i][col] == '~' for i in range(length)):
                    for i in range(length):
                        grid[row + i][col] = ship_name[0]
                    placed = True

    def _render_board(self) -> str:
        """
        Render the board for both players.
        
        Returns:
            str: The rendered board.
        """
        # Prepare header for both players and column numbers
        view = []
        view.append("   " + "Player 0's Ships".center(self.grid_size * 3) + "        " + "Player 1's Ships".center(self.grid_size * 3))
        view.append("   " + " ".join([f"{i:2}" for i in range(self.grid_size)]) + "      " +
                    "   " + " ".join([f"{i:2}" for i in range(self.grid_size)]))
        
        for i in range(self.grid_size):
            # Row labels (letters) and grid display for both players' grids
            row_label = chr(i + ord('A'))
            row_player1 = " ".join(f"{cell:2}" for cell in self.board[0][i])
            row_player2 = " ".join(f"{cell:2}" for cell in self.board[1][i])
            view.append(f"{row_label}   {row_player1}     {row_label}   {row_player2}")
        
        # Join all lines into a single string with newlines
        return "\n".join(view)
    
    def _render_player_view(self, player_id: int) -> str:
        """
        Render the player's view of the game.
        
        Returns:
            str: The rendered player view.
        """
        # Determine which player's view to return
        if player_id == 0:
            own_grid = self.board[0]
            tracking_grid = self.tracking_board[0]
            player_label = "Player 0"
        else:
            own_grid = self.board[1]
            tracking_grid = self.tracking_board[1]
            player_label = "Player 1"
        
        # Prepare header with Player ID and column numbers
        view = []
        view.append(f"\n{player_label}'s View".center(self.grid_size * 4 + 15))
        view.append("   " + "Your Ships".center(self.grid_size * 3) + "        " + "Your Hits on Opponent".center(self.grid_size * 3))
        view.append("   " + " ".join([f"{i:2}" for i in range(self.grid_size)]) + "      " +
                    "   " + " ".join([f"{i:2}" for i in range(self.grid_size)]))
        
        for i in range(self.grid_size):
            # Row labels (letters) and grid display for both player's ships and tracking grid
            row_label = chr(i + ord('A'))
            row_own_grid = " ".join(f"{cell:2}" for cell in own_grid[i])
            row_tracking_grid = " ".join(f"{cell:2}" for cell in tracking_grid[i])
            view.append(f"{row_label}   {row_own_grid}     {row_label}   {row_tracking_grid}")
        
        # Join all lines into a single string with newlines
        return "\n".join(view)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the player prompt.
        
        Args:
            player_id (int): ID of the player.
        
        Returns:
            str: Player prompt.
        """
        prompt = (
            f"You are Player {player_id}. You are playing the Battleship game ({self.difficulty} level).\n"
            "Your goal is to sink all of your opponent's ships before they sink yours.\n"
            "On your turn, consider the observations made by your opponent, but do not repeat them exactly.\n"
            "Focus on new insights or strategies based on your understanding of the opponent's moves and the current state of the game.\n"
            "You may mention coordinates in the format B3 or C8. Only when you have decided to fire a missile to a specified coordinate, then you must enter the row and column values in square brackets like [A4]. This is to avoid submitting a wrong coordinate to the game environment.\n"
            "If the missile hits a ship, it is marked with 'X'. If it misses, it is marked with 'O'. In either scenarios, the game environment will inform you of your hits. If you have sunk a boat, the game environment will tell you!\n"
            "The game ends when all of one player's ships have been sunk.\n"
            "Your initial board will show all of your ships placed and your opponent's hits on you, and your hits and misses on your opponent's board without showing your opponent's ships.\n"
            "Here is the initial board:\n"
        )


        prompt += self._render_player_view(player_id)

        return prompt
    
    def step(self, action: str) -> Tuple[Optional[ta.Rewards], bool, bool, ta.Info]: 
        """
        Take a step in the environment.
        
        Args:
            action (str): Action taken by the player.
        
        Returns:
            Rewards: Rewards for the players.
            bool: Whether the game is over.
            bool: Whether the turn is over.
            Info: Additional information.
        """
        player_id = self.state.current_player_id
        ## validate the action
        self.state.check_action_format(
            action=action,
        )

        ## update the observations
        self.state.add_observation(
            from_id=player_id,
            to_id=player_id, ## send the observation to the player who took the action, since this is a private observation
            message=action,
            for_logging=True
        )

        ## action search pattern
        action_search_pattern = re.compile(r"\[([A-Z])(\d+)\]") # e.g. [A4]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid move format. Player {player_id} did not respond with a valid coordinate in square brackets."]
            )
        
        else:
            row = ord(match.group(1)) - ord('A') # convert letter to row index
            col = int(match.group(2))
            
            opponent_id = 1 - player_id
            opponent_board = self.board[opponent_id]
            tracking_board = self.tracking_board[player_id]

            ## check if the move is valid
            if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
                self.state.set_invalid_move(
                    player_ids=[player_id],
                    reasons=[f"Invalid move. The coordinate {match.group()[1:3]} is outside the board."]
                )
            elif tracking_board[row][col] != '~':
                self.state.set_invalid_move(
                    player_ids=[player_id],
                    reasons=[f"Invalid move. The coordinate {match.group()[1:3]} has already been fired upon."]
                )
            else:
                if opponent_board[row][col] != '~':
                    tracking_board[row][col] = 'X'
                    ship_initial = opponent_board[row][col]
                    opponent_board[row][col] = 'X'
                    if not any(ship_initial in row for row in opponent_board):
                        self.state.add_observation(
                            from_id=ta.GAME_ID,
                            to_id=player_id,
                            message=f"Sunk! You sunk a ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=player_id)}",
                            for_logging=False
                        )
                        self.state.add_observation(
                            from_id=ta.GAME_ID,
                            to_id=opponent_id,
                            message=f"Opponent sunk your ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=opponent_id)}",
                            for_logging=False
                        )
                    else:
                        self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=f"Hit! You hit a ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=player_id)}",
                            for_logging=False
                        )
                        self.state.add_observation(
                            from_id=-1,
                            to_id=opponent_id,
                            message=f"Opponent hit your ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=opponent_id)}",
                            for_logging=False
                        )
                else:
                    tracking_board[row][col] = 'O'
                    opponent_board[row][col] = 'O'
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=f"Miss! You missed the ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=player_id)}",
                        for_logging=False
                    )
                    self.state.add_observation(
                        from_id=-1,
                        to_id=opponent_id,
                        message=f"Opponent missed your ship at {match.group()[1:3]}! Your updated board:\n{self._render_player_view(player_id=opponent_id)}",
                        for_logging=False
                    )
            
            ## check if the game is over
            if self._check_win(player_id):
                self.state.set_winners(
                    player_ids=[player_id],
                    reason=[f"Player {player_id} has sunk all of their opponent's ships!"]
                )

        ## update the rendered board
        self.state.game_state["rendered_board"] = self._render_board()

        return self.state.step()
    
    def _check_win(self, player_id: int) -> bool:
        """
        Check if the game is over.
        
        Args:
            player_id (int): ID of the player.
        
        Returns:
            bool: Whether the game is over.
        """
        opponent_id = 1 - player_id
        opponent_board = self.board[opponent_id]
        abbreviations = {name[0] for name in self.ships.keys()}
        return not any(any(cell in abbreviations for cell in row) for row in opponent_board)
    
    def render(self) -> str:
        """
        Render the environment.
        
        Returns:
            str: Rendered environment.
        """
        return self.state.game_state["rendered_board"]
