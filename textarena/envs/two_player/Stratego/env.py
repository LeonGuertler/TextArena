from typing import Optional, Dict, Tuple, List, Any
import random
import textarena as ta
import re

class StrategoEnv(ta.Env):
    """
    A two-player implementation of the board game Stratego.
    """
    def __init__(
        self
    ):
        """
        Initialize the environment.
        """
        self.environment_name = "Stratego"

        ## initialise the state
        self.state = ta.State(
            num_players=2,
            max_turns=None
        )

        ## set up the board items
        self.piece_counts = {
            'Flag': 1, 'Bomb': 6, 'Spy': 1, 'Scout': 8, 'Miner': 5,
            'Sergeant': 4, 'Lieutenant': 4, 'Captain': 4, 'Major': 3,
            'Colonel': 2, 'General': 1, 'Marshal': 1
        }
        self.piece_ranks = {
            'Flag': 0, 'Bomb': 11, 'Spy': 1, 'Scout': 2, 'Miner': 3,
            'Sergeant': 4, 'Lieutenant': 5, 'Captain': 6, 'Major': 7,
            'Colonel': 8, 'General': 9, 'Marshal': 10
        }
        self.lakes = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
        self.player_pieces = {
            0: [],
            1: []
        }
        self.board = [[None for _ in range(10)] for _ in range(10)]

        ## add render objects
        # self.board_state_render = ta.envs.two_player.Stratego.render.BoardStateRender
        self.render_keys = ["rendered_board"]

    def reset(
        self,
        seed: Optional[int] = None
    ):
        """
        Reset the environment to start a new game.

        Args:
            seed (Optional[int]): The seed for the random number generator.
        """
        ## set the seed
        random.seed(seed)

        ## populate the board
        self.board = self._populate_board()

        ## initialise the game state
        self.state.reset(
            game_state={
                "board": self.board,
                "player_pieces": self.player_pieces,
                "rendered_board": self._render_board(player_id=None, full_board=True)
            },
            player_prompt_function=self._generate_player_prompt,
            seed=seed
        )
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]):
        """
        Generates the player prompt for the current player.

        Args:
            player_id (int): The ID of the current player.
            game_state (Dict[str, Any]): The current game state.
        """
        prompt = (
            f"You are Player {player_id}. You are playing the Stratego game.\n"
            "Your goal is to capture your opponent's Flag or eliminate all of their movable pieces.\n"
            "At the start of the game, you have placed your army on the board, including your Flag, Bombs, and other pieces of varying ranks.\n"
            "\n"
            "### Gameplay Instructions\n"
            "1. **Movement Rules:**\n"
            "   - On your turn, you can move one piece by one step to an adjacent square (up, down, left, or right).\n"
            "   - Example: A piece can move from A1 to B1 or A1 to A2.\n"
            "   - If the selected piece is a Bomb or a Flag, it cannot be moved.\n"
            "   - **Scout Movement:** Scouts can move multiple steps in a straight line (horizontally or vertically).\n"
            "       - Scouts cannot jump over any piece (your own or your opponent's).\n"
            "       - Example: If there is a piece between the Scout and its destination, the Scout cannot move to the destination.\n"
            "2. **Battles:**\n"
            "   - If you move onto a square occupied by an opponent's piece, a battle will occur:\n"
            "     - The piece with the higher rank wins and eliminates the opponent's piece.\n"
            "     - If the ranks are equal, both pieces are removed from the board.\n"
            "     - **Special Cases:**\n"
            "       - Bombs eliminate most attacking pieces except Miners, which defuse Bombs.\n"
            "       - Spies can defeat the Marshal if the Spy attacks first but lose to all other pieces.\n"
            "3. **Strategic Goals:**\n"
            "   - Identify your opponent's pieces through their movements and battles.\n"
            "   - Protect your Flag while attempting to capture your opponent's Flag.\n"
            "   - Use Scouts strategically to gain information about your opponent's pieces and attack weak ones.\n"
            "\n"
            "### How to Make a Move:\n"
            "1. Specify the coordinates of the piece you want to move and its destination.\n"
            "2. Use the format: [A0 B0], where A0 is the source position, and B0 is the destination.\n"
            "   - Example: To move a piece from row 0, column 0 to row 1, column 0, input [A0 B0].\n"
            "3. Ensure the destination is valid according to the movement rules above.\n"
            "\n"
            "### Important Notes:\n"
            "- The board will show your pieces and their positions.\n"
            "- The board will also show known positions of your opponent's pieces without revealing their ranks.\n"
            "- Invalid moves will not be allowed. Plan carefully!\n"
            "\n"
            "Here is the current board state:\n"
        )


        prompt += self._render_board(player_id=player_id, full_board=False)

        return prompt

    
    def _populate_board(self):
        """
        Populates the board with pieces for each player strategically.
        """
        for player in range(2):
            # Define rows for each player
            back_rows = range(0, 2) if player == 0 else range(8, 10)
            front_rows = range(2, 4) if player == 0 else range(7, 9)
            all_rows = range(0, 4) if player == 0 else range(6, 10)

            # Place the Flag strategically
            while True:
                row = random.choice(back_rows)
                col = random.randint(0, 9)
                if (row, col) not in self.lakes and self.board[row][col] is None:
                    self.board[row][col] = {'rank': 'Flag', 'player': player}
                    self.player_pieces[player].append((row, col))
                    flag_position = (row, col)
                    break

            # Place Bombs around the Flag if possible
            bombs_to_place = self.piece_counts['Bomb']
            bomb_positions = [
                (flag_position[0] + dr, flag_position[1] + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Adjacent cells
                if 0 <= flag_position[0] + dr < 10 and 0 <= flag_position[1] + dc < 10
            ]

            for pos in bomb_positions:
                if bombs_to_place > 0 and self.board[pos[0]][pos[1]] is None and pos not in self.lakes:
                    self.board[pos[0]][pos[1]] = {'rank': 'Bomb', 'player': player}
                    self.player_pieces[player].append(pos)
                    bombs_to_place -= 1

            # Place remaining Bombs at the frontline
            for _ in range(bombs_to_place):
                while True:
                    row = random.choice(front_rows)
                    col = random.randint(0, 9)
                    if self.board[row][col] is None and (row, col) not in self.lakes:
                        self.board[row][col] = {'rank': 'Bomb', 'player': player}
                        self.player_pieces[player].append((row, col))
                        break

            # Place other pieces randomly
            for piece, count in self.piece_counts.items():
                if piece in ['Flag', 'Bomb']:
                    continue  # Skip already placed pieces
                for _ in range(count):
                    while True:
                        row = random.choice(all_rows)
                        col = random.randint(0, 9)
                        if self.board[row][col] is None and (row, col) not in self.lakes:
                            self.board[row][col] = {'rank': piece, 'player': player}
                            self.player_pieces[player].append((row, col))
                            break

        return self.board


    
    def _render_board(self, player_id, full_board: bool = False):
        """
        Renders the board state with fixed-width formatting for uniform alignment.

        Args:
            player_id (int): The player viewing the board.
            full_board (bool): Whether to render the full board or just the visible pieces.
        """
        # Define abbreviations for each piece
        piece_abbreviations = {
            'Flag': 'FL', 'Bomb': 'BM', 'Spy': 'SP', 'Scout': 'SC', 'Miner': 'MN',
            'Sergeant': 'SG', 'Lieutenant': 'LT', 'Captain': 'CP', 'Major': 'MJ',
            'Colonel': 'CL', 'General': 'GN', 'Marshal': 'MS'
        }

        res = []
        column_headers = "   " + " ".join([f"{i:>3}" for i in range(10)])  # Align column numbers
        res.append(column_headers + "\n")

        for row in range(10):
            row_label = chr(row + 65)  # Convert row index to a letter (A, B, C, ...)
            row_render = [f"{row_label:<3}"]  # Add row label with fixed width
            for col in range(10):
                if (row, col) in self.lakes:
                    cell = "  ~ "  # Lakes
                elif self.board[row][col] is None:
                    cell = "  . "  # Empty space
                else:
                    piece = self.board[row][col]
                    abbreviation = piece_abbreviations[piece['rank']]
                    if full_board:
                        cell = f" {abbreviation.lower() if piece['player'] == 0 else abbreviation.upper()} "  # Full board view
                    elif piece['player'] == player_id:
                        displayed_piece = abbreviation.upper()
                        cell = f" {displayed_piece} "
                    else:
                        cell = "  ? "  # Hidden opponent piece
                row_render.append(cell)

            res.append("".join(row_render) + "\n")

        return "".join(res)



    def step(
        self,
        action: str
    ):
        """
        Execute an action in the environment.

        Args:
            action (str): The action to be executed

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

        ## update the observation
        self.state.add_observation(
            from_id=player_id,
            to_id=player_id, ## send the observation to the same player since this is a private observation
            message=action,
            for_logging=True
        )

        ## action search pattern
        action_search_pattern = re.compile(r"\[([A-J])([0-9]) ([A-J])([0-9])\]") ## e.g. [A1 B1]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} did not input a move in the format [A0 B0]."]
            )
        
        else:
            src_row, src_col, dest_row, dest_col = match.groups()
            source = f"{src_row}{src_col}"
            dest = f"{dest_row}{dest_col}"
            src_row, src_col = ord(src_row) - 65, int(src_col)
            dest_row, dest_col = ord(dest_row) - 65, int(dest_col)
             

            ## check if the source and destination are valid
            if self._validate_move(player_id=player_id, src_row=src_row, src_col=src_col, dest_row=dest_row, dest_col=dest_col):

                attacking_piece = self.board[src_row][src_col]
                target_piece = self.board[dest_row][dest_col]

                if target_piece is None:
                    ## move to an empty square
                    self.board[dest_row][dest_col] = attacking_piece
                    self.board[src_row][src_col] = None
                    self.player_pieces[player_id].remove((src_row, src_col))
                    self.player_pieces[player_id].append((dest_row, dest_col))
                    
                    ## add the observation to both players separately
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=(
                            f"You have moved your piece from {source} to {dest}. Here is the updated board state:\n"
                            f"{self._render_board(player_id=player_id, full_board=False)}"
                        ),
                        for_logging=False
                    )

                    self.state.add_observation(
                        from_id=-1,
                        to_id=1 - player_id,
                        message=(
                            f"Player {player_id} has moved a piece from {source} to {dest}. Here is the updated board state:\n"
                            f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                        ),
                        for_logging=False
                    )

                else:
                    ## battle
                    attacking_rank = self.piece_ranks[attacking_piece['rank']]
                    target_rank = self.piece_ranks[target_piece['rank']]
                    if attacking_rank == target_rank:
                        ## both pieces are removed
                        self.board[src_row][src_col] = None
                        self.board[dest_row][dest_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=(
                                f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the ranks are the same, both pieces lost. Here is the updated board state:\n"
                                f"{self._render_board(player_id=player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                        self.state.add_observation(
                            from_id=-1,
                            to_id=1 - player_id,
                            message=(
                                f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the ranks are the same, both pieces lost. Here is the updated board state:\n"
                                f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                    elif target_piece['rank'] == 'Bomb':
                        if attacking_piece['rank'] == 'Miner':
                            ## Miner defuses the bomb
                            self.board[dest_row][dest_col] = attacking_piece
                            self.board[src_row][src_col] = None
                            self.player_pieces[player_id].remove((src_row, src_col))
                            self.player_pieces[player_id].append((dest_row, dest_col))

                            ## add the observation to both players separately
                            self.state.add_observation(
                                from_id=-1,
                                to_id=player_id,
                                message=(
                                    f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As miners can defuse bombs, you won the battle. Here is the updated board state:\n"
                                    f"{self._render_board(player_id=player_id, full_board=False)}"
                                ),
                                for_logging=False
                            )

                            self.state.add_observation(
                                from_id=-1,
                                to_id=1 - player_id,
                                message=(
                                    f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As miners can defuse bombs, you lost the battle. Here is the updated board state:\n"
                                    f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                                ),
                                for_logging=False
                            )

                        else:
                            ## attacking piece is destroyed
                            self.board[src_row][src_col] = None
                            self.player_pieces[player_id].remove((src_row, src_col))

                            ## add the observation to both players separately
                            self.state.add_observation(
                                from_id=-1,
                                to_id=player_id,
                                message=(
                                    f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is not a miner, you lost the battle. Here is the updated board state:\n"
                                    f"{self._render_board(player_id=player_id, full_board=False)}"
                                ),
                                for_logging=False
                            )

                            self.state.add_observation(
                                from_id=-1,
                                to_id=1 - player_id,
                                message=(
                                    f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is not a miner, you won the battle. Here is the updated board state:\n"
                                    f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                                ),
                                for_logging=False
                            )

                    elif target_piece['rank'] == 'Flag':
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))
                        ## game over
                        self.state.set_winners(
                            player_ids=[player_id],
                            reason=[f"Player {player_id} has captured the opponent's flag!"]
                        )
                    elif attacking_piece['rank'] == 'Spy' and target_piece['rank'] == 'Marshal':
                        ## Spy beats Marshal only if spy attacks first
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=(
                                f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a spy and the destination is a marshall, you won the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                        self.state.add_observation(
                            from_id=-1,
                            to_id=1 - player_id,
                            message=(
                                f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a spy and the destination is a marshall, you lost the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                    elif attacking_rank > target_rank:
                        ## attacker wins
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=(
                                f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a higher rank than the destination, you won the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                        self.state.add_observation(
                            from_id=-1,
                            to_id=1 - player_id,
                            message=(
                                f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a higher rank than the destination, you lost the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                    else:
                        ## defender wins
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))

                        ## add the observation to both players separately
                        self.state.add_observation(
                            from_id=-1,
                            to_id=player_id,
                            message=(
                                f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a lower rank than the destination, you lost the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

                        self.state.add_observation(
                            from_id=-1,
                            to_id=1 - player_id,
                            message=(
                                f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a lower rank than the destination, you won the battle. Here is the updated board state:\n"
                                f"{self._render_board(player_id=1 - player_id, full_board=False)}"
                            ),
                            for_logging=False
                        )

        ## check if the game is over
        if self._check_winner():
            self.state.set_winners(
                player_ids=[self._check_winner()],
                reason=[f"Player {self._check_winner()} wins! Player {1 - self._check_winner()} has no more movable pieces."]
            )

        ## update the rendered board
        self.state.game_state["rendered_board"] = self._render_board(player_id=player_id, full_board=True)

        return self.state.step()
    
    def _validate_move(self, player_id, src_row, src_col, dest_row, dest_col):
        """
        Validates the move based on the game rules.

        Args:
            player_id (int): The ID of the player making the move.
            src_row (int): The row of the source position.
            src_col (int): The column of the source position.
            dest_row (int): The row of the destination position.
            dest_col (int): The column of the destination position.
        """
        if not (0 <= src_row < 10 and 0 <= src_col < 10 and 0 <= dest_row < 10 and 0 <= dest_col < 10):
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} did not input valid coordinates."]
            )
            return False
        
        if self.board[src_row][src_col] is None or self.board[src_row][src_col]['player'] != player_id:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} must move one of their own pieces."]
            )
            return False
        
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1 and self.board[src_row][src_col]['rank'].lower() == 'scout':
            ## check if there's a piece in between the source and destination
            if src_row == dest_row:
                for col in range(min(src_col, dest_col) + 1, max(src_col, dest_col)):
                    if self.board[src_row][col] is not None:
                        self.state.set_invalid_move(
                            player_ids=[player_id],
                            reasons=[f"Invalid action format. Player {player_id} cannot move a scout through other pieces."]
                        )
                        return False
            elif src_col == dest_col:
                for row in range(min(src_row, dest_row) + 1, max(src_row, dest_row)):
                    if self.board[row][src_col] is not None:
                        self.state.set_invalid_move(
                            player_ids=[player_id],
                            reasons=[f"Invalid action format. Player {player_id} cannot move a scout through other pieces."]
                        )
                        return False
            else:
                self.state.set_invalid_move(
                    player_ids=[player_id],
                    reasons=[f"Invalid action format. Player {player_id} cannot move a scout diagonally."]
                )
                return False
            
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1 and self.board[src_row][src_col]['rank'].lower() != 'scout':
            ## !  - by right, only scouts can move more than one square at a time but we are not implementing that yet
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Pieces, apart from scouts, can only move one square at a time."]
            )
            return False
        
        if self.board[dest_row][dest_col] is not None and self.board[dest_row][dest_col]['player'] == player_id:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} cannot move onto their own piece."]
            )
            return False
        
        if (dest_row, dest_col) in self.lakes:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} cannot move into the lake."]
            )
            return False
        
        if self.board[src_row][src_col]['rank'].lower() in ['bomb','flag']:
            self.state.set_invalid_move(
                player_ids=[player_id],
                reasons=[f"Invalid action format. Player {player_id} cannot move a bomb or flag."]
            )
            return False
        
        return True
    
    def _check_winner(self):
        """
        determine which player has no more pieces that are not bombs or flags.
        """
        for player in range(2):
            if all([self.board[row][col]['rank'] in ['Bomb', 'Flag'] for row, col in self.player_pieces[player]]):
                return 1 - player
        return None
    
    def render(
        self
    ):
        """
        Render the environment.
        """
        return self._render_board(player_id=self.state.current_player, full_board=True)

