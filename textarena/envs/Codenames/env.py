import re
import nltk, random
from nltk.corpus import words
from nltk import pos_tag
from typing import Any, Dict, Optional, Tuple, List, Union
import textarena as ta

nltk.download("words")
nltk.download('averaged_perceptron_tagger_eng')

class CodenamesEnv(ta.Env):
    """Environment for Codenames game."""
    def __init__(self, hardcore: Optional[bool] = False):
        self._load_word_list(hardcore=hardcore)

    def _load_word_list(self, hardcore: bool = False) -> None:
        """Load a set of words for the game."""
        # Get word list
        word_list = words.words("en") if hardcore else words.words("en-basic")

        # Filter words based on POS tags
        self.word_list = [
            word for word in word_list if pos_tag([word])[0][1] in ["NN"] and len(word) < 8
        ]
    def reset(self, num_players: int = 4, seed: Optional[int] = None):
        """ Reset the game state """
        self.state = ta.State(num_players=num_players, min_players=4, max_players=4)
        self._setup_board()
        
        self.state.reset(
            seed=seed,
            game_state={
                "turn": 0,
                "team_turn": 0,  # 0 for Red, 1 for Blue
                "guessed_words": set(),
                "last_clue": None,  # Initialize last clue
                "last_number": 0,    # Initialize last number
            },
            player_prompt_function=self._generate_player_prompt
        )

    def _setup_board(self):
        """Set up the board with words and random team assignments."""
        selected_words = random.sample(self.word_list, 25)  # Select 25 random words

        # Create a list of 25 assignments: 8 Red (R), 8 Blue (B), 8 Neutral (N), and 1 Assassin (A)
        assignments = ["R"] * 8 + ["B"] * 8 + ["N"] * 8 + ["A"]

        # Shuffle the assignments to randomize their placement
        random.shuffle(assignments)

        # Assign each word to a team
        self.board = {word: team for word, team in zip(selected_words, assignments)}


    def _render_board(self, spymaster: bool = False, guessed_words: set = None):
        """Render the board with team labels for spymasters and update guessed words for operatives."""
        board = "Codenames Board:"
        words = list(self.board.keys())
        cell_width = 11  # 8 characters for the word + 3 spaces padding
        board += "\n" + "-" * (cell_width * 5 + (3*4) + 4)  # 5 words per row, 2 and 2 padding at the ends
        for i in range(5):
            row = []
            for word in words[i * 5:(i + 1) * 5]:
                label = self.board[word] if spymaster or (guessed_words and word in guessed_words) else " "
                formatted_word = f"{word:<8}{label:<3}" if label else f"{word:<11}"
                row.append(formatted_word)
            board += "\n" + "| " + " | ".join(row) + " |"
        board += "\n" + "-" * (cell_width * 5 + (3*4) + 4)
        board += "\n"
        return board
    
    def _render_player_view(self, spymaster: bool = False, guessed_words: set = None):
        """Render a list view of the words instead of a board format."""
        words = list(self.board.keys())
        view = "Codenames Words:\n"
        
        for word in words:
            if spymaster:
                # Show the team label for spymasters
                view += f"{word:<8} {self.board[word]}\n"
            else:
                # Hide labels for operatives, but mark guessed words
                if guessed_words and word in guessed_words:
                    view += f"{word:<8} {self.board[word]}\n"  # Add a checkmark for guessed words
                else:
                    view += f"{word:<8}\n"

        return view



    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate a prompt for the player, explaining game rules and rendering the board."""
        prompt = (
            "Codenames is a word-based deduction game where two teams, Red (R) and Blue (B), compete to uncover all of their team's secret words before the other team does. Each team consists of:\n"
            "- One Spymaster who provides cryptic, one-word clues.\n"
            "- One Operative who deciphers the clues and makes guesses.\n\n"
            "Game Rules:\n"
            "1. The Spymaster knows the identity of all words on the board and provides a one-word clue followed by a number indicating how many words relate to the clue.\n"
            "2. The Operative must then guess words based on the clue while avoiding:\n"
            "   - Opponent’s words (which help the other team).\n"
            "   - Neutral (N) words (which waste a turn).\n"
            "   - The deadly Assassin (A) word, which immediately ends the game in a loss for their team.\n"
            "3. Each turn, the Operative can make up to N+1 guesses (where N is the number given by the Spymaster).\n"
            "4. The game ends when:\n"
            "   - One team successfully guesses all of their words.\n"
            "   - An Operative selects the Assassin word, instantly losing the game for their team.\n"
            "How to Play:\n"
            "- Spymaster: Enter a one-word clue followed by a number in square brackets, e.g., [wind 2].\n"
            "- Operative: Guess a word by entering it in square brackets, e.g., [breeze].\n\n"
        )
        
        if player_id in [0, 2]:  # Spymasters
            return prompt + self._render_player_view(spymaster=True) + f"You are Player {player_id}, the Spymaster for {'Red' if player_id == 0 else 'Blue'} team. Provide a one-word clue followed by a number."
        else:  # Operatives
            return prompt + self._render_player_view(spymaster=False) + f"You are Player {player_id}, the Operative for {'Red' if player_id == 1 else 'Blue'} team. Guess a word based on your spymaster's clues."

    def step(self, action: Union[List[Union[str, int]], str]) -> Tuple[bool, ta.Info]:
        """Process the player's action."""
        game_state = self.state.game_state
        player_id = self.state.current_player_id
        current_team = "R" if player_id < 2 else "B"

        # Log the player's action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,  # broadcast to all
            message=action,
            for_logging=True
        )

        if self.state.current_player_id in [0, 2]:  # Spymasters give clues
            match = re.search(r"\[(\w+)\s+(\d+)\]", action)
            
            if match:
                word = match.group(1)  # Extracts the word
                number = int(match.group(2))  # Extracts the number as an integer

                # check that clue word is not a word/ subset of word on the board
                if any(word in board_word or board_word in word for board_word in self.board.keys()):
                    # if current team said a subset to cheat then the other team automatically wins
                    self.state.set_winners(
                            player_ids=[0, 1] if current_team == "B" else [2, 3],
                            reason=f"Player {player_id} mentioned a clue that is a subset/ exact match of words on the board."
                        )
                    return self.state.step()

                # add the clue to the game state
                game_state["last_clue"] = word
                game_state["last_number"] = number
                game_state["remaining_guesses"] = number + 1 # Operatives can make up to N+1 guesses

                message = f"Spymaster of {"Red" if current_team=="R" else "Blue"} team, Player {player_id}, submitted [{word} {number}]."
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=message,
                    for_logging=False
                )

                return self.state.step()
            else:
                reason = "Invalid clue. Provide a word and a number (e.g., [dust 2])."
                self.state.set_invalid_move(player_id, reason)
                return self.state.step()
        else:  # Operatives guess words, 1 3 indices
            match = re.search(r"\[(\w+)\]", action)

            if match:
                guessed_word = match.group(1).lower()

                # check 0: if guessed word exists on the board
                if guessed_word not in self.board:
                    reason = "Invalid move. Word is not on the board."
                    self.state.set_invalid_move(player_id, reason)
                    return self.state.step()

                # check 1: if guessed word is in the guessed words set
                if guessed_word in game_state["guessed_words"]:
                    reason = "Word has already been guessed."
                    self.state.set_invalid_move(player_id, reason)
                    return self.state.step()
                
                game_state["guessed_words"].add(guessed_word)

                # check 2: if guessed word is the assassin word
                if self.board[guessed_word] == "A":
                    self.state.set_winners(
                        # the other team wins
                        # if 1 says assasin, 2 and 3 win
                        # if 3 says assasin, 0 and 1 win
                        player_ids= [2, 3] if current_team == "R" else [0, 1],
                        reason=f"Player {player_id} selected the assassin word."
                    )
                    return self.state.step()
                
                # check 3: if guessed word is correct
                elif self.board[guessed_word] == current_team:
                    # Check if all words of the current team are guessed
                    if all(word in game_state["guessed_words"] for word, team in self.board.items() if team == current_team):
                        self.state.set_winners(
                            player_ids=[0, 1] if current_team == "R" else [2, 3],
                            reason=f"Player {player_id} guessed all their team's words!"
                        )
                        return self.state.step()
                    message = f"Operator of {"Red" if current_team=="R" else "Blue"} team, Player {player_id}, correctly guessed [{guessed_word}]."
                    message += "\n"
                    message += self._render_player_view(spymaster=False, guessed_words=game_state["guessed_words"])
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=message,
                        for_logging=False
                    )

                    game_state["remaining_guesses"] -= 1

                    if game_state["remaining_guesses"] > 0:
                        return self.state.step(rotate_player=False)
                    else:
                        game_state["remaining_guesses"] = 0
                        return self.state.step()
                    
                # check 4: if guessed word is incorrect [opponent's word or neutral word]
                else: # self.board[guessed_word] != current_team:
                    # Check if all words of the opposing team are guessed
                    opponent_team = "B" if current_team == "R" else "R"
                    if all(word in game_state["guessed_words"] for word, team in self.board.items() if team == opponent_team):
                        self.state.set_winners(
                            player_ids=[0, 1] if opponent_team == "R" else [2, 3],
                            reason=f"Player {player_id} guessed the opponent team's last word!"
                        )
                        return self.state.step()
                    opponent_team_name = "Red" if opponent_team == "R" else "Blue"
                    message = f"Operator of {"Red" if current_team=="R" else "Blue"} team, Player {player_id}, wrongly guessed [{guessed_word}]. It is a {opponent_team_name + " Team" if self.board[guessed_word]==opponent_team else "Neutral"} word."
                    message += "\n"
                    message += self._render_player_view(spymaster=False, guessed_words=game_state["guessed_words"])
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=message,
                        for_logging=False
                    )
                    game_state["remaining_guesses"] = 0
                    return self.state.step()

            else:
                reason = "Provide a word in square brackets (e.g., [apple])."
                self.state.set_invalid_move(player_id, reason)
                return self.state.step()
  