import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.utils.word_lists import EnglishDictionary


class LetterAuctionEnv(ta.Env):
    """
    The environment for Letter Auction Game.
    """

    def __init__(
        self,
        difficulty: Optional[str] = "easy",
    ):
        """
        Initialize the environment for Letter Auction Game.

        Args:
            difficulty (str, optional): The difficulty level of the game. Defaults to "easy".
        """
        self.environment_name = "LetterAuction"
        self.difficulty = difficulty
        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=False)

        ## Load the game settings based on difficulty
        if self.difficulty == "easy":
            ## Easy difficulty settings
            self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            self.letter_values = [1 for _ in self.letters]
            self.starting_coins = 100
        elif self.difficulty == "medium":
            ## Medium difficulty settings
            self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            self.letter_values = [1 for _ in self.letters]
            self.starting_coins = 50
        elif self.difficulty == "hard":
            ## Hard difficulty settings
            self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            self.letter_values = [1 for _ in self.letters]
            self.starting_coins = 25
        else:
            raise ValueError(f"Invalid difficulty level: {self.difficulty}")

        ## Initialize the game state
        self.state = ta.State(
            num_players=2,
        )

    @property
    def terminal_render_keys(self):
        return ["rendered_text", "turn"]

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Optional[ta.Observations]:
        """
        Reset the environment to start a new game.

        Args:
            seed (int, optional): The random seed for the game. Defaults to None.

        Returns:
            Optional[ta.Observations]: The initial observations for the players.
        """
        if seed is not None:
            random.seed(seed)

        ## Initialize the player state
        self.player_states = {
            0: {
                "coins": self.starting_coins,
                "letters": [],
                "letter_values": [],
                "letter_bid_history": {i: None for i in range(len(self.letters))},
                "word": None,
                "word_value": 0,
            },
            1: {
                "coins": self.starting_coins,
                "letters": [],
                "letter_values": [],
                "letter_bid_history": {i: None for i in range(len(self.letters))},
                "word": None,
                "word_value": 0,
            },
        }

        ## Initialize the game
        self.current_player = 0
        random.shuffle(self.letters)
        self.round_number = 0
        self.round_letter = self.letters[self.round_number]
        self.bid_amount = self.letter_values[self.round_number]

        ## intialize the game states
        return self.state.reset(
            game_state={
                "player_states": self.player_states,
                "rendered_text": self.render_text(),
                "turn": self.current_player,
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[int, Any]
    ) -> str:
        """
        Generate the prompt for the current player.

        Returns:
            str: The prompt for the current player.
        """
        prompt = (
            f"You are Player {player_id}. You are currently in the Letter Auction game.\n"
            "The goal of the game is to strategically bid on letters to form the highest value word. This is how the game works.\n"
            "You must listen to the gamemaster for guidance to play the game.\n"
            "The game consists of a series of rounds. In each round, a letter will be put up for auction.\n"
            "You can bid on the letter using your coins. The player with the highest bid wins the letter.\n"
            "The letter will be added to your collection, and the coins you bid will be deducted from your total.\n"
            "This bidding of letters will repeat till all the letters have been auctioned off. You are not rewarded for saving your coins.\n"
            "After all the letters have been auctioned, you will use the letters to form the highest value english word from the letters won.\n"
            "The player with the highest value word wins the game.\n"
            "If you want to bid, submit your bid amount in square brackets like [bid 2] or [bid 10].\n"
            "If you do not want to bid, submit [pass].\n"
            "For the submission of the highest value word, you will be prompted at the end of the game to submit them in square brackets like [dog].\n"
            "Here is your starting information:\n"
            f"Your current coins: {self.player_states[player_id]['coins']}\n"
            f"Your current letters: {self.player_states[player_id]['letters']}\n"
            "\n"
            f"[Game] Player 0 will go first. The first letter for bid: {self.round_letter}.\n"
            f"Starting bid is {self.bid_amount} coin. You can bid any amount of coins, or choose not to bid.\n"
        )

        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Execute the player's action in the environment.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action taken by the player.

        Returns:
            Tuple[Optional[ta.Observations], Optional[ta.Rewards], bool, bool, ta.Info]: The observations, rewards, done flag, info, etc.
        """
        player_id = self.state.current_player_id

        ## Check if the player is valid
        if player_id != self.current_player:
            raise ValueError(
                f"Invalid player ID: {player_id}. It is not the turn of player {player_id}."
            )

        ## update the observations
        self.state.add_observation(
            from_id=player_id, to_id=-1, message=action, for_logging=True
        )

        self.auction_over_prompt = ""

        if self.round_number < len(self.letters):
            ## the auction is still in play

            ## validate the action
            action_search_pattern = re.compile(r"\[(bid \d+|pass)\]", re.IGNORECASE)
            match = action_search_pattern.search(action)

            if not match:
                ## invalid action
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid action: {action}. Please enter a valid action: '[bid <amount>]' or '[pass]'.",
                )

            else:
                ## valid action
                action_text = match.group(1).lower()
                if "pass" in action_text:
                    ## player passed the bid

                    ## indicate the player has passed the letter
                    if (
                        self.player_states[player_id]["letter_bid_history"][
                            self.round_number
                        ]
                        is None
                    ):
                        self.player_states[player_id]["letter_bid_history"][
                            self.round_number
                        ] = "pass"

                    self._pass_bid(player_id)  ## TODO

                else:
                    ## player bids on the letter
                    bid_amount = int(action_text.split()[1])
                    ## indicate the player wants to bid
                    if (
                        self.player_states[player_id]["letter_bid_history"][
                            self.round_number
                        ]
                        is None
                    ):
                        self.player_states[player_id]["letter_bid_history"][
                            self.round_number
                        ] = "bid"

                    self._place_bid(player_id, bid_amount)  ## TODO

        else:
            ## the auction is over, calculate the word values

            ## validate the action
            action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]")  # e.g. [plane]
            match = action_search_pattern.search(action)

            if not match:
                ## invalid action
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid action: {action}. Please enter a valid action: '[<word>]'.",
                )

            else:
                ## valid action
                action_text = match.group(1).lower()
                self._calculate_word_value(player_id, action_text)  ## TODO

                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=-1,
                    message=f"Player {player_id} chooses the word '{action_text}' with a value of {self.player_states[player_id]['word_value']}.",
                    for_logging=True,
                )

        ## update the game state
        self.state.game_state["rendered_text"] = self.render_text()

        ## check if the game is done
        if self._check_game_done():  ##TODO
            if (
                self.player_states[0]["word_value"]
                > self.player_states[1]["word_value"]
            ):
                self.state.set_winners(
                    player_ids=[0],
                    reason=f"Player 0 wins with a score of {self.player_states[0]['word_value']}",
                )
            elif (
                self.player_states[1]["word_value"]
                > self.player_states[0]["word_value"]
            ):
                self.state.set_winners(
                    player_ids=[1],
                    reason=f"Player 1 wins with a score of {self.player_states[1]['word_value']}",
                )
            else:
                self.state.set_draw(reason="It's a draw!")

        return self.state.step()

    def _pass_bid(
        self,
        player_id: int,
    ) -> None:
        """
        Pass on the current letter, allowing opponent to bid for it if it has not
        """
        opponent_id = 1 - player_id

        prompt = f"Player {player_id} passes on the letter '{self.round_letter}'."

        if (
            self.player_states[opponent_id]["letter_bid_history"][self.round_number]
            is None
        ):
            ## the opponent has not bid on the letter, we ask it

            ## Keep to the same round but move to next player
            next_prompt = self._turn_manager(next_round=False, next_player=True)

            ## we ask the current to bid on the next letter
            prompt += next_prompt

        elif (
            self.player_states[opponent_id]["letter_bid_history"][self.round_number]
            == "bid"
        ):
            ## the opponent has bid on the letter, they win it
            prompt += f" Player {opponent_id} will have '{self.round_letter}' for {self.bid_amount}."

            ## assign the letter
            self._assign_letter(opponent_id, self.round_letter, self.bid_amount)

            ## move to the next round but keep the current player
            next_prompt = self._turn_manager(next_round=True, next_player=False)

            ## we ask the current to bid on the next letter
            prompt += next_prompt

        else:
            ## the opponent has passed as well, no one will gain the letter and we move to the next round
            prompt += f" Player {opponent_id} also passes on the letter '{self.round_letter}'. So, no one will gain the letter."

            ## move to the next round but keep the current player
            next_prompt = self._turn_manager(next_round=True, next_player=False)

            ## we ask the current to bid on the next letter
            prompt += next_prompt

        self.state.add_observation(
            from_id=ta.GAME_ID, to_id=-1, message=prompt, for_logging=True
        )

    def _place_bid(
        self,
        player_id: int,
        bid_amount: int,
    ) -> None:
        """
        Place a bid on the current letter.
        """
        opponent_id = 1 - player_id

        prompt = (
            f"Player {player_id} bids {bid_amount} on the letter '{self.round_letter}'."
        )

        if self.player_states[player_id]["coins"] < bid_amount:
            ## the player does not have enough coins
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid bid: {bid_amount}. You do not have enough coins.",
            )
            return

        elif (
            self.player_states[opponent_id]["letter_bid_history"][self.round_number]
            is None
        ):
            ## the opponent has not bid on the letter, we ask it

            ## update the bid amount
            self.bid_amount = bid_amount

            ## Keep to the same round but move to next player
            next_prompt = self._turn_manager(next_round=False, next_player=True)

            prompt += next_prompt

        elif (
            self.player_states[opponent_id]["letter_bid_history"][self.round_number]
            == "bid"
        ):
            ## the opponent has bid on the letter, we compare the bids

            if bid_amount < self.bid_amount:
                ## the bid is not enough, means the opponent wins the letter
                prompt += f" Player {opponent_id} will have '{self.round_letter}' for {self.bid_amount}."

                ## assign the letter
                self._assign_letter(opponent_id, self.round_letter, self.bid_amount)

                ## Move to next round and move to next player
                next_prompt = self._turn_manager(next_round=True, next_player=True)

                ## we ask the current to bid on the next letter
                prompt += next_prompt

            else:
                ## the bid is enough, we ask the opponent to bid

                ## update the bid amount
                self.bid_amount = bid_amount

                ## Keep to the same round and move to next player
                next_prompt = self._turn_manager(next_round=False, next_player=True)

                prompt += next_prompt

        else:
            ## the opponent has passed, we ask the current to bid
            prompt += f" Since Player {opponent_id} passes on the letter '{self.round_letter}', Player {player_id} will have it for {bid_amount}."

            ## assign the letter
            self._assign_letter(player_id, self.round_letter, bid_amount)

            ## Move to next round and move to next player
            next_prompt = self._turn_manager(next_round=True, next_player=True)

            ## we ask the current to bid on the next letter
            prompt += next_prompt

        self.state.add_observation(
            from_id=ta.GAME_ID, to_id=-1, message=prompt, for_logging=True
        )

    def _assign_letter(
        self,
        player_id: int,
        letter: str,
        bid_amount: int,
    ) -> None:
        """
        Assign the letter to the player.
        """
        self.player_states[player_id]["letters"].append(letter)
        self.player_states[player_id]["letter_values"].append(bid_amount)
        self.player_states[player_id]["coins"] -= bid_amount

    def _turn_manager(
        self,
        next_round: bool = False,
        next_player: Optional[bool] = False,
    ) -> str:
        """
        Manage the turns and rounds in the game, and return the prompt for the next player or announces end of auction.

        Args:
            next_round (bool, optional): Move to the next round. Defaults to False.
            next_player (bool, optional): Move to the next player. Defaults to False.

        Returns:
            str: The prompt for the next player or the end of auction.
        """
        next_player = True
        if next_player:
            ## we switch the player
            self.current_player = 1 - self.current_player

        if next_round:
            ## we advance to the next round if within the rounds
            self.round_number += 1
            if self.round_number < len(self.letters):
                self.round_letter = self.letters[self.round_number]
                self.bid_amount = self.letter_values[self.round_number]
                next_prompt = f" Player {self.current_player}, do you want to start bid on the letter '{self.round_letter}' for {self.bid_amount}?"
            else:
                ## the auction is over
                next_prompt = "The auction is over. Now, players will use the letters they've won to form the highest value english word from the letters won. The player with the highest value word wins the game. To submit the word, submit it in square brackets like [dog]."

        else:
            next_prompt = f" Player {self.current_player}, do you want to bid on the letter '{self.round_letter}' for more than {self.bid_amount}?"

        return next_prompt

    def _calculate_word_value(
        self,
        player_id: int,
        word: str,
    ) -> None:
        """
        Calculate the value of the player's chosen word based on the bids.
        """
        ## check if the word is valid
        word = word.upper()

        if not self.dictionary.is_english_word(word):
            self.player_states[player_id]["word"] = ""
            self.player_states[player_id]["word_value"] = 0

            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid word: {word}. Please enter a valid English word.",
            )
            return

        ## check if the word is valid based on the letters
        for letter in word:
            if letter not in self.player_states[player_id]["letters"]:
                self.player_states[player_id]["word"] = ""
                self.player_states[player_id]["word_value"] = 0

                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid word: {word}. You do not have the letter '{letter}'.",
                )
                return

        ## calculate the word value
        word_value = sum(
            self.player_states[player_id]["letter_values"][
                self.player_states[player_id]["letters"].index(letter)
            ]
            for letter in word
        )
        self.player_states[player_id]["word"] = word
        self.player_states[player_id]["word_value"] = word_value

        ## move to the next round
        self._turn_manager(next_round=False, next_player=True)

    def _check_game_done(self) -> bool:
        """
        Check if the game is done.
        """
        ## check if the game is done
        for player_id in self.player_states:
            if self.player_states[player_id]["word"] is None:
                return False

        return True

    def render_text(self) -> str:
        """
        Render the game state.

        Returns:
            str: The rendered game state.
        """
        rendered_text = f"Round {self.round_number + 1}/{len(self.letters) + 1}\n"  # +1 for the word phase
        rendered_text += f"All letters: {self.letters}\n"
        rendered_text += f"Current letter: {self.round_letter}\n"
        rendered_text += f"Player 0: {self.player_states[0]['coins']} coins, {self.player_states[0]['letters']}\n"
        rendered_text += f"Player 1: {self.player_states[1]['coins']} coins, {self.player_states[1]['letters']}\n"
        rendered_text += f"Current player: {self.current_player}\n"
        return rendered_text

    def render(self):
        """
        Render the game state.
        """
        print(self.state.game_state["rendered_text"])
