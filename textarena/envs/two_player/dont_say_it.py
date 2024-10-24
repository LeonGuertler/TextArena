"""
Don't Say It Game

Each player is given a secret word that they must try to get the other player to say during the course of the game. 
Players take turns speaking to each other, attempting to subtly guide the conversation towards the other's secret word. 
If a player successfully gets the other player to say their secret word, they win the game. 
If a player accidentally says the other player's secret word, they lose the game. 
The game ends either when one player says the other player's word or when the maximum number of turns is reached. 
If the turn limit is reached without either player saying the other's word, the game ends in a draw.

Key Rules:
1. Each player has a secret word that they must protect while trying to make the other player say their word.
2. Players can converse freely but should be subtle to avoid giving away their secret word.
3. The game ends if either player says the other's secret word or when the maximum number of turns is reached.

Parameters:
- `max_turns`: The maximum number of turns allowed before the game ends in a draw.
- `render`: If set to True, the game state and actions are printed for debugging purposes.
- `data_path`: The path to the JSON file containing the list of possible secret words.

Game Outcomes:
- A player wins if they successfully make the other player say their secret word.
- A player loses if they accidentally say the other player's secret word.
- The game is a draw if the turn limit is reached without either player saying the other's word.
"""

import random
from typing import Any, Dict, Optional, Tuple

# nltk is used to get the words
import nltk
from nltk import pos_tag
from nltk.corpus import words

import textarena as ta

nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")


class DontSayItEnv(ta.Env):
    """Environment for Don't say it game"""

    def __init__(
        self,
        hardcore: Optional[bool] = False,
        max_turns: Optional[int] = None,
    ):
        """
        Initialize the Don't Say It Game.
        Args:
            hardcore (bool): If True, use full English word-set. Otherwise, simplified wordset
            max_turns (int): Maximum number of turns before the game ends in a draw
        """
        self.environment_name = (
            "Don't Say It" if not hardcore else "Don't Say It (hardcore)"
        )

        # load the word list
        self._load_word_list(hardcore=hardcore)

        # Initialize game state (mostly used by wrappers (especially rendering))
        self.state = ta.State(
            num_players=2,
            max_turns=max_turns,
            render_keys=["target_words"],
        )


    def _load_word_list(self, hardcore: bool = False) -> None:
        """
        Load the word list as specified
        """
        # get word list
        if hardcore:
            word_list = words.words("en")
        else:
            word_list = words.words("en-basic")

        # Filter words based on POS tags
        # NN: Noun, VB: Verb, JJ: Adjective
        self.word_list = [
            word for word in word_list if pos_tag([word])[0][1] in ["NN", "VB", "JJ"]
        ]

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[
        Optional[ta.Observations]
    ]:  # player-wise observations and info
        """
        Reset the game to its initial state.
        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.
        Returns:
            Tuple[str, str, Dict[str, str]]: Initial observations for both players and their secret words.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()


        # reset the state
        target_words = {
            0: random.choice(self.word_list),
            1: random.choice(self.word_list),
        }
        while target_words[0] == target_words[1]:
            target_words[1] = random.choice(self.word_list)


        self.state.reset(
            game_state={
                "target_words": target_words,
            },
            initial_logs=[(ta.GAME_ID, "Game started!")]
        )

        # Generate the initial player-wise observations for both players and return them
        return (
            {
                0: [(ta.GAME_ID, self._generate_player_prompt(player_id=0))],
                1: [(ta.GAME_ID, self._generate_player_prompt(player_id=1))],
            },
            {
                "player_0_secret_word": self.state.game_state["target_words"][0],
                "player_1_secret_word": self.state.game_state["target_words"][1],
            },
        )

    def _generate_player_prompt(self, player_id: int) -> ta.Message:
        """
        Generate the initial prompt for a player, providing them with their secret word and instructions.
        Args:
            player_id (int): ID of the player (0 or 1).
        Returns:
            str: Initial prompt for the player.
        """
        prompt = (
            f"You are playing 'Don't Say It'. You are Player {player_id}\n"
            f"Your secret word is: '{self.state.game_state['target_words'][player_id]}'.\n"
            "Your goal is to get the other player to say your secret word before you say theirs.\n"
            "You can converse freely, but try to be subtle to avoid making it obvious.\n"
            "On your turn, simply type your message.\n"
            "Turn: 1"
        )
        if self.state.max_turns:
            prompt += (
                f"The game lasts for {self.state.max_turns} turns in total.\n"
            )
        return prompt

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observations],  # player-wise observations
        Optional[ta.Rewards],  # player-wise reward
        bool,  # truncated
        bool,  # terminated
        Optional[ta.Info],  # info
    ]:
        """
        Process the player's action. Checks if the player's action mentions the opponent's secret word.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's message or action.

        Returns:
            tuple: (observation, reward, done, info)
                - observation: The action taken.
                - reward: Dictionary with rewards for both players.
                - truncated: Boolean indicating if the game has reach the turn limit.
                - terminated: Boolean indicating if the game has concluded.
                - info: Additional information about the game state.
        """
        assert isinstance(
            action, str
        ), f"Actions are required to be strings. Received dtype: {type(action)}"

        assert (
            player_id == self.state.current_player
        ), f"The passed player_id is not as expected. Player id received: {player_id}; Expected: {self.state.current_player}"

        terminated, truncated = False, False
        self.step_logs = []
        observations = {0: [], 1: []}


        # update step logs
        self.step_logs.append((player_id, action))

        # check observation mode here (not relevant for this env)
        observations[player_id].append((player_id, action))
        observations[1-player_id].append((player_id, action))

            

        # Check if the action mentions the opponent's secret word
        if self.state.game_state["target_words"][1 - player_id].lower() in action.lower():
            # Opponent's secret word was mentioned, player loses
            reward = {player_id: -1, 1 - player_id: 1}
            terminated = True
            info = {"reason": f"Player {player_id} mentioned the hidden word."}
            self.step_logs.append((-1, f"{info['reason']}. Player {1 - player_id} wins!"))

        # Check if the maximum number of turns has been reached
        elif (
            self.state.max_turns
            and self.state.turn >= self.state.max_turns
        ):
            reward = {0: 0, 1: 0}
            truncated = True
            info = {"reason": "The turn limit has been reached. The game is a draw."}
            self.step_logs.append((-1, f"{info['reason']}"))

        # Normal turn, neither word mentioned
        else:
            reward = None
            info = {"info": f"Player {player_id}: {action}"}


        # step the game state to updated turn count, current player and game state
        self.state.step(
            logging_messages=self.step_logs,
            game_state_updates=None
        )
        return observations, reward, truncated, terminated, info


    def render(self):
        """
        Render minimal game state.
        """
        turn_info = f"Turn {self.state.turn}/{self.state.max_turns if self.state.max_turns else '∞'}"
        print(turn_info)
        for player_id, log in self.step_logs:
            if player_id == ta.GAME_ID:
                print(f"GAME: {log}")
            else:
                print(f"Player {player_id}: {log}")
