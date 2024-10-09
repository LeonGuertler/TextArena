"""20 Questions is a game where one player thinks of an object,
and the other player has to guess what it is by asking yes-or-no questions.
The player who knows the object can only respond with "yes", "no", or "I don't know".
The game ends when the guessing player either correctly guesses the object or runs out of questions.
The guessing player can also choose to give up and make a final guess."""

import random
import textarena as ta
import re

# nltk is used to get the words
import nltk
from nltk.corpus import words
from nltk import pos_tag

nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")

TWENTY_QUESTIONS_PROMPT = """You are playing '20 Questions'. You are Player {player_id}.
You have to guess the object by asking yes-or-no questions.
The game will last for a maximum of 20 questions.
Make your final guess by ending your response with Guess: <object>."""

JUDGE_PROMPT = """You are playing '20 Questions'. You have chosen the object: {target_word}.
    Your opponent is trying to guess the object by asking yes-or-no questions.
    Please respond with 'Response: yes', 'Response: no', or 'Response: I don't know' to their questions.
    Their question was: {question}"""

FINAL_GUESS_REGEX = r"Guess: (.+)"
RESPONSE_REGEX = r"Response: (yes|no|I don't know)"
FLAGS = re.IGNORECASE | re.MULTILINE


class TwentyQuestions(ta.Env):
    """Environment for playing the game 20 Questions."""

    def __init__(
        self,
        judge_generate_fn,
        num_questions: int = 20,
        basic_word_list: bool = False,
    ):
        """
        Initialize the 20 Questions game.
        Args:
            num_questions (int): Maximum number of questions allowed before the game ends
        """
        self.ENVIRONMENT_NAME = "20 Questions"

        self.num_questions = num_questions
        self.game_state = {
            "turn": 0,
            "max_turns": num_questions,
            "target_word": None,
            "logs": [],
            "render": [
                "turn",
                "max_turns",
                "target_word",
            ],
        }

        self.judge_generate_fn = judge_generate_fn
        self.word_list = load_word_list(basic_word_list=basic_word_list)

    def reset(self, seed: int | None = None):
        """
        Reset the environment to its initial state.
        Args:
            seed (int | None): Seed for random number generator to ensure reproducibility.
        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial observations for both players and info.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        self.game_state["turn"] = 0
        self.game_state["target_word"] = random.choice(self.word_list)
        self.game_state["logs"] = []

        return (
            {
                0: TWENTY_QUESTIONS_PROMPT.format(player_id=0),
            },
            {
                "target_word": self.game_state["target_word"],
            },
        )

    def _get_prompt(self, player_id: int) -> str:
        """Generate the prompt for each player, providing them with instructions.
        Args:
            player_id (int): ID of the player (0 or 1).
        Returns:
            str: The prompt for the player.
        """
        history = "\n\n".join(self.game_state["logs"])
        end_q = f"\n\n\nPlayer {player_id}, ask a yes-or-no question about the object."
        return (
            TWENTY_QUESTIONS_PROMPT.format(player_id=0) + "history:\n" + history + end_q
        )

    def step(
        self, player_id: int, action: str
    ) -> tuple[dict[int, str], dict[int, any], bool, bool, dict[int, any]]:
        """
        Take a step in the environment.
        Args:
            player_id (int): ID of the player taking the action.
            action (dict[int, str]): The action taken by the player.
        Returns:
            Tuple containing:
            - observations (Optional[Dict[int, str]]): New observations for each player.
            - rewards (Optional[Dict[int, int]]): Reward for each player.
            - truncated (bool): Whether the episode has been truncated (e.g., time limit reached).
            - terminated (bool): Whether the episode has been terminated (e.g., goal reached).
            - info (Dict[str, Any]): Additional information about the environment.
        """
        # Get the current turn
        turn = self.game_state["turn"]
        self.game_state["logs"].append(f"Player {player_id}: {action}")

        # Check if the player has made a final guess
        final_guess = re.search(FINAL_GUESS_REGEX, action, flags=FLAGS)
        if final_guess:
            guess = final_guess.group(1).strip().lower()
            if guess == self.game_state["target_word"].lower():
                reward = 1
                terminated = True
            else:
                reward = 0
                terminated = True
            return (
                {player_id: f"Final guess: {guess}"},
                {player_id: reward},
                False,
                terminated,
                {
                    "reason": f"Player made a final guess {guess}. The target word was {self.game_state['target_word']}",
                    "outcome": (
                        "You guessed correctly!"
                        if reward == 1
                        else "You guessed incorrectly."
                    ),
                },
            )

        # Increment the turn
        self.game_state["turn"] += 1
        # Check if the game has ended
        if turn >= self.num_questions:
            return (
                {player_id: "You have run out of questions. You lose."},
                {player_id: -1},
                True,  # truncated
                False,
                {
                    "reason": "Player ran out of questions",
                    "outcome": "You lost!",
                },
            )

        # Get response from judge to the question
        judge_prompt = JUDGE_PROMPT.format(
            target_word=self.game_state["target_word"],
            question=action,
        )
        judge_response = self.judge_generate_fn(judge_prompt)
        judge_response_match = re.match(RESPONSE_REGEX, judge_response, flags=FLAGS)
        if judge_response_match:
            response = judge_response_match.group(1)
            if response == "yes":
                self.game_state["logs"].append("Judge: Yes")
            elif response == "no":
                self.game_state["logs"].append("Judge: No")
            else:
                self.game_state["logs"].append("Judge: I don't know")
        else:
            self.game_state["logs"].append("Judge: Invalid response")
            return (
                {player_id: "Invalid response. Please ask a yes-or-no question."},
                {player_id: 0},
                False,
                False,
                {
                    "reason": "Invalid response from the judge",
                },
            )

        return (
            {player_id: self._get_prompt(player_id)},
            {player_id: 0},
            False,
            False,
            {
                "reason": "Player asked a question",
            },
        )

    def render(self):
        # print the logs
        turn_info = f"Turn {self.game_state['turn']}/{self.game_state['max_turns'] if self.game_state['max_turns'] else '∞'}, Target word: {self.game_state['target_word']}"
        print(turn_info)
        print("Last actions:")
        for log in self.game_state["logs"][-1:]:
            print(log)


def load_word_list(basic_word_list: bool = False) -> None:
    """
    Load the word list as specified
    """
    # get word list
    if basic_word_list:
        word_list = words.words("en-basic")
    else:
        word_list = words.words("en")

    # Filter words based on POS tags
    # NN: Noun
    return [word for word in word_list if pos_tag([word])[0][1] in ["NN"]]
