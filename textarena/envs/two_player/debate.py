"""
Debate Game

In this game, two players engage in a debate on a given topic. Each player is assigned a side—Affirmative or Negative.

**Gameplay:**

1. **Topic Selection**: A random debate topic is chosen.
2. **Role Assignment**: Players are randomly assigned to the Affirmative or Negative side.
3. **Pre-Debate Voting**: Simulated judges cast their initial votes based solely on the topic.
4. **Debate Turns**: Players take turns presenting their arguments over a fixed number of turns.
5. **Post-Debate Voting**: Judges cast their votes again after the debate.
6. **Winner Determination**: The player whose side gains the most votes compared to the pre-debate votes wins.

**Key Rules:**

- Players must argue for their assigned side.
- The debate lasts for a predetermined number of turns.
- The winner is determined based on the change in votes from pre-debate to post-debate.

**Parameters:**

- `max_turns`: Number of turns each player has to present their arguments.
- `num_judges`: Number of simulated judges evaluating the debate.
- `topics_path`: Path to the JSON file containing debate topics.

**Game Outcomes:**

- **Win**: The player whose side gains more votes after the debate wins.
- **Tie**: If both sides gain an equal number of votes, the game is a tie.
"""

import json
import os
import random
from typing import Any, Dict, Optional, Tuple
import textarena.utils as gen_utils

import textarena as ta


class DebateEnv(ta.Env):
    """Environment for the Debate game."""

    def __init__(
        self,
        max_turns: Optional[int] = 4,
        num_judges: Optional[int] = 11,
        topics_path: Optional[str] = None,
    ):
        """
        Initialize the Debate game.

        Args:
            max_turns (int): Number of turns per player.
            num_judges (int): Number of judges evaluating the debate.
            topics_path (str): Path to the JSON file containing debate topics.
        """
        self.environment_name = "Debate"

        # Load debate topics
        self._load_topics(topics_path)

        # define the judge models
        self.judge_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
        # Initialize game state
        self.game_state = ta.State(
            {
                "turn": 0,
                "max_turns": max_turns,
                "num_judges": num_judges,
                "topic": None,
                "sides": {},
                "arguments": {0: [], 1: []},
                "pre_votes": {"Affirmative": 0, "Negative": 0},
                "post_votes": {"Affirmative": 0, "Negative": 0},
                "logs": [],
                "render": ["turn", "max_turns", "topic", "sides"],
            }
        )

    def _load_topics(self, topics_path: Optional[str]):
        """
        Load debate topics from the JSON file.

        Args:
            topics_path (str): Path to the JSON file containing debate topics.
        """
        if topics_path is None:
            topics_path = os.path.join(
                "textarena", "envs", "two_player", "data", "debate_topics.json"
            )

        if not os.path.exists(topics_path):
            raise FileNotFoundError(f"Debate topics file not found at {topics_path}")

        with open(topics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "topics" not in data or not isinstance(data["topics"], list):
            raise ValueError(
                "Invalid format for debate topics JSON. Expected a key 'topics' with a list of topics."
            )

        self.topics = data["topics"]
        if not self.topics:
            raise ValueError("Debate topics list is empty.")

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[Dict[int, str]], Dict[int, Any]]:
        """
        Reset the game to its initial state.

        Args:
            seed (Optional[int]): Seed for random number generator to ensure reproducibility.

        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        self.game_state["turn"] = 0
        self.game_state["arguments"] = {0: [], 1: []}
        self.game_state["logs"] = []

        # Select a random topic
        self.game_state["topic"] = random.choice(self.topics)

        # Assign sides randomly
        if random.choice([True, False]):
            self.game_state["sides"] = {0: "Affirmative", 1: "Negative"}
        else:
            self.game_state["sides"] = {0: "Negative", 1: "Affirmative"}

        # Conduct pre-debate voting
        self.game_state["pre_votes"] = self._evaluate_debate(debate_transcript=None)
        self.game_state["logs"].append(
            (-1, f"Pre-Debate Votes: {self.game_state['pre_votes']}")
        )

        # Generate initial prompts for both players
        observations = self._generate_initial_observation([0, 1])

        info = {
            "topic": self.game_state["topic"],
            "sides": self.game_state["sides"],
            "pre_votes": self.game_state["pre_votes"],
        }

        self.game_state["logs"].append("[GAME] New game started.")

        return observations, info

    def _generate_initial_observation(self, ids: list[int]) -> ta.Observation:
        """
        Generate the initial prompt for a player based on their side.

        Args:
            player_id (int): The player's ID (0 or 1).

        Returns:
            str: The initial prompt for the player.
        """
        observations = {}
        for player_id in ids:
            prompt = (
                f"You are Player {player_id} in the Debate game.\n"
                f"Topic: {self.game_state['topic']}\n"
                f"Your position: {self.game_state['sides'][player_id]}\n"
                f"You will have {self.game_state['max_turns']} turns to present your arguments.\n"
                "On your turn, simply type your argument."
            )
            observations[player_id] = [(ta.GAME_ID, prompt)]
        return observations

    def _evaluate_debate(self, debate_transcript: Optional[str]) -> Dict[str, int]:
        """
        Conduct evaluation by simulated judges.

        Args:
            debate_transcript (str): The transcript of the debate. If None, judges evaluate based on the topic only.

        Returns:
            Dict[str, int]: A dictionary with sides as keys and their corresponding vote counts.
        """
        votes = {"Affirmative": 0, "Negative": 0}

        for i in range(self.game_state["num_judges"]):
            model = self.judge_models[i % len(self.judge_models)]
            prompt = f"Debate Topic: {self.game_state['topic']}\n"
            if debate_transcript:
                prompt += f"Debate Transcript:\n{debate_transcript}\n"
                prompt += "Based on the debate, please vote for either 'Affirmative' or 'Negative'. Provide only the side you vote for."
            else:
                prompt += "No debate has occurred yet. Please vote for either 'Affirmative' or 'Negative' based solely on the topic. Provide only the side you vote for."

                judge_decision = (
                    gen_utils.batch_open_router_generate(
                        texts=[prompt],
                        model_string=f"openai/{model}",
                        message_history=None,
                    )
                    .lower()
                    .strip()
                )
                if "affirmative" in judge_decision:
                    votes["Affirmative"] += 1
                elif "negative" in judge_decision:
                    votes["Negative"] += 1
                else:
                    self.game_state["logs"].append(
                        (
                            -1,
                            f"[ERROR] Judge returned illegal evaluation: {judge_decision}",
                        )
                    )
        return votes

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observation],  # observations
        Optional[ta.Reward],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's argument.

        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        other_player_id = 1 - player_id
        terminated = False
        truncated = False
        reward = None
        info = {}
        message = (player_id, action)

        self.game_state["turn"] += 1

        # Store the argument
        self.game_state["arguments"][player_id].append(action)
        self.game_state["logs"].append([message])

        # Observations for both players
        observations = {player_id: [message], other_player_id: [message]}

        # Check if debate is over
        if self.game_state["turn"] >= self.game_state["max_turns"]:
            # Compile the debate transcript
            debate_transcript = ""
            for i in range(len(self.game_state["arguments"][0])):
                if i < len(self.game_state["arguments"][0]):
                    debate_transcript += f"Player 0 ({self.game_state['sides'][0]}): {self.game_state['arguments'][0][i]}\n"
                if i < len(self.game_state["arguments"][1]):
                    debate_transcript += f"Player 1 ({self.game_state['sides'][1]}): {self.game_state['arguments'][1][i]}\n"

            # Conduct post-debate voting
            self.game_state["post_votes"] = self._evaluate_debate(
                debate_transcript=debate_transcript
            )
            self.game_state["logs"].append(
                (-1, f"[GAME] Post-Debate Votes: {self.game_state['post_votes']}")
            )

            # Calculate gains
            gain_affirmative = (
                self.game_state["post_votes"]["Affirmative"]
                - self.game_state["pre_votes"]["Affirmative"]
            )
            gain_negative = (
                self.game_state["post_votes"]["Negative"]
                - self.game_state["pre_votes"]["Negative"]
            )

            # Determine the winner
            if gain_affirmative > gain_negative:
                winner_side = "Affirmative"
            elif gain_negative > gain_affirmative:
                winner_side = "Negative"
            else:
                winner_side = None  # It's a tie

            if winner_side:
                winner_id = [
                    pid
                    for pid, side in self.game_state["sides"].items()
                    if side == winner_side
                ][0]
                reward = {winner_id: 1, 1 - winner_id: -1}
                info["reason"] = f"Player {winner_id} ({winner_side}) wins the debate."
                self.game_state["logs"].append((-1, f"[GAME] {info['reason']}"))
            else:
                # It's a tie
                reward = {0: 0, 1: 0}
                info["reason"] = "The debate is a tie."
                self.game_state["logs"].append((-1, f"[GAME] {info['reason']}"))

            terminated = True

        return observations, reward, truncated, terminated, info

    def render(self):
        """
        Render the current game state.
        """
        print(f"Turn: {self.game_state['turn']}/{self.game_state['max_turns'] * 2}")
        print(f"Topic: {self.game_state['topic']}")
        print(f"Player Sides: {self.game_state['sides']}")
        print("Game Logs:")
        for player_id, log in self.game_state["logs"]:
            if player_id == ta.GAME_ID:
                print(f"GAME: {log}")
            else:
                print(f"Player {player_id}: {log}")
        print("\n")
