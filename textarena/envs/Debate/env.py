import os, json, random
import importlib.resources
from typing import Optional, Dict, Any, Tuple

import textarena as ta
from textarena.envs.Debate.renderer import create_board_str

class DebateEnv(ta.Env):
    """
    Environment for a two-player Debate game. Two players (Affirmative and Negative)
    alternate turns to debate a randomly chosen topic. A jury votes before and after
    the debate, and the side that gains the most support is declared the winner.
    """

    def __init__(
        self,
        max_turns: Optional[int] = 4,
        jury_class: Optional[Any] = None,
        jury_size: Optional[int] = 5,
        topics_path: Optional[str] = None,
    ):
        """
        Initialize the Debate game.

        Args:
            max_turns (int, optional): Number of turns total (for both players). Must be even.
            jury_class (Any, optional): A Jury class or factory function that returns a Jury-like object.
                Defaults to OpenRouterJury if None.
            jury_size (int, optional): Number of models in the jury. Defaults to 5.
            topics_path (str, optional): Path to the JSON file containing debate topics.
                Defaults to "textarena/envs/two_player/Debate/topics.json".
        """
        if jury_class is None:
            from textarena.utils import OpenRouterJury  # or from your local import
            jury_class = OpenRouterJury

        assert max_turns % 2 == 0, f"Please use an even number of max turns. Current: {max_turns}"

        self.max_turns = max_turns

        # Load debate topics
        self._load_topics(topics_path)

        # Initialize jury
        self.jury = jury_class(jury_size=jury_size, options=["Affirmative", "Negative"])

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _load_topics(self, topics_path: Optional[str]):
        """
        Load debate topics from a JSON file.

        The JSON must have the format:
        {
            "topics": [
                "Topic 1",
                "Topic 2",
                ...
            ]
        }

        Args:
            topics_path (str, optional): Path to the JSON file containing debate topics.

        Raises:
            FileNotFoundError: If the `topics_path` does not exist.
            ValueError: If the JSON file has an invalid format or is empty.
        """
        try:
            if topics_path is not None:
                # Use provided path
                if not os.path.exists(topics_path):
                    raise FileNotFoundError(f"Topics data file not found at: {topics_path}")
                with open(topics_path, "r", encoding="utf-8") as file:
                    self.topics = json.load(file)["topics"]
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.Debate').joinpath('topics.json').open('r') as file:
                    self.topics = json.load(file)["topics"]
        except Exception as e:
            raise FileNotFoundError(f"Failed to load topics data: {str(e)}")
        
        if not self.topics:
            raise ValueError("Debate topics list is empty.")

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the game to its initial state """
        # Initialize game state
        self.state = ta.State(num_players=2, min_players=2, max_players=2, max_turns=self.max_turns, check_truncated=False)

        # Assign sides randomly
        affirmative_player_id = random.choice([0, 1])

        # Initialize game state
        game_state = {
            "arguments": {0: [], 1: []},
            "topic": random.choice(self.topics),
            "sides": {
                affirmative_player_id: "Affirmative",
                1 - affirmative_player_id: "Negative"
            },
            "votes": {
                "pre-debate": {"Affirmative": 0, "Negative": 0},
                "post-debate": {"Affirmative": 0, "Negative": 0}
            }
        }

        # Get pre-debate jury vote
        game_state["votes"]["pre-debate"] = self._evaluate_debate(
            topic=game_state["topic"],
            debate_transcript=None,
        )

        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the prompt for a player when it is their turn """
        prompt = (
            f"You are Player {player_id} in the Debate game.\n"
            f"Topic: {game_state['topic']}\n"
            f"Your position: {game_state['sides'][player_id]}\n"
            f"You will have {self.max_turns} total turns (shared between both players) "
            f"to present your arguments. On your turn, type your argument.\n"
        )
        return prompt

    def _evaluate_debate(self, topic: str, debate_transcript: Optional[str]=None) -> Dict[str, float]:
        """
        Conduct evaluation by the simulated jury.

        If `debate_transcript` is None, the jury simply votes based on the topic alone.
        Otherwise, the jury votes based on the entire debate transcript.

        Args:
            topic (str): The debate topic.
            debate_transcript (Optional[str]): The transcript of the debate so far.

        Returns:
            Dict[str, float]: A dictionary with sides ('Affirmative' or 'Negative')
            as keys, and normalized votes (0-1) as values.
        """
        prompt = f"Debate Topic: {topic}\n"
        if debate_transcript:
            prompt += (
                f"Debate Transcript:\n{debate_transcript}\n"
                "Please vote for either 'Affirmative' or 'Negative'."
            )
        else:
            prompt += (
                "No debate has occurred yet. Please vote based solely on the topic.\n"
                "Vote for either 'Affirmative' or 'Negative'."
            )

        return self.jury.evaluate(context=prompt)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Take a single step in the environment """
        current_player_id = self.state.current_player_id

        # Record the player's action
        self.state.add_observation(from_id=current_player_id, to_id=-1, message=action)
        self.state.game_state["arguments"][current_player_id].append(action)

        # Check if the debate has ended
        if self.state.turn >= self.state.max_turns - 1:
            winner_id = self._determine_debate_winner()
            if winner_id is None:
                self.state.set_draw(reason="The jury's opinion did not favor either side more.")
            else:
                reason=f"Player {winner_id} wins by gaining more support."
                self.state.set_winners(player_ids=[winner_id], reason=reason)

        return self.state.step()

    def _determine_debate_winner(self) -> Optional[int]:
        """
        Determine the winner of the debate by comparing pre- and post-debate jury votes.

        Returns:
            Optional[int]: The winning player's ID, or None if there is a tie.
        """
        # Compile the debate transcript
        transcript_lines = []
        max_rounds = max(
            len(self.state.game_state["arguments"][0]),
            len(self.state.game_state["arguments"][1])
        )
        for i in range(max_rounds):
            if i < len(self.state.game_state["arguments"][0]):
                transcript_lines.append(
                    f"Player 0 ({self.state.game_state['sides'][0]}): "
                    f"{self.state.game_state['arguments'][0][i]}"
                )
            if i < len(self.state.game_state["arguments"][1]):
                transcript_lines.append(
                    f"Player 1 ({self.state.game_state['sides'][1]}): "
                    f"{self.state.game_state['arguments'][1][i]}"
                )

        debate_transcript = "\n".join(transcript_lines)

        # Conduct post-debate voting
        post_votes = self._evaluate_debate(
            topic=self.state.game_state["topic"],
            debate_transcript=debate_transcript
        )
        self.state.game_state["votes"]["post-debate"] = post_votes

        # Calculate vote gains
        pre_votes = self.state.game_state["votes"]["pre-debate"]
        gain_aff = post_votes["Affirmative"] - pre_votes["Affirmative"]
        gain_neg = post_votes["Negative"] - pre_votes["Negative"]

        # Determine winner or tie
        if gain_aff > gain_neg:
            winner_side = "Affirmative"
        elif gain_neg > gain_aff:
            winner_side = "Negative"
        else:
            return None  # tie

        # Map winning side to player ID
        for pid, side in self.state.game_state["sides"].items():
            if side == winner_side:
                return pid

        return None
