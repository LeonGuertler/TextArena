"""Environment for playing the game iterated prisoners dilemma with chat

We have two players, each of whom can choose to cooperate or defect.
The players are rewarded based on their choices and the choices of the other player.
The game is played for a random number of rounds.
The players can communicate with each other using chat messages.
"""

import re
import random
import textarena as ta

MIN_TURNS = 1

IPD_PROMPT = """You are playing 'Iterated Prisoners Dilemma' with chat. You are Player {player_id}.
You can choose to 'cooperate' or 'defect' in each round.
Your goal is to maximize your score by choosing the best strategy.
You can also send chat messages to the other player.
The game will last for a random number of rounds.
You receive points based on the following rules:
- If both players cooperate, both get 3 points.
- If one player cooperates and the other defects, the defector gets 5 points and the cooperator gets 0 points.
- If both players defect, both get 1 point.
Good luck!"""

CHOICE_REGEX = r"Choice: (cooperate|defect)"


class IteratedPrisonersDilemma(ta.Env):
    """Environment for playing the game iterated prisoners dilemma with chat."""

    def __init__(self, chat_turns_per_round, max_turns=30) -> None:
        """
        Initialize the environment.

        Args:
            chat_turns_per_round (int): Number of chat turns allowed per round.
            max_turns (int): Maximum number of rounds to play.
        """
        self.chat_turns_per_round = chat_turns_per_round + 1
        self.max_turns = max_turns
        self.num_rounds = max(int(random.betavariate(2, 5) * self.max_turns), MIN_TURNS)
        self.environment_name = "Iterated Prisoners Dilemma"

        self.game_state = ta.State(
            {
                "turn": 0,
                "sub_turn": 0,
                "max_turns": self.max_turns,
                "logs": [],
                "player_scores": {0: 0, 1: 0},
                "player_choices": {0: None, 1: None},
                "render": ["player_scores"],
            }
        )

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
        self.game_state["player_scores"] = {0: 0, 1: 0}
        self.game_state["sub_turn"] = 0
        self.game_state["logs"] = []
        self.game_state["player_choices"] = {0: None, 1: None}

        return (
            {
                0: [self._initial_prompt(player_id=0)],
                1: [self._initial_prompt(player_id=1)],
            },
            {
                "player_0_score": self.game_state["player_scores"][0],
                "player_1_score": self.game_state["player_scores"][1],
            },
        )

    def _initial_prompt(self, player_id: int) -> ta.Message:
        """Generate the prompt for each player, providing them with instructions.
        Args:
            player_id (int): ID of the player (0 or 1).
        Returns:
            str: Initial prompt for the player."""

        base_prompt = IPD_PROMPT.format(player_id=player_id)
        turn_prompt = self._get_turn_prompt()
        return ta.GAME_ID, "\n\n".join([base_prompt, turn_prompt])

    def _get_turn_prompt(self) -> str:
        chat_turn = self.game_state["turn"] % self.chat_turns_per_round != 0
        if chat_turn:
            return "Chat turn: You can send a message to the other player."
        else:
            return "Make your choice: 'cooperate' or 'defect'. (reply with either Choice: cooperate or Choice: defect)"

    def compute_scores(self) -> tuple[int, int]:
        """Compute the scores for the current round based on the players' choices.

        Returns:
            Tuple[int, int]: Scores for player 0 and player 1.
        """
        assert self.game_state["player_choices"][0] is not None
        assert self.game_state["player_choices"][1] is not None
        choices = self.game_state["player_choices"]
        if choices[0] == "cooperate" and choices[1] == "cooperate":
            return 3, 3
        if choices[0] == "cooperate" and choices[1] == "defect":
            return 0, 5
        if choices[0] == "defect" and choices[1] == "cooperate":
            return 5, 0
        if choices[0] == "defect" and choices[1] == "defect":
            return 1, 1

    def step(
        self, player_id: int, action: str
    ) -> tuple[ta.Observation, ta.Reward, bool, bool, ta.Info]:
        """Execute a step in the environment.

        Args:
            player_id (int): ID of the player making the action.
            action (str): Action taken by the player.

        Returns:
            Tuple[Dict[int, str], Dict[int, Any], bool, bool, Dict[int, Any]]: Observations, info, truncated, and extra info.
        """
        reward, info, scores = (
            None,
            None,
            None,
        )
        message = (player_id, action)
        ## first figure out if it's a chat turn or a choice turn
        chat_turn = self.game_state["turn"] % self.chat_turns_per_round != 0
        if chat_turn:
            self.game_state["sub_turn"] = 1 - self.game_state["sub_turn"]
            observation = {
                player_id: [message],
                1 - player_id: [message],
            }  # public chat
        else:  # they should have made a choice
            action_match = re.search(CHOICE_REGEX, action)
            if action_match is None:
                terminated = True
                info = {
                    "reason": f"Player {player_id} made an invalid choice: {action}. It should be 'cooperate' or 'defect'."
                }
                reward = {player_id: -1, 1 - player_id: -1}
                return (
                    {
                        player_id: [message],
                    },
                    reward,
                    None,
                    terminated,
                    info,
                )
            action = action_match.group(1)
            self.game_state["player_choices"][player_id] = action
            self.game_state["sub_turn"] = 1 - self.game_state["sub_turn"]
            if self.game_state["sub_turn"] == 0:
                scores = self.compute_scores()
                self.game_state["player_scores"][0] += scores[0]
                self.game_state["player_scores"][1] += scores[1]
                res_message = (
                    ta.GAME_ID,
                    (
                        f"Player {player_id} chose {action} and Player {1 - player_id} chose {self.game_state['player_choices'][1 - player_id]}.",
                        f"Player {player_id} scored {scores[0]} and Player {1 - player_id} scored {scores[1]}.",
                    ),
                )
                self.game_state.logs.append(res_message)
                self.game_state["player_choices"] = {0: None, 1: None}
                observation = {
                    player_id: [message, res_message],
                    1 - player_id: [message, res_message],
                }
            # NO OBSERVATION IF NOT END OF SUBTURN

        truncated = self.game_state["turn"] >= self.num_rounds - 1
        terminated = False
        if truncated:
            info = {"reason": "Game over: maximum number of rounds reached."}
            reward = self.game_state["player_scores"]
        elif self.game_state["sub_turn"] == 1:
            self.game_state["turn"] += 1
            next_turn_message = (ta.GAME_ID, self._get_turn_prompt())
            self.game_state["logs"].append(next_turn_message)
            for player_messages in observation.values():
                player_messages.append(
                    next_turn_message
                )  # add new instruction for new turn
        return observation, reward, truncated, terminated, info

    def render(self):
        """Render minimal game state."""
        print(f"Scores: {self.game_state['player_scores']}")
        print(f"Turn: {self.game_state['turn']}")
        for player_id, log in self.game_state["logs"]:
            if player_id == ta.GAME_ID:
                print(log)
            else:
                print(f"Player {player_id}: {log}")
        print("\n")
