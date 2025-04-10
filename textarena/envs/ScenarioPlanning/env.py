import os, json, random
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena import utils
from textarena.envs.ScenarioPlanning.renderer import create_board_str


class ScenarioPlanningEnv(ta.Env):
    """ Environment for the Scenario Planning Game """

    def __init__(
        self,
        jury_class: Optional[Any] = None,
        jury_size: Optional[int] = 5,
        scenarios_path: Optional[str] = None,
    ):
        """
        Initialize the Scenario Planning game environment.

        Args:
            num_judges (int): Number of judges evaluating the strategies.
            judge_class (ta.JudgeVote): The judge evaluation class.
            scenarios_path (str): Path to the JSON file containing scenarios.
        """
        if jury_class is None:
            from textarena.utils import OpenRouterJury  # or from your local import
            jury_class = OpenRouterJury

        # Load scenarios
        self._load_scenarios(scenarios_path)

        # Initialize judges
        self.judge = jury_class(jury_size=jury_size, options=["Player 0", "Player 1"])

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _load_scenarios(self, scenarios_path: Optional[str]):
        """
        Load scenarios from the JSON file.

        Args:
            scenarios_path (str): Path to the JSON file containing scenarios.
        """
        if scenarios_path is None:
            scenarios_path = os.path.join(
                "textarena",
                "envs",
                "ScenarioPlanning",
                "scenarios.json",
            )

        if not os.path.exists(scenarios_path):
            raise FileNotFoundError(f"Scenarios file not found at {scenarios_path}")

        with open(scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "scenarios" not in data or not isinstance(data["scenarios"], list):
            raise ValueError(
                "Invalid format for scenarios JSON. Expected a key 'scenarios' with a list of scenarios."
            )

        self.scenarios = data["scenarios"]
        if not self.scenarios:
            raise ValueError("Scenarios list is empty.")

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the Scenario Planning game to its initial state """
        # Initialize game state variables
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2)

        # Select a random scenario
        self.selected_scenario = random.choice(self.scenarios)

        # reset game state 
        game_state = {
            "strategies": {0: None, 1: None},
            "scenario": self.selected_scenario,
            "votes": {
                0: {"Votes": 0},
                1: {"Votes": 0}

            }
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player based on the scenario """
        prompt = (
            f"You are Player {player_id} in the Scenario Planning game.\n"
            f"Scenario: {self.selected_scenario}\n"
            "Your goal is to propose a strategy for survival in this scenario.\n"
            "After both players submit their strategies, a panel of judges will evaluate them.\n"
            "On your turn, simply type your strategy."
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's strategy """

        player_id = self.state.current_player_id

        # update the log
        self.state.add_log(from_id=player_id, message=action)
        
        # Store the strategy
        self.state.game_state["strategies"][player_id] = action
        
        # check if both players have submitted their strategies
        if all(
            strategy is not None for strategy in self.state.game_state["strategies"].values()
        ):
            # Conduct judging
            votes = self._evaluate_strategies()
            self.state.game_state["votes"] = {
                0: {"Votes": votes["Player 0"]},
                1: {"Votes": votes["Player 1"]}
            }
            
            # check for draw firs
            if votes["Player 0"] == votes["Player 1"]:
                self.state.set_draw(reson="An equal number of judges voted for each option.")
            else:
                # get winner id
                winner_id = 0 if votes["Player 0"] > votes["Player 1"] else 1
                self.state.set_winners(
                    player_ids=[winner_id],
                    reason=f"Player {winner_id} wins by convincing the judges."
                )
            
        return self.state.step()


    def _evaluate_strategies(self) -> Dict[str, int]:
        """
        Conduct evaluation by judges based on the submitted strategies.

        Returns:
            Dict[str, int]: A dictionary with 'Player 0' and 'Player 1' as keys and their corresponding vote counts.
        """
        prompt = (
            f"Scenario: {self.state.game_state['scenario']}\n\n"
            f"Player 0's Strategy:\n{self.state.game_state['strategies'][0]}\n\n"
            f"Player 1's Strategy:\n{self.state.game_state['strategies'][1]}\n\n"
            f"Based on the above strategies, which player's strategy is more effective and feasible for survival?\n"
            f"Vote for 'Player 0' or 'Player 1'. Provide only the player you vote for."
        )
        votes = self.judge.evaluate(context=prompt)
        return votes