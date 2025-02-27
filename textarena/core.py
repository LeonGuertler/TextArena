from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

import random

GAME_ID = -1  # literal for use in game messages
Message = Tuple[int, str]  # maps role to content
Observations = dict[int, List[Message]]  # consists of the message seen by each player after the action
Rewards = Dict[int, int]  # maps player ID to reward
Info = Dict[str, Any]  # additional information about the environment


class State:
    """
    A class to manage the state of the game,
    including observations, rewards, and some game logic.
    """
    def __init__(
        self,
        num_players: int,
        min_players: Optional[int] = None,
        max_players: Optional[int] = None,
        even_player_num: Optional[int] = False,
        odd_player_num: Optional[int] = False,
        max_turns: Optional[int] = None,
        role_mapping: Optional[Dict[int, str]] = {},
        check_truncated: Optional[bool] = True,
        error_allowance: Optional[int] = 1,
    ):
        """
        Initialize the State object.

        Args:
            num_players (int): Number of players in the game.
            max_turns (Optional[int]): Maximum number of turns before the game is truncated.
            role_mapping (Optional[Dict[int, str]]): Mapping from player IDs to role names.
            check_truncated (Optional[bool]): Whether to check for truncated games.
            error_allowance (Optional[int]): Number of errors allowed before a player loses the game.
        """
        # assert the number of players
        assert min_players<=num_players<=max_players, \
            f"The number of players needs to be in range {min_players}<=num_player<={max_players}. You provided {num_players}"

        if even_player_num:
            assert num_players%2==0, \
                f"The number of players needs to be even. You provided {num_players}"

        if odd_player_num:
            assert num_players%2==1, \
                f"The number of players needs to be odd. You provided {num_players}"

        self.num_players = num_players
        self.max_turns = max_turns
        self.check_truncated = check_truncated
        self.error_allowance = error_allowance
        self.error_count = 0

        # set standard state parameters
        self.logs = []
        self.turn = 0
        self.current_player_id = 0

        self.role_mapping = role_mapping
        self.role_mapping[-1] = "GAME"


    def reset(
        self,
        game_state: Dict[str, Any],
        player_prompt_function: Callable,
        executable_on_reset: Optional[List[Callable]] = None,
        seed: Optional[int] = None,
        role_mapping = None
    ):
        """
        Reset the game state.

        Args:
            game_state (Dict[str, Any]): Initial game state to be set.
        """
        self.game_state = game_state
        self.current_player_id = 0
        self.turn = 0
        self.prevent_player_change = False 

        if not role_mapping is None:
            for pid, role in role_mapping.items():
                self.role_mapping[pid] = role

        self.logs.append((GAME_ID, "Game started."))

        if seed is not None:
            random.seed(seed)

        self._reset_game_parameters()

        # generate the player prompts
        for player_id in range(self.num_players):
            self.add_observation(
                from_id=GAME_ID,
                to_id=player_id,
                message=player_prompt_function(
                    player_id=player_id,
                    game_state=self.game_state
                ),
                for_logging=False,
            )
        # try to execute relevant functions
        if executable_on_reset is not None:
            for executable in executable_on_reset:
                executable()

    def _reset_game_parameters(self):
        """
        Reset the game parameters at the start of the game or after each step.
        """
        self.done = False 
        self.info = {}
        self.rewards = None
        self.observations = {pid: [] for pid in range(self.num_players)}

    def add_observation(self, from_id: int, to_id: int, message: str, for_logging: bool = True):
        """
        Add an observation message to the observations and logs.

        Args:
            from_id (int): The ID of the sender (player or game).
            to_id (int): The ID of the receiver (-1 for all players).
            message (str): The message content.
            for_logging (bool): If True, the message is added to the logs.
        """
        if for_logging:
            # log the observation
            self.logs.append((from_id, message))

        # add to observations
        if to_id == -1:
            for pid in range(self.num_players): #self.observations:
                self.observations[pid].append((from_id, message))
        else:
            assert (
                to_id in self.observations
            ), f"The provided 'to_id' {to_id} does not exists. ({list(self.observations.keys())})"
            self.observations[to_id].append((from_id, message))

    def add_log(self, from_id: int, message: str):
        """TODO"""
        self.logs.append((from_id, message))

    def step(self, rotate_player : bool = True):
        """
        Advance the game state by one turn.

        Args:
            rotate_player  (bool): Whether to rotate the current player after the step.

        Returns:
            Tuple[
                bool,                               # done
                Dict[str, Any],                     # info
            ]: The updated game state after the step.
        """
        if self.done:  # if game happens to be terminated on last turn...
            return (True, self.info)

        # increment turn counter
        if not self.prevent_player_change:
            self.turn += 1

        # check if the turn limit has been reached
        if (
            self.max_turns is not None
            and self.turn >= self.max_turns
            and self.check_truncated
        ):
            # set the rewards
            self.rewards = {pid: 0 for pid in range(self.num_players)}

            # log the reason & update info
            reason = "Turn limit reached"
            self.logs.append((GAME_ID, reason))
            self.info["detailed_reason"] = reason
            self.info["reason"] = "Draw."
            self.done = True

        info = self.info
        done = self.done
        rewards = self.rewards

        # update current player
        if rotate_player and not self.prevent_player_change:
            prev_player = self.current_player_id
            self.current_player_id = (self.current_player_id + 1) % self.num_players
            self.error_count = 0

        self.info = {}
        self.reward = None
        self.prevent_player_change = False

        return (done, info)

    def manually_update_current_player(self, new_player_id):
        if not self.prevent_player_change:
            self.current_player_id = new_player_id
            self.error_count = 0

    def get_current_player_observation(self):
        current_player_observation = self.observations[self.current_player_id]
        # reset observations
        self.observations[self.current_player_id] = []
        return current_player_observation

    def close(self):
        return self.rewards

    def set_winners(self, player_ids: List[int], reason: Optional[str]):
        """
        Set the winners of the game.

        Args:
            player_ids (List[int]): List of player IDs who have won.
            reason (Optional[str]): Reason for winning.
        """

        # set the rewards
        self.rewards = {}
        for player_id in range(self.num_players):
            if player_id in player_ids:
                self.rewards[player_id] = 1
            else:
                self.rewards[player_id] = -1

        # log the reason & update info
        self.logs.append((GAME_ID, reason))
        self.add_observation(
            from_id=GAME_ID,
            to_id=-1,
            message=f"Player {player_ids[0]} won the game. Reason: {reason}",
            for_logging=False
        )
        self.info["reason"] = reason
        self.done = True

    def set_draw(self, reason: Optional[str]):
        """
        Declare the game as a draw.

        Args:
            reason (Optional[str]): Reason for the draw.
        """
        # set the rewards
        self.rewards = {pid: 0 for pid in range(self.num_players)}

        # log the reason & update info
        self.logs.append((GAME_ID, reason))
        self.add_observation(
            from_id=GAME_ID,
            to_id=-1,
            message=f"The game ended in a draw. Reason: {reason}",
            for_logging=False
        )
        self.info["reason"] = reason
        self.done = True

    def set_invalid_move(self, player_id: int, reason: Optional[str]):
        """
        Handle an invalid move made by a player.

        Args:
            player_ids (int): Invalid move player id.
            reason (str): Reason for the invalid move.
        """
        if self.error_allowance > self.error_count:
            # increment error count
            self.error_count += 1
            self.prevent_player_change = True 
            self.add_observation(
                from_id=GAME_ID,
                to_id=-1, # Broadcast (would only be an issue if info is secret)
                message=(
                    f"Player {player_id} attempted an invalid move. Reason: {reason} "
                    f"Please resubmit a valid move and remember to follow the game rules to avoid penalties."
                ),
                for_logging=True
            )
        else:
            # set the rewards
            self.rewards = {}
            for pid in range(self.num_players):
                if pid == player_id:
                    self.rewards[pid] = -1
                else:
                    self.rewards[pid] = 0

            # log the reason & update info
            self.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=f"Player {player_id} lost the game by way of invalid move. Reason: {reason}",
                for_logging=True
            )
            self.info["reason"] = f"Invalid Move: {reason}"
            self.done = True

class Env(ABC):
    """
    Abstract base class for text-based game environments.

    This class outlines the interface for the environment, including methods for resetting the environment,
    stepping through the environment (taking actions), and rendering the environment state.
    """
    game_state: State  # the state of the environment

    @abstractmethod
    def reset(self, num_players: int, seed: Optional[int]=None):
        """
        Resets the environment to an initial state.

        Args:
            num_players (int): Number of players in the game.
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> Tuple[bool, Info]:
        """
        Performs a single step in the environment.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be taken by the player.

        Returns:
            Tuple containing:
                - done (bool): Whether the episode has concluded
                - info (Dict[str, Any]): Additional information about the environment.
        """
        raise NotImplementedError
    
    @property
    def offline_renderer(self):
        raise NotImplementedError
    
    @property
    def terminal_render_keys(self):
        # If this instance is wrapping another environment, delegate to it
        if hasattr(self, 'env') and self.env is not None:
            return self.env.terminal_render_keys
        # Default for non-wrapped (base) environments that don't override
        return []

    def get_observation(self):
        return self.state.current_player_id, self.state.get_current_player_observation()

    def close(self):
        rewards = self.state.close()
        return rewards


class Wrapper(Env):
    """ Base class for environment wrappers. """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, num_players: int , seed: Optional[int] = None):
        return self.env.reset(num_players=num_players, seed=seed)


    def step(self, action: str) -> Tuple[bool, Info]:
        return self.env.step(action=action)

    def get_observation(self):
        return self.env.get_observation()

    def close(self):
        return self.env.close()


class ObservationWrapper(Wrapper):

    def get_observation(self):
        player_id, observation = self.env.get_observation()
        return player_id, self.observation(player_id, observation)

    def observation(self):
        raise NotImplementedError


class RenderWrapper(Wrapper):
    def step(self, action: str) -> Tuple[bool, Optional[Info]]:
        return self.env.step(action=action)

    
    def reset(self, num_players: int , seed: Optional[int] = None):
        self.reset_render()
        return self.env.reset(num_players=num_players, seed=seed)


    def reset_render(self):
        raise NotImplementedError

class ActionWrapper(Wrapper):
    def step(self, action: str) -> Tuple[bool, Optional[Info]]:
        return self.env.step(action=self.action(action))

    def action(self, action: str) -> str:
        """
        Transforms the action.

        Args:
            action (str): The original action.

        Returns:
            str: The transformed action.
        """
        raise NotImplementedError


class Agent(ABC):
    """
    Generic agent class that defines the basic structure of an agent.
    """
    @abstractmethod
    def __call__(self, observation: str) -> str:
        """
        Process the observation and return the action.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The response generated by the agent.
        """
        pass


class AgentWrapper(Agent):
    """ TODO """
    def __init__(self, agent: Agent):
        """ TODO """
        self.agent = agent 
        assert isinstance(agent, Agent)

    def __getattr__(self, name):
        """ TODO """
        return getattr(self.agent, name)

    def __call__(self, observation: str) -> str:
        return self.agent(observation=observation)