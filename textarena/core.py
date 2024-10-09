from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

GAME_ID = -1  # literal for use in game messages
Message = tuple[int, str]  # maps role to content
Observation = dict[
    int, list[Message]
]  # consists of the message seen by each player after the action
Reward = dict[int, int]  # maps player ID to reward
Info = dict[str, Any]  # additional information about the environment


class State(dict):
    """
    A class to represent the state of the environment, allowing for both
    dictionary-style and attribute-style access.

    Default keys include:
    - 'render': a mapping of what to render
    - 'logs': a list of messages where -1 indicates a game message
    - 'player_map': a mapping of player IDs to names (e.g. Black/White in chess)

    Note that using kwargs is incompatible with the 'state_dict' parameter.
    """

    def __init__(
        self,
        state_dict: Optional[dict[str, Any]] = None,
        render: Optional[dict] = None,
        logs: Optional[list[Message]] = None,
        player_map: Optional[dict[int, str]] = None,
        **kwargs,
    ):
        # Initialize from a dict if provided
        if state_dict:
            super().__init__(state_dict)
        else:
            super().__init__(**kwargs)

        # Ensure 'render' and 'logs' have defaults, but allow them to be overridden
        self.render = render if render is not None else self.get("render", {})
        self.logs = logs if logs is not None else self.get("logs", [])
        self.player_map = player_map

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'State' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'State' object has no attribute '{item}'")


class Env(ABC):
    """
    Abstract base class for text-based game environments.

    This class outlines the interface for the environment, including methods for resetting the environment,
    stepping through the environment (taking actions), and rendering the environment state.
    """

    environment_name: str  # the name of the environment
    game_state: State  # the state of the environment

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None
    ) -> tuple[Optional[Observation], Optional[Info]]:
        """
        Resets the environment to an initial state.

        Args:
            observations (Optional[Dict[int, str]]): Initial observations for the players.
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.

        Returns:
            Depending on implementation, should return initial observations and any additional info.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        player_id: int,
        action: str,
    ) -> tuple[
        Observation,  # player-wise observations
        Reward,  # player-wise reward
        bool,  # truncated
        bool,  # terminated
        Info,  # info
    ]:
        """
        Performs a single step in the environment.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be taken by the player.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): New observations for each player.
                - rewards (Optional[Dict[int, int]]): Reward for each player.
                - truncated (bool): Whether the episode has been truncated (e.g., time limit reached).
                - terminated (bool): Whether the episode has been terminated (e.g., goal reached).
                - info (Dict[str, Any]): Additional information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """
        Renders the current state of the environment.

        This method should output a representation of the environment's state to the console or other output.
        """
        raise NotImplementedError


class Wrapper(Env):
    """
    Base class for environment wrappers.

    This class wraps an environment to allow modular transformations or extensions of its functionality.
    """

    def __init__(self, env: Env):
        """
        Initialize the Wrapper.

        Args:
            env (Env): The environment to wrap.
        """
        self.env = env
        self.environment_name = env.environment_name
        self.game_state = env.game_state
        assert isinstance(env, Env)

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped environment.

        This method is called only if the attribute wasn't found the usual ways.
        It allows the wrapper to pass through attributes and methods to the underlying environment.

        Args:
            name (str): The attribute name.

        Returns:
            The attribute from the wrapped environment.
        """
        return getattr(self.env, name)

    def reset(
        self, seed: Optional[int] = None
    ) -> tuple[Optional[Observation], Optional[Info]]:
        """
        Resets the environment and returns initial observations.

        Args:
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): Initial observations for each player.
                - info (Optional[Dict[str, Any]]): Additional information about the environment.
        """
        return self.env.reset(seed=seed)

    def step(
        self,
        player_id: int,
        action: str,
    ) -> tuple[
        Optional[Observation],  # player-wise observations
        Optional[Reward],  # player-wise reward
        bool,  # truncated
        bool,  # terminated
        Info,  # info
    ]:
        """
        Performs a step in the environment with the given action.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be taken.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): Observations after the action.
                - rewards (Optional[Dict[int, int]]): Rewards for each player.
                - truncated (bool): Whether the episode is truncated.
                - terminated (bool): Whether the episode is terminated.
                - info (Optional[Dict[str, Any]]): Additional information.
        """
        return self.env.step(player_id=player_id, action=action)

    def render(self):
        """
        Renders the environment.

        This method calls the render method of the wrapped environment.
        """
        return self.env.render()


class ObservationWrapper(Wrapper):
    """
    Abstract base class for observation wrappers.

    This class is used to modify the observations returned by the environment.
    Subclasses should implement the `observation` method to define how observations are transformed.
    """

    def reset(
        self, _: Optional[int] = None
    ) -> tuple[Observation, Info]:  # observations and info
        """
        Resets the environment and applies the observation transformation.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): Transformed observations.
                - info (Dict[str, Any]): Additional information.
        """
        raise NotImplementedError
        # observations, info = self.env.reset(seed=seed)
        # return self.observation(observations), info

    def step(
        self,
        player_id: int,
        action: str,
    ) -> tuple[
        Observation,  # player-wise observations
        Reward,  # player-wise reward
        bool,  # truncated
        bool,  # terminated
        Info,  # info
    ]:
        """
        Performs a step in the environment with the given action.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be taken.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): Observations after the action.
                - rewards (Optional[Dict[int, int]]): Rewards for each player.
                - truncated (bool): Whether the episode is truncated.
                - terminated (bool): Whether the episode is terminated.
                - info (Optional[Dict[str, Any]]): Additional information.
        """
        observations, reward, truncated, terminated, info = self.env.step(
            player_id=player_id, action=action
        )
        return self.observation(observations), reward, truncated, terminated, info

    def observation(
        self, observations: Observation  # player-wise observations
    ) -> Optional[Dict[int, str]]:
        """Transforms the observations.

        Args:
            observations (Optional[Dict[int, str]]): The observations to transform.

        Returns:
            Optional[Dict[int, str]]: The transformed observations.
        """
        raise NotImplementedError


class RenderWrapper(Wrapper):
    """
    Abstract base class for render wrappers.

    This class is used to modify the rendering of the environment.
    Subclasses should implement the `render` method to define custom rendering behavior.
    """

    def render(self):
        """
        Renders the environment.

        Subclasses should implement this method to provide custom rendering behavior.
        """
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """
    Abstract base class for action wrappers.

    This class is used to modify the actions before they are passed to the environment.
    Subclasses should implement the `action` method to define how actions are transformed.
    """

    def step(self, player_id: int, action: str) -> tuple[
        Optional[Observation],
        Optional[Reward],
        bool,
        bool,
        Optional[Info],
    ]:
        """
        Performs a step in the environment with the transformed action.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be transformed and taken.

        Returns:
            Tuple containing:
                - observations (Optional[Dict[int, str]]): Observations after the action.
                - rewards (Optional[Dict[int, int]]): Rewards for each player.
                - truncated (bool): Whether the episode is truncated.
                - terminated (bool): Whether the episode is terminated.
                - info (Optional[Dict[str, Any]]): Additional information.
        """
        return self.env.step(player_id=player_id, action=self.action(action))

    def action(self, action: str) -> str:
        """
        Transforms the action.

        Args:
            action (str): The original action.

        Returns:
            str: The transformed action.
        """
        raise NotImplementedError
