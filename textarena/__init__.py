""" Root __init__ of textarena """

from textarena.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    RenderWrapper,
    ActionWrapper,
    State,
    Message,
    Observation,
    Reward,
    Info,
    GAME_ID,
)
from textarena.envs.registration import (
    make,
    register,
    pprint_registry,
    pprint_registry_detailed,
    find_highest_version,
    check_env_exists,
)

from textarena import wrappers

__all__ = [
    # core
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "RenderWrapper",
    "ActionWrapper",
    # registration
    "make",
    "register",
    "pprint_registry",
    "pprint_registry_detailed",
    "find_highest_version",
    "check_env_exists",
    # module folders
    "envs",
    "utils",
    "wrappers",
]
__version__ = "1.0.0"
