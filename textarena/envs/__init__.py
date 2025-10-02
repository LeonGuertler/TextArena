""" Register VendingMachine environment """ 

from textarena.envs.registration import register_with_versions
from textarena.wrappers import ActionFormattingWrapper

# Vending Machine (2 players, simple)
from textarena.envs.VendingMachine.wrapper import VendingMachineObservationWrapper
register_with_versions(
    id="VendingMachine-v0",
    entry_point="textarena.envs.VendingMachine.env:VendingMachineEnv",
    wrappers={"default": [VendingMachineObservationWrapper, ActionFormattingWrapper], "-train": [VendingMachineObservationWrapper, ActionFormattingWrapper]},
)