from textarena.core import ActionWrapper, Env

__all__ = ["ActionFormattingWrapper"]


class ActionFormattingWrapper(ActionWrapper):
    """ TODO """

    def __init__(self, env: Env):
        """ TODO """
        super().__init__(env)

    def action(self, action: str) -> str:
        """ TODO """
        if "[" not in action and "]" not in action:
            return f"[{action}]"
        else:
            return action