from textarena.core import ActionWrapper, Env

__all__ = ["ClipWordsActionWrapper", "ClipCharactersActionWrapper"]


class ClipWordsActionWrapper(ActionWrapper):
    """
    Action wrapper that limits actions to a maximum number of words.

    This wrapper truncates the player's action if it exceeds the specified maximum number of words.
    """

    def __init__(self, env: Env, max_num_words: int):
        """
        Initialize the ClipWordsActionWrapper.

        Args:
            env (Env): The environment to wrap.
            max_num_words (int): The maximum number of words allowed in an action.
        """
        super().__init__(env)
        self.max_num_words = max_num_words

    def action(self, action: str) -> str:
        """
        Truncates the action to the maximum number of words.

        Args:
            action (str): The original action.

        Returns:
            str: The truncated action.
        """
        word_list = action.split()
        if len(word_list) <= self.max_num_words:
            return action
        else:
            # Truncate and return
            return " ".join(word_list[: self.max_num_words])


class ClipCharactersActionWrapper(ActionWrapper):
    """
    Action wrapper that limits actions to a maximum number of characters.

    This wrapper truncates the player's action if it exceeds the specified maximum number of characters.
    """

    def __init__(self, env: Env, max_num_characters: int):
        """
        Initialize the ClipCharactersActionWrapper.

        Args:
            env (Env): The environment to wrap.
            max_num_characters (int): The maximum number of characters allowed in an action.
        """
        super().__init__(env)
        self.max_num_characters = max_num_characters

    def action(self, action: str) -> str:
        """
        Truncates the action to the maximum number of characters.

        Args:
            action (str): The original action.

        Returns:
            str: The truncated action.
        """
        if len(action) <= self.max_num_characters:
            return action
        else:
            # Truncate and return
            return action[: self.max_num_characters]
