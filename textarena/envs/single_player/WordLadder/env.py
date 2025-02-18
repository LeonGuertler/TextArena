"""Word Ladder environment."""

import random
import re
from typing import Any, Dict, Optional, Tuple

import networkx as nx

import textarena as ta
from textarena.envs.utils.word_lists import EnglishDictionary


class WordLadderEnv(ta.Env):
    """
    Word Ladder environment.
    """

    def __init__(
        self,
        difficulty: str = "easy",
    ):
        """
        Initialize the Word Ladder environment.

        Args:
            hardcore: Whether to play in hardcore mode.
            word_len: The length of the words to use.

        """

        super().__init__()
        self.environment_name = "WordLadder"
        self.difficulty = difficulty

        ## initialize the game state
        self.state = ta.State(
            num_players=1,
        )

        ## load the word list (to be sampled from)
        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)

        ## Set the difficulty parameters
        if self.difficulty == "easy":
            self.min_distance = 5
            self.max_distance = 7
        elif self.difficulty == "medium":
            self.min_distance = 8
            self.max_distance = 12
        elif self.difficulty == "hard":
            self.min_distance = 13
            self.max_distance = 15
        self.current_word = ...  # needs to be initialized in reset
        self.start_word = ...  # needs to be initialized in reset
        self.target_word = ...  # needs to be initialized in reset
        self.history = ...  # needs to be initialized in reset
        self.word_graph = ...  # needs to be initialized in reset

    @property
    def offline_renderer(self):
        pass

    @property
    def terminal_render_keys(self):
        return ["rendered_text"]

    def reset(self, seed: Optional[int] = None) -> Optional[ta.Observations]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for the environment.

        Returns:
            Observations: Initial observations for the player.

        """

        ## seed the random number generator
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        ## initialize the game state
        self.word_graph = self._generate_word_graph()
        self.start_word, self.target_word = self._generate_words()
        self.current_word = self.start_word
        self.history = [self.start_word]

        ## reset the game state
        return self.state.reset(
            game_state={
                "start_word": self.start_word,
                "target_word": self.target_word,
                "rendered_text": self._render_text(),
            },
            player_prompt_function=self._generate_player_prompt,
        )

    def _generate_player_prompt(
        self, player_id: int, game_state: Dict[int, Any]
    ) -> str:
        """
        Generate the prompt for the player based on the current state of the game.

        Args:
            player_id: The player id.

        Returns:
            str: The prompt for the player.

        """
        prompt = (
            f"You are Player {player_id}. You are playing Word Ladder ({self.difficulty}).\n"
            "The objective of the game is to convert the start word to the target word by changing one letter at a time.\n"
            f"The start word is: {self.start_word}\n"
            f"The target word is: {self.target_word}\n"
            "You may only submit one word at a time. To submit your word, you must wrap it in square brackets, e.g. [word].\n"
            "As you play, the history of your choices will be appended below. Use the information to win the game.\n"
        )

        return prompt

    def _render_text(self) -> str:
        """
        Render the text for the player based on the current state of the game.

        Returns:
            str: The rendered text for the player.

        """
        ## render the history and also the target words
        return (
            f"Word Ladder History: {' -> '.join(self.history)}. Target Word: {self.target_word}\n",
        )

    def _generate_word_graph(self, min_length=3, max_length=11):
        """
        Creates a dictionary of NetworkX graphs for word lengths between min_length and max_length.
        Each graph represents words of the same length, with edges connecting words differing by one letter.

        Returns:
            dict: A dictionary mapping word lengths to their respective graphs.
        """
        word_graphs = {}

        for length in range(min_length, max_length + 1):
            filtered_words = [  # should maybe upgrade to get_all_words...
                w.lower() for w in self.dictionary.get_basic_words() if len(w) == length
            ]

            # Create a graph for this word length
            G = nx.Graph()
            G.add_nodes_from(filtered_words)
            # Add edges for words differing by one letter using a more efficient approach
            buckets = {}
            for word in filtered_words:
                for i in range(len(word)):
                    bucket = word[:i] + "_" + word[i + 1 :]
                    if bucket not in buckets:
                        buckets[bucket] = []
                    buckets[bucket].append(word)

            for bucket, words in buckets.items():
                for i, word1 in enumerate(words):
                    for word2 in words[i + 1 :]:
                        G.add_edge(word1, word2)

            # Store the graph
            word_graphs[length] = G

            # print(f"Graph for {length}-letter words: {G.number_of_nodes()} words, {G.number_of_edges()} edges.")

        return word_graphs

    def one_letter_difference(self, word1, word2):
        """Returns True if word1 and word2 differ by exactly one letter."""
        if len(word1) != len(word2):
            return False
        return sum(a != b for a, b in zip(word1, word2)) == 1

    def words_with_at_least_n_difference(
        self, graphs, min_steps, max_steps, max_pairs=100000
    ):
        """
        Finds word pairs with path lengths between min_steps and max_steps within the graphs,
        using a more efficient sampling approach.

        Args:
            graphs (dict): A dictionary of graphs created by `create_word_graphs`.
            min_steps (int): Minimum number of steps required.
            max_steps (int): Maximum number of steps allowed.
            max_pairs (int): Maximum number of pairs to return.

        Returns:
            list: A list of tuples (word1, word2, path) with path lengths between min_steps and max_steps.
        """
        word_pairs = []

        # For each graph (representing words of a specific length)
        for length, G in graphs.items():
            # If graph is too small, skip it
            if G.number_of_nodes() < 2:
                continue

            # Sample starting words rather than examining all pairs
            sample_size = min(100, G.number_of_nodes())
            start_words = random.sample(list(G.nodes()), sample_size)

            for start_word in start_words:
                # Use single-source shortest paths from each start word
                lengths = nx.single_source_shortest_path_length(G, start_word)

                # Filter words by path length
                candidates = [
                    (word, length)
                    for word, length in lengths.items()
                    if min_steps <= length <= max_steps and word != start_word
                ]

                # Randomly sample from candidates if there are many
                if candidates:
                    sample_count = min(10, len(candidates))
                    sampled_candidates = random.sample(candidates, sample_count)

                    for target_word, _ in sampled_candidates:
                        # Only compute the actual path for the selected candidates
                        path = nx.shortest_path(
                            G, source=start_word, target=target_word
                        )
                        word_pairs.append((start_word, target_word, path))

                        # Early exit if we've found enough pairs
                        if len(word_pairs) >= max_pairs:
                            return word_pairs

        return word_pairs

    def _generate_words(self) -> Tuple[str, str]:
        """
        Generate a start and target word pair with exactly 10 steps between them.

        Returns:
            Tuple[str, str]: The start and target words.
        """
        word_pairs = self.words_with_at_least_n_difference(
            self.word_graph, self.min_distance, self.max_distance
        )
        start_word, target_word, _ = random.choice(word_pairs)
        return start_word, target_word

    def _validate_solution_existence(self, graph, start_word, target_word) -> bool:
        """
        Check if there is a path from start_word to target_word in the graph.

        Args:
            graph: The graph to search.
            start_word: The start word.
            target_word: The target word.

        Returns:
            bool: Whether a path exists between the two words.
        """
        return nx.has_path(graph, start_word, target_word)

    def step(self, action: str) -> Tuple[
        Optional[ta.Observations],  # observations
        Optional[ta.Rewards],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action and update the environment state.

        Args:
            player_id (int): The ID of the player making the move.
            action (str): The action taken by the player.

        Returns:
            Observations: Observations for the player after the action.
            Rewards: Rewards for the player after the action.
            bool: Whether the game was truncated.
            bool: Whether the game is terminated.
            Info: Additional information about the game state

        """
        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(
            from_id=player_id, to_id=-1, message=action, for_logging=True
        )

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]")  # e.g. [word]
        match = action_search_pattern.search(action)

        if match is None:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Invalid move format. Player {player_id} did not respond with a valid word format in square brackets.",
            )

        else:
            next_word = match.group(1)
            if len(next_word) != len(self.target_word):
                ## check if the word is of the correct length
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move format. Player {player_id} did not respond with a word of the correct length.",
                )
            elif not self.dictionary.is_english_word(next_word):
                ## check if the word is in the word list
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move format. Player {player_id} did not respond with a valid word.",
                )
            elif not self._is_one_alphabet_different(next_word):
                ## check if word is a move that is one letter away from the current word
                self.state.set_invalid_move(
                    player_id=player_id,
                    reason=f"Invalid move format. Player {player_id}'s word choice of '{next_word}' is not one alphabet different from the previous word.",
                )

            else:
                ## is a valid move
                self.current_word = next_word
                self.history.append(next_word)
                if next_word == self.target_word:
                    ## player found the target word - game is over
                    self.state.set_winners(
                        player_ids=[player_id],
                        reason=f"Congratulations! Player {player_id} has found the target word.",
                    )
                else:
                    ## game is not over
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=f"You've selected a valid word.\n{self._render_text()}",
                        for_logging=False,
                    )

            ## update the game board
            self.state.game_state["rendered_text"] = self._render_text()

        return self.state.step()

    def _is_one_alphabet_different(self, next_word: str) -> bool:
        """
        Checks if `next_word` is a valid move from `self.current_word`,
        ensuring that the words differ by exactly one letter.

        Args:
            next_word (str): The word to change to.

        Returns:
            bool: True if `next_word` is exactly one letter different from `self.current_word`, otherwise False.
        """
        next_word = next_word.lower()

        # Count the number of differing letters
        difference_count = sum(a != b for a, b in zip(self.current_word, next_word))

        # Move is valid only if there is exactly one letter difference
        return difference_count == 1
