import re, random, copy
import networkx as nx
from typing import Any, Dict, List, Tuple, Optional, Union

import textarena as ta
from textarena.envs.WordLadder.renderer import create_board_str
from textarena.utils.word_lists import EnglishDictionary

## use nltk to get the words
import nltk
from nltk.corpus import words
nltk.download('words')


class WordLadderEnv(ta.Env):
    """ Word Ladder environment """

    def __init__(self, min_distance: int=5, max_distance: int=7, max_turns: int=100):
        """
        Initialize the Word Ladder environment.

        Args:
            min_distance (int): TODO
            max_distance (int): TODO 
            max_turns (int): The maximum number of turns
        """
        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_turns = max_turns

        ## load the word list (to be sampled from)
        self.word_list = words.words("en-basic")

        ## load the universal word list
        self.universal_word_list = self.load_universal_word_list()

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
    
    def load_universal_word_list(self):
        """
        Load a universal word list that includes words from the NLTK word list and US and UK spell-check dictionaries.
        """
        dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)
        return dictionary.get_all_words()


    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the environment to its initial state """
        ## initialize the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## initialize the game state
        self.word_graph = self._generate_word_graph()
        self.start_word, self.target_word = self._generate_words()
        self.current_word = self.start_word
        self.history = [self.start_word]

        ## reset the game state
        game_state={
            "start_word": self.start_word,
            "target_word": self.target_word,
            "rendered_text": self._render_text() 
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the prompt for the player based on the current state of the game """
        prompt = (
            f"You are Player {player_id}. You are playing Word Ladder.\n"
            "The objective of the game is to convert the start word to the target word by changing one letter at a time.\n"
            f"The start word is: {self.start_word}\n"
            f"The target word is: {self.target_word}\n"
            "You may only submit one word at a time. To submit your word, you must wrap it in square brackets, e.g. [word].\n"
            "As you play, the history of your choices will be appended below. Use the information to win the game.\n"
        )
        return prompt
    
    def _render_text(self) -> str:
        return f"Word Ladder History: {' -> '.join(self.history)}. Target Word: {self.target_word}\n"
    
    def _generate_word_graph(self, min_length=3, max_length=11):
        """
        Creates a dictionary of NetworkX graphs for word lengths between min_length and max_length.
        Each graph represents words of the same length, with edges connecting words differing by one letter.
        
        Returns:
            dict: A dictionary mapping word lengths to their respective graphs.
        """
        word_graphs = {}

        for length in range(min_length, max_length + 1):
            filtered_words = [w.lower() for w in self.word_list if len(w) == length]

            # Create a graph for this word length
            G = nx.Graph()
            G.add_nodes_from(filtered_words)

            # Add edges for words differing by one letter
            for i, word1 in enumerate(filtered_words):
                for word2 in filtered_words[i+1:]:
                    if self.one_letter_difference(word1, word2):
                        G.add_edge(word1, word2)

            # Store the graph
            word_graphs[length] = G

        return word_graphs

    def one_letter_difference(self, word1, word2):
        """Returns True if word1 and word2 differ by exactly one letter."""
        if len(word1) != len(word2):
            return False
        return sum(a != b for a, b in zip(word1, word2)) == 1

    def words_with_at_least_n_difference(self, graphs, min, max):
        """
        Finds all word pairs with at least 'n' letter differences within the graphs.

        Args:
            graphs (dict): A dictionary of graphs created by `create_word_graphs`.
            n (int): Minimum number of letter differences required.

        Returns:
            list: A list of tuples (word1, word2, path) where path length is at least 'n'.
        """
        word_pairs = []

        for length, G in graphs.items():
            for word1 in G.nodes:
                for word2 in G.nodes:
                    if word1 < word2:  # Avoid duplicate pairs
                        try:
                            path = nx.shortest_path(G, source=word1, target=word2)
                            steps = len(path) - 1  # Path length is number of transformations
                            if steps >= min and steps <= max:
                                word_pairs.append((word1, word2, path))
                        except nx.NetworkXNoPath:
                            continue  # Ignore words with no connection

        return word_pairs

    def _generate_words(self) -> Tuple[str, str]:
        """
        Generate a start and target word pair with exactly 10 steps between them.

        Returns:
            Tuple[str, str]: The start and target words.
        """
        word_pairs = self.words_with_at_least_n_difference(self.word_graph, self.min_distance, self.max_distance)
        start_word, target_word, path = random.choice(word_pairs)
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
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action and update the environment state """
        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]") # e.g. [word]
        match = action_search_pattern.search(action)


        if match is None:
            reason=f"Invalid move format. Player {player_id} did not respond with a valid word format in square brackets."
            self.state.set_invalid_move(player_id=player_id, reason=reason)

        else:
            next_word = match.group(1)
            if len(next_word) != len(self.target_word):
                ## check if the word is of the correct length
                reason=f"Invalid move format. Player {player_id} did not respond with a word of the correct length."
                self.state.set_invalid_move(player_id=player_id, reason=reason)

            elif next_word not in self.universal_word_list:
                ## check if the word is in the word list
                reason=f"Invalid move format. Player {player_id} did not respond with a valid word."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            elif not self._is_one_alphabet_different(next_word):
                ## check if word is a move that is one letter away from the current word
                reason=f"Invalid move format. Player {player_id}'s word choice of '{next_word}' is not one alphabet different from the previous word."
                self.state.set_invalid_move(player_id=player_id, reason=reason)

            else:
                ## is a valid move
                self.current_word = next_word
                self.history.append(next_word)
                if next_word == self.target_word:
                    ## player found the target word - game is over
                    reason=f"Congratulations! Player {player_id} has found the target word."
                    self.state.set_winners(player_ids=[player_id], reason=reason)
                else:
                    ## game is not over
                    message=f"You've selected a valid word.\n{self._render_text()}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message, for_logging=False)

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
