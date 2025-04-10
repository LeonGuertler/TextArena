import re, random, json, importlib
from typing import Any, Dict, Optional, Tuple

import textarena as ta

class GuessWhoEnv(ta.Env):
    """ Guess Who game environment """
    
    def __init__(self, max_turns: int = 40):
        """ Initialize the environment """
        super().__init__()
        self.max_turns = max_turns

        # Initialize the gamemaster
        self.gamemaster = ta.agents.OpenRouterAgent(model_name="openai/gpt-4o")
        self.gamemaster_options = ["Yes", "No", "I don't know"]
        self.gamemaster_context = None
        self.gamemaster_history = []

        # Load character list
        self.characters = self._load_characters()


    def _load_characters(self, characters_path: Optional[str] = None):
        """
        Load characters from a JSON file.
        
        The JSON file must contain a list of character data objects.
        
        Args:
            characters_path (str, optional): Path to the JSON file containing character data.
            
        Returns:
            list: A list of character data objects.
            
        Raises:
            FileNotFoundError: If the `characters_path` does not exist.
            ValueError: If the JSON file has an invalid format or is empty.
        """
        try:
            if characters_path is not None:
                # Use provided path
                if not os.path.exists(characters_path):
                    raise FileNotFoundError(f"Characters data file not found at: {characters_path}")
                with open(characters_path, 'r', encoding='utf-8') as file:
                    characters = json.load(file)
            else:
                # Use package resource
                with importlib.resources.files('textarena.envs.GuessWho').joinpath('characters.json').open('r') as file:
                    characters = json.load(file)
                    
            if not characters:
                raise ValueError("Characters list is empty.")
                
            return characters
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load characters data: {str(e)}")

    @property
    def terminal_render_keys(self):
        return ["rendered_text"]

    def get_gamemaster_response(self, action: str) -> str:
        """
        Get the gamemaster's response based on the provided action.

        Args:
            action (str): The player's question or statement.

        Returns:
            str: The gamemaster's response.
        """
        # Validate gamemaster state
        if self.gamemaster_context is None:
            raise ValueError("Gamemaster context is not set.")
        if self.gamemaster_history is None:
            raise ValueError("History is not set.")
        if self.gamemaster_options is None:
            raise ValueError("Gamemaster options are not set.")

        # Format available response options
        options = ", ".join(f"'{opt}'" for opt in self.gamemaster_options)

        # Construct conversation history
        history = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.gamemaster_history)

        # Create prompt
        prompt = (
            f"{self.gamemaster_context}\n"
            f"{history}\n\n"
            f"Q: {action}\n"
            f"Options: {options}\n\n"
            "Please respond with the most appropriate option."
        )

        # Get response from the gamemaster agent
        response = self.gamemaster(prompt).strip()

        # Validate response
        if any(option.lower() in response.lower() for option in self.gamemaster_options):
            self.gamemaster_history.append((action, response))  # Store valid responses
        else:
            response = "I'm sorry, I don't understand. Please try asking again."
            self.gamemaster_history.append((action, response))  # Log fallback response

        return response

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """

        # Initialize the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)
        
        ## select a random character
        self.target_character = random.choice(self.characters)

        ## update the gamemaster context
        self.gamemaster_context = (
            f"You are the gamemaster for the game of 'Guess Who'.\n"
            f"You will provide responses to the player's questions that guides them into guessing the target character with the following name and traits: {self.target_character}.\n"
        )

        ## reset the game state
        game_state = {
            "target_character": self.target_character,
            "rendered_text": self._render_text()
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the player prompt """
        prompt = (
            f"You are Player {player_id}. You are playing Guess Who.\n"
            "The gamemaster has chosen one target character from the list of characters that you will be shown below.\n"
            "You have to guess the target character by asking yes-or-no questions about the target character's traits.\n"
            "You can ask questions like 'Is the character male?' or 'Does the character have a beard?'.\n"
            "You can also guess the name of the target character at any time by ensuring that you wrap their name in square brackets, e.g. [Zach].\n"
            "As you play, the history of your questions and gamemaster's responses will be displayed."
            "Here is the list of characters you can ask questions about:\n"
        )
        prompt += self._characters_to_string()
        return prompt
    
    def _characters_to_string(self) -> str:
        """
        Render the text for the game.
        
        Returns:
            The rendered text
        """
        formatted_descriptions = []
        for i, char in enumerate(self.characters, start=1):
            # Format the description in a narrative style
            accessories = ", ".join(char["accessories"]) if char["accessories"] else "no accessories"
            description = (
                f"{i}. {char['name']} is a {char['age_range']} {char['gender']} with {char['hair_style']} "
                f"{char['hair_color']} hair and {char['eye_color']} eyes. {char['name']} has a {char['complexion']} complexion, "
                f"{char['skin_tone']} skin tone, and {char['smile_type']} smile. They wear {accessories}, "
                f"have {char['facial_hair']} facial hair, and their clothing style is {char['clothing_style']}. "
                f"{char['name']} has {char['hair_texture']} hair texture, {char['eyewear_style']} glasses style, "
                f"a {char['nose_shape']} nose, {char['ear_size']} ears, and {char['cheek_features']} on their cheeks."
            )
            formatted_descriptions.append(description)
        
        # Join all descriptions into a single text block
        return "\n\n".join(formatted_descriptions)
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action and update the environment state """
        player_id = self.state.current_player_id
        
        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## validate the action
        action_search_pattern = re.compile(r"\[([a-zA-Z]+)\]") # e.g. [zach]
        action_match = action_search_pattern.search(action)

        if not action_match:
            ## if the action is not a guess, then it is a question
            gamemaster_response = self.get_gamemaster_response(action)
            self.state.add_observation(from_id=-1, to_id=player_id, message=gamemaster_response)
        
        else:
            ## if the action is a guess
            action_text = action_match.group(1).lower()
            if action_text == self.target_character["name"].lower():
                reason=f"Congratulations! Player {player_id} guessed the target character."
                self.state.set_winners(player_ids=[player_id], reason=reason)
            else:
                reason=f"Invalid guess. Player {player_id} guessed incorrectly."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
            
            self.state.game_state["rendered_text"] = self._render_text()
        return self.state.step()
    
    def _render_text(self) -> str:
        """
        Render the target character and their traits.
        
        Returns:
            string: The rendered text
        """
        res = ""
        for key, value in self.target_character.items():
            res += f"{key}: {value}\n"
        return res
