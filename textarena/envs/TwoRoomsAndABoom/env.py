import re
import random
from typing import Any, Dict, Optional, Tuple, List, Set
from collections import deque

import textarena as ta


class TwoRoomsAndABoomEnv(ta.Env):
    """
    Two Rooms and a Boom game environment for the textarena framework.

    A social deduction game where players are split between two teams (Red and Blue) and
    placed in two different rooms. The Red Team aims to get the Bomber and President in
    the same room by the end, while the Blue Team wants to keep them in different rooms.
    """

    # Message patterns for player actions and card reveals (improved with more specific patterns)
    # Support both [Player X] and [X] formats consistently
    target_pattern = re.compile(r'.*\[(?:player\s*)?(\d+)\].*', re.IGNORECASE)
    reveal_keyword_pattern = re.compile(r'.*\b(?:reveal|show)\b.*(?:\bcard\b|\brole\b).*', re.IGNORECASE)

    # Maximum number of role reveals per player per game
    MAX_REVEALS_PER_PLAYER = 5

    # Maximum message history per room
    MAX_MESSAGE_HISTORY = 200

    # Maximum recursion depth for player transitions
    MAX_RECURSION_DEPTH = 10

    def __init__(self, num_rounds: int = 3, cards_per_room: int = 3, discussion_rounds: int = 2):
        """
        Initialize the Two Rooms and a Boom environment.

        Args:
            num_rounds (int): Number of rounds to play (default: 3)
            cards_per_room (int): Number of cards to initially place in each room (default: 3)
            discussion_rounds (int): Number of discussion turns per player per round (default: 2)
        """
        # Game configuration parameters
        self.num_rounds = num_rounds
        self.cards_per_room = cards_per_room
        self.discussion_rounds = discussion_rounds

        # For tracking recursion depth in _transition_current_pid to avoid infinite recursion
        self.transition_recursion_depth = 0

        # History tracking for messages and actions
        self.history = []

        # Role definitions with team affiliations and descriptions
        self.roles = {
            "Red": {
                "team": "Red Team",
                "description": "Member of the Red Team. Your goal is to make sure the Bomber and President are in the same room at the end of the game."
            },
            "Blue": {
                "team": "Blue Team",
                "description": "Member of the Blue Team. Your goal is to make sure the Bomber and President are in different rooms at the end of the game."
            },
            "Bomber": {
                "team": "Red Team",
                "description": "You are the Bomber on the Red Team. Your goal is to be in the same room as the President at the end of the game."
            },
            "President": {
                "team": "Blue Team",
                "description": "You are the President on the Blue Team. Your goal is to be in a different room from the Bomber at the end of the game."
            }
        }

    @property
    def terminal_render_keys(self):
        """Keys to render in the terminal visualization"""
        return ["round", "rooms", "player_roles", "leaders", "current_phase"]

    def reset(self, num_players: int, seed: Optional[int] = None):
        """
        Reset the environment for a new game

        Args:
            num_players (int): Number of players in the game
            seed (Optional[int]): Random seed for reproducibility
        """
        # Validate minimum and maximum players
        min_players = 6  # Absolute minimum for gameplay
        max_players = 20  # Reasonable maximum for communication

        if num_players < min_players:
            raise ValueError(f"Game requires at least {min_players} players")
        if num_players > max_players:
            raise ValueError(f"Game supports a maximum of {max_players} players")

        # Initialize state
        self.state = ta.State(num_players=num_players, min_players=min_players, max_players=max_players)

        # Initialize game state with seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Initialize game components and assign roles
        self._assign_roles_and_rooms(num_players)

        # Reset recursion depth counter
        self.transition_recursion_depth = 0

        # Initialize message history
        self.history = []

        # Track which roles each player has seen
        self.revealed_roles = {i: [] for i in range(num_players)}

        # Count reveals per player
        self.reveal_counts = {i: 0 for i in range(num_players)}

        # Store initial room assignments for reference
        self.original_room_assignments = {}
        for room_idx, room_players in enumerate(self.rooms):
            for pid in room_players:
                self.original_room_assignments[pid] = room_idx

        # Create initial game state
        game_state = {
            "round": 1,
            "current_phase": "Discussion",
            "rooms": self.rooms,
            "player_roles": self.player_roles,
            "leaders": self.leaders,
            "hostages_to_trade": {},
            "message_history": {},  # Track message history per room
            "revealed_roles": self.revealed_roles,
            "reveal_counts": self.reveal_counts,
            "original_room_assignments": self.original_room_assignments,
            "team_discussions": {},  # Private team discussions
            "revealing_player": None,  # To track who initiated a role reveal
        }

        # Reset state with game setup
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

        # Start with Discussion phase
        self._phase_transition_player_prompts(new_phase="Discussion")
        self._transition_current_pid()

        # Final validation after initialization
        self._validate_game_state()

    def _validate_game_state(self):
        """
        Validate game state consistency to catch bugs and perform recovery actions
        """
        try:
            # Check if every player is in exactly one room
            all_players = set(range(self.state.num_players))
            room_players = set(self.state.game_state["rooms"][0] + self.state.game_state["rooms"][1])

            # All players should be in a room
            if all_players != room_players:
                missing = all_players - room_players
                duplicates = []

                # Check for duplicates
                for p in all_players:
                    if (p in self.state.game_state["rooms"][0] and
                        p in self.state.game_state["rooms"][1]):
                        duplicates.append(p)

                # Handle missing players
                if missing:
                    for p in missing:
                        # Add to smaller room for balance
                        room_idx = 0 if len(self.state.game_state["rooms"][0]) <= len(self.state.game_state["rooms"][1]) else 1
                        self.state.game_state["rooms"][room_idx].append(p)
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=f"Game state recovery: Players {missing} were not assigned to any room and have been placed."
                    )

                # Handle duplicate players
                if duplicates:
                    for p in duplicates:
                        # Remove from the larger room
                        if len(self.state.game_state["rooms"][0]) >= len(self.state.game_state["rooms"][1]):
                            self.state.game_state["rooms"][0].remove(p)
                        else:
                            self.state.game_state["rooms"][1].remove(p)
                    self.state.add_observation(
                        from_id=ta.GAME_ID,
                        to_id=-1,
                        message=f"Game state recovery: Players {duplicates} were assigned to multiple rooms and have been fixed."
                    )

            # Verify both rooms have leaders
            for room_idx in range(2):
                # Skip if room is empty
                if not self.state.game_state["rooms"][room_idx]:
                    self.state.game_state["leaders"][room_idx] = None
                    continue

                # If the leader is None or not in this room, assign a new one
                if (self.state.game_state["leaders"][room_idx] is None or
                    self.state.game_state["leaders"][room_idx] not in self.state.game_state["rooms"][room_idx]):
                    # Choose a new leader, prioritizing regular players over special roles
                    regular_players = [
                        pid for pid in self.state.game_state["rooms"][room_idx]
                        if self.state.game_state["player_roles"][pid] not in ["President", "Bomber"]
                    ]

                    if regular_players:
                        new_leader = random.choice(regular_players)
                    else:
                        new_leader = random.choice(self.state.game_state["rooms"][room_idx])

                    self.state.game_state["leaders"][room_idx] = new_leader

                    # Notify players in the room
                    message = f"Room {room_idx} has a new leader: Player {new_leader}"
                    for pid in self.state.game_state["rooms"][room_idx]:
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=message)

            # Verify special roles exist
            president_exists = False
            bomber_exists = False

            for pid, role in self.state.game_state["player_roles"].items():
                if role == "President":
                    president_exists = True
                elif role == "Bomber":
                    bomber_exists = True

            # Critical error if special roles are missing - this should never happen
            # but we have a recovery path just in case
            if not president_exists or not bomber_exists:
                error_msg = f"Critical: Special roles missing: President={president_exists}, Bomber={bomber_exists}"
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

                # Emergency recovery - assign missing roles if needed
                all_pids = list(range(self.state.num_players))

                if not president_exists:
                    # Find a Blue player to make President
                    blue_players = [
                        pid for pid, role in self.state.game_state["player_roles"].items()
                        if "Blue" in self.roles[role]["team"] and role != "Bomber"
                    ]
                    if blue_players:
                        new_president = random.choice(blue_players)
                    else:
                        # Last resort - pick any non-Bomber player
                        non_bombers = [
                            pid for pid, role in self.state.game_state["player_roles"].items()
                            if role != "Bomber"
                        ]
                        new_president = random.choice(non_bombers if non_bombers else all_pids)

                    self.state.game_state["player_roles"][new_president] = "President"
                    recovery_msg = f"Recovery: Player {new_president} has been assigned as President."
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=recovery_msg)

                if not bomber_exists:
                    # Find a Red player to make Bomber
                    red_players = [
                        pid for pid, role in self.state.game_state["player_roles"].items()
                        if "Red" in self.roles[role]["team"] and role != "President"
                    ]
                    if red_players:
                        new_bomber = random.choice(red_players)
                    else:
                        # Last resort - pick any non-President player
                        non_presidents = [
                            pid for pid, role in self.state.game_state["player_roles"].items()
                            if role != "President"
                        ]
                        new_bomber = random.choice(non_presidents if non_presidents else all_pids)

                    self.state.game_state["player_roles"][new_bomber] = "Bomber"
                    recovery_msg = f"Recovery: Player {new_bomber} has been assigned as Bomber."
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=recovery_msg)

            # Verify room balance is reasonable (no completely empty rooms if possible)
            if not self.state.game_state["rooms"][0] and self.state.game_state["rooms"][1]:
                # Room 0 is empty but Room 1 has players
                if len(self.state.game_state["rooms"][1]) >= 2:
                    # Move half the players to balance rooms
                    players_to_move = self.state.game_state["rooms"][1][:len(self.state.game_state["rooms"][1])//2]
                    for p in players_to_move:
                        self.state.game_state["rooms"][0].append(p)
                        self.state.game_state["rooms"][1].remove(p)

                    balance_msg = f"Room balance adjusted: {len(players_to_move)} players moved to Room 0"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=balance_msg)

            elif not self.state.game_state["rooms"][1] and self.state.game_state["rooms"][0]:
                # Room 1 is empty but Room 0 has players
                if len(self.state.game_state["rooms"][0]) >= 2:
                    # Move half the players to balance rooms
                    players_to_move = self.state.game_state["rooms"][0][:len(self.state.game_state["rooms"][0])//2]
                    for p in players_to_move:
                        self.state.game_state["rooms"][1].append(p)
                        self.state.game_state["rooms"][0].remove(p)

                    balance_msg = f"Room balance adjusted: {len(players_to_move)} players moved to Room 1"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=balance_msg)

            # Check for extreme imbalance between rooms
            if (self.state.game_state["rooms"][0] and self.state.game_state["rooms"][1] and
                (len(self.state.game_state["rooms"][0]) > 3 * len(self.state.game_state["rooms"][1]) or
                 len(self.state.game_state["rooms"][1]) > 3 * len(self.state.game_state["rooms"][0]))):
                # Identify larger and smaller rooms
                larger_room = 0 if len(self.state.game_state["rooms"][0]) > len(self.state.game_state["rooms"][1]) else 1
                smaller_room = 1 - larger_room

                # Calculate number of players to move for better balance
                imbalance = len(self.state.game_state["rooms"][larger_room]) - len(self.state.game_state["rooms"][smaller_room])
                players_to_move = self.state.game_state["rooms"][larger_room][:imbalance//2]

                # Move players
                for p in players_to_move:
                    self.state.game_state["rooms"][smaller_room].append(p)
                    self.state.game_state["rooms"][larger_room].remove(p)

                if players_to_move:
                    rebalance_msg = f"Extreme room imbalance corrected: {len(players_to_move)} players moved to Room {smaller_room}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=rebalance_msg)

        except Exception as e:
            # Catch-all for unexpected errors in validation
            error_msg = f"Game state validation error: {str(e)}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

    def _assign_roles_and_rooms(self, num_players: int):
        """
        Assign roles to players and distribute them into two rooms,
        ensuring initial role separation for fairness.

        Args:
            num_players (int): Number of players in the game
        """
        self.player_roles = {}
        self.rooms = [[], []]  # Two rooms
        self.leaders = [None, None]  # Leaders for each room

        # Calculate team sizes (roughly equal)
        half_players = num_players // 2
        red_team_size = half_players
        blue_team_size = num_players - red_team_size

        # Create player roles with appropriate team balance
        role_pool = ["Red"] * red_team_size + ["Blue"] * blue_team_size

        # Assign special roles (President and Bomber)
        blue_indices = [i for i, role in enumerate(role_pool) if role == "Blue"]
        red_indices = [i for i, role in enumerate(role_pool) if role == "Red"]

        # Randomly select one Blue team member to be President
        president_idx = random.choice(blue_indices)
        role_pool[president_idx] = "President"

        # Randomly select one Red team member to be Bomber
        bomber_idx = random.choice(red_indices)
        role_pool[bomber_idx] = "Bomber"

        # Shuffle and assign roles - directly map to player IDs to avoid confusion
        for i in range(num_players):
            self.player_roles[i] = role_pool[i]

        # Find the President and Bomber IDs after direct assignment
        president_id = None
        bomber_id = None
        for pid, role in self.player_roles.items():
            if role == "President":
                president_id = pid
            elif role == "Bomber":
                bomber_id = pid

        # Ensure both special roles were assigned
        if president_id is None or bomber_id is None:
            raise ValueError("Failed to assign special roles properly")

        # Initially distribute players to rooms - ensuring President and Bomber in different rooms
        all_players = list(range(num_players))
        random.shuffle(all_players)

        # Remove President and Bomber from initial list to place them separately
        all_players.remove(president_id)
        all_players.remove(bomber_id)

        # Calculate balanced room sizes (adjusting for President and Bomber)
        room0_size = (num_players - 2) // 2

        # Distribute regular players
        self.rooms[0] = all_players[:room0_size]
        self.rooms[1] = all_players[room0_size:]

        # Add President to Room 0 and Bomber to Room 1 (initial separation)
        self.rooms[0].append(president_id)
        self.rooms[1].append(bomber_id)

        # Additional shuffle to mix positions in room
        random.shuffle(self.rooms[0])
        random.shuffle(self.rooms[1])

        # Assign leaders for each room, avoiding special roles if possible
        for room_idx in range(2):
            # Prefer regular players as leaders
            regular_players = [pid for pid in self.rooms[room_idx]
                              if self.player_roles[pid] not in ["President", "Bomber"]]

            if regular_players:
                self.leaders[room_idx] = random.choice(regular_players)
            else:
                # If no regular players, assign any player from the room
                self.leaders[room_idx] = random.choice(self.rooms[room_idx])

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """
        Generate the initial prompt for each player, including their role and objectives

        Args:
            player_id (int): The player's ID
            game_state (Dict[str, Any]): Current game state

        Returns:
            str: Personalized prompt for the player
        """
        # Get player's role and team info
        role = game_state["player_roles"][player_id]
        role_info = self.roles[role]

        # Determine which room the player is in
        player_room = None
        if player_id in game_state["rooms"][0]:
            player_room = 0
        elif player_id in game_state["rooms"][1]:
            player_room = 1
        else:
            # Failsafe for missing room assignment
            player_room = "unknown"

        # Determine if player is a leader
        is_leader = player_id in game_state["leaders"]
        leader_status = "You are the Leader of your room." if is_leader else ""

        # Basic prompt for all players
        prompt = (
            f"Welcome to Two Rooms and a Boom! You are Player {player_id}.\n"
            f"Your role: {role}\n"
            f"Team: {role_info['team']}\n"
            f"Description: {role_info['description']}\n\n"
            f"You are currently in Room {player_room}.\n"
            f"{leader_status}\n\n"
            f"The game progresses through {self.num_rounds} rounds:\n"
            f"• In each round, players in the same room can talk to each other\n"
            f"• Room Leaders can choose one player to trade to the other room\n"
            f"• During discussions, you can choose to privately reveal your card to another player\n"
            f"• At the end of all rounds, the game checks which room contains the President and Bomber\n\n"
            f"The Red Team wins if the President and Bomber are in the same room at the end.\n"
            f"The Blue Team wins if the President and Bomber are in different rooms at the end.\n\n"
        )

        # Add role-specific information
        if role == "Bomber":
            prompt += (
                "As the Bomber, you are a crucial member of the Red Team.\n"
                "Your goal is to end up in the same room as the President.\n"
                "You may choose whether to reveal your identity to others or keep it secret.\n\n"
            )
        elif role == "President":
            prompt += (
                "As the President, you are a crucial member of the Blue Team.\n"
                "Your goal is to end up in a different room from the Bomber.\n"
                "You may choose whether to reveal your identity to others or keep it secret.\n\n"
            )

        # Add leader-specific information
        if is_leader:
            prompt += (
                "As a Room Leader, you have special responsibilities:\n"
                "• You'll choose one player from your room to trade with the other room\n"
                "• You'll receive information from other players in your room\n"
                "• Use this information to make strategic decisions for your team\n"
                "• Leaders cannot trade themselves to the other room\n\n"
            )

        # Add information about revealing roles
        prompt += (
            "Role Revealing:\n"
            "• During discussions, you can say 'reveal card' or 'show role' to initiate revealing your role\n"
            "• The game will then prompt you to select which player to reveal to\n"
            f"• You can reveal your role up to {self.MAX_REVEALS_PER_PLAYER} times per game\n"
            "• This is a way to build trust, but be careful who you reveal to!\n\n"
        )

        # Add information about team coordination
        prompt += (
            f"Team Coordination:\n"
            f"• When you're with teammates, strategize on how to achieve your team's goal\n"
            f"• Blue team wants President and Bomber in different rooms\n"
            f"• Red team wants President and Bomber in the same room\n"
        )

        return prompt

    def _phase_transition_player_prompts(self, new_phase):
        """
        During a phase transition, provide relevant prompts to all players
        and update game state

        Args:
            new_phase (str): The new game phase to transition to
        """
        # Validate game state before any phase transition to catch issues early
        self._validate_game_state()

        if new_phase == "Discussion":
            # Reset any role reveal state
            self.state.game_state["revealing_player"] = None

            # All players in each room can discuss with each other
            for room_idx, room_players in enumerate(self.state.game_state["rooms"]):
                # Skip empty rooms
                if not room_players:
                    continue

                # Initialize message history for this room if not present
                if str(room_idx) not in self.state.game_state["message_history"]:
                    self.state.game_state["message_history"][str(room_idx)] = []

                # Create a player list string
                player_list = ", ".join([f"Player {pid}" for pid in room_players])

                # List roles that have been revealed to each player
                for pid in room_players:
                    revealed_roles = []
                    for revealed_pid in self.state.game_state["revealed_roles"].get(pid, []):
                        role = self.state.game_state["player_roles"][revealed_pid]
                        revealed_roles.append(f"Player {revealed_pid}: {role}")

                    # Build previous messages summary if any exist
                    previous_messages = ""
                    if self.state.game_state["message_history"].get(str(room_idx)):
                        previous_messages = "\nPrevious discussions in this room:\n"
                        # Only show last 10 messages to avoid overwhelming
                        recent_messages = self.state.game_state["message_history"][str(room_idx)][-10:]
                        for msg in recent_messages:
                            previous_messages += f"Player {msg['from']}: {msg['message']}\n"

                    # Create the discussion observation
                    discussion_observation = (
                        f"Round {self.state.game_state['round']}: Discussion phase has started.\n"
                        f"You are in Room {room_idx} with: {player_list}.\n"
                        f"You can talk freely with the other players in your room.\n"
                        f"To reveal your role to someone, say 'reveal card' or 'show role' during your turn.\n"
                    )

                    # Add revealed roles if any
                    if revealed_roles:
                        discussion_observation += "\nPlayers who have revealed their roles to you:\n"
                        discussion_observation += "\n".join(revealed_roles) + "\n"

                    # Add previous messages if any
                    discussion_observation += previous_messages

                    # Add advice for leader
                    if pid in self.state.game_state["leaders"]:
                        discussion_observation += (
                            "\nAs the room leader, pay close attention to discussions. "
                            "You'll be selecting a player to trade after this phase.\n"
                        )

                        # Add team-specific advice
                        player_role = self.state.game_state["player_roles"][pid]
                        team = self.roles[player_role]["team"]
                        if "Red" in team:
                            discussion_observation += (
                                "Remember: Your Red Team's goal is to get the Bomber and President in the same room.\n"
                            )
                        else:
                            discussion_observation += (
                                "Remember: Your Blue Team's goal is to keep the Bomber and President in different rooms.\n"
                            )

                    # Send the tailored observation to each player
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=discussion_observation)

            # Set up player order for discussion (each player speaks a few times)
            self.next_player_ids = []

            for _ in range(self.discussion_rounds):  # Each player gets multiple turns to speak
                for room_players in self.state.game_state["rooms"]:
                    # Skip empty rooms
                    if not room_players:
                        continue

                    # Shuffle players within each room for variety
                    shuffled_players = room_players.copy()
                    random.shuffle(shuffled_players)
                    self.next_player_ids.extend(shuffled_players)

        elif new_phase == "Role_Reveal":
            # Only proceed if we have a valid player who wants to reveal
            revealing_player = self.state.game_state.get("revealing_player")
            if revealing_player is None:
                # This shouldn't happen, but just in case
                self.state.game_state["current_phase"] = "Discussion"
                self._phase_transition_player_prompts(new_phase="Discussion")
                return

            # Determine which room the revealing player is in
            player_room = None
            if revealing_player in self.state.game_state["rooms"][0]:
                player_room = 0
            elif revealing_player in self.state.game_state["rooms"][1]:
                player_room = 1
            else:
                # Error - player not in any room
                self.state.game_state["current_phase"] = "Discussion"
                self._phase_transition_player_prompts(new_phase="Discussion")
                return

            # Check if player has reveals left
            current_reveals = self.state.game_state["reveal_counts"].get(revealing_player, 0)
            if current_reveals >= self.MAX_REVEALS_PER_PLAYER:
                error_msg = f"You have already used all {self.MAX_REVEALS_PER_PLAYER} of your allowed role reveals."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=revealing_player, message=error_msg)
                # Return to Discussion
                self.state.game_state["current_phase"] = "Discussion"
                self._transition_current_pid()
                return

            # Get list of players in the same room
            room_players = [
                pid for pid in self.state.game_state["rooms"][player_room]
                if pid != revealing_player
            ]

            # Create selection prompt
            if not room_players:
                # No one to reveal to
                error_msg = "There are no other players in your room to reveal your role to."
                self.state.add_observation(from_id=ta.GAME_ID, to_id=revealing_player, message=error_msg)
                # Return to Discussion
                self.state.game_state["current_phase"] = "Discussion"
                self._transition_current_pid()
                return

            player_list = ", ".join([f"Player {pid}" for pid in room_players])
            selection_options = ", ".join([f"[{pid}]" for pid in room_players])

            reveal_prompt = (
                f"You've chosen to reveal your role.\n"
                f"Players in your room: {player_list}\n\n"
                f"To whom would you like to reveal your role?\n"
                f"Simply reply in the following format: '[Player X]' or '[X]'\n"
                f"Valid options: {selection_options}\n\n"
                f"Note: This will be your reveal #{current_reveals + 1} out of {self.MAX_REVEALS_PER_PLAYER} allowed reveals."
            )

            self.state.add_observation(from_id=ta.GAME_ID, to_id=revealing_player, message=reveal_prompt)

            # Only the revealing player should act in this phase
            self.next_player_ids = [revealing_player]

        elif new_phase == "Leader_Selection":
            # Reset any role reveal state
            self.state.game_state["revealing_player"] = None

            # Check each room for leader status
            for room_idx in range(2):
                # Skip empty rooms
                if not self.state.game_state["rooms"][room_idx]:
                    self.state.game_state["leaders"][room_idx] = None
                    continue

                # If the leader is no longer in this room (traded), assign a new one
                current_leader = self.state.game_state["leaders"][room_idx]
                if current_leader not in self.state.game_state["rooms"][room_idx]:
                    # Select a new leader from the room, preferring regular players
                    regular_players = [
                        pid for pid in self.state.game_state["rooms"][room_idx]
                        if self.state.game_state["player_roles"][pid] not in ["President", "Bomber"]
                    ]

                    if regular_players:
                        new_leader = random.choice(regular_players)
                    else:
                        new_leader = random.choice(self.state.game_state["rooms"][room_idx])

                    self.state.game_state["leaders"][room_idx] = new_leader

                    # Notify all players in the room about the new leader
                    leader_change_msg = f"Room {room_idx} has a new leader: Player {new_leader}"
                    for pid in self.state.game_state["rooms"][room_idx]:
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=leader_change_msg)

            # Leaders select players to trade
            for room_idx, leader_id in enumerate(self.state.game_state["leaders"]):
                # Skip empty rooms or None leaders
                if leader_id is None or not self.state.game_state["rooms"][room_idx]:
                    continue

                # Get all players in the room except the leader
                room_players = [pid for pid in self.state.game_state["rooms"][room_idx] if pid != leader_id]

                # Only proceed if there are players to select
                if room_players:
                    player_options = ", ".join([f"[{pid}]" for pid in room_players])

                    # Get team information to provide strategic context
                    leader_role = self.state.game_state["player_roles"][leader_id]
                    leader_team = "Red" if "Red" in self.roles[leader_role]["team"] else "Blue"

                    # Get known information about players in the room
                    player_intel = []
                    for pid in room_players:
                        if pid in self.state.game_state["revealed_roles"].get(leader_id, []):
                            role = self.state.game_state["player_roles"][pid]
                            player_intel.append(f"Player {pid}: {role}")

                    intel_info = "\n".join(player_intel) if player_intel else "No players have revealed their roles to you."

                    # Create leader observation with strategic information
                    leader_observation = (
                        f"Round {self.state.game_state['round']}: As the Leader of Room {room_idx}, "
                        f"you must select one player to trade with the other room.\n"
                        f"Your team: {leader_team} Team\n\n"
                        f"Known player roles:\n{intel_info}\n\n"
                        f"Simply reply in the following format: '[Player X]' or '[X]'\n"
                        f"Valid options: {player_options}\n\n"
                    )

                    # Add team-specific strategic guidance
                    if leader_team == "Red":
                        leader_observation += (
                            "Strategic reminder: Red Team wants the President and Bomber in the same room at the end.\n"
                            "If you know who the President is, consider your strategy carefully.\n"
                        )
                    else:
                        leader_observation += (
                            "Strategic reminder: Blue Team wants the President and Bomber in different rooms at the end.\n"
                            "If you know who the Bomber is, consider your strategy carefully.\n"
                        )

                    self.state.add_observation(from_id=ta.GAME_ID, to_id=leader_id, message=leader_observation)

            # Leaders act in sequence, filtering out None values
            self.next_player_ids = [leader for leader in self.state.game_state["leaders"] if leader is not None]

            # If no leaders are left or no valid leaders, force trade with random selection
            if not self.next_player_ids and self.state.game_state["round"] < self.num_rounds:
                # Force random selection for rooms without leaders
                for room_idx in range(2):
                    if (self.state.game_state["leaders"][room_idx] is None and
                        self.state.game_state["rooms"][room_idx]):
                        eligible_players = [
                            pid for pid in self.state.game_state["rooms"][room_idx]
                            if pid != self.state.game_state["leaders"][room_idx]
                        ]

                        if eligible_players:
                            hostage = random.choice(eligible_players)
                            self.state.game_state["hostages_to_trade"][room_idx] = hostage

                            # Inform the room
                            message = f"With no leader, Player {hostage} was randomly selected to be traded."
                            for pid in self.state.game_state["rooms"][room_idx]:
                                self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=message)

                # Move to trade execution
                self.state.game_state["current_phase"] = "Trade_Execution"
                self._phase_transition_player_prompts(new_phase="Trade_Execution")

        elif new_phase == "Trade_Execution":
            # Reset any role reveal state
            self.state.game_state["revealing_player"] = None

            # Ensure both rooms have selected hostages (crucial game mechanic)
            for room_idx in range(2):
                # Skip empty rooms
                if not self.state.game_state["rooms"][room_idx]:
                    continue

                if room_idx not in self.state.game_state["hostages_to_trade"]:
                    # Room has players but no hostage selected - make random selection
                    eligible_players = [
                        p for p in self.state.game_state["rooms"][room_idx]
                        if p != self.state.game_state["leaders"][room_idx]
                    ]

                    # Only proceed if there are eligible players
                    if eligible_players:
                        random_hostage = random.choice(eligible_players)
                        self.state.game_state["hostages_to_trade"][room_idx] = random_hostage

                        # Inform all players in the room
                        random_selection_msg = (
                            f"Since no hostage was selected for Room {room_idx}, "
                            f"Player {random_hostage} was randomly chosen to be traded."
                        )
                        for pid in self.state.game_state["rooms"][room_idx]:
                            self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=random_selection_msg)

            # Execute the trade and inform players
            room0_hostage = self.state.game_state["hostages_to_trade"].get(0)
            room1_hostage = self.state.game_state["hostages_to_trade"].get(1)

            # Track if trade was executed
            trade_executed = False

            # Both rooms have hostages - standard trade
            if room0_hostage is not None and room1_hostage is not None:
                # Verify both hostages are actually in their respective rooms
                if (room0_hostage in self.state.game_state["rooms"][0] and
                    room1_hostage in self.state.game_state["rooms"][1]):

                    # Remove players from their current rooms
                    self.state.game_state["rooms"][0].remove(room0_hostage)
                    self.state.game_state["rooms"][1].remove(room1_hostage)

                    # Add players to their new rooms
                    self.state.game_state["rooms"][0].append(room1_hostage)
                    self.state.game_state["rooms"][1].append(room0_hostage)

                    trade_executed = True

                    # Inform all players about the trade
                    trade_observation = (
                        f"Round {self.state.game_state['round']}: The Leaders have exchanged hostages.\n"
                        f"Player {room0_hostage} moved from Room 0 to Room 1.\n"
                        f"Player {room1_hostage} moved from Room 1 to Room 0."
                    )

                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=trade_observation)
                else:
                    # Invalid hostage selection - hostages not in their rooms
                    error_msg = "Trade error: Selected hostages are not in their expected rooms."
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

                    # Try to find replacement hostages
                    if (room0_hostage not in self.state.game_state["rooms"][0] and
                        len(self.state.game_state["rooms"][0]) > 1):
                        # Find a new hostage from room 0
                        eligible_players = [
                            p for p in self.state.game_state["rooms"][0]
                            if p != self.state.game_state["leaders"][0]
                        ]
                        if eligible_players:
                            new_room0_hostage = random.choice(eligible_players)
                            self.state.game_state["hostages_to_trade"][0] = new_room0_hostage

                    if (room1_hostage not in self.state.game_state["rooms"][1] and
                        len(self.state.game_state["rooms"][1]) > 1):
                        # Find a new hostage from room 1
                        eligible_players = [
                            p for p in self.state.game_state["rooms"][1]
                            if p != self.state.game_state["leaders"][1]
                        ]
                        if eligible_players:
                            new_room1_hostage = random.choice(eligible_players)
                            self.state.game_state["hostages_to_trade"][1] = new_room1_hostage

                    # Try trade again with new hostages
                    room0_hostage = self.state.game_state["hostages_to_trade"].get(0)
                    room1_hostage = self.state.game_state["hostages_to_trade"].get(1)

                    if (room0_hostage is not None and room1_hostage is not None and
                        room0_hostage in self.state.game_state["rooms"][0] and
                        room1_hostage in self.state.game_state["rooms"][1]):

                        # Execute trade with new hostages
                        self.state.game_state["rooms"][0].remove(room0_hostage)
                        self.state.game_state["rooms"][1].remove(room1_hostage)
                        self.state.game_state["rooms"][0].append(room1_hostage)
                        self.state.game_state["rooms"][1].append(room0_hostage)

                        trade_executed = True

                        recovery_msg = (
                            f"Trade recovery: New hostages were selected.\n"
                            f"Player {room0_hostage} moved from Room 0 to Room 1.\n"
                            f"Player {room1_hostage} moved from Room 1 to Room 0."
                        )
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=recovery_msg)

            # One-sided trades (if one room is empty)
            elif room0_hostage is not None and not self.state.game_state["rooms"][1]:
                # Only room 0 has players, just announce no trade
                no_trade_msg = (
                    f"Round {self.state.game_state['round']}: No trade occurred as Room 1 is empty."
                )
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=no_trade_msg)

            elif room1_hostage is not None and not self.state.game_state["rooms"][0]:
                # Only room 1 has players, just announce no trade
                no_trade_msg = (
                    f"Round {self.state.game_state['round']}: No trade occurred as Room 0 is empty."
                )
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=no_trade_msg)

            # If no trade happened and both rooms have players, force a trade
            elif (not trade_executed and
                  self.state.game_state["rooms"][0] and
                  self.state.game_state["rooms"][1]):

                # Try to identify eligible players for forced trade
                eligible_room0 = [
                    p for p in self.state.game_state["rooms"][0]
                    if p != self.state.game_state["leaders"][0]
                ]
                eligible_room1 = [
                    p for p in self.state.game_state["rooms"][1]
                    if p != self.state.game_state["leaders"][1]
                ]

                # Only proceed if both rooms have eligible players
                if eligible_room0 and eligible_room1:
                    force_room0_hostage = random.choice(eligible_room0)
                    force_room1_hostage = random.choice(eligible_room1)

                    # Remove players from their current rooms
                    self.state.game_state["rooms"][0].remove(force_room0_hostage)
                    self.state.game_state["rooms"][1].remove(force_room1_hostage)

                    # Add players to their new rooms
                    self.state.game_state["rooms"][0].append(force_room1_hostage)
                    self.state.game_state["rooms"][1].append(force_room0_hostage)

                    # Inform all players about the forced trade
                    forced_trade_msg = (
                        f"Round {self.state.game_state['round']}: A trade had to be forced to continue the game.\n"
                        f"Player {force_room0_hostage} moved from Room 0 to Room 1.\n"
                        f"Player {force_room1_hostage} moved from Room 1 to Room 0."
                    )
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=forced_trade_msg)
                else:
                    # Cannot force a trade - leaders are the only players
                    no_trade_msg = (
                        f"Round {self.state.game_state['round']}: No trade occurred as there are not enough eligible players."
                    )
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=no_trade_msg)

            # Reset hostages for next round
            self.state.game_state["hostages_to_trade"] = {}

            # No player needs to take action in this phase
            self.next_player_ids = []

            # Check if we need to advance to the next round
            if self.state.game_state["round"] >= self.num_rounds:
                # This was the last round, determine winner
                self._determine_winner()
            else:
                # Advance to the next round
                self.state.game_state["round"] += 1

                # Move to Discussion phase for the next round
                self.state.game_state["current_phase"] = "Discussion"
                self._phase_transition_player_prompts(new_phase="Discussion")
        else:
            raise Exception(f"{new_phase} phase not recognized.")

    def _transition_current_pid(self):
        """
        Handle player transitions and phase changes with safeguards against infinite recursion
        """
        # Only transition if not invalid move
        if self.state.prevent_player_change:
            return

        # Increment recursion depth counter
        self.transition_recursion_depth += 1

        # Prevent infinite recursion
        if self.transition_recursion_depth > self.MAX_RECURSION_DEPTH:
            error_msg = f"Recursion depth exceeded in player transitions. Resetting to next phase."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

            # Emergency phase transition
            current_phase = self.state.game_state["current_phase"]
            if current_phase == "Discussion":
                self.state.game_state["current_phase"] = "Leader_Selection"
                self._phase_transition_player_prompts(new_phase="Leader_Selection")
            elif current_phase == "Leader_Selection":
                self.state.game_state["current_phase"] = "Trade_Execution"
                self._phase_transition_player_prompts(new_phase="Trade_Execution")
            elif current_phase == "Role_Reveal":
                self.state.game_state["current_phase"] = "Discussion"
                self._phase_transition_player_prompts(new_phase="Discussion")

            # Reset recursion counter
            self.transition_recursion_depth = 0
            return

        # Check if list is empty or doesn't exist
        if not hasattr(self, 'next_player_ids') or not self.next_player_ids:
            # Transition phase and replenish list
            current_phase = self.state.game_state["current_phase"]

            if current_phase == "Discussion":
                new_phase = "Leader_Selection"
                self.state.game_state["current_phase"] = new_phase
                self._phase_transition_player_prompts(new_phase=new_phase)
            elif current_phase == "Leader_Selection":
                new_phase = "Trade_Execution"
                self.state.game_state["current_phase"] = new_phase
                self._phase_transition_player_prompts(new_phase=new_phase)
            elif current_phase == "Role_Reveal":
                # If role reveal completes or fails, go back to discussion
                new_phase = "Discussion"
                self.state.game_state["current_phase"] = new_phase
                self._phase_transition_player_prompts(new_phase=new_phase)
            elif current_phase == "Trade_Execution":
                # Trade_Execution phase is handled automatically by _phase_transition_player_prompts
                pass

            # If we still don't have player IDs and we're in the final round, end the game
            if not hasattr(self, 'next_player_ids') or not self.next_player_ids:
                if self.state.game_state["round"] >= self.num_rounds:
                    self._determine_winner()
                    self.transition_recursion_depth = 0
                    return
                elif current_phase == "Trade_Execution":
                    # If we're in Trade_Execution phase and still have no players,
                    # go to Discussion for the next round
                    self.state.game_state["current_phase"] = "Discussion"
                    self._phase_transition_player_prompts(new_phase="Discussion")

                # If we still don't have players, return to avoid infinite recursion
                if not hasattr(self, 'next_player_ids') or not self.next_player_ids:
                    self.transition_recursion_depth = 0
                    return

        # Safety check - if next_player_ids exists but is empty, move to next phase
        if hasattr(self, 'next_player_ids') and not self.next_player_ids:
            # Move to the next phase directly
            current_phase = self.state.game_state["current_phase"]
            if current_phase == "Discussion":
                self.state.game_state["current_phase"] = "Leader_Selection"
                self._phase_transition_player_prompts(new_phase="Leader_Selection")
            elif current_phase == "Leader_Selection":
                self.state.game_state["current_phase"] = "Trade_Execution"
                self._phase_transition_player_prompts(new_phase="Trade_Execution")
            elif current_phase == "Role_Reveal":
                self.state.game_state["current_phase"] = "Discussion"
                self._phase_transition_player_prompts(new_phase="Discussion")

            self.transition_recursion_depth = 0
            return

        # Pop next pid and update state if we have players
        if hasattr(self, 'next_player_ids') and self.next_player_ids:
            next_pid = self.next_player_ids.pop(0)

            # Validate the player still exists in the game before updating
            all_players = set(self.state.game_state["rooms"][0] + self.state.game_state["rooms"][1])
            if next_pid in all_players:
                self.state.manually_update_current_player(new_player_id=next_pid)
            else:
                # Skip this player - they're somehow not in a room
                # Recursively call to get the next player
                self._transition_current_pid()

        # Reset recursion depth counter after successful transition
        self.transition_recursion_depth = 0

    def _determine_winner(self):
        """
        Determine which team wins based on the final positions of President and Bomber
        """
        # Find which rooms the President and Bomber are in
        president_room = None
        bomber_room = None
        president_id = None
        bomber_id = None

        for room_idx, room_players in enumerate(self.state.game_state["rooms"]):
            for pid in room_players:
                role = self.state.game_state["player_roles"][pid]
                if role == "President":
                    president_room = room_idx
                    president_id = pid
                elif role == "Bomber":
                    bomber_room = room_idx
                    bomber_id = pid

        # Create final game state observation
        final_state = (
            f"===== GAME OVER =====\n\n"
            f"Final room positions:\n"
        )

        for room_idx, players in enumerate(self.state.game_state["rooms"]):
            player_list = ", ".join([f"Player {p}" for p in players])
            final_state += f"Room {room_idx}: {player_list}\n"

        if president_id is not None:
            final_state += f"\nThe President (Player {president_id}) is in Room {president_room}.\n"

        if bomber_id is not None:
            final_state += f"The Bomber (Player {bomber_id}) is in Room {bomber_room}.\n\n"

        # Determine winner
        if president_room is not None and bomber_room is not None:
            if president_room == bomber_room:
                # Red team wins
                red_team_pids = [pid for pid, role in self.state.game_state["player_roles"].items()
                               if self.roles[role]["team"] == "Red Team"]

                reason = "The Red Team wins! The Bomber and President are in the same room."
                final_state += reason
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=final_state)
                self.state.set_winners(player_ids=red_team_pids, reason=reason)
            else:
                # Blue team wins
                blue_team_pids = [pid for pid, role in self.state.game_state["player_roles"].items()
                                if self.roles[role]["team"] == "Blue Team"]

                reason = "The Blue Team wins! The Bomber and President are in different rooms."
                final_state += reason
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=final_state)
                self.state.set_winners(player_ids=blue_team_pids, reason=reason)
        else:
            # This should never happen with proper validation, but as a failsafe:
            missing_roles = []
            if president_room is None:
                missing_roles.append("President")
            if bomber_room is None:
                missing_roles.append("Bomber")

            error_msg = f"Game could not determine a winner. Missing roles: {', '.join(missing_roles)}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

            # Default to Blue team win as in original logic, but with explanation
            blue_team_pids = [pid for pid, role in self.state.game_state["player_roles"].items()
                            if self.roles[role]["team"] == "Blue Team"]
            reason = "The Blue Team wins by default due to missing special roles."
            self.state.set_winners(player_ids=blue_team_pids, reason=reason)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process a single step (action) from the current player
        with improved validation and error handling

        Args:
            action (str): The player's action

        Returns:
            Tuple[bool, ta.Info]: Game state update
        """
        # Validate game state consistency
        try:
            self._validate_game_state()
        except ValueError as e:
            # Report error but try to continue
            error_msg = f"Game state inconsistency detected: {str(e)}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=error_msg)

        current_pid = self.state.current_player_id
        current_phase = self.state.game_state["current_phase"]

        # Verify player is in a valid room
        player_in_room0 = current_pid in self.state.game_state["rooms"][0]
        player_in_room1 = current_pid in self.state.game_state["rooms"][1]

        if not (player_in_room0 or player_in_room1):
            # Player is not in any room - this is an error
            error_msg = f"Player {current_pid} is not assigned to any room. This is a game state error."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)

            # Try to fix by adding player to a room
            room_to_add = 0 if len(self.state.game_state["rooms"][0]) <= len(self.state.game_state["rooms"][1]) else 1
            self.state.game_state["rooms"][room_to_add].append(current_pid)

            fixed_msg = f"Fixed by adding Player {current_pid} to Room {room_to_add}."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=fixed_msg)

            # Prevent this player from taking action this turn
            self.state.prevent_player_change = True
            return self.state.step(rotate_player=False)

        # Handle different phases
        if current_phase == "Discussion":
            # Check if the player wants to reveal their role
            if self.reveal_keyword_pattern.search(action):
                # Player wants to reveal their role - start the role reveal phase
                self.state.game_state["revealing_player"] = current_pid
                self.state.game_state["current_phase"] = "Role_Reveal"
                self._phase_transition_player_prompts(new_phase="Role_Reveal")
            else:
                # Normal discussion
                self._handle_discussion(current_pid=current_pid, action=action)

        elif current_phase == "Role_Reveal":
            # Handle role reveal target selection
            self._handle_role_reveal_selection(current_pid=current_pid, action=action)

        elif current_phase == "Leader_Selection":
            # Handle leader selection of hostages
            self._handle_leader_selection(current_pid=current_pid, action=action)

        # Trade_Execution phase doesn't require player actions

        # Only transition if the move wasn't invalid
        if not self.state.prevent_player_change:
            self._transition_current_pid()

        return self.state.step(rotate_player=False)

    def _handle_role_reveal_selection(self, current_pid, action):
        """
        Handle the selection of a player to reveal role to during the Role_Reveal phase

        Args:
            current_pid (int): ID of the player making the selection
            action (str): The selection action
        """
        # Verify this is the player who initiated the reveal
        if current_pid != self.state.game_state["revealing_player"]:
            error_msg = "Only the player who initiated the role reveal can select a target."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Determine which room the player is in
        player_room = None
        if current_pid in self.state.game_state["rooms"][0]:
            player_room = 0
        elif current_pid in self.state.game_state["rooms"][1]:
            player_room = 1
        else:
            # This shouldn't happen due to earlier validation
            error_msg = "You are not in any room. Cannot process role reveal."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Extract target player ID - check both formats
        target_pid = None

        # Check for [Player X] format
        if target_pid is None:
            match = self.target_pattern.search(action)
            if match:
                try:
                    target_pid = int(match.group(1))
                except ValueError:
                    pass

        # If we couldn't extract a player ID
        if target_pid is None:
            error_msg = "Could not determine which player you want to reveal your role to. Please reply in the following format: '[Player X]' or '[X]'."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Check if player has reveals left (double-check)
        current_reveals = self.state.game_state["reveal_counts"].get(current_pid, 0)
        if current_reveals >= self.MAX_REVEALS_PER_PLAYER:
            error_msg = f"You have already used all {self.MAX_REVEALS_PER_PLAYER} of your allowed role reveals."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Check if target is in the same room
        if target_pid not in self.state.game_state["rooms"][player_room]:
            error_msg = f"Player {target_pid} is not in your room. You can only reveal your role to players in the same room."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Process the role reveal
        true_role = self.state.game_state["player_roles"][current_pid]

        # Add this player to the target's revealed_roles list
        if target_pid not in self.state.game_state["revealed_roles"]:
            self.state.game_state["revealed_roles"][target_pid] = []

        # Update reveal count
        self.state.game_state["reveal_counts"][current_pid] = current_reveals + 1

        # Add to revealed roles if not already revealed
        if current_pid not in self.state.game_state["revealed_roles"][target_pid]:
            self.state.game_state["revealed_roles"][target_pid].append(current_pid)

        # Send private message to the target
        reveal_msg = (
            f"[PRIVATE] Player {current_pid} has revealed their card to you. "
            f"Their true role is: {true_role}"
        )
        self.state.add_observation(from_id=current_pid, to_id=target_pid, message=reveal_msg)

        # Send confirmation to the revealing player
        reveals_left = self.MAX_REVEALS_PER_PLAYER - (current_reveals + 1)
        confirm_msg = (
            f"You revealed your role ({true_role}) to Player {target_pid}. "
            f"You have {reveals_left} reveals remaining."
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=confirm_msg)

        # Create public version of the message without revealing the role
        public_msg = f"I am revealing my card to Player {target_pid}."

        # Process as normal discussion after handling the reveal
        self._handle_discussion(current_pid=current_pid, action=public_msg)

        # Return to discussion phase
        self.state.game_state["current_phase"] = "Discussion"
        self._phase_transition_player_prompts(new_phase="Discussion")

    def _handle_discussion(self, current_pid, action):
        """
        Handle discussion phase - broadcast message to all players in the same room
        and store message history with improved validation

        Args:
            current_pid (int): ID of the speaking player
            action (str): The message being sent
        """
        # Determine which room the player is in
        player_room = None
        if current_pid in self.state.game_state["rooms"][0]:
            player_room = 0
        elif current_pid in self.state.game_state["rooms"][1]:
            player_room = 1
        else:
            # This shouldn't happen due to earlier validation, but just in case
            error_msg = f"Player {current_pid} is not in any room. Cannot process discussion."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=error_msg)
            self.state.set_invalid_move(player_id=current_pid, reason=error_msg)
            return

        # Sanitize the message for safety
        # This could be expanded to filter out inappropriate content or exploitative messages
        sanitized_action = action.strip()

        # Limit message length if needed
        MAX_MESSAGE_LENGTH = float('inf')  # Characters
        if len(sanitized_action) > MAX_MESSAGE_LENGTH:
            sanitized_action = sanitized_action[:] + "... (message truncated)"
            truncate_notice = "Your message was truncated due to length."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=truncate_notice)

        # Store message in history with room-specific tracking
        if str(player_room) not in self.state.game_state["message_history"]:
            self.state.game_state["message_history"][str(player_room)] = []

        # Add message to history
        self.state.game_state["message_history"][str(player_room)].append({
            "from": current_pid,
            "message": sanitized_action,
            "round": self.state.game_state["round"]
        })

        # Limit message history size per room
        if len(self.state.game_state["message_history"][str(player_room)]) > self.MAX_MESSAGE_HISTORY:
            # Remove oldest messages
            self.state.game_state["message_history"][str(player_room)] = (
                self.state.game_state["message_history"][str(player_room)][-self.MAX_MESSAGE_HISTORY:]
            )

        # Broadcast message to all players in the same room
        for pid in self.state.game_state["rooms"][player_room]:
            if pid != current_pid:  # Don't send to self
                self.state.add_observation(from_id=current_pid, to_id=pid, message=sanitized_action)

        # Send confirmation to the speaking player
        confirm_msg = "Your message was sent to all players in your room."
        self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=confirm_msg)

        # If this player is sharing with their team's leader, add strategic info
        for team_leader_pid in self.state.game_state["leaders"]:
            # Skip if leader is None or self
            if team_leader_pid is None or team_leader_pid == current_pid:
                continue

            # Check if leader is in same room and same team
            leader_in_same_room = team_leader_pid in self.state.game_state["rooms"][player_room]

            if leader_in_same_room:
                # Check if they're on the same team
                player_role = self.state.game_state["player_roles"][current_pid]
                leader_role = self.state.game_state["player_roles"][team_leader_pid]

                player_team = "Red" if "Red" in self.roles[player_role]["team"] else "Blue"
                leader_team = "Red" if "Red" in self.roles[leader_role]["team"] else "Blue"

                if player_team == leader_team:
                    # Same team - send strategic context to leader
                    team_msg = (
                        f"[TEAM INFO] Player {current_pid} ({player_role}) on your team said: {sanitized_action}"
                    )
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=team_leader_pid, message=team_msg)

    def _handle_leader_selection(self, current_pid, action):
        """
        Handle leader selection of hostages to trade with improved validation

        Args:
            current_pid (int): ID of the leader making a selection
            action (str): The selection action
        """
        # Verify this is actually a leader
        if current_pid not in self.state.game_state["leaders"]:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="Only room leaders can select hostages."
            )
            return

        # Determine which room the leader is in
        room_idx = None
        if current_pid == self.state.game_state["leaders"][0]:
            room_idx = 0
        elif current_pid == self.state.game_state["leaders"][1]:
            room_idx = 1
        else:
            # Should never happen due to earlier check, but just in case
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="Leader room assignment error."
            )
            return

        # Try to extract player ID using different formats
        selected_pid = None

        # Try [Player X] format
        if selected_pid is None:
            match = self.target_pattern.search(action)
            if match:
                try:
                    selected_pid = int(match.group(1))
                except ValueError:
                    pass

        # If we couldn't extract a player ID
        if selected_pid is None:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="Could not determine which player you selected. Please reply in the following format: '[Player X]' or '[X]'."
            )
            return

        # Verify the selected player is in the leader's room
        if selected_pid not in self.state.game_state["rooms"][room_idx]:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason=f"Player {selected_pid} is not in your room. You can only select players from your own room."
            )
            return

        # Verify the selected player is not the leader themselves
        if selected_pid == current_pid:
            self.state.set_invalid_move(
                player_id=current_pid,
                reason="You cannot select yourself as a hostage."
            )
            return

        # Verify not trading the President and Bomber directly (if leader knows their identities)
        if selected_pid in self.state.game_state["revealed_roles"].get(current_pid, []):
            selected_role = self.state.game_state["player_roles"][selected_pid]

            # Check other room's selected hostage if already chosen
            other_room = 1 - room_idx
            other_hostage = self.state.game_state["hostages_to_trade"].get(other_room)

            if other_hostage is not None and other_hostage in self.state.game_state["revealed_roles"].get(current_pid, []):
                other_role = self.state.game_state["player_roles"][other_hostage]

                # Prevent direct President-Bomber trade if leader knows both roles
                if ((selected_role == "President" and other_role == "Bomber") or
                    (selected_role == "Bomber" and other_role == "President")):
                    warning_msg = (
                        "Warning: You are about to trade the President and Bomber directly. "
                        "This may help the other team. Proceeding anyway..."
                    )
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=warning_msg)

        # Record the selection
        self.state.game_state["hostages_to_trade"][room_idx] = selected_pid

        # Get selected player's role for leader's information
        selected_role = self.state.game_state["player_roles"][selected_pid]

        # Inform all players in the room about the selection
        selection_message = f"[LEADER] I have selected Player {selected_pid} to be traded with the other room."

        # Separately inform the leader about the player's role (if known)
        if selected_pid in self.state.game_state["revealed_roles"].get(current_pid, []):
            leader_info = f"[PRIVATE] You've selected Player {selected_pid} who revealed to you as: {selected_role}"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=current_pid, message=leader_info)

        # Broadcast decision to all players in the room
        for pid in self.state.game_state["rooms"][room_idx]:
            self.state.add_observation(from_id=current_pid, to_id=pid, message=selection_message)
