import re, random
from typing import Any, Dict, Optional, Tuple, List, Set

import textarena as ta

class TwoRoomsAndABoomEnv(ta.Env):
    # Message patterns for player actions
    target_pattern = re.compile(r'.*\[(?:player\s*)?(\d+)\].*', re.IGNORECASE)

    def __init__(self, num_rounds: int = 3, cards_per_room: int = 3):
        """
        Initialize the Two Rooms and a Boom environment.

        Args:
            num_rounds (int): Number of rounds to play (default: 3)
            cards_per_room (int): Number of cards to initially place in each room (default: 3)
        """
        self.num_rounds = num_rounds
        self.cards_per_room = cards_per_room

        # Role definitions
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
        return ["round", "rooms", "player_roles", "current_phase"]

    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """
        self.state = ta.State(num_players=num_players, min_players=6, max_players=20)

        # Initialize game state
        self._assign_roles_and_rooms(num_players)

        game_state = {
            "round": 1,
            "current_phase": "Discussion",
            "rooms": self.rooms,
            "player_roles": self.player_roles,
            "leaders": self.leaders,
            "hostages_to_trade": {},
        }

        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

        # Start with the Discussion phase
        self._phase_transition_player_prompts(new_phase="Discussion")
        self._transition_current_pid()

    def _assign_roles_and_rooms(self, num_players: int):
        """
        Assign roles to players and distribute them into two rooms.

        Args:
            num_players (int): Number of players in the game
        """
        self.player_roles = {}
        self.rooms = [[], []]  # Two rooms
        self.leaders = [None, None]  # Leaders for each room

        # Determine how many players per team
        half_players = num_players // 2

        # Create player roles (equal number of Red and Blue team members)
        role_pool = ["Red"] * half_players + ["Blue"] * (num_players - half_players)

        # Assign special roles (President and Bomber)
        blue_indices = [i for i, role in enumerate(role_pool) if role == "Blue"]
        red_indices = [i for i, role in enumerate(role_pool) if role == "Red"]

        # Randomly select one Blue team member to be President
        president_idx = random.choice(blue_indices)
        role_pool[president_idx] = "President"

        # Randomly select one Red team member to be Bomber
        bomber_idx = random.choice(red_indices)
        role_pool[bomber_idx] = "Bomber"

        # Shuffle and assign roles
        random.shuffle(role_pool)
        for i in range(num_players):
            self.player_roles[i] = role_pool[i]

        # Distribute players to rooms (initially equal distribution)
        all_players = list(range(num_players))
        random.shuffle(all_players)
        room_size = num_players // 2

        self.rooms[0] = all_players[:room_size]
        self.rooms[1] = all_players[room_size:]

        # Assign leaders for each room
        self.leaders[0] = random.choice(self.rooms[0])
        self.leaders[1] = random.choice(self.rooms[1])

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for each player, including their role and objectives """
        role = game_state["player_roles"][player_id]
        role_info = self.roles[role]

        # Determine which room the player is in
        player_room = 0 if player_id in game_state["rooms"][0] else 1

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
                "• Use this information to make strategic decisions for your team\n\n"
            )

        return prompt

    def _phase_transition_player_prompts(self, new_phase):
        """ During a phase transition, provide relevant prompts to all players """
        if new_phase == "Discussion":
            # All players in each room can discuss with each other
            for room_idx, room_players in enumerate(self.state.game_state["rooms"]):
                player_list = ", ".join([f"Player {pid}" for pid in room_players])
                discussion_observation = (
                    f"Round {self.state.game_state['round']}: Discussion phase has started.\n"
                    f"You are in Room {room_idx} with: {player_list}.\n"
                    f"You can talk freely with the other players in your room."
                )

                # Send observation to all players in the room
                for pid in room_players:
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=pid, message=discussion_observation)

            # Set up player order for discussion (each player speaks a few times)
            discussion_rounds = 2  # Each player gets to speak twice during discussion
            self.next_player_ids = []

            for _ in range(discussion_rounds):
                for room_players in self.state.game_state["rooms"]:
                    # Shuffle players within each room for variety
                    shuffled_players = room_players.copy()
                    random.shuffle(shuffled_players)
                    self.next_player_ids.extend(shuffled_players)

        elif new_phase == "Leader_Selection":
            # Leaders select players to trade
            for room_idx, leader_id in enumerate(self.state.game_state["leaders"]):
                # Get all players in the room except the leader
                room_players = [pid for pid in self.state.game_state["rooms"][room_idx] if pid != leader_id]
                player_options = ", ".join([f"'[{pid}]'" for pid in room_players])

                leader_observation = (
                    f"Round {self.state.game_state['round']}: As the Leader of Room {room_idx}, "
                    f"you must select one player to trade with the other room.\n"
                    f"Simply reply in the following format: '[Player X]' or '[X]'\n"
                    f"Valid options: {player_options}"
                )

                self.state.add_observation(from_id=ta.GAME_ID, to_id=leader_id, message=leader_observation)

            # Leaders act in sequence
            self.next_player_ids = self.state.game_state["leaders"].copy()

        elif new_phase == "Trade_Execution":
            # Execute the trade and inform players
            room0_hostage = self.state.game_state["hostages_to_trade"].get(0)
            room1_hostage = self.state.game_state["hostages_to_trade"].get(1)

            if room0_hostage is not None and room1_hostage is not None:
                # Remove players from their current rooms
                self.state.game_state["rooms"][0].remove(room0_hostage)
                self.state.game_state["rooms"][1].remove(room1_hostage)

                # Add players to their new rooms
                self.state.game_state["rooms"][0].append(room1_hostage)
                self.state.game_state["rooms"][1].append(room0_hostage)

                # Inform all players about the trade
                trade_observation = (
                    f"Round {self.state.game_state['round']}: The Leaders have exchanged hostages.\n"
                    f"Player {room0_hostage} moved from Room 0 to Room 1.\n"
                    f"Player {room1_hostage} moved from Room 1 to Room 0."
                )

                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=trade_observation)

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

        else:
            raise Exception(f"{new_phase} phase not recognized.")

    def _transition_current_pid(self):
        """ Handle player transitions and phase changes """
        # Only transition if not invalid move
        if self.state.prevent_player_change:
            return

        # Check if list is empty
        if not self.next_player_ids:
            # Transition phase and replenish list
            current_phase = self.state.game_state["current_phase"]

            if current_phase == "Discussion":
                new_phase = "Leader_Selection"
            elif current_phase == "Leader_Selection":
                new_phase = "Trade_Execution"
            elif current_phase == "Trade_Execution":
                new_phase = "Discussion"

            self.state.game_state["current_phase"] = new_phase
            self._phase_transition_player_prompts(new_phase=new_phase)

        if not self.next_player_ids:
            self._transition_current_pid()
        else:
            # Pop next pid and update state
            next_pid = self.next_player_ids.pop(0)
            self.state.manually_update_current_player(new_player_id=next_pid)

    def _determine_winner(self):
        """ Determine which team wins based on the final positions of President and Bomber """
        # Find which rooms the President and Bomber are in
        president_room = None
        bomber_room = None

        for room_idx, room_players in enumerate(self.state.game_state["rooms"]):
            for pid in room_players:
                role = self.state.game_state["player_roles"][pid]
                if role == "President":
                    president_room = room_idx
                elif role == "Bomber":
                    bomber_room = room_idx

        # Determine winner
        if president_room == bomber_room:
            # Red team wins
            red_team_pids = [pid for pid, role in self.state.game_state["player_roles"].items()
                          if role == "Red" or role == "Bomber"]
            reason = "The Red Team wins! The Bomber and President are in the same room."
            self.state.set_winners(player_ids=red_team_pids, reason=reason)
        else:
            # Blue team wins
            blue_team_pids = [pid for pid, role in self.state.game_state["player_roles"].items()
                           if role == "Blue" or role == "President"]
            reason = "The Blue Team wins! The Bomber and President are in different rooms."
            self.state.set_winners(player_ids=blue_team_pids, reason=reason)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process a single step (action) from the current player """
        current_pid = self.state.current_player_id

        # Check game phase
        if self.state.game_state["current_phase"] == "Discussion":
            self._handle_discussion(current_pid=current_pid, action=action)

        elif self.state.game_state["current_phase"] == "Leader_Selection":
            self._handle_leader_selection(current_pid=current_pid, action=action)

        # Rotate players
        self._transition_current_pid()

        return self.state.step(rotate_player=False)

    def _handle_discussion(self, current_pid, action):
        """ Handle discussion phase - broadcast message to all players in the same room """
        # Determine which room the player is in
        player_room = 0 if current_pid in self.state.game_state["rooms"][0] else 1

        # Broadcast message to all players in the same room
        for pid in self.state.game_state["rooms"][player_room]:
            if pid != current_pid:  # Don't send to self
                self.state.add_observation(from_id=current_pid, to_id=pid, message=action)

    def _handle_leader_selection(self, current_pid, action):
        """ Handle leader selection of hostages to trade """
        # Verify this is actually a leader
        room_idx = 0 if current_pid == self.state.game_state["leaders"][0] else 1

        # Extract and validate selection
        match = self.target_pattern.search(action)
        if not match:
            # Invalid selection format
            self.state.set_invalid_move(player_id=current_pid, reason="The selection was not submitted in the correct format.")
            return

        selected_pid = int(match.group(1))

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

        # Record the selection
        self.state.game_state["hostages_to_trade"][room_idx] = selected_pid

        # Inform all players in the room about the selection
        selection_message = f"[LEADER] I have selected Player {selected_pid} to be traded with the other room."
        for pid in self.state.game_state["rooms"][room_idx]:
            self.state.add_observation(from_id=current_pid, to_id=pid, message=selection_message)
