import re, random
from typing import Any, Dict, Optional, Tuple, List, Set

import textarena as ta
from textarena.envs.games.SecretMafia.renderer import create_board_str


class SecretMafiaEnv(ta.Env):
    # Message patterns for player actions
    voting_pattern = re.compile(r'.*\[(?:player\s*)?(\d+)\].*', re.IGNORECASE)

    def __init__(self, mafia_ratio: float = 0.25, discussion_rounds: int = 3):
        """
        Args:
            mafia_ratio (float): Ratio of Mafia members to total players (default: 0.25)
            discussion_rounds (int): The number of discussion rounds
        """
        self.mafia_ratio = mafia_ratio
        self.discussion_rounds = discussion_rounds
        self.action_space = lambda player_id: re.compile(r".*", re.DOTALL)

        # Role definitions
        self.roles = {
            "Villager": {"team": "Village", "description": "A regular villager. Your goal is to identify and eliminate all Mafia members through voting during the day."},
            "Mafia": {"team": "Mafia", "description": "A Mafia member. Your goal is to eliminate enough villagers to gain majority. During the night phase, you can communicate secretly with other Mafia members and vote to eliminate a villager."},
            "Doctor": {"team": "Village", "description": "A villager with medical skills. During the night phase, you can choose one player to protect from Mafia elimination."},
            "Detective": {"team": "Village", "description": "A villager with investigative skills. During the night phase, you can investigate one player to learn if they are a Mafia member."}
        }

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
        
    def reset(self, num_players: int, seed: Optional[int] = None):
        assert 5<=num_players<=15, f"The number of players has to be 5<=x<=15, received {num_players}"

        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        self._assign_roles(num_players)
        game_state = {"phase": "Night-Mafia", "day_number": 1, "alive_players": list(range(num_players)), "player_roles": self.player_roles, "votes": {}, "to_be_eliminated": None, "num_discussion_rounds": 3}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)

        # the game starts with the mafia making their first vote
        self._phase_transition_player_prompts(new_phase="Night-Mafia")
        self._transition_current_pid()

    def _assign_roles(self, num_players: int):
        """ Assign roles to players based on the number of players and mafia ratio """
        self.player_roles = {}
        role_pool = ["Mafia"] * max(1, round(num_players * self.mafia_ratio)) + ["Doctor"] + ["Detective"] # Create the role pool
        role_pool.extend(["Villager"] * (num_players - len(role_pool)))
        
        # Shuffle and assign roles
        random.shuffle(role_pool)
        for i in range(num_players): 
            self.player_roles[i] = role_pool[i]
        
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for each player, including their role and objectives """
        role = game_state["player_roles"][player_id]
        player_list = ", ".join([f"Player {i}" for i in range(self.state.num_players)])
        prompt = (
            f"Welcome to Secret Mafia! You are Player {player_id}.\nYour role: {role}\nTeam: {self.roles[role]['team']}\nDescription: {self.roles[role]['description']}\n\n"
            f"Players: {player_list}\n\nThe game progresses through Day and Night phases:\n"
            f"- During the Day phase, there are {game_state['num_discussion_rounds']} rounds of discussion followed by voting\n"
            f"- During discussions, everything you say is automatically broadcasted to all players\n"
            f"- After discussions, all players must vote to eliminate one player\n"
            f"- During the Night phase, special roles perform their actions\n\n"
            f"The game ends when either all Mafia members are eliminated (Village wins) or\n"
            f"Mafia members equal or outnumber Villagers (Mafia wins)\n\n"
        )
        
        # Add role-specific information and abilities
        if role == "Mafia":
            mafia_members = [f"Player {pid}" for pid, r in game_state["player_roles"].items() if r == "Mafia"]
            prompt += (
                f"You are part of the Mafia team. Your teammates are: {', '.join(mafia_members)}.\n\nYour abilities:\n"
                "  During DAY phase:\n"
                "    - Everything you say is automatically shared with all players\n"
                "    - You'll vote to eliminate a player at the end of discussions\n\n"
                "  During NIGHT phase:\n"
                "    - '[Player X]' - Vote to eliminate Player X (must be a non-Mafia player)\n\n"
                "Your goal is to eliminate enough villagers until Mafia members equal or outnumber the Villagers.\n\n"
            )
        elif role == "Doctor":
            prompt += (
                "Your abilities:\n"
                "  During DAY phase:\n"
                "    - Everything you say is automatically shared with all players\n"
                "    - You'll vote to eliminate a player at the end of discussions\n\n"
                "  During NIGHT phase:\n"
                "    - '[Player X]' - Protect Player X from Mafia elimination tonight\n"
                "Your goal is to help identify and eliminate all Mafia members.\n\n"
            )
        elif role == "Detective":
            prompt += (
                "Your abilities:\n"
                "  During DAY phase:\n"
                "    - Everything you say is automatically shared with all players\n"
                "    - You'll vote to eliminate a player at the end of discussions\n\n"
                "  During NIGHT phase:\n"
                "    - '[Player X]' - Investigate whether Player X is a Mafia member\n"
                "      (You'll receive immediate results of your investigation)\n\n"
                "Your goal is to help identify and eliminate all Mafia members.\n\n"
            )
        else:  # Villager
            prompt += (
                "Your abilities:\n"
                "  During DAY phase:\n"
                "    - Everything you say is automatically shared with all players\n"
                "    - You'll vote to eliminate a player at the end of discussions\n\n"
                "  During NIGHT phase:\n"
                "    - You have no special actions during the night phase\n\n"
                "Your goal is to help identify and eliminate all Mafia members.\n\n"
            )
        return prompt


    def _phase_transition_player_prompts(self, new_phase):
        """ During a phase transition, provide relevant prompts to all players """
        if new_phase == "Night-Mafia":
            # all mafia players receive a prompt to vote whom to kill
            mafia_pids = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Mafia" and pid in self.state.game_state["alive_players"]]
            remaining_non_mafia = [pid for pid, role in self.state.game_state["player_roles"].items() if role!="Mafia" and pid in self.state.game_state["alive_players"]]
            valid_votes = ", ".join([f"'[{rpid}]'" for rpid in remaining_non_mafia])
            for pid in mafia_pids: # send observations to all relevant players
                self.state.add_observation(to_id=pid, message=f"The Night phase has started, please vote who you would like to kill. Only votes in the format '[Player X]' or '[X]' are valid.Valid votes: {valid_votes}", observation_type=ta.ObservationType.GAME_MESSAGE)

            # update new player orders - initially next pids is just voting of mafia (so one mafia each)
            self.next_player_ids = [pid for pid, role in self.state.game_state["player_roles"].items() if role == "Mafia" and pid in self.state.game_state["alive_players"]]
            random.shuffle(self.next_player_ids) # shuffle order

        elif new_phase == "Night-Doctor":
            d_pid = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Doctor"][0] # get doctor pid 
            valid_player_options = [pid for pid, role in self.state.game_state["player_roles"].items() if role!="Doctor" and pid in self.state.game_state["alive_players"]]
            valid_options = ", ".join([f"'[{rpid}]'" for rpid in valid_player_options])
            self.state.add_observation(to_id=d_pid, message=f"We are in the Night phase. Since you are the doctor, you can decide which player to save. Simply reply in the following format: '[Player X]' or '[X]' valid options: {valid_options}", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.next_player_ids = [d_pid]

        elif new_phase == "Night-Detective":
            d_pid = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Detective"][0] # get detective pid 
            valid_player_options = [pid for pid, role in self.state.game_state["player_roles"].items() if role!="Detective" and pid in self.state.game_state["alive_players"]]
            
            # check if detective is still alive 
            if d_pid in self.state.game_state["alive_players"]:
                valid_options = ", ".join([f"'[{rpid}]'" for rpid in valid_player_options])
                self.state.add_observation(to_id=d_pid, message=f"We are in the Night phase. Since you are the detective, you can decide which player to investigate. Simply reply in the following format: '[Player X]' or '[X]'. valid options: {valid_options}", observation_type=ta.ObservationType.GAME_MESSAGE)
                self.next_player_ids = [d_pid]

        elif new_phase == "Day-Discussion": # TODO add who was killed
            self.state.add_observation(to_id=-1, message=f"For the next {self.state.game_state['num_discussion_rounds']} you can converse freely with the other players to decide who you ultimatly want to vote out.", observation_type=ta.ObservationType.GAME_MESSAGE)
            next_players = self.state.game_state["alive_players"]
            random.shuffle(next_players)
            self.next_player_ids = next_players * self.state.game_state['num_discussion_rounds']

        elif new_phase == "Day-Voting":
            valid_options = ", ".join([f"'[{rpid}]'" for rpid in self.state.game_state['alive_players']])
            self.state.add_observation(to_id=-1, message=f"The voting phase has began. On your turn, submit your vote for which player you want to vote out. Simply reply in the following format: '[Player X]' or '[X]'. valid options: {valid_options}", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.next_player_ids = self.state.game_state["alive_players"].copy()
            random.shuffle(self.next_player_ids)

        else:
            raise Exception(f"{new_phase} phase not recognized.")

    def _transition_current_pid(self):
        """ this should iterate over the pids to call, then if empty, update the phase and call the phase transition prompt function """
        # only transition if not invalid move 
        if self.state.made_invalid_move: return
        
        if not self.next_player_ids: # check if list is empty 
            # transition phase and replenish list
            doctor_pid = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Doctor"][0]
            detective_pid = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Detective"][0]
            if self.state.game_state["phase"] == "Night-Mafia":
                # check if doctor is still alive
                if doctor_pid in self.state.game_state["alive_players"]:        new_phase = "Night-Doctor" # transition to doctor phase
                elif detective_pid in self.state.game_state["alive_players"]:   new_phase = "Night-Detective"
                else:                                                           new_phase = "Day-Discussion"
            elif self.state.game_state["phase"] == "Night-Doctor":
                if detective_pid in self.state.game_state["alive_players"]:     new_phase = "Night-Detective"
                else:                                                           new_phase = "Day-Discussion"
            elif self.state.game_state["phase"] == "Night-Detective":           new_phase = "Day-Discussion"
            elif self.state.game_state["phase"] == "Day-Discussion":            new_phase = "Day-Voting"
            elif self.state.game_state["phase"] == "Day-Voting":                new_phase = "Night-Mafia"

            # check for winning conditions on relevant transition phases
            if new_phase=="Day-Discussion" or new_phase=="Night-Mafia":
                # add observation
                tbe_pid = self.state.game_state['to_be_eliminated']
                if tbe_pid is None: observation = f"No player has been eliminated."
                else: observation = f"Player {tbe_pid} has been eliminated."
                self.state.add_observation(message=observation, observation_type=ta.ObservationType.GAME_MESSAGE)
                self.state.game_state["alive_players"] = [pid for pid in self.state.game_state["alive_players"] if pid != tbe_pid]  # remove player from alive
                self.state.game_state["votes"] = {} # reset votes
                self.state.game_state["to_be_eliminated"] = None # reset to be eliminated
                self._check_winning_conditions() # check winning condition

            self.state.game_state["phase"] = new_phase
            self._phase_transition_player_prompts(new_phase=new_phase)

        if not self.next_player_ids:    self._transition_current_pid()
        else:                           self.state.manually_set_current_player_id(new_player_id=self.next_player_ids.pop()) # pop next pid and update state

    def _check_winning_conditions(self):
        # winning condition 1, all mafia members are eliminated
        alive_mafia_members = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Mafia" and pid in self.state.game_state["alive_players"]]

        if not alive_mafia_members: # villagers win
            villager_pids = [pid for pid, role in self.state.game_state["player_roles"].items() if role!="Mafia"]
            self.state.set_winners(player_ids=villager_pids, reason=f"The villagers win by eliminating all members of the mafia.")

        if len(alive_mafia_members) >= len(self.state.game_state["alive_players"])/2: # mafia wins
            mafia_pids = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Mafia"]
            self.state.set_winners(player_ids=mafia_pids, reason=f"The Mafia wins by outnumbering the villagers")

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process a single step (action) from the current player """
        # check game phase 
        if self.state.game_state["phase"] == "Day-Discussion":      self._day_discussion(current_pid=self.state.current_player_id, action=action)
        elif self.state.game_state["phase"] == "Day-Voting":        self._day_voting(current_pid=self.state.current_player_id, action=action)
        elif self.state.game_state["phase"] == "Night-Mafia":       self._night_mafia(current_pid=self.state.current_player_id, action=action) 
        elif self.state.game_state["phase"] == "Night-Doctor":      self._night_doctor(current_pid=self.state.current_player_id, action=action)
        elif self.state.game_state["phase"] == "Night-Detective":   self._night_detective(current_pid=self.state.current_player_id, action=action)
        else: raise
        self._transition_current_pid() # rotate players
        return self.state.step(rotate_player=False)

    def _evaluate_votes(self):
        """ returns pid with most votes or None & resets the votes"""
        # Count votes for each player
        print(f"votes: ", self.state.game_state["votes"])
        vote_counts = {}
        for voter, target in self.state.game_state["votes"].items():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        # reset votes
        self.state.game_state["votes"] = {}
        if not vote_counts:
            return None

        # Find player(s) with most votes
        max_votes = max(vote_counts.values())
        top_candidates = [pid for pid, count in vote_counts.items() if count == max_votes]

        # If there's a tie (more than one player with the most votes), return None
        if len(top_candidates) > 1: return None
        else: return top_candidates[0]

    def _day_discussion(self, current_pid, action):
        """ simply broadcast messages to all """
        self.state.add_observation(from_id=current_pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

    def _day_voting(self, current_pid, action):
        """ validate voting and add to votes until no next pid """
        # extract and validate vote 
        match = self.voting_pattern.search(action)
        if not match: # raise invalid 
            if self.state.set_invalid_move(reason=f"The vote was not submitted in the correct format."):
                self.state.set_winners(player_ids=[pid for pid in range(self.state.num_players) if pid != self.state.current_player_id], reason=f"Player {self.state.current_player_id} made an invalid move.")
        else:
            self.state.game_state["votes"][current_pid] = int(match.group(1)) # count vote and broadcast
            self.state.add_observation(from_id=current_pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

            # Store a copy of the alive players before checking if everyone voted
            # This can help identify if the alive_players list is being modified
            alive_before = list(self.state.game_state["alive_players"])

            # check if everybody has voted
            if not self.next_player_ids:
                # evaluate votes and update observations accordingly 
                self.state.game_state["to_be_eliminated"] = self._evaluate_votes()

    def _night_mafia(self, current_pid, action):
        """ basically the same as day phase voting """
        # extract and validate vote
        match = self.voting_pattern.search(action)
        if not match: # raise invalid
            if self.state.set_invalid_move(reason=f"The vote was not submitted in the correct format."):
                self.state.set_winners(player_ids=[pid for pid in range(self.state.num_players) if pid != self.state.current_player_id], reason=f"Player {self.state.current_player_id} made an invalid move.")
        else:
            self.state.game_state["votes"][current_pid] = int(match.group(1))
            # count vote and broadcast to all mafia players
            mafia_pids = [pid for pid, role in self.state.game_state["player_roles"].items() if role=="Mafia" and pid in self.state.game_state["alive_players"]]
            for pid in mafia_pids:
                self.state.add_observation(from_id=current_pid, to_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

            # check if everybody has voted
            if not self.next_player_ids:
                # evaluate votes and update observations accordingly
                self.state.game_state["to_be_eliminated"] = self._evaluate_votes()

    def _night_doctor(self, current_pid, action):
        """ check who the doctor whould like to save """
        # extract and validate vote
        match = self.voting_pattern.search(action)
        if not match: # raise invalid
            if self.state.set_invalid_move(reason=f"The action was not submitted in the correct format."):
                self.state.set_winners(player_ids=[pid for pid in range(self.state.num_players) if pid != self.state.current_player_id], reason=f"Player {self.state.current_player_id} made an invalid move.")
        else: # check if voted_pid is to_be_eliminated
            self.state.add_observation(from_id=current_pid, to_id=current_pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
            if int(match.group(1)) == self.state.game_state["to_be_eliminated"]:
                self.state.game_state["to_be_eliminated"] = None # save

    def _night_detective(self, current_pid, action):
        """ can check status of a single player """
        match = self.voting_pattern.search(action)

        if not match: # raise invalid 
            if self.state.set_invalid_move(reason=f"The action was not submitted in the correct format."):
                self.state.set_winners(player_ids=[pid for pid in range(self.state.num_players) if pid != self.state.current_player_id], reason=f"Player {self.state.current_player_id} made an invalid move.")
        else:
            voted_pid = int(match.group(1))
            mafia_set = [pid for pid,role in self.state.game_state["player_roles"].items() if role == "Mafia"]
            if voted_pid in mafia_set:  observation = f"Player {voted_pid} is part of the Mafia"
            else:                       observation = f"Player {voted_pid} is NOT part of the Mafia"
            self.state.add_observation(to_id=current_pid, message=observation, observation_type=ta.ObservationType.GAME_MESSAGE)

 