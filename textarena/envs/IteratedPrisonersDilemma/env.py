import re
from typing import Optional, Tuple, Dict, Any

import textarena as ta
from textarena.envs.IteratedPrisonersDilemma.renderer import create_board_str

class IteratedPrisonersDilemmaEnv(ta.Env):
    """Environment for Iterated Prisoner's Dilemma with communication rounds."""
    
    def __init__(
        self,
        num_rounds: int = 5,
        communication_turns: int = 3,
        cooperate_reward: int = 3,
        defect_reward: int = 5,
        sucker_reward: int = 0,
        mutual_defect_reward: int = 1,
    ):
        """
        Initialize the Iterated Prisoner's Dilemma environment.
        
        Args:
            num_rounds (int): Number of decision rounds
            communication_turns (int): Number of communication turns between decisions
            cooperate_reward (int): Reward for mutual cooperation (R)
            defect_reward (int): Reward for defecting while other cooperates (T)
            sucker_reward (int): Reward for cooperating while other defects (S)
            mutual_defect_reward (int): Reward for mutual defection (P)
        """
        # Game parameters
        self.num_rounds = num_rounds
        self.communication_turns = communication_turns*2
        
        # Payoff matrix parameters
        self.cooperate_reward = cooperate_reward  # R
        self.defect_reward = defect_reward        # T
        self.sucker_reward = sucker_reward        # S
        self.mutual_defect_reward = mutual_defect_reward  # P
        self.max_turns = num_rounds * (communication_turns + 2)
        
        # Action patterns
        self.cooperate_pattern = re.compile(r"\[Cooperate\]", re.IGNORECASE)
        self.defect_pattern = re.compile(r"\[Defect\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the game to initial state."""
        # Initialize game state variables
        self.state = ta.State(num_players=num_players,  min_players=2, max_players=2, max_turns=self.max_turns, check_truncated=True)
        game_state={"current_round": 1, "current_comm_turn": 0, "is_decision_phase": False, "scores": {0: 0, 1: 0}, 
            "round_decisions": {0: None, 1: None}, "history": []}
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the initial prompt for a player."""
        prompt = (
            f"You are Player {player_id} in an Iterated Prisoner's Dilemma game.\n\n"
            f"Game Structure:\n"
            f"- The game consists of {self.num_rounds} decision rounds\n"
            f"- Before each decision, you have {self.communication_turns} turns to communicate\n"
            f"- After communication, both players simultaneously choose to Cooperate or Defect\n\n"
            f"Payoff Matrix:\n"
            f"- Both Cooperate: Both get {self.cooperate_reward} points\n"
            f"- Both Defect: Both get {self.mutual_defect_reward} points\n"
            f"- One Defects, One Cooperates: Defector gets {self.defect_reward}, Cooperator gets {self.sucker_reward}\n\n"
            f"How to Play:\n"
            f"- During communication: Simply type your message\n"
            f"- During decision phase: Use [Cooperate] or [Defect]\n"
            f"You can include additional text before or after these tokens.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[ bool, ta.Info]:
        """Process player actions for both communication and decision phases."""
        # Add action to log/observations based on phase
        if self.state.game_state["is_decision_phase"]:
            # During decision phase, only log the action (keep it hidden)
            self.state.add_log(from_id=self.state.current_player_id, message=action)
        else:
            # During communication phase, broadcast to all
            self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action, for_logging=True)
        
        # Handle the appropriate phase
        if self.state.game_state["is_decision_phase"]:
            self._handle_decision_phase(self.state.current_player_id, action)
        else:
            self._handle_communication_phase()
            
        return self.state.step()

    def _handle_decision_phase(self, player_id: int, action: str):
        """Handle player decisions during the decision phase."""
        # Parse decision
        decision = None
        if self.cooperate_pattern.search(action):
            decision = "cooperate"
        elif self.defect_pattern.search(action):
            decision = "defect"
            
        if decision is None:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"Player {player_id} did not provide a valid decision."
            )
            return
            
        # Record decision
        self.state.game_state["round_decisions"][player_id] = decision
        
        # If both players have decided, calculate rewards and reveal decisions
        if all(d is not None for d in self.state.game_state["round_decisions"].values()):
            self._calculate_round_rewards()
            
            # Record round history
            self.state.game_state["history"].append({
                "round": self.state.game_state["current_round"],
                "decisions": self.state.game_state["round_decisions"].copy()
            })
            
            # Check if game is complete
            if self.state.game_state["current_round"] == self.num_rounds:
                self._determine_winner()
            else:
                # Reset for next round
                self.state.game_state["current_round"] += 1
                self.state.game_state["current_comm_turn"] = 0
                self.state.game_state["is_decision_phase"] = False
                self.state.game_state["round_decisions"] = {0: None, 1: None}
                
                message=f"Starting Round {self.state.game_state['current_round']}"
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

    def _handle_communication_phase(self):
        """Handle the communication phase between decisions."""
        self.state.game_state["current_comm_turn"] += 1
        
        # Check if communication phase is complete
        if self.state.game_state["current_comm_turn"] >= self.communication_turns:
            self.state.game_state["is_decision_phase"] = True
            self.state.game_state["round_decisions"] = {0: None, 1: None}
            
            message = f"Communication phase complete. Round {self.state.game_state['current_round']}: Please make your decisions."
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

    def _calculate_round_rewards(self):
        """Calculate and apply rewards based on player decisions."""
        p0_decision = self.state.game_state["round_decisions"][0]
        p1_decision = self.state.game_state["round_decisions"][1]
        
        # Calculate rewards
        if p0_decision == "cooperate" and p1_decision == "cooperate":
            p0_reward = p1_reward = self.cooperate_reward
            outcome = "Both players cooperated"
        elif p0_decision == "defect" and p1_decision == "defect":
            p0_reward = p1_reward = self.mutual_defect_reward
            outcome = "Both players defected"
        elif p0_decision == "cooperate" and p1_decision == "defect":
            p0_reward = self.sucker_reward
            p1_reward = self.defect_reward
            outcome = "Player 0 cooperated, Player 1 defected"
        else:  # p0 defects, p1 cooperates
            p0_reward = self.defect_reward
            p1_reward = self.sucker_reward
            outcome = "Player 0 defected, Player 1 cooperated"
            
        # Update scores
        self.state.game_state["scores"][0] += p0_reward
        self.state.game_state["scores"][1] += p1_reward
        
        # Reveal round results to all players
        message=(
            f"Round {self.state.game_state['current_round']} Results:\n"
            f"{outcome}\n"
            f"Player 0 scored {p0_reward}, total: {self.state.game_state['scores'][0]}\n"
            f"Player 1 scored {p1_reward}, total: {self.state.game_state['scores'][1]}"
        )
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

    def _determine_winner(self):
        """Determine the winner based on final scores."""
        p0_score = self.state.game_state["scores"][0]
        p1_score = self.state.game_state["scores"][1]
        
        if p0_score == p1_score:
            self.state.set_draw(reason=f"Game ended in a draw. Both players scored {p0_score} points.")
        else:
            winner_id = 0 if p0_score > p1_score else 1
            reason = f"Player {winner_id} won with {max(p0_score, p1_score)} points vs {min(p0_score, p1_score)} points"
            self.state.set_winners(player_ids=[winner_id], reason=reason)