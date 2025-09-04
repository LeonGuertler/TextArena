import re
from typing import Dict, Any, Optional, Tuple

import textarena as ta
from textarena.envs.PublicGoodsGame.renderer import create_board_str

class PublicGoodsGameEnv(ta.Env):
    def __init__(self, 
                 num_rounds: int = 5, 
                 communication_turns: int = 3,
                 endowment: int = 20,
                 multiplication_factor: float = 1.5,
                 num_players: int = 4):
        """
        Initialize the Public Goods Game environment.
        
        Args:
            num_rounds: Number of rounds to play
            communication_turns: Number of communication turns before each decision
            endowment: Number of tokens each player starts with each round
            multiplication_factor: Factor by which total contributions are multiplied
            num_players: Number of players in the game (default 4)
        """
        self.num_rounds = num_rounds
        self.communication_turns = communication_turns
        self.endowment = endowment
        self.multiplication_factor = multiplication_factor
        self.default_num_players = num_players
        
        # Action regex - matches numbers in brackets like [15] or [0]
        self.contribution_pattern = re.compile(r"\[(\d+)\]", re.IGNORECASE)
        
        # Public message regex - matches messages in curly braces like {Hello everyone!}
        self.public_message_pattern = re.compile(r"\{([^}]*)\}", re.DOTALL)

    def get_board_str(self):
        gs = self.state.game_state
        return create_board_str(gs)

    def reset(self, num_players: Optional[int] = None, seed: Optional[int] = None):
        if num_players is None:
            num_players = self.default_num_players
            
        self.state = ta.FFAMultiPlayerState(
            num_players=num_players, 
            seed=seed,
            max_turns=self.num_rounds * num_players * (self.communication_turns + 1),
            error_allowance=2  # Allow 2 errors before elimination
        )
        
        game_state = {
            "round": 1,
            "num_rounds": self.num_rounds,
            "phase": "conversation",
            "conversation_round": 0,
            "total_conversation_rounds": self.communication_turns,
            "contributions": {i: None for i in range(num_players)},
            "total_scores": {i: 0 for i in range(num_players)},
            "round_scores": {i: 0 for i in range(num_players)},
            "endowment": self.endowment,
            "multiplication_factor": self.multiplication_factor,
            "history": [],
            "eliminations": [],  # Track eliminated players
            "pending_messages": {},  # Store messages until all players have acted
            "pending_contributions": {}  # Store contributions until all players have acted
        }
        
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        if game_state.get("phase") == "conversation":
            phase_round_description = (
                f"Round {game_state['round']} - Communication Round {game_state['conversation_round'] + 1} "
                f"of {game_state['total_conversation_rounds']}\n"
            )
        else:
            phase_round_description = (
                f"Round {game_state['round']} - Decision Phase"
            )
            
        return (
            f"You are Player {player_id} in a Public Goods Game spanning "
            f"{game_state['num_rounds']} rounds.\n\n"
            f"Game Structure:\n"
            f"- Each round, you receive {game_state['endowment']} tokens.\n"
            f"- Before each decision you have {game_state['total_conversation_rounds']} "
            f"turns to communicate with other players.\n"
            f"- Communication is SIMULTANEOUS: Messages from a turn are revealed only after all players submit.\n"
            f"- After communication, all players SIMULTANEOUSLY choose how many tokens to contribute "
            f"to the public pot (0 to {game_state['endowment']}).\n"
            f"- Contributions are revealed only after all players have decided.\n\n"
            f"Payoff Calculation:\n"
            f"- Your payoff = (tokens kept) + (your share of public good)\n"
            f"- Public good = (sum of all contributions) × {game_state['multiplication_factor']}\n"
            f"- The public good is divided equally among all {self.state.num_players} players\n\n"
            f"Example:\n"
            f"If everyone contributes 10 tokens:\n"
            f"- Total contributions: {10 * self.state.num_players}\n"
            f"- Public good: {10 * self.state.num_players} × {game_state['multiplication_factor']} = {10 * self.state.num_players * game_state['multiplication_factor']}\n"
            f"- Each player gets: {10 * self.state.num_players * game_state['multiplication_factor'] / self.state.num_players:.1f} from public good\n"
            f"- Plus 10 tokens kept = {10 + (10 * self.state.num_players * game_state['multiplication_factor'] / self.state.num_players):.1f} total\n\n"
            f"How to Play:\n"
            f"- You can think internally and reason about your strategy (this won't be shared).\n"
            f"- Your goal is to maximize your total score across all rounds.\n"
            f"- During conversation: send public messages using {{message}} format.\n"
            f"  Example: 'I think we should cooperate. {{Let me propose we all contribute 15 tokens}}'\n"
            f"  Only the text in curly braces will be visible to other players.\n"
            f"- During decision phase: include '[X]' where X is your contribution (0-{game_state['endowment']}).\n"
            f"  Example: 'Based on the discussion, I will contribute [15] tokens to the public good.'\n"
            f"- Invalid moves (wrong format or out of range) will result in warnings, then elimination.\n"
            f"- If you don't send any public message during conversation (no {{}} format), others will see that you remained silent.\n\n" 
            f"Game Status: \n"
            f"{phase_round_description}\n"
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(
            to_id=self.state.current_player_id,
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION
        )
        
        match self.state.game_state["phase"]:
            case "conversation":
                self._handle_conversation_phase(action)
            case "decision":
                self._handle_decision_phase(action)
                
        return self.state.step()

    def _extract_public_message(self, action: str) -> Optional[str]:
        """Extract and validate public message from action using {message} format."""
        matches = self.public_message_pattern.findall(action)
        if matches:
            valid_messages = [match.strip() for match in matches if match.strip()]
            if valid_messages:
                return " ".join(valid_messages)
        return None

    def _handle_conversation_phase(self, action: str):
        # Extract only the public message portion
        public_message = self._extract_public_message(action)
        
        # Store the public message (or None if no public message was sent)
        self.state.game_state["pending_messages"][self.state.current_player_id] = public_message
        
        # Check if all alive players have submitted their actions
        alive_players = [p for p in range(self.state.num_players) if self.state.is_player_alive(p)]
        
        if all(p in self.state.game_state["pending_messages"] for p in alive_players):
            # All alive players have submitted - now broadcast public messages simultaneously
            messages_to_broadcast = []
            
            for sender_id in alive_players:
                public_msg = self.state.game_state["pending_messages"][sender_id]
                if public_msg:
                    messages_to_broadcast.append(f"Player {sender_id}: {public_msg}")
                else:
                    messages_to_broadcast.append(f"Player {sender_id}: [remained silent]")
            
            # Broadcast all messages to all players at once
            if messages_to_broadcast:
                full_message = "Messages from this turn:\n" + "\n".join(messages_to_broadcast)
                for receiver_id in alive_players:
                    self.state.add_observation(
                        to_id=receiver_id,
                        from_id=ta.GAME_ID,
                        message=full_message,
                        observation_type=ta.ObservationType.GAME_MESSAGE
                    )
            
            # Clear pending messages
            self.state.game_state["pending_messages"] = {}
            
            # Advance conversation round
            self.state.game_state["conversation_round"] += 1
            
            if self.state.game_state["conversation_round"] >= self.state.game_state["total_conversation_rounds"]:
                # Switch to decision phase
                self.state.game_state["phase"] = "decision"
                self.state.add_observation(
                    message=f"Conversation finished for round {self.state.game_state['round']}.\n"
                           f"Decision phase: Submit '[X]' where X is your contribution (0-{self.state.game_state['endowment']}).\n"
                           f"Contributions will be revealed after all players decide.",
                    observation_type=ta.ObservationType.GAME_BOARD
                )

    def _handle_decision_phase(self, action: str):
        # Extract contribution amount
        match = self.contribution_pattern.search(action)
        if match:
            contribution = int(match.group(1))
            # Validate contribution
            if 0 <= contribution <= self.state.game_state["endowment"]:
                self.state.game_state["pending_contributions"][self.state.current_player_id] = contribution
            else:
                # Invalid contribution - use the state's invalid move handling
                eliminated = self.state.set_invalid_move(
                    f"Invalid contribution {contribution}. Must be between 0 and {self.state.game_state['endowment']}."
                )
                if not eliminated:
                    return  # Player gets another chance
                else:
                    # Player eliminated - default to 0 contribution
                    self.state.game_state["pending_contributions"][self.state.current_player_id] = 0
                    self.state.game_state["eliminations"].append(self.state.current_player_id)
                    self.state.add_observation(
                        message=f"Player {self.state.current_player_id} eliminated for too many invalid moves.",
                        observation_type=ta.ObservationType.GAME_MESSAGE
                    )
        else:
            # No valid contribution found - use the state's invalid move handling
            eliminated = self.state.set_invalid_move(
                f"No valid contribution found. Please use format '[X]' where X is 0-{self.state.game_state['endowment']}."
            )
            if not eliminated:
                return  # Player gets another chance
            else:
                # Player eliminated - default to 0 contribution
                self.state.game_state["pending_contributions"][self.state.current_player_id] = 0
                self.state.game_state["eliminations"].append(self.state.current_player_id)
                self.state.add_observation(
                    message=f"Player {self.state.current_player_id} eliminated for too many invalid moves.",
                    observation_type=ta.ObservationType.GAME_MESSAGE
                )
        
        # Check if all alive players have made their decisions
        alive_players = [p for p in range(self.state.num_players) if self.state.is_player_alive(p)]
        if all(p in self.state.game_state["pending_contributions"] for p in alive_players):
            # All players have decided - now reveal all contributions
            self.state.game_state["contributions"] = self.state.game_state["pending_contributions"].copy()
            self.state.game_state["pending_contributions"] = {}
            
            # Announce all contributions at once
            contrib_msg = "All players have made their decisions:\n"
            for pid in sorted(alive_players):
                contrib_msg += f"Player {pid}: {self.state.game_state['contributions'][pid]} tokens\n"
            self.state.add_observation(
                message=contrib_msg,
                observation_type=ta.ObservationType.GAME_MESSAGE
            )
            
            # Now resolve the round
            self._resolve_round()
            
            # Advance to next round or finish
            self.state.game_state["round"] += 1
            if self.state.game_state["round"] > self.state.game_state["num_rounds"]:
                self._determine_winner()
            else:
                # Reset for next round - clear both contributions and pending data
                self.state.game_state["contributions"] = {i: None for i in range(self.state.num_players)}
                self.state.game_state["pending_messages"] = {}
                self.state.game_state["pending_contributions"] = {}
                self.state.game_state["phase"] = "conversation"
                self.state.game_state["conversation_round"] = 0
                self.state.add_observation(
                    message=f"--- Starting Round {self.state.game_state['round']} ---\n"
                           f"Communication phase: Submit your message using {{message}} format for public communication.",
                    observation_type=ta.ObservationType.GAME_MESSAGE
                )

    def _resolve_round(self):
        contributions = self.state.game_state["contributions"]
        alive_players = [p for p in range(self.state.num_players) if self.state.is_player_alive(p)]
        
        # Calculate total contribution from alive players only
        total_contribution = sum(contributions.get(p, 0) for p in alive_players)
        public_good = total_contribution * self.state.game_state["multiplication_factor"]
        
        # Share is divided among alive players only
        share_per_player = public_good / len(alive_players) if alive_players else 0
        
        # Calculate payoffs for each player
        round_info = {
            "round": self.state.game_state["round"],
            "contributions": {p: contributions.get(p, 0) for p in alive_players},
            "total_contribution": total_contribution,
            "public_good": public_good,
            "payoffs": {}
        }
        
        result_message = f"Round {self.state.game_state['round']} results:\n"
        result_message += f"Alive players: {len(alive_players)}\n"
        result_message += f"Contributions: {', '.join([f'P{i}: {contributions.get(i, 0)}' for i in alive_players])}\n"
        result_message += f"Total contribution: {total_contribution}\n"
        result_message += f"Public good: {total_contribution} × {self.state.game_state['multiplication_factor']} = {public_good:.1f}\n"
        result_message += f"Share per alive player: {share_per_player:.1f}\n\n"
        
        for player_id in range(self.state.num_players):
            if self.state.is_player_alive(player_id):
                tokens_kept = self.state.game_state["endowment"] - contributions.get(player_id, 0)
                payoff = tokens_kept + share_per_player
                round_info["payoffs"][player_id] = payoff
                
                self.state.game_state["round_scores"][player_id] = payoff
                self.state.game_state["total_scores"][player_id] += payoff
                
                result_message += f"Player {player_id}: kept {tokens_kept} + share {share_per_player:.1f} = {payoff:.1f} (total: {self.state.game_state['total_scores'][player_id]:.1f})\n"
            else:
                result_message += f"Player {player_id}: ELIMINATED\n"
        
        # Save round history
        self.state.game_state["history"].append(round_info)
        
        # Announce results
        self.state.add_observation(
            message=result_message,
            observation_type=ta.ObservationType.GAME_MESSAGE
        )

    def _determine_winner(self):
        # Only consider alive players for winning
        alive_players = [p for p in range(self.state.num_players) if self.state.is_player_alive(p)]
        scores = {p: self.state.game_state["total_scores"][p] for p in alive_players}
        
        if not scores:  # No alive players
            self.state.set_draw(reason="All players eliminated!")
            return
            
        max_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == max_score]
        
        final_message = f"Game Over! Final scores:\n"
        for player_id in range(self.state.num_players):
            score = self.state.game_state["total_scores"][player_id]
            status = " (eliminated)" if not self.state.is_player_alive(player_id) else ""
            final_message += f"Player {player_id}: {score:.1f}{status}\n"
        
        if len(winners) == 1:
            self.state.set_winners(
                player_ids=winners,
                reason=f"{final_message}\nPlayer {winners[0]} wins with {max_score:.1f} points!"
            )
        else:
            # Multiple winners - use set_game_outcome with normalized rewards
            reward_dict = {}
            for p in range(self.state.num_players):
                if not self.state.is_player_alive(p):
                    reward_dict[p] = -1  # Eliminated players get -1
                elif p in winners:
                    reward_dict[p] = 1   # Winners get 1
                else:
                    reward_dict[p] = 0   # Other players get 0
                    
            self.state.set_game_outcome(
                reward_dict=reward_dict,
                reason=f"{final_message}\nDraw! Players {', '.join(map(str, winners))} tied with {max_score:.1f} points."
            )