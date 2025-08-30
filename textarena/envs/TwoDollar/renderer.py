"""
Rendering functions for the Two Dollar negotiation game environment.

Provides functions to display game state, negotiation history, and role information
in a clear and organized format for players and observers.
"""

from typing import Dict, List, Any, Optional


def render_game_state(current_proposal: Dict[str, Any], 
                     total_amount: float,
                     negotiation_history: List[Dict[str, Any]],
                     max_history_items: int = 5) -> str:
    """
    Render the current game state including proposal and recent history.
    
    Args:
        current_proposal: Current proposal dict with 'amount' and 'proposer'
        total_amount: Total amount being negotiated
        negotiation_history: List of all negotiation actions
        max_history_items: Maximum number of history items to show
        
    Returns:
        Formatted string showing current state and recent history
    """
    output = []
    
    # Current proposal section
    if current_proposal.get("amount") is not None:
        proposer_id = current_proposal["proposer"]
        amount = current_proposal["amount"]
        other_amount = total_amount - amount
        
        output.append("=== CURRENT PROPOSAL ===")
        output.append(f"Player {proposer_id + 1} wants: ${amount:.2f}")
        output.append(f"Player {2 - proposer_id} gets: ${other_amount:.2f}")
        output.append("")
    else:
        output.append("=== NO CURRENT PROPOSAL ===")
        output.append("Waiting for a player to make a proposal...")
        output.append("")
    
    # Recent negotiation history
    if negotiation_history:
        output.append("=== RECENT NEGOTIATION HISTORY ===")
        recent_history = negotiation_history[-max_history_items:]
        
        for action in recent_history:
            player_id = action["player_id"]
            action_type = action["action_type"]
            round_num = action["round"]
            message = action.get("message", "")
            
            if action_type == "propose":
                amount = action.get("amount", 0)
                other_amount = total_amount - amount
                action_desc = f"proposed ${amount:.2f} for self, ${other_amount:.2f} for opponent"
            elif action_type == "accept":
                action_desc = "accepted the proposal"
            elif action_type == "reject":
                action_desc = "rejected the proposal"
            else:
                action_desc = action_type
            
            # Format the history entry
            entry = f"Round {round_num + 1}: Player {player_id + 1} {action_desc}"
            if message:
                entry += f" (said: \"{message}\")"
            
            output.append(entry)
        
        if len(negotiation_history) > max_history_items:
            output.append(f"... and {len(negotiation_history) - max_history_items} earlier actions")
    else:
        output.append("=== NEGOTIATION HISTORY ===")
        output.append("No actions taken yet.")
    
    return "\n".join(output)


def render_role_instructions(role: Dict[str, Any], 
                           player_id: int,
                           total_amount: float,
                           max_rounds: int,
                           current_round: int,
                           player_deadline: Optional[int] = None) -> str:
    """
    Render role instructions for a specific player.
    
    Args:
        role: Role dictionary with instructions and conditions
        player_id: Player ID (0 or 1)
        total_amount: Total amount being negotiated
        max_rounds: Maximum number of rounds
        current_round: Current round number
        player_deadline: Deadline for x_rounds role (if applicable)
        
    Returns:
        Formatted string with role instructions
    """
    output = []
    
    output.append(f"=== YOUR SECRET ROLE: {role.get('name', 'Unknown').upper()} ===")
    output.append("")
    
    # Role instructions
    instructions = role.get("instructions", "No instructions provided.")
    
    # Replace placeholders for x_rounds role
    if role.get("name") == "x_rounds" and player_deadline is not None:
        instructions = instructions.replace("{deadline}", str(player_deadline))
        instructions = instructions.replace("{total_rounds}", str(max_rounds))
    
    output.append("INSTRUCTIONS:")
    output.append(instructions)
    output.append("")
    
    # Victory and failure conditions
    output.append("SUCCESS CONDITION:")
    output.append(role.get("victory_condition", "Not specified"))
    output.append("")
    
    output.append("FAILURE CONDITION:")
    output.append(role.get("failure_condition", "Not specified"))
    output.append("")
    
    # Special warnings for enforceable roles
    enforcement = role.get("enforcement", "none")
    if enforcement == "action_validation":
        output.append("⚠️  WARNING: This role has STRICT REQUIREMENTS that are automatically enforced!")
        if role.get("name") == "say_little":
            max_words = role.get("behavioral_rules", {}).get("max_words_per_message", 15)
            output.append(f"   - You can only use {max_words} words or less before each [Action]")
            output.append("   - Exceed this limit 3 times and you'll be forced to accept any deal")
        elif role.get("name") == "high_tension":
            max_concession = role.get("behavioral_rules", {}).get("max_concession", 0.01)
            output.append(f"   - You can only reduce your proposals by ${max_concession:.2f} or less at a time")
            output.append("   - Violate this 3 times and you'll be forced to accept any deal")
        output.append("")
    
    elif enforcement == "end_game_check":
        if role.get("threshold") is not None:
            threshold = role.get("threshold", 0)
            output.append(f"⚠️  CRITICAL: You MUST receive at least ${threshold:.2f} or you get $0.00!")
            output.append("")
        elif role.get("name") == "x_rounds" and player_deadline is not None:
            output.append(f"⚠️  DEADLINE: You MUST accept a deal by round {player_deadline} or you get $0.00!")
            remaining_rounds = player_deadline - current_round
            if remaining_rounds <= 3:
                output.append(f"   🚨 URGENT: Only {remaining_rounds} rounds left!")
            output.append("")
    
    return "\n".join(output)


def render_proposal_analysis(current_proposal: Dict[str, Any],
                           total_amount: float,
                           player_id: int,
                           role: Dict[str, Any]) -> str:
    """
    Render analysis of current proposal from player's perspective.
    
    Args:
        current_proposal: Current proposal dict
        total_amount: Total amount being negotiated
        player_id: Player ID analyzing the proposal
        role: Player's role information
        
    Returns:
        Formatted analysis of the proposal
    """
    if current_proposal.get("amount") is None:
        return "No current proposal to analyze."
    
    proposer_id = current_proposal["proposer"]
    proposer_amount = current_proposal["amount"]
    
    # Calculate what this player would get
    if proposer_id == player_id:
        player_amount = proposer_amount
        opponent_amount = total_amount - proposer_amount
        perspective = "your own proposal"
    else:
        player_amount = total_amount - proposer_amount
        opponent_amount = proposer_amount
        perspective = "opponent's proposal"
    
    output = []
    output.append(f"=== PROPOSAL ANALYSIS ({perspective.upper()}) ===")
    output.append(f"You would receive: ${player_amount:.2f}")
    output.append(f"Opponent would receive: ${opponent_amount:.2f}")
    output.append("")
    
    # Role-specific analysis
    enforcement = role.get("enforcement", "none")
    
    if enforcement == "end_game_check" and role.get("threshold") is not None:
        threshold = role.get("threshold", 0)
        if player_amount >= threshold:
            output.append(f"✅ MEETS your minimum requirement of ${threshold:.2f}")
        else:
            shortfall = threshold - player_amount
            output.append(f"❌ FAILS your minimum requirement of ${threshold:.2f}")
            output.append(f"   You need ${shortfall:.2f} more to avoid getting $0.00!")
    
    elif role.get("name") in ["battle_ax", "imaginative"]:
        # Honor system roles - provide guidance
        percentage = (player_amount / total_amount) * 100
        if percentage >= 60:
            output.append(f"💪 Strong position: You get {percentage:.1f}% of the total")
        elif percentage >= 40:
            output.append(f"⚖️  Balanced split: You get {percentage:.1f}% of the total")
        else:
            output.append(f"⚠️  Weak position: You only get {percentage:.1f}% of the total")
    
    return "\n".join(output)


def render_negotiation_summary(negotiation_history: List[Dict[str, Any]],
                             player_proposal_history: Dict[int, List[float]],
                             total_amount: float) -> str:
    """
    Render a summary of the entire negotiation for post-game analysis.
    
    Args:
        negotiation_history: Complete negotiation history
        player_proposal_history: All proposals made by each player
        total_amount: Total amount negotiated
        
    Returns:
        Formatted negotiation summary
    """
    output = []
    output.append("=== NEGOTIATION SUMMARY ===")
    output.append("")
    
    if not negotiation_history:
        output.append("No negotiation took place.")
        return "\n".join(output)
    
    # Overall statistics
    total_actions = len(negotiation_history)
    proposals = [a for a in negotiation_history if a["action_type"] == "propose"]
    accepts = [a for a in negotiation_history if a["action_type"] == "accept"]
    rejects = [a for a in negotiation_history if a["action_type"] == "reject"]
    
    output.append(f"Total actions: {total_actions}")
    output.append(f"Proposals made: {len(proposals)}")
    output.append(f"Acceptances: {len(accepts)}")
    output.append(f"Rejections: {len(rejects)}")
    output.append("")
    
    # Player-specific statistics
    for player_id in [0, 1]:
        player_actions = [a for a in negotiation_history if a["player_id"] == player_id]
        player_proposals = player_proposal_history.get(player_id, [])
        
        output.append(f"Player {player_id + 1} Statistics:")
        output.append(f"  Actions taken: {len(player_actions)}")
        output.append(f"  Proposals made: {len(player_proposals)}")
        
        if player_proposals:
            first_proposal = player_proposals[0]
            last_proposal = player_proposals[-1]
            output.append(f"  First proposal: ${first_proposal:.2f}")
            output.append(f"  Final proposal: ${last_proposal:.2f}")
            
            if len(player_proposals) > 1:
                total_concession = first_proposal - last_proposal
                if total_concession > 0:
                    output.append(f"  Total concessions: ${total_concession:.2f}")
                elif total_concession < 0:
                    output.append(f"  Became more demanding: ${abs(total_concession):.2f}")
                else:
                    output.append("  No concessions made")
        
        output.append("")
    
    # Proposal progression
    if proposals:
        output.append("Proposal Progression:")
        for i, proposal in enumerate(proposals):
            player_id = proposal["player_id"]
            amount = proposal["amount"]
            other_amount = total_amount - amount
            round_num = proposal["round"]
            
            output.append(f"  Round {round_num + 1}: Player {player_id + 1} → "
                         f"${amount:.2f} for self, ${other_amount:.2f} for opponent")
        output.append("")
    
    # Final outcome
    if accepts:
        final_accept = accepts[-1]
        output.append("OUTCOME: Deal accepted")
        output.append(f"Accepted by: Player {final_accept['player_id'] + 1}")
        output.append(f"Final round: {final_accept['round'] + 1}")
    else:
        output.append("OUTCOME: No deal reached")
    
    return "\n".join(output)


def render_final_results(final_amounts: Dict[int, float],
                        player_roles: Dict[int, Dict[str, Any]],
                        total_amount: float) -> str:
    """
    Render final results showing what each player received and role compliance.
    
    Args:
        final_amounts: Final dollar amounts for each player
        player_roles: Role information for each player
        total_amount: Total amount that was negotiated
        
    Returns:
        Formatted final results
    """
    output = []
    output.append("=== FINAL RESULTS ===")
    output.append("")
    
    # Show final amounts
    for player_id in [0, 1]:
        amount = final_amounts.get(player_id, 0.0)
        role = player_roles.get(player_id, {})
        role_name = role.get("name", "Unknown")
        
        output.append(f"Player {player_id + 1} ({role_name}):")
        output.append(f"  Final amount: ${amount:.2f}")
        
        # Check role compliance
        enforcement = role.get("enforcement", "none")
        
        if enforcement == "end_game_check" and role.get("threshold") is not None:
            threshold = role.get("threshold", 0)
            if amount >= threshold:
                output.append(f"  ✅ Met minimum requirement of ${threshold:.2f}")
            else:
                output.append(f"  ❌ Failed minimum requirement of ${threshold:.2f}")
                output.append(f"     Result: Received $0.00 instead of negotiated amount")
        
        elif enforcement == "action_validation":
            if amount > 0:
                output.append(f"  ✅ Successfully followed behavioral constraints")
            else:
                output.append(f"  ❌ Violated behavioral constraints too many times")
        
        elif role.get("name") == "x_rounds":
            if amount > 0:
                output.append(f"  ✅ Met deadline requirement")
            else:
                output.append(f"  ❌ Failed to meet deadline")
        
        else:
            # Honor system role
            if amount > 0:
                percentage = (amount / total_amount) * 100
                output.append(f"  Received {percentage:.1f}% of total amount")
        
        output.append("")
    
    # Overall outcome
    total_distributed = sum(final_amounts.values())
    if total_distributed == total_amount:
        output.append("✅ Successful negotiation - all money distributed")
    elif total_distributed == 0:
        output.append("❌ Failed negotiation - no money distributed")
    else:
        lost_amount = total_amount - total_distributed
        output.append(f"⚠️  Partial success - ${lost_amount:.2f} lost due to role failures")
    
    return "\n".join(output)
