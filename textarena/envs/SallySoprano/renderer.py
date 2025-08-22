from typing import Dict, Any, Optional


def render_negotiation_state(game_state: Dict[str, Any], player_id: Optional[int] = None) -> str:
    """
    Render the current state of the Sally Soprano negotiation.
    
    Args:
        game_state: Current game state
        player_id: ID of the player viewing the state (affects what information is shown)
        
    Returns:
        String representation of the negotiation state
    """
    lines = []
    
    # Header
    lines.append("SALLY SOPRANO NEGOTIATION")
    lines.append("=" * 40)
    lines.append("")
    
    # Current proposal
    current_proposal = game_state.get("current_proposal")
    if current_proposal:
        proposer_name = "Sally's Agent" if current_proposal["proposer"] == 0 else "Business Manager"
        lines.append("CURRENT PROPOSAL:")
        lines.append(f"From: {proposer_name}")
        lines.append(f"Amount: ${current_proposal['amount']:,}")
        
        if current_proposal.get("rationale"):
            lines.append(f"Rationale: {current_proposal['rationale']}")
        lines.append("")
    else:
        lines.append("No current proposal on the table.")
        lines.append("")
    
    # Proposal history (last 3)
    proposal_history = game_state.get("proposal_history", [])
    if proposal_history:
        lines.append("RECENT PROPOSALS:")
        for i, proposal in enumerate(proposal_history[-3:]):
            proposer_name = "Sally's Agent" if proposal["proposer"] == 0 else "Business Manager"
            lines.append(f"{i+1}. {proposer_name}: ${proposal['amount']:,} (Round {proposal['round']})")
        lines.append("")
    
    # Non-monetary promises
    promises = game_state.get("non_monetary_promises", [])
    if promises:
        lines.append("NON-MONETARY PROMISES MADE:")
        for promise in promises[-3:]:  # Show last 3
            player_name = "Sally's Agent" if promise["player"] == 0 else "Business Manager"
            lines.append(f"- {player_name} (Round {promise['round']}): {promise['text'][:100]}...")
        lines.append("")
    
    # Final deal (if reached)
    final_deal = game_state.get("final_deal")
    if final_deal:
        proposer_name = "Sally's Agent" if final_deal["proposer"] == 0 else "Business Manager"
        accepter_name = "Sally's Agent" if final_deal["accepter"] == 0 else "Business Manager"
        lines.append("DEAL REACHED:")
        lines.append(f"Final Salary: ${final_deal['amount']:,}")
        lines.append(f"Proposed by: {proposer_name}")
        lines.append(f"Accepted by: {accepter_name}")
        if final_deal.get("acceptance_rationale"):
            lines.append(f"Acceptance rationale: {final_deal['acceptance_rationale']}")
        lines.append("")
    
    # Instructions for negotiating players
    if player_id is not None and player_id < 2:
        lines.append("AVAILABLE ACTIONS:")
        lines.append("- [Propose] amount - Make a salary proposal")
        lines.append("- [Accept] - Accept the current proposal (if not your own)")
        lines.append("- Free text - Argue your position and make non-monetary promises")
        lines.append("")
    
    return "\n".join(lines)


def render_judge_summary(game_state: Dict[str, Any]) -> str:
    """
    Render a summary for the judge showing the complete negotiation.
    
    Args:
        game_state: Current game state
        
    Returns:
        String representation for judge evaluation
    """
    lines = []
    
    lines.append("NEGOTIATION SUMMARY FOR JUDGE")
    lines.append("=" * 50)
    lines.append("")
    
    # Final outcome
    final_deal = game_state.get("final_deal")
    if final_deal:
        lines.append("FINAL DEAL:")
        lines.append(f"Salary: ${final_deal['amount']:,}")
        proposer_name = "Sally's Agent" if final_deal["proposer"] == 0 else "Business Manager"
        accepter_name = "Sally's Agent" if final_deal["accepter"] == 0 else "Business Manager"
        lines.append(f"Proposed by: {proposer_name}")
        lines.append(f"Accepted by: {accepter_name}")
        lines.append("")
    else:
        lines.append("NO DEAL REACHED - Maximum rounds exceeded")
        lines.append("")
    
    # All proposals made
    proposal_history = game_state.get("proposal_history", [])
    if proposal_history:
        lines.append("ALL PROPOSALS MADE:")
        for i, proposal in enumerate(proposal_history):
            proposer_name = "Sally's Agent" if proposal["proposer"] == 0 else "Business Manager"
            lines.append(f"{i+1}. {proposer_name}: ${proposal['amount']:,} (Round {proposal['round']})")
            if proposal.get("rationale"):
                lines.append(f"   Rationale: {proposal['rationale']}")
        lines.append("")
    
    # All non-monetary promises
    promises = game_state.get("non_monetary_promises", [])
    if promises:
        lines.append("NON-MONETARY PROMISES:")
        for promise in promises:
            player_name = "Sally's Agent" if promise["player"] == 0 else "Business Manager"
            lines.append(f"- {player_name} (Round {promise['round']}): {promise['text']}")
        lines.append("")
    
    lines.append("EVALUATION CRITERIA:")
    lines.append("- Consider each party's confidential instructions and constraints")
    lines.append("- Evaluate the final salary in context of market rates and alternatives")
    lines.append("- Consider any non-monetary promises and their value")
    lines.append("- Determine who achieved a better outcome relative to their position")
    
    return "\n".join(lines)