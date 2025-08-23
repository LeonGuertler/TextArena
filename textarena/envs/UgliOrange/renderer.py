def render_negotiation_state(game_state, player_id=None):
    """Render the current negotiation state for display."""
    
    lines = []
    lines.append("=== UGLI ORANGE NEGOTIATION ===")
    lines.append("")
    
    # Show objectives
    lines.append("OBJECTIVES:")
    lines.append("- Roland: Get 3000 orange rinds (budget: $250,000)")
    lines.append("- Jones: Get 3000 orange juice (budget: $250,000)")
    lines.append("")
    
    # Show current proposals
    if game_state["proposals"]:
        lines.append("PROPOSALS:")
        for proposal in game_state["proposals"]:
            proposer_name = "Roland" if proposal["proposer"] == 0 else "Jones"
            lines.append(f"#{proposal['number']} ({proposer_name}): {proposal['text']}")
        lines.append("")
    
    # Show deal status
    if game_state["deal_reached"]:
        deal = game_state["final_deal"]
        proposer_name = "Roland" if deal["proposer"] == 0 else "Jones"
        accepter_name = "Roland" if deal["accepter"] == 0 else "Jones"
        lines.append(f"DEAL REACHED: {accepter_name} accepted Proposal #{deal['proposal_number']} from {proposer_name}")
        lines.append("")
    elif game_state["negotiation_complete"]:
        lines.append("NEGOTIATION COMPLETE - No deal reached")
        lines.append("")
    else:
        lines.append("NEGOTIATION IN PROGRESS")
        lines.append("")
    
    return "\n".join(lines)