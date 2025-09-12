def create_board_str(game_state: dict) -> str:
    """Create a visual representation of the Public Goods Game state."""
    lines = []
    
    # Determine phase info
    phase = game_state.get("phase", "conversation")
    current_round = game_state.get("round", 1)
    total_rounds = game_state.get("num_rounds", 5)
    
    if phase == "conversation":
        conv_round = game_state.get("conversation_round", 0) + 1
        total_conv = game_state.get("total_conversation_rounds", 3)
        phase_display = f"💬 Communication ({conv_round}/{total_conv})"
    else:
        phase_display = "🎯 Decision Phase"
    
    # Header with game info
    lines.append("╭─────────────────── PUBLIC GOODS GAME ────────────────────╮")
    lines.append(f"│ Round {current_round:>2}/{total_rounds:<2} │ {phase_display:<35} │")
    
    endowment = game_state.get("endowment", 20)
    multiplier = game_state.get("multiplication_factor", 1.5)
    lines.append(f"│ Endowment: {endowment:>2} tokens │ Multiplier: {multiplier}x                │")
    lines.append("╰───────────────────────────────────────────────────────────╯")
    
    # Get alive vs eliminated players
    eliminations = game_state.get("eliminations", [])
    
    # Current round status
    lines.append("┌─ 🎮 CURRENT ROUND STATUS ─────────────────────────────────┐")
    
    if phase == "conversation":
        # Show who has submitted messages this conversation round
        pending_messages = game_state.get("pending_messages", {})
        lines.append("│ Player Messages Status:                                   │")
        
        # Assume players 0 to num_players-1, need to infer from total_scores or other data
        total_scores = game_state.get("total_scores", {})
        num_players = len(total_scores) if total_scores else 4
        
        for player_id in range(num_players):
            if player_id in eliminations:
                status = "❌ ELIMINATED"
            elif player_id in pending_messages:
                msg = pending_messages[player_id]
                if msg:
                    # Truncate long messages
                    msg_display = msg[:25] + "..." if len(msg) > 25 else msg
                    status = f"💬 \"{msg_display}\""
                else:
                    status = "🤐 Silent"
            else:
                status = "⏳ Waiting..."
            lines.append(f"│ Player {player_id}: {status:<45} │")
            
    else:  # decision phase
        # Show contribution status
        contributions = game_state.get("contributions", {})
        pending_contributions = game_state.get("pending_contributions", {})
        lines.append("│ Player Contribution Status:                               │")
        
        total_scores = game_state.get("total_scores", {})
        num_players = len(total_scores) if total_scores else 4
        
        for player_id in range(num_players):
            if player_id in eliminations:
                status = "❌ ELIMINATED"
            elif player_id in contributions and contributions[player_id] is not None:
                amount = contributions[player_id]
                status = f"✅ {amount} tokens"
            elif player_id in pending_contributions:
                amount = pending_contributions[player_id]
                status = f"⏳ {amount} tokens (pending)"
            else:
                status = "⏳ Deciding..."
            lines.append(f"│ Player {player_id}: {status:<45} │")
    
    lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Round results (if decision phase and contributions revealed)
    if phase == "decision" and game_state.get("contributions"):
        contributions = game_state.get("contributions", {})
        if any(c is not None for c in contributions.values()):
            # Calculate what we can show
            alive_contribs = {pid: amt for pid, amt in contributions.items() 
                            if amt is not None and pid not in eliminations}
            if alive_contribs:
                total_contrib = sum(alive_contribs.values())
                public_good = total_contrib * multiplier
                num_alive = len(alive_contribs)
                share_per_player = public_good / num_alive if num_alive > 0 else 0
                
                lines.append("┌─ 💰 ROUND CALCULATION ────────────────────────────────────┐")
                lines.append(f"│ Total Contributions: {total_contrib:>3} tokens                        │")
                lines.append(f"│ Public Good: {total_contrib} × {multiplier} = {public_good:>6.1f}                    │")
                lines.append(f"│ Share per Player: {share_per_player:>6.1f} tokens                     │")
                lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Player standings
    total_scores = game_state.get("total_scores", {})
    if total_scores:
        lines.append("┌─ 🏆 PLAYER STANDINGS ─────────────────────────────────────┐")
        
        # Sort by score, but put eliminated players at bottom
        alive_scores = [(pid, score) for pid, score in total_scores.items() 
                       if pid not in eliminations]
        eliminated_scores = [(pid, score) for pid, score in total_scores.items() 
                           if pid in eliminations]
        
        alive_scores.sort(key=lambda x: x[1], reverse=True)
        eliminated_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Show alive players first
        for rank, (player_id, score) in enumerate(alive_scores, 1):
            rank_icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            lines.append(f"│ {rank_icon:<3} Player {player_id}: {score:>6.1f} points                    │")
        
        # Then eliminated players
        for player_id, score in eliminated_scores:
            lines.append(f"│ ❌  Player {player_id}: {score:>6.1f} points (eliminated)         │")
        
        lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Compact round history
    history = game_state.get("history", [])
    if history:
        lines.append("┌─ 📊 ROUND HISTORY ────────────────────────────────────────┐")
        lines.append("│ Rnd │ Contributions  │ Total │ Public │ Avg Payoff        │")
        lines.append("├─────┼────────────────┼───────┼────────┼───────────────────┤")
        
        for round_info in history[-5:]:  # Show last 5 rounds only
            round_num = round_info.get("round", "?")
            contributions = round_info.get("contributions", {})
            
            # Create compact contribution display
            contrib_list = [f"{pid}:{amt}" for pid, amt in sorted(contributions.items())]
            contrib_str = ",".join(contrib_list)
            if len(contrib_str) > 14:
                contrib_str = contrib_str[:11] + "..."
            
            total = round_info.get("total_contribution", 0)
            public_good = round_info.get("public_good", 0)
            payoffs = round_info.get("payoffs", {})
            avg_payoff = sum(payoffs.values()) / len(payoffs) if payoffs else 0
            
            lines.append(f"│ {round_num:>3} │ {contrib_str:<14} │ {total:>5} │ {public_good:>6.1f} │ {avg_payoff:>8.1f}         │")
        
        if len(history) > 5:
            lines.append(f"│     │ ... ({len(history)-5} more rounds shown above)          │")
        
        lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Game mechanics reminder (compact)
    lines.append("┌─ ℹ️  QUICK REFERENCE ──────────────────────────────────────┐")
    lines.append("│ Communication: {message}  │  Decision: [amount]           │")
    lines.append(f"│ Payoff = kept_tokens + share_of_public_good               │")
    lines.append("└───────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)