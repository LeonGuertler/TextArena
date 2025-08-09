import textwrap

def create_board_str(game_state: dict) -> str:
    """Create a visual representation of the Market Entry Game state."""
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
    lines.append("╭─────────────────── MARKET ENTRY GAME ────────────────────╮")
    lines.append(f"│ Round {current_round:>2}/{total_rounds:<2} │ {phase_display:<35} │")
    
    capacity = game_state.get("market_capacity", 2)
    entry_profit = game_state.get("entry_profit", 15)
    overcrowding = game_state.get("overcrowding_penalty", -5)
    safe = game_state.get("safe_payoff", 5)
    
    lines.append(f"│ Market Capacity: {capacity} │ Entry: {entry_profit:+3} │ Crowd: {overcrowding:+3} │ Safe: {safe:+3} │")
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
                    # Truncate long messages - increased limit
                    max_len = 40  # Increased from 25
                    msg_display = msg[:max_len] + "..." if len(msg) > max_len else msg
                    status = f"💬 \"{msg_display}\""
                else:
                    status = "🤐 Silent"
            else:
                status = "⏳ Waiting..."
            lines.append(f"│ Player {player_id}: {status:<45} │")
            
    else:  # decision phase
        # Show decision status
        decisions = game_state.get("decisions", {})
        pending_decisions = game_state.get("pending_decisions", {})
        lines.append("│ Player Decision Status:                                   │")
        
        total_scores = game_state.get("total_scores", {})
        num_players = len(total_scores) if total_scores else 4
        
        for player_id in range(num_players):
            if player_id in eliminations:
                status = "❌ ELIMINATED"
            elif player_id in decisions and decisions[player_id] is not None:
                decision = decisions[player_id]
                status = f"✅ {'ENTER' if decision == 'E' else 'STAY OUT'}"
            elif player_id in pending_decisions:
                decision = pending_decisions[player_id]
                status = f"⏳ {'ENTER' if decision == 'E' else 'STAY OUT'} (pending)"
            else:
                status = "⏳ Deciding..."
            lines.append(f"│ Player {player_id}: {status:<45} │")
    
    lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Show full messages if in conversation phase and messages exist
    if phase == "conversation" and pending_messages:
        has_messages = [pid for pid, msg in pending_messages.items() if msg]
        if has_messages:
            lines.append("┌─ 💬 FULL MESSAGES ────────────────────────────────────────┐")
            for player_id in sorted(has_messages):
                msg = pending_messages[player_id]
                if msg:
                    # Word wrap long messages
                    wrapped = textwrap.wrap(msg, width=56)
                    lines.append(f"│ P{player_id}: {wrapped[0]:<53} │")
                    for line in wrapped[1:]:
                        lines.append(f"│     {line:<53} │")
            lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Market status visualization (if decisions have been made)
    if phase == "decision" and game_state.get("decisions"):
        decisions = game_state.get("decisions", {})
        alive_entries = [pid for pid, dec in decisions.items() 
                        if dec == 'E' and pid not in eliminations]
        if decisions and any(d is not None for d in decisions.values()):
            num_entries = len(alive_entries)
            
            lines.append("┌─ 🏪 MARKET STATUS ────────────────────────────────────────┐")
            
            # Visual representation of market capacity
            capacity_bar = ""
            for i in range(capacity):
                if i < num_entries:
                    capacity_bar += "🟢"  # Filled spot
                else:
                    capacity_bar += "⚪"  # Empty spot
            
            # Add overflow if overcrowded
            if num_entries > capacity:
                overflow = num_entries - capacity
                capacity_bar += " +" + "🔴" * overflow + " (OVERCROWDED!)"
            
            lines.append(f"│ Market Occupancy: {capacity_bar:<40} │")
            lines.append(f"│ Entrants: {num_entries}/{capacity} │ Status: {'❌ Overcrowded' if num_entries > capacity else '✅ Profitable' if num_entries > 0 else '⚫ Empty':<25} │")
            
            # Show who entered
            if alive_entries:
                entrant_list = f"Players {', '.join(map(str, sorted(alive_entries)))}"
                lines.append(f"│ Who entered: {entrant_list:<44} │")
            
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
            lines.append(f"│ {rank_icon:<3} Player {player_id}: {score:>4} points                          │")
        
        # Then eliminated players
        for player_id, score in eliminated_scores:
            lines.append(f"│ ❌  Player {player_id}: {score:>4} points (eliminated)              │")
        
        lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Compact round history
    history = game_state.get("history", [])
    if history:
        lines.append("┌─ 📊 ROUND HISTORY ────────────────────────────────────────┐")
        lines.append("│ Rnd │ Entries │ Market │ Winners           │ Avg Payoff  │")
        lines.append("├─────┼─────────┼────────┼───────────────────┼─────────────┤")
        
        for round_info in history[-5:]:  # Show last 5 rounds only
            round_num = round_info.get("round", "?")
            num_entrants = round_info.get("num_entrants", 0)
            is_overcrowded = round_info.get("is_overcrowded", False)
            decisions = round_info.get("decisions", {})
            payoffs = round_info.get("payoffs", {})
            
            # Determine who won this round
            if payoffs:
                max_payoff = max(payoffs.values())
                round_winners = [str(pid) for pid, pay in payoffs.items() if pay == max_payoff]
                winner_str = ",".join(round_winners[:3])  # Show first 3 winners
                if len(round_winners) > 3:
                    winner_str += f"+{len(round_winners)-3}"
            else:
                winner_str = "-"
            
            # Market status
            market_status = "❌" if is_overcrowded else "✅" if num_entrants > 0 else "⚫"
            
            avg_payoff = sum(payoffs.values()) / len(payoffs) if payoffs else 0
            
            lines.append(f"│ {round_num:>3} │    {num_entrants}    │   {market_status}    │ P{winner_str:<15} │ {avg_payoff:>8.1f}    │")
        
        if len(history) > 5:
            lines.append(f"│     │ ... ({len(history)-5} more rounds above) ...                       │")
        
        lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Strategy insights (based on history)
    if history and len(history) >= 2:
        lines.append("┌─ 📈 STRATEGY INSIGHTS ────────────────────────────────────┐")
        
        # Calculate average entries and success rate
        total_entries = sum(r.get("num_entrants", 0) for r in history)
        avg_entries = total_entries / len(history)
        overcrowded_rounds = sum(1 for r in history if r.get("is_overcrowded", False))
        overcrowd_rate = (overcrowded_rounds / len(history)) * 100
        
        lines.append(f"│ Avg entries/round: {avg_entries:.1f} │ Overcrowded: {overcrowd_rate:.0f}% of rounds   │")
        
        # Find most successful strategy
        player_strategies = {}
        for round_info in history:
            decisions = round_info.get("decisions", {})
            payoffs = round_info.get("payoffs", {})
            for pid, decision in decisions.items():
                if pid not in player_strategies:
                    player_strategies[pid] = {"enters": 0, "stays": 0, "total_payoff": 0}
                if decision == 'E':
                    player_strategies[pid]["enters"] += 1
                else:
                    player_strategies[pid]["stays"] += 1
                player_strategies[pid]["total_payoff"] += payoffs.get(pid, 0)
        
        # Find most aggressive and most conservative players
        if player_strategies:
            entry_rates = {pid: (stats["enters"] / (stats["enters"] + stats["stays"])) 
                          for pid, stats in player_strategies.items() 
                          if (stats["enters"] + stats["stays"]) > 0}
            if entry_rates:
                most_aggressive = max(entry_rates.items(), key=lambda x: x[1])
                most_conservative = min(entry_rates.items(), key=lambda x: x[1])
                
                lines.append(f"│ Most aggressive: P{most_aggressive[0]} ({most_aggressive[1]*100:.0f}% entry)          │")
                lines.append(f"│ Most conservative: P{most_conservative[0]} ({most_conservative[1]*100:.0f}% entry)        │")
        
        lines.append("└───────────────────────────────────────────────────────────┘")
    
    # Game mechanics reminder (compact)
    lines.append("┌─ ℹ️  QUICK REFERENCE ──────────────────────────────────────┐")
    lines.append("│ Communication: {message}  │  Decision: [E] or [S]         │")
    lines.append(f"│ Enter: {entry_profit:+3} if ≤{capacity} players, {overcrowding:+3} if >{capacity} │ Stay Out: {safe:+3}     │")
    lines.append("└───────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)