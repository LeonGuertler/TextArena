from typing import List, Optional

def create_board_str(scores: List[int], turn_total: int, current_player: int, current_roll: Optional[int], goal: int = 100) -> str:
    # Dice face rendering
    dice_faces = {
        1: "┌─────────┐\n│         │\n│    ●    │\n│         │\n└─────────┘",
        2: "┌─────────┐\n│ ●       │\n│         │\n│       ● │\n└─────────┘",
        3: "┌─────────┐\n│ ●       │\n│    ●    │\n│       ● │\n└─────────┘",
        4: "┌─────────┐\n│ ●     ● │\n│         │\n│ ●     ● │\n└─────────┘",
        5: "┌─────────┐\n│ ●     ● │\n│    ●    │\n│ ●     ● │\n└─────────┘",
        6: "┌─────────┐\n│ ●     ● │\n│ ●     ● │\n│ ●     ● │\n└─────────┘",
        None: "┌─────────┐\n│         │\n│         │\n│         │\n└─────────┘"
    }

    dice_art = dice_faces.get(current_roll, dice_faces[None]).splitlines()
    
    # Format output
    return f"""
┌─ SCORES ─────────────┐                 ┌─ CURRENT TURN ───────────┐
│                      │                 │                          │
│  Player 0: {scores[0]:>3} pts   │   {dice_art[0]}   │  Player {current_player} rolling        │
│  Player 1: {scores[1]:>3} pts   │   {dice_art[1]}   │  Current roll: [{current_roll if current_roll is not None else ' '}]       │
│                      │   {dice_art[2]}   │  Turn total: {turn_total:>3} pts     │
│  Goal: {goal} points    │   {dice_art[3]}   │  Options: Roll or Hold   │
│                      │   {dice_art[4]}   │                          │
│                      │                 │                          │
└──────────────────────┘                 └──────────────────────────┘
    """.rstrip()
