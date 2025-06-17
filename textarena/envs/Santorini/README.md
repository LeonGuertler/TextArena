# Santorini Base Fixed Worker Environment

This is an implementation of the Santorini board game supporting 2-3 players with fixed initial worker positions.

## Game Description

Santorini is an abstract strategy game where players build towers and try to climb to victory. Players take turns moving one of their workers and then building a level on an adjacent space.

### Components
- 5x5 grid board
- Building levels (0-3) and domes
- Two workers per player
- Supports 2-3 players

### Initial Setup
Fixed worker positions optimized for balanced gameplay:

2 Players:
- Player 0 (Blue): C2, B3
- Player 1 (White): D3, C4

3 Players:
- Player 0 (Blue): C3, B3
- Player 1 (White): D3, B4
- Player 2 (Grey): D2, D4

### Turn Structure
On your turn, you must:
1. Move one worker to an adjacent space (including diagonals)
2. Build one level on an adjacent space to where you moved

### Movement Rules
- Can move to any of the 8 adjacent spaces
- Can move up one level
- Can move down any number of levels
- Cannot move up more than one level
- Cannot move to occupied spaces
- Cannot move to spaces with domes

### Building Rules
- Build in any of the 8 spaces adjacent to the moved worker
- Build one level at a time (0→1→2→3)
- Place a dome (◊) on level 3 to complete the tower
- Cannot build on occupied spaces or completed towers

### Winning
- Win by moving a worker up to level 3
- Win if next player has no legal moves

## Usage

### Action Format
Actions should be in the format: `[worker_num source dest build]`

Example:
- `[1C2C3B3]` - Move worker 1 from C2 to C3 and build at B3
- `[2D3E3E4]` - Move worker 2 from D3 to E3 and build at E4

### Board Representation
```
     1     2     3     4     5
  ┌─────┬─────┬─────┬─────┬─────┐
A │ 0   │ 0   │ 0   │ 0   │ 0   │ A
  ├─────┼─────┼─────┼─────┼─────┤
B │ 0   │ 0 ① │ 0 ② │ 0   │ 0   │ B
  ├─────┼─────┼─────┼─────┼─────┤
C │ 0   │ 0   │ 1   │ 0 ❷ │ 0   │ C
  ├─────┼─────┼─────┼─────┼─────┤
D │ 0   │ 0   │ 0 ❶ │ 0   │ 0   │ D
  ├─────┼─────┼─────┼─────┼─────┤
E │ 0   │ 0   │ 0   │ 0   │ 0   │ E
  └─────┴─────┴─────┴─────┴─────┘
     1     2     3     4     5
```

Legend:
- Numbers (0-3): Building levels
- ◊: Dome (completed tower)
- ①②: Blue player's workers
- ❶❷: White player's workers
- ⓵⓶: Grey player's workers (3-player game only)

## Implementation Notes

This is a simplified version that:
- Uses fixed initial worker positions optimized for balanced gameplay
- Has no worker placement phase
- Does not include God Powers

Future versions may add:
- Worker placement phase
- Support for 4 players
- God Powers
