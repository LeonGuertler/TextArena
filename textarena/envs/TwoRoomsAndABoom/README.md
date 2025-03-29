# Two Rooms and a Boom Environment Documentation

## Overview
**Two Rooms and a Boom** is a social deduction game where players are divided into two teams (Red and Blue) and physically separated into two rooms. Players do not initially know other players' teams or roles. The Red Team's goal is to have the Red Team Bomber and Blue Team President in the same room at the end of the game, while the Blue Team's goal is to keep them in separate rooms. The game involves discussion, identity revelation, and strategic hostage exchanges between rooms.

## Action Space

- **Format:** Actions are strings that depend on the current game phase and player role:
  - **Discussion Phase (All Players):**
    - Free-form text communication with other players in the same room
  - **Leader Selection Phase (Room Leaders Only):**
    - **Select Hostage:** `[Player X]` or `[X]` where X is a player ID in the leader's room

- **Examples:**
  - Discussion: `I am on the Blue team, and I'm not the President.`
  - Leader selection: `[Player 3]` or `[3]` to select Player 3 as a hostage

- **Notes:** The game automatically handles hostage exchanges and room transitions. Leaders cannot select themselves as hostages.

## Observation Space

**Reset Observations**
On reset, each player receives a prompt containing their role, team, and available actions:

```plaintext
Welcome to Two Rooms and a Boom! You are Player 2.
Your role: Blue
Team: Blue Team
Description: Member of the Blue Team. Your goal is to make sure the Bomber and President are in different rooms at the end of the game.

You are currently in Room 0.
You are the Leader of your room.

The game progresses through 3 rounds:
• In each round, players in the same room can talk to each other
• Room Leaders can choose one player to trade to the other room
• At the end of all rounds, the game checks which room contains the President and Bomber

The Red Team wins if the President and Bomber are in the same room at the end.
The Blue Team wins if the President and Bomber are in different rooms at the end.

As a Room Leader, you have special responsibilities:
• You'll choose one player from your room to trade with the other room
• You'll receive information from other players in your room
• Use this information to make strategic decisions for your team
```

**Step Observations**
During gameplay, players receive observations based on the current phase and actions:

```plaintext
# Discussion Phase
[GAME] Round 1: Discussion phase has started.
You are in Room 0 with: Player 0, Player 2, Player 4, Player 6.
You can talk freely with the other players in your room.
[Player 0] I'm on the Blue team, but I'm not the President.
[Player 4] I'm on the Red team, just a regular member.

# Leader Selection Phase
[GAME] Round 1: As the Leader of Room 0, you must select one player to trade with the other room.
Simply reply in the following format: '[Player X]' or '[X]'
Valid options: '[0]', '[4]', '[6]'
[LEADER] I have selected Player 4 to be traded with the other room.

# Trade Execution
[GAME] Round 1: The Leaders have exchanged hostages.
Player 4 moved from Room 0 to Room 1.
Player 5 moved from Room 1 to Room 0.
```

## Gameplay

- **Players:** 6-20 players
- **Initial Setup:** Players are assigned roles and divided into two rooms with a leader for each room
- **Game Progression:** Multiple rounds of discussion followed by hostage exchanges
- **Objective:**
  - **Red Team:** Have the Bomber and President in the same room at the end
  - **Blue Team:** Keep the Bomber and President in different rooms at the end

## Key Rules

1. **Roles:**
   - **Blue Team Member:** Regular Blue Team member
   - **Red Team Member:** Regular Red Team member
   - **President:** Special Blue Team role (target for the Red Team)
   - **Bomber:** Special Red Team role (must reach the President for Red Team to win)
   - **Leader:** A player in each room designated as the leader (can be any role)

2. **Communication:**
   - Players can only communicate with others in the same room
   - Players decide how much information about their identity to reveal

3. **Hostage Exchange:**
   - Each round, leaders select one player from their room to trade
   - Selected players swap rooms
   - Leaders cannot select themselves as hostages

4. **Victory Conditions:**
   - **Red Team Wins:** The Bomber and President are in the same room at the end
   - **Blue Team Wins:** The Bomber and President are in different rooms at the end

## Rewards

| Outcome          | Reward for Winners | Reward for Others |
|------------------|:------------------:|:-----------------:|
| **Red Team Win** | `+1`               | `-1`              |
| **Blue Team Win**| `+1`               | `-1`              |
| **Invalid Move** | `-1`               | `0`               |

## Parameters

- `num_rounds` (`int`, default: `3`):
  - **Description:** Number of rounds to play
  - **Impact:** More rounds give players more information but also more opportunities for strategic moves

- `cards_per_room` (`int`, default: `3`):
  - **Description:** Initial number of players per room
  - **Impact:** Affects the starting distribution of players

## Game Phases

1. **Discussion:** Players in each room discuss freely to gather information
2. **Leader Selection:** Room leaders select a hostage to trade
3. **Trade Execution:** Selected hostages swap rooms and the game either advances to the next round or ends

## Implementation Notes

- The game maintains two rooms with distinct sets of players
- Special roles (President and Bomber) are assigned to random players on the respective teams
- One leader is designated for each room
- The game automatically handles hostage exchanges and tracking which players are in which room
- Communication is strictly limited to players in the same room
- Winning is determined by the final positions of the President and Bomber

## Example Game Flow

1. Game starts with players randomly assigned to roles and rooms
2. Leaders are randomly assigned in each room
3. Players discuss within their rooms to gather information
4. Leaders select hostages to trade
5. Hostages swap rooms
6. Steps 3-5 repeat for the specified number of rounds
7. Game ends and winner is determined based on President and Bomber locations

## Variants

| Env-id                     | num_rounds | cards_per_room |
|----------------------------|:----------:|:--------------:|
| `TwoRoomsAndABoom-v0`      |    `3`     |      `3`       |

### Credit
Based on the party game "Two Rooms and a Boom" by Tuesday Knight Games.

### Contact
If you have questions or face issues with this specific environment, please reach out directly to the Textarena team.
