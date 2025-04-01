# Codenames Environment Documentation

## Overview
**Codenames** is a 4-player cooperative-competitive word game where two teams—Red and Blue—compete to uncover their team's secret words based on clues given by their Spymaster. Each team has a **Spymaster** and an **Operative**. The game board contains 25 words, each secretly assigned to one of four categories: Red team word, Blue team word, Neutral, or Assassin. The first team to guess all their own words wins—unless someone accidentally reveals the deadly **Assassin** word, which causes an immediate loss.

## Action Space

- **Format:** Actions are strings in square brackets.
  - **Spymasters (Player 0 & 2):** `[clue N]`  
    A one-word clue followed by a number indicating how many words relate to the clue.
  - **Operatives (Player 1 & 3):** `[word]`  
    A single word guess from the board.

- **Examples:**
  - Spymaster clue: `[animal 3]`
  - Operative guess: `[lion]`

- **Rules:**
  - Clue **must not** be a subset or match of any word on the board.
  - Operatives can make up to **N+1 guesses** after receiving a clue.

## Observation Space

### Reset Observations
On reset, players receive an instructional prompt tailored to their role:

```
Welcome to Codenames! You are Player 2, the Spymaster for Blue team.

Codenames is a word-based deduction game where two teams, Red (R) and Blue (B), compete to uncover all of their team's secret words before the other team does.

Your role: Spymaster  
Team: Blue

Description:
- Provide a one-word clue and a number (e.g., [wind 2]) to guide your teammate.
- You can see the true identity (Red, Blue, Neutral, Assassin) of all 25 words.

Codenames Words:
candle     B  
rocket     R  
...
```

### Step Observations
Each move is followed by a broadcast observation (to all players), including feedback and a view of the guessed words:

```
[GAME] Spymaster of Blue team, Player 2, submitted [cold 2].

[GAME] Operator of Blue team, Player 3, correctly guessed [ice].

[GAME] Operator of Blue team, Player 3, wrongly guessed [flame]. It is a Red Team word.
```

## Gameplay

- **Players:** Exactly 4 players
  - Player 0: Red Spymaster
  - Player 1: Red Operative
  - Player 2: Blue Spymaster
  - Player 3: Blue Operative

- **Initial Setup:**
  - 25 random English nouns selected (basic or full list depending on `hardcore`)
  - Each word secretly assigned to one of:  
    - 8 Red (R)  
    - 8 Blue (B)  
    - 8 Neutral (N)  
    - 1 Assassin (A)

- **Game Flow:**
  - Turns alternate between teams (Red → Blue → Red → ...).
  - Spymaster gives a clue → Operative makes up to N+1 guesses.
  - Correct guesses allow continued guessing; incorrect guesses end the turn.

- **End Conditions:**
  - One team correctly identifies all their team’s words → **Win**
  - An operative guesses the **Assassin** word → **Immediate Loss** for that team

## Key Rules

1. **Roles:**
   - **Spymaster (P0 & P2):** Sees the board assignments and gives clues.
   - **Operative (P1 & P3):** Guesses words based on the clues.

2. **Clue Restrictions:**
   - Clues must not be words on the board or substrings/supersets of board words.

3. **Guesses:**
   - Operatives can make **up to N+1 guesses**.
   - Words can only be guessed **once**.

4. **Teams:**
   - Red Team: P0 (Spymaster), P1 (Operative)
   - Blue Team: P2 (Spymaster), P3 (Operative)

5. **Turn Rotation:**
   - Alternates between Red and Blue team.
   - Turn ends early if Operative guesses incorrectly.

6. **Winning Conditions:**
   - Correctly guess all team words → **Win**
   - Guess the **Assassin** → **Lose**

## Rewards

| Outcome                  | Winners Reward | Others Reward |
|--------------------------|----------------|----------------|
| All words guessed        | `+1`           | `-1`           |
| Assassin guessed         | `+1`           | `-1`           |
| Invalid clue or move     | `-1`           | `0`            |

## Parameters

- `hardcore` (`bool`, default: `False`)
  - **Description:** Whether to use a larger, more complex word list
  - **Impact:** Increases difficulty by using a wider vocabulary

## Example Game Flow

1. Game begins with 25 words shown to all players (without labels).
2. Spymaster (Player 0) submits `[cold 2]`
3. Operative (Player 1) guesses `[ice]` and `[snow]`
4. Both are correct → team continues guessing
5. Next, Spymaster (Player 2) submits `[space 1]`
6. Operative (Player 3) guesses `[rocket]`, but it is a **Red** word → Red gains advantage
7. Eventually, Blue team guesses the **Assassin** → Red team wins

## Variants

| Env-id                   | hardcore |
|--------------------------|:--------:|
| `Codenames-v0`           | `False`  |
| `Codenames-v0-hardcore`  | `True`   |

## Implementation Notes

- Spymasters have access to full board info
- The environment auto-validates clue legality
- Word list is POS-filtered to include short nouns only (`< 8 characters`)
- Supports full replayability via randomized word board and assignments


## Contact
If you have questions or suggestions, reach out to **ananyabalehithlu@gmail.com**

