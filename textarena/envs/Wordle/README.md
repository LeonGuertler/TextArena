# Wordle Environment Documentation

## Overview
**Wordle** is a classic single-player word-guessing game adapted for the TextArena environment. The player attempts to guess a secret English word of a fixed length (e.g., 5 or 7 letters) within a limited number of attempts. After each guess, the player receives feedback indicating whether each letter is correct and in the correct position (**green**), correct but in the wrong position (**yellow**), or not in the word at all (**gray**).

## Action Space

- **Format:** `[guess]` — A guessed word wrapped in square brackets.

- **Examples:**
  - `[apple]`
  - `[shines]`

- **Rules:**
  - Words must be exactly the specified length (e.g., 5 or 7 letters).
  - Words must be valid English nouns.
  - Players receive feedback after each guess.

## Feedback System

For each letter in the guess:
- **G (Green):** Correct letter in the correct position
- **Y (Yellow):** Correct letter in the wrong position
- **X (Gray):** Letter is not in the word

### Example
```
APPLE
G X X Y G
```
This means:
- A and E are in the correct positions
- L is in the word but misplaced
- P and P are incorrect

## Observation Space

### Reset Observations
When the game starts, the player sees:
```
You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be given as follows:
  - G (green): correct letter in the correct position
  - Y (yellow): letter exists in the word but in the wrong position
  - X (wrong): letter is not in the word
Enter your guess to begin.
```

### Step Observations
After each guess, players see:
```
Player 0 submitted [apple].
Feedback:
A P P L E
G X X Y G
You have 3 guesses left.
```

## Gameplay

- **Players:** Single-player only
- **Objective:** Guess the secret word before running out of guesses
- **Guess Limit:** Defaults to 6 (or 9 in long mode)

### Game Flow
1. Player is prompted with instructions
2. Player enters a guess: `[apple]`
3. Game provides letter-by-letter feedback
4. If the word is correct → player wins
5. If guess limit is reached → game ends in a draw

## Key Rules

1. **Input Format:** Guesses must be wrapped in square brackets.
2. **Length Check:** Guesses must match the word length exactly.
3. **Dictionary Check:** Words must be valid English nouns (excluding proper nouns).
4. **Win Condition:** All letters are green (G) in a single guess.
5. **Loss Condition:** Number of guesses used equals maximum allowed.

## Rewards

| Outcome             | Reward |
|---------------------|--------|
| Word guessed        | `+1`   |
| Turn limit reached  | `0`    |
| Invalid move        | `-1`   |

## Parameters

- `word_length` (`int`, default: `5`)
  - Description: Number of letters in the target word

- `num_guesses` (`int`, default: `6`)
  - Description: Maximum number of allowed guesses

- `hardcore` (`bool`, default: `False`)
  - Description: If True, uses a larger, more difficult vocabulary

## Variants

| Env-id                    | hardcore | word_length | num_guesses |
|---------------------------|:--------:|:-----------:|:-----------:|
| `Wordle-v0`               | False    | 5           | 6           |
| `Wordle-v0-hardcore`      | True     | 5           | 6           |
| `Wordle-v0-long`          | False    | 7           | 9           |
| `Wordle-v0-long-hardcore` | True     | 7           | 9           |

## Implementation Notes

- POS-tag filtering ensures selected words are common English nouns.
- The board is rendered as a sequence of guesses and feedback tokens.
- Guesses are evaluated in two passes to handle multiple instances of the same letter.
- Feedback is consistent with the classic Wordle logic.

## Contact
For questions or improvements, reach out to **ananyabalehithlu@gmail.com**