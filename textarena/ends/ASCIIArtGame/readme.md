# ASCII Art Guessing Game

## Overview
A collaborative two-player game where players take turns creating and guessing ASCII art representations. Success is shared, with cumulative rewards earned over multiple rounds.

## Game Rules

### Player Setup
- Two players (A and B) participate.
- Players alternate roles between artist and guesser.
- Rewards are shared based on mutual success.
- The game consists of three rounds.

### Round Structure
- Each round, both players receive a term to depict using ASCII symbols.
- Players switch roles between artist and guesser in alternating turns.
- A valid symbol set is provided for each turn.
- Points earned contribute to a shared score.

### Drawing Mechanics
- Players must adhere to the given ASCII symbol constraints.
- Any symbol outside the provided set results in an invalid move.
- No words or emojis are allowed.
- Drawings must be purely visual representations of the given term.

### Guessing System
- Players have multiple attempts to guess the correct term.
- Points are awarded based on:
  - Number of attempts required
  - Complexity of the term
  - Efficiency in using the allowed symbols

### Scoring & Rewards
- Both players share a cumulative score.
- Points accumulate over all three rounds.
- Bonus points are awarded for effective communication.
- Penalties apply for using invalid symbols.

## Technical Details

### Move Validation
- The system verifies adherence to the allowed symbol set.
- Invalid symbol usage results in move rejection.
- A turn management system ensures correct role alternation.

### Game State Tracking
- Tracks:
  - Current round number
  - Player roles (artist or guesser)
  - Allowed symbols per turn
  - Shared score
  - Individual round performance

### Implementation Requirements
- Symbol set generation system
- Turn-based drawing interface
- Move validation logic
- Shared reward distribution system
- Round and term database with complexity ratings
- Performance tracking metrics

## Success Criteria
- Players correctly alternate roles.
- Symbol constraints are strictly enforced.
- Shared rewards incentivize cooperation.
- Invalid moves are properly detected.
- The game state is accurately tracked across all three rounds.

## Getting Started

### Prerequisites
- A web browser (for the front-end interface)
- A server environment supporting the game logic (if applicable)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ascii-art-guessing-game.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ascii-art-guessing-game
   ```
3. Install dependencies (if applicable):
   ```sh
   npm install  # or pip install -r requirements.txt
   ```

### Running the Game
- Start the server (if required):
  ```sh
  npm start  # or python server.py
  ```
- Open the game in a browser or run via the command line interface.

## Contributing
Contributions are welcome! Please submit a pull request with any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

