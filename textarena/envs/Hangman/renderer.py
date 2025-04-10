from typing import Dict, Any

# def create_board_str(game_state: Dict[str, Any]) -> str:
#     HANGMAN_STAGES = ["💀", "🪓", "🔪", "⚔️", "🪣", "🧢", "🙂"]
#     board = game_state.get("board", [])
#     tries_left = game_state.get("tries_left", 6)
#     guessed_letters = sorted(game_state.get("guessed_letters", []))  # Optional, default empty
#     board_display = " ".join(letter if letter != "_" else "_" for letter in board)
#     guessed_display = " ".join(guessed_letters) if guessed_letters else "None yet"
#     stage_emoji = HANGMAN_STAGES[max(0, min(tries_left, 6))]
#     return (f"🎯 Word: {board_display}\n🕵️ Guessed letters: {guessed_display}\n❤️ Tries left: {tries_left} {stage_emoji}\n")
def create_board_str(game_state: Dict[str, Any]) -> str:
    """
    Render the Hangman board showing the current guessed word and a hangman drawing based on tries left.
    """
    board = game_state.get("board", [])
    tries_left = game_state.get("tries_left", 6)

    # Join the current word with spaces
    word_display = " ".join(board)

    # Hangman ASCII art (indexed by remaining tries: 6 down to 0)
    hangman_stages = [
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |      /|\   ",
            " |      / \   ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |      /|\   ",
            " |      /     ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |      /|\   ",
            " |            ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |      /|    ",
            " |            ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |       |    ",
            " |            ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |      (X)   ",
            " |            ",
            " |            ",
            " |            ",
            "_|___         "
        ],
        [
            "  _______     ",
            " |/      |    ",
            " |            ",
            " |            ",
            " |            ",
            " |            ",
            "_|___         "
        ],
    ]

    hangman_drawing = hangman_stages[::-1][max(0, min(6, 6 - tries_left))]

    return (
        f"🎯 Word: {word_display}\n\n" +
        "\n".join(hangman_drawing) +
        f"\n\n❤️ Tries left: {tries_left}"
    )
