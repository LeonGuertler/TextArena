from textarena.envs.registration import register, register_with_versions
from textarena.wrappers import LLMObservationWrapper, ActionFormattingWrapper, GameMessagesAndCurrentBoardObservationWrapper, GameMessagesObservationWrapper, GameBoardObservationWrapper, ClipCharactersActionWrapper


DEFAULT_WRAPPERS = [LLMObservationWrapper, ActionFormattingWrapper]

# Reasoning Gym
REASONING_GYM_GAMES = [
    "ab", "advanced_geometry", "aiw", "arc_1d", "arc_agi", "base_conversion", "basic_arithmetic", "bf", "binary_alternation", "binary_matrix", "bitwise_arithmetic", "boxnet", "caesar_cipher", "calendar_arithmetic", "chain_sum", "circuit_logic", "codeio", "color_cube_rotation",
    "complex_arithmetic", "count_bits", "count_primes", "countdown", "course_schedule", "cryptarithm", "decimal_arithmetic", "decimal_chain_sum", "dice", "emoji_mystery", "family_relationships", "figlet_font", "fraction_simplification", "futoshiki", "game_of_life", "game_of_life_halting",
    "gcd", "graph_color", "intermediate_integration", "isomorphic_strings", "jugs", "kakurasu", "knight_swap", "knights_knaves", "largest_island", "lcm", "leg_counting", "letter_counting", "letter_jumble", "mahjong_puzzle", "manipulate_matrix", "maze", "mini_sudoku", "modulo_grid",
    "n_queens", "needle_haystack", "number_filtering", "number_format", "number_sequence", "number_sorting", "palindrome_generation", "palindrome_partitioning", "polynomial_equations", "polynomial_multiplication", "pool_matrix", "power_function", "prime_factorization", "products",
    "propositional_logic", "puzzle24", "quantum_lock", "ransom_note", "rearc", "rectangle_count", "rotate_matrix", "rotten_oranges", "rubiks_cube", "rush_hour", "self_reference", "sentence_reordering", "shortest_path", "simple_equations", "simple_geometry", "simple_integration",
    "sokoban", "spell_backward", "spiral_matrix", "string_insertion", "string_manipulation", "string_splitting", "string_synthesis", "sudoku", "survo", "syllogism", "time_intervals", "tower_of_hanoi", "tsumego", "word_ladder", "word_sequence_reversal", "word_sorting", "zebra_puzzles",
]
for g in REASONING_GYM_GAMES: 
    name = "".join([w.capitalize() for w in g.split("_")]) # add both names for ease of use
    register_with_versions(id=f"ReasoningGym-{g}", entry_point="textarena.external_envs.ReasoningGym.env:ReasoningGymEnv", wrappers={"default": DEFAULT_WRAPPERS, "-train": DEFAULT_WRAPPERS}, reasoning_gym_env_id=g)
    register_with_versions(id=f"ReasoningGym-{name}", entry_point="textarena.external_envs.ReasoningGym.env:ReasoningGymEnv", wrappers={"default": DEFAULT_WRAPPERS, "-train": DEFAULT_WRAPPERS}, reasoning_gym_env_id=g)



# ArcAGI-3
