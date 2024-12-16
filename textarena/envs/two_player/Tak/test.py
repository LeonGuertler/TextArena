import unittest
from parameterized import parameterized
from textarena.envs.two_player.Tak.env import TakEnv

## write test functions that are populated with the test cases
def generate_correct_move_sequence():
    """
    Generates an action sequence leading to a correct move in Tak.
    """
    return ["[place () {(0,2): [W1]}]", "[place () {(1,3): [F0]}]"]

def generate_invalid_move_sequence():
    """
    Generates an action sequence leading to an invalid move in Tak.
    """
    return ["[move () {(1,3): ['F0']}]"]

def generate_out_of_bounds_sequence():
    """
    Generates an action sequence with out-of-bounds coordinates in Tak.
    """
    return ["[place () {(0,200): [W1]}]", "[place () {(10,30): [F0]}]"]

def generate_player_winning_seqeunce():
    """
    Generates an action sequence leading to a win condition in Tak.
    """
    return ["[place () {(0,0): [F0]}]", "[place () {(3,0): [F1]}]", "[place () {(0,1): [F0]}]", "[place () {(3,1): [W1]}]", "[place () {(1,1): [F0]}]", "[place () {(3,2): [F1]}]", "[place () {(1,2): [F0]}]", "[place () {(3,3): [F1]}]", "[place () {(1,3): [F0]}]"]

class TestTakEnv(unittest.TestCase):
    """
    Test class for Tak
    """
    ## define environment variants as class attributes
    env_variants = [
        {"difficulty": "easy"},
        {"difficulty": "medium"},
        {"difficulty": "hard"},
    ]

    test_cases = {
        "correct_move": {
            "difficulty": "easy",
            "actions": generate_correct_move_sequence()
        },
        "invalid_move": {
            "difficulty": "easy",
            "actions": generate_invalid_move_sequence()
        },
        "out_of_bounds": {
            "difficulty": "easy",
            "actions": generate_out_of_bounds_sequence()
        },
        "player_winning": {
            "difficulty": "easy",
            "actions": generate_player_winning_seqeunce()
        }
    }

    @parameterized.expand([
        (name, details['difficulty'], details['actions'])
        for name, details in test_cases.items()
    ])
    def test_battleship_outcomes(self, name, difficulty, actions):
        """
        Test the outcomes of the Tak game.
        
        Args:
            name (str): Name of the test case.
            difficulty (str): Difficulty level of the game.
            actions (list): List of actions to take in the game.
        """
        env_config = next((env for env in self.env_variants if env['difficulty'] == difficulty), None)
        self.assertIsNotNone(env_config, f"Environment configuration not found for difficulty level: {difficulty}")

        env = TakEnv(difficulty=env_config['difficulty'])

        observations = env.reset(seed=490)

        terminated = False
        truncated = False
        rewards = {0: 0, 1: 0}

        for i, action in enumerate(actions):
            if terminated or truncated:
                break
            player_id = i % 2

            reward, truncated, terminated, info = env.step(action=action)

            ## update rewards
            if reward:
                rewards.update(reward)

        if "correct_move" in name:
            self.assertFalse(truncated, "Game should not truncate for correct moves.")
            self.assertFalse(terminated, "Game should not terminate until the puzzle is complete.")
        elif "invalid_move" in name:
            self.assertTrue(truncated or terminated, "Game should terminate due to an invalid move.")
            self.assertEqual(rewards[player_id], -1, "Player should receive -1 reward for an invalid move.")
        elif "out_of_bounds" in name:
            self.assertTrue(truncated or terminated, "Game should terminate due to an invalid move.")
            self.assertEqual(rewards[player_id], -1, "Player should receive -1 reward for an invalid move.")
        elif "player_winning" in name:
            self.assertTrue(terminated, "Game should terminate due to a player winning.")
            self.assertEqual(rewards[player_id], 1, f"Player {player_id} should have received +1 for winning.")
            self.assertEqual(rewards[1 - player_id], -1, f"Player {1 - player_id} should have received -1 for losing.")
        else:
            raise ValueError(f"Invalid test case name: {name}")
        
    def run_unit_test():
        print("Running Tak tests...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)