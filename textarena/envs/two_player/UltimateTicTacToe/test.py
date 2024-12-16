import unittest
from parameterized import parameterized
from textarena.envs.two_player.UltimateTicTacToe.env import UltimateTicTacToeEnv

## write test functions that are populated with the test cases
def generate_correct_move_sequence():
    """
    Generates an action sequence leading to a correct move in Ultimate Tic Tac Toe.
    """
    return ["[0 1 1]", "[4 1 1]"]

def generate_invalid_move_sequence():
    """
    Generates an action sequence leading to an invalid move in Ultimate Tic Tac Toe.
    """
    return ["(0 1 1)", "[4, 1, 1]"]

def generate_out_of_bounds_sequence():
    """
    Generates an action sequence with out-of-bounds coordinates in Ultimate Tic Tac Toe.
    """
    return ["[9 9 9]"]

def generate_player_winning_seqeunce():
    """
    Generates an action sequence leading to a win condition in Ultimate Tic Tac Toe.
    """
    return ["[4 1 1]", "[4 1 0]", "[3 1 1]", "[4 0 0]", "[0 1 1]", "[4 2 0]", "[6 2 0]", "[6 2 1]", "[7 2 1]", "[7 0 0]", "[0 2 1]", "[7 1 0]", "[3 2 1]", "[7 2 0]", "[6 0 1]", "[1 0 0]", "[0 0 1]", "[1 1 0]", "[3 0 1]", "[1 2 0]"]

class TestUltimateTicTacToeEnv(unittest.TestCase):
    """
    Test class for Ultimate Tic Tac Toe
    """
    test_cases = {
        "correct_move": {
            "actions": generate_correct_move_sequence()
        },
        "invalid_move": {
            "actions": generate_invalid_move_sequence()
        },
        "out_of_bounds": {
            "actions": generate_out_of_bounds_sequence()
        },
        "player_winning": {
            "actions": generate_player_winning_seqeunce()
        }
    }

    @parameterized.expand([
        (name, details['actions'])
        for name, details in test_cases.items()
    ])
    def test_ultimate_tic_tac_toe_env(self, name, actions):
        """
        Test function for Ultimate Tic Tac Toe environment.
        """

        env = UltimateTicTacToeEnv()

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
        print("Running Ultimate Tic Tac Toe tests...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)