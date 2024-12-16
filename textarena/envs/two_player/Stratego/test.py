import unittest
from parameterized import parameterized
from textarena.envs.two_player.Stratego.env import StrategoEnv

## write test functions that are populated with the test cases
def generate_correct_move_sequence():
    """
    Generates an action sequence leading to a correct move in Stratego.
    """
    return ["[D0 E0]", "[G4 F4]"]

def generate_invalid_move_sequence():
    """
    Generates an action sequence leading to an invalid move in Stratego.
    """
    return ["[D0 D1]", "[D8 E8]"]

def generate_out_of_bounds_sequence():
    """
    Generates an action sequence with out-of-bounds coordinates in Stratego.
    """
    return ["[Z100 Y100]"]

def generate_player_winning_seqeunce():
    """
    Generates an action sequence leading to a win condition in Stratego.
    """
    return ["[D0 E0]", "[G9 F9]", "[D1 E1]", "[F9 F8]", "[E0 F0]", "[H9 G9]", "[E1 F1]", "[G9 F9]", "[F0 G0]", "[F9 E9]", "[F1 G1]", "[E9 E8]", "[G0 H0]", "[E8 D8]", "[D4 E4]", "[D8 C8]", "[D5 E5]", "[C8 B8]"]

class TestStrategoEnv(unittest.TestCase):
    """
    Test class for Stratego
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
    def test_stratego_env(self, name, actions):
        """
        Test function for Stratego environment.
        """

        env = StrategoEnv()

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
        print("Running Stratego tests...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)