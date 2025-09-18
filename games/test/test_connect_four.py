import unittest
import numpy as np
from games.connect_four.connect_four import ConnectFour

class TestConnectFour(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFour()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertEqual(state.shape, (6, 7))
        self.assertTrue(np.all(state == 0))

    def test_get_next_state(self):
        state = self.game.get_initial_state()
        action = 3  # Drop in column 3
        player = 1
        next_state = self.game.get_next_state(state, action, player)
        self.assertEqual(next_state[5, 3], player)  # Bottom row, column 3
        self.assertTrue(np.all(next_state[:5, 3] == 0))  # Rows above are empty

    def test_get_valid_actions(self):
        state = self.game.get_initial_state()
        valid_actions = self.game.get_valid_actions(state)
        self.assertEqual(valid_actions.shape, (42,))  # Action space size is 42
        self.assertTrue(all(action in [0, 1] for action in valid_actions))  # Valid actions are binary (0 or 1)
        self.assertEqual(sum(valid_actions), 7)  # All columns are valid initially

    def test_is_valid_action(self):
        state = self.game.get_initial_state()
        self.assertTrue(self.game.is_valid_action(state, 3))  # Column 3 is valid initially
        for row in range(6):
            state[row, 3] = 1  # Fill column 3
        self.assertFalse(self.game.is_valid_action(state, 3))  # Column 3 is now full

    def test_check_win_vertical(self):
        state = self.game.get_initial_state()
        for i in range(4):
            state[5 - i, 3] = 1  # Fill column 3
        action = 3
        self.assertTrue(self.game.check_win(state, action))

    def test_check_win_horizontal(self):
        state = self.game.get_initial_state()
        for i in range(4):
            state[5, i] = 1  # Fill row 5
        action = 3
        self.assertTrue(self.game.check_win(state, action))

    def test_check_win_diagonal(self):
        state = self.game.get_initial_state()
        for i in range(4):
            state[5 - i, i] = 1  # Fill diagonal
        action = 3
        self.assertTrue(self.game.check_win(state, action))

    def test_check_win_complex_scenario(self):
        state = self.game.get_initial_state()
        state[2, 0] = 1  # X
        state[2, 6] = 1  # X
        state[2, 1] = -1  # O
        state[2, 2] = -1  # O

        state[3, 0] = 1  # X
        state[3, 1] = 1  # X
        state[3, 2] = -1  # O
        state[3, 4] = 1  # X
        state[3, 6] = -1  # O

        state[4, 0] = -1  # O
        state[4, 1] = 1  # X
        state[4, 2] = 1  # X
        state[4, 4] = -1  # O
        state[4, 6] = -1  # O

        state[5, 0] = 1  # X
        state[5, 1] = -1  # O
        state[5, 2] = 1  # X
        state[5, 3] = 1  # X
        state[5, 4] = -1  # O
        state[5, 6] = -1  # O

        action = 0  # Last move in column 0
        self.assertTrue(self.game.check_win(state, action))

    def test_check_win_anti_diagonal(self):
        state = self.game.get_initial_state()
        for i in range(4):
            state[5 - i, 6 - i] = 1  # Fill anti-diagonal
        action = 6
        self.assertTrue(self.game.check_win(state, action))

    def test_no_win(self):
        state = self.game.get_initial_state()
        state[5, 3] = 1
        state[4, 3] = 2
        action = 3
        self.assertFalse(self.game.check_win(state, action))

    def test_full_column(self):
        state = self.game.get_initial_state()
        for row in range(6):
            state[row, 3] = 1  # Fill column 3
        valid_actions = self.game.get_valid_actions(state)
        self.assertEqual(valid_actions[3], 0)  # Column 3 should be invalid

    def test_full_board(self):
        state = self.game.get_initial_state()
        for col in range(7):
            for row in range(6):
                state[row, col] = 1 if (row + col) % 2 == 0 else 2
        valid_actions = self.game.get_valid_actions(state)
        self.assertTrue(np.all(valid_actions == 0))  # No valid actions on a full board

if __name__ == "__main__":
    unittest.main()