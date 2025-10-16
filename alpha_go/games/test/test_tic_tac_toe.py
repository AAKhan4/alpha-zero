import unittest
import numpy as np
from games.tic_tac_toe.tic_tac_toe import TicTacToe

class TestTicTacToe(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertTrue(np.array_equal(state, np.zeros((3, 3))), "Initial state should be an empty 3x3 board.")

    def test_get_next_state(self):
        state = self.game.get_initial_state()
        next_state = self.game.get_next_state(state, 0, 1)
        expected_state = np.array([[1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        self.assertTrue(np.array_equal(next_state, expected_state), "Next state should reflect the player's move.")

    def test_get_valid_actions(self):
        state = self.game.get_initial_state()
        valid_actions = self.game.get_valid_actions(state)
        self.assertTrue(np.array_equal(valid_actions, np.ones(9, dtype=np.uint8)), "All actions should be valid on an empty board.")

        state[0, 0] = 1
        valid_actions = self.game.get_valid_actions(state)
        expected_valid_actions = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
        self.assertTrue(np.array_equal(valid_actions, expected_valid_actions), "Valid actions should exclude occupied cells.")

    def test_is_valid_action(self):
        state = self.game.get_initial_state()
        self.assertTrue(self.game.is_valid_action(state, 0), "Action 0 should be valid on an empty board.")
        state[0, 0] = 1
        self.assertFalse(self.game.is_valid_action(state, 0), "Action 0 should be invalid if the cell is occupied.")

    def test_check_win_row(self):
        state = self.game.get_initial_state()
        state[0, :] = 1
        self.assertTrue(self.game.check_win(state, 2), "Player 1 should win with a complete row.")

    def test_check_win_column(self):
        state = self.game.get_initial_state()
        state[:, 1] = 2
        self.assertTrue(self.game.check_win(state, 4), "Player 2 should win with a complete column.")

    def test_check_win_main_diagonal(self):
        state = self.game.get_initial_state()
        np.fill_diagonal(state, 1)
        self.assertTrue(self.game.check_win(state, 8), "Player 1 should win with a complete main diagonal.")

    def test_check_win_anti_diagonal(self):
        state = self.game.get_initial_state()
        state[0, 2] = state[1, 1] = state[2, 0] = 2
        self.assertTrue(self.game.check_win(state, 6), "Player 2 should win with a complete anti-diagonal.")

    def test_no_win(self):
        state = self.game.get_initial_state()
        state[0, 0] = 1
        state[0, 1] = 2
        state[0, 2] = 1
        self.assertFalse(self.game.check_win(state, 2), "There should be no winner in this state.")

    def test_full_board_no_win(self):
        state = np.array([[1, 2, 1],
                          [2, 1, 2],
                          [2, 1, 2]])
        self.assertFalse(self.game.check_win(state, None), "There should be no winner on a full board with no winning line.")

    def test_full_board_with_win(self):
        state = np.array([[1, 1, 1],
                          [2, 2, 0],
                          [0, 0, 0]])
        self.assertTrue(self.game.check_win(state, 2), "Player 1 should win with a complete row on a full board.")

if __name__ == "__main__":
    unittest.main()