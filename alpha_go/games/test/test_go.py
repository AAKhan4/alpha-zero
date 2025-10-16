import unittest
import numpy as np
from games.go.go import Go

class TestGo(unittest.TestCase):
    def setUp(self):
        self.game = Go(board_size=9, komi=6.5)

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertEqual(state.shape, (9, 9))
        self.assertTrue(np.all(state == 0))

    def test_get_neighbors(self):
        state = self.game.get_initial_state()
        neighbors_idx, neighbors_val = self.game.get_neighbors(state, 0, 0)
        self.assertEqual(len(neighbors_idx), 2)  # Top-left corner has 2 neighbors
        self.assertTrue(np.all(neighbors_val == 0))

        neighbors_idx, neighbors_val = self.game.get_neighbors(state, 4, 4)
        self.assertEqual(len(neighbors_idx), 4)  # Center has 4 neighbors

    def test_count_liberties(self):
        state = self.game.get_initial_state()
        state[4, 4] = 1
        group, liberties = self.game.count_liberties(state, 4, 4)
        self.assertEqual(len(group), 1)
        self.assertEqual(len(liberties), 4)

        state[4, 5] = -1
        group, liberties = self.game.count_liberties(state, 4, 4)
        self.assertEqual(len(liberties), 3)  # One liberty blocked

    def test_remove_adj_dead_stones(self):
        state = self.game.get_initial_state()
        state[4, 4] = 1
        state[3, 4] = -1
        state[5, 4] = -1
        state[4, 3] = -1
        state[4, 5] = -1
        new_state = self.game.remove_adj_dead_stones(state, 40)
        self.assertEqual(new_state[4, 4], 0)  # Stone at (4, 4) should be removed

    def test_detect_suicide_moves(self):
        state = self.game.get_initial_state()
        state[3, 4] = -1
        state[5, 4] = -1
        state[4, 3] = -1
        state[4, 5] = -1
        suicide_moves = self.game.detect_suicide_moves(state, 1)
        self.assertEqual(suicide_moves[4, 4], 1)  # Move at (4, 4) is a suicide move

    def test_detect_ko(self):
        state = self.game.get_initial_state()
        self.game.last_2_boards = [state.copy(), state.copy()]
        ko_moves = self.game.detect_ko(state, 1)
        self.assertTrue(np.all(ko_moves == 0))  # No Ko moves in an empty board

    def test_get_valid_actions(self):
        state = self.game.get_initial_state()
        valid_actions = self.game.get_valid_actions(state)
        self.assertEqual(valid_actions.sum(), 82)  # 81 board positions + 1 pass move

        state[4, 4] = 1
        valid_actions = self.game.get_valid_actions(state)
        self.assertEqual(valid_actions[40], 0)  # Position (4, 4) is no longer valid

    def test_is_valid_action(self):
        state = self.game.get_initial_state()
        self.assertTrue(self.game.is_valid_action(state, 0))  # Top-left corner
        self.assertTrue(self.game.is_valid_action(state, 81))  # Pass move

        state[4, 4] = 1
        self.assertFalse(self.game.is_valid_action(state, 40))  # Position (4, 4) is occupied

    def test_get_next_state(self):
        state = self.game.get_initial_state()
        next_state = self.game.get_next_state(state, 40, 1)
        self.assertEqual(next_state[4, 4], 1)  # Stone placed at (4, 4)

    def test_calc_territory(self):
        state = self.game.get_initial_state()
        state[0, 0] = 1
        state[1, 1] = -1
        territory = self.game.calc_territory(state)
        self.assertEqual(territory[0, 1], 0)  # Neutral territory

    def test_remove_dead_stones_end(self):
        state = self.game.get_initial_state()
        state[4, 4] = 1
        state[3, 4] = -1
        state[5, 4] = -1
        state[4, 3] = -1
        state[4, 5] = -1
        new_state = self.game.remove_dead_stones_end(state)
        self.assertEqual(new_state[4, 4], 0)  # Dead stone removed

    def test_calc_score(self):
        state = self.game.get_initial_state()
        state[0, 0] = 1
        state[1, 1] = -1
        score = self.game.calc_score(state)
        self.assertTrue(score < 0)  # White wins due to komi

    def test_pass_move(self):
        state = self.game.get_initial_state()
        next_state = self.game.get_next_state(state, 81, 1)  # Pass move
        self.assertTrue(np.array_equal(state, next_state))  # State should remain unchanged

    def test_ko_rule(self):
        state = self.game.get_initial_state()
        state[4, 4] = 1
        state[4, 5] = -1
        state[3, 4] = -1
        state[5, 4] = -1
        state[4, 3] = -1
        self.game.last_2_boards = [state.copy(), state.copy()]
        ko_moves = self.game.detect_ko(state, 1)
        self.assertTrue(np.all(ko_moves == 0))  # No Ko moves detected

if __name__ == "__main__":
    unittest.main()