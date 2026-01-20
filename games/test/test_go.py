import unittest
import numpy as np
from games.go.go import Go, GoState

class TestGo(unittest.TestCase):
    def setUp(self):
        self.game = Go(board_size=9, komi=6.5)
        self.state = GoState(self.game)

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

    def test_detect_suicide_moves(self):
        state = self.game.get_initial_state()
        state[3, 4] = -1
        state[5, 4] = -1
        state[4, 3] = -1
        state[4, 5] = -1
        suicide_moves = self.game.detect_suicide_moves(state)
        self.assertEqual(suicide_moves[4, 4], 1)  # Move at (4, 4) is a suicide move

    def test_detect_ko(self):
        state = self.game.get_initial_state()
        prev_state = state.copy()
        ko_moves = self.game.detect_ko(state, prev_state)
        self.assertTrue(np.all(ko_moves == 0))  # No Ko moves in an empty board

    def test_get_valid_actions(self):
        game_info = self.state.get_info()
        valid_actions = self.game.get_valid_actions(game_info)
        self.assertEqual(valid_actions.sum(), 82)  # 81 board positions + 1 pass move

        game_info["board"][4, 4] = 1
        valid_actions = self.game.get_valid_actions(game_info)
        self.assertEqual(valid_actions[40], 0)  # Position (4, 4) is no longer valid

    def test_is_valid_action(self):
        game_info = self.state.get_info()
        self.assertTrue(self.game.is_valid_action(game_info, 0))  # Top-left corner
        self.assertTrue(self.game.is_valid_action(game_info, 81))  # Pass move

        game_info["board"][4, 4] = 1
        self.assertFalse(self.game.is_valid_action(game_info, 40))  # Position (4, 4) is occupied

    def test_get_next_state(self):
        game_info = self.state.get_info()
        next_state = self.game.get_next_state(game_info, 40)
        self.assertEqual(next_state["board"][4, 4], 1)  # Stone placed at (4, 4)

    def test_is_terminal(self):
        game_info = self.state.get_info()
        game_info["last_moves"]["1"] = 81  # Pass
        game_info["last_moves"]["-1"] = 81  # Pass
        score, is_terminal = self.game.is_terminal(game_info)
        self.assertTrue(is_terminal)  # Game should end after two passes

    def test_check_win(self):
        game_info = self.state.get_info()
        game_info["board"][0, 0] = 1
        game_info["board"][1, 1] = -1
        score = self.game.check_win(game_info)
        self.assertTrue(score > 0)  # Black wins due to more area

    def test_change_perspective(self):
        game_info = self.state.get_info()
        game_info["board"][0, 0] = 1
        game_info["board"][1, 1] = -1
        new_perspective = self.game.change_perspective(game_info)
        self.assertTrue(np.array_equal(new_perspective["board"], -game_info["board"]))

if __name__ == "__main__":
    unittest.main()
