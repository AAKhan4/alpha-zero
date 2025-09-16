import numpy as np
from games.base_game import BaseGame  # Assuming BaseGame is in a file named base_game.py

class ConnectFour(BaseGame):
    def __init__(self):
        super().__init__()
        # Initialize board dimensions and action space size
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.row_count * self.col_count
        self.win_length = 4

    def __repr__(self):
        return "ConnectFour"
    
    def get_initial_state(self):
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count))
    
    def get_next_state(self, state, action, player):
        # Convert flat action to column
        col = action % self.col_count
        # Find the lowest empty row in the selected column
        empty_rows = np.where(state[:, col] == 0)[0]
        if empty_rows.size > 0:
            row = empty_rows[-1]
            state[row, col] = player
        return state

    def get_valid_actions(self, state):
        # Return a (42,) mask: 1 for the top empty cell in each column, 0 elsewhere
        mask = np.zeros(self.action_size, dtype=np.uint8)
        top_empty_cells = np.where(state[0, :] == 0)[0]
        for col in top_empty_cells:
            row = np.max(np.where(state[:, col] == 0))
            mask[row * self.col_count + col] = 1
        return mask
    
    def check_win(self, state, action):
        # Check if the last action resulted in a win
        if action is None:
            return False

        col = action % self.col_count
        row = np.max(np.where(state[:, col] != 0))
        player = state[row, col]

        def count_direction(delta_row, delta_col):
            r, c = row, col
            count = 0
            while True:
                r += delta_row
                c += delta_col
                if (
                    0 <= r < self.row_count and
                    0 <= c < self.col_count and
                    state[r, c] == player
                ):
                    count += 1
                else:
                    break
            return count
        
        return (
            count_direction(1, 0) + count_direction(-1, 0) + 1 >= self.win_length or  # Vertical
            count_direction(0, 1) + count_direction(0, -1) + 1 >= self.win_length or  # Horizontal
            count_direction(1, 1) + count_direction(-1, -1) + 1 >= self.win_length or  # Main diagonal
            count_direction(1, -1) + count_direction(-1, 1) + 1 >= self.win_length  # Anti-diagonal
        )
