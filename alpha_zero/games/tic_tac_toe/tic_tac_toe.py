import numpy as np
from games.base_game import BaseGame  # Assuming BaseGame is in a file named base_game.py

class TicTacToe(BaseGame):
    def __init__(self):
        super().__init__()
        # Initialize board dimensions and action space size
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.row_count * self.col_count
    
    def __repr__(self):
        return "TicTacToe"
    
    def get_initial_state(self):
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count))
    
    def get_next_state(self, state, action, player):
        # Apply action to the board and return the updated state
        next_state = state.copy()
        row, col = divmod(action, self.col_count)
        next_state[row, col] = player
        return next_state

    def get_valid_actions(self, state):
        # Return valid actions (empty cells) as a binary mask
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def is_valid_action(self, state, action):
        return self.get_valid_actions(state)[action] == 1
    
    def check_win(self, state, action):
        # Check if the last action resulted in a win
        if action is None:
            return False

        row = action // self.col_count
        col = action % self.col_count
        player = state[row, col]

        return (
            (np.all(state[row, :] == player) or  # Check row
             np.all(state[:, col] == player) or  # Check column
             (row == col and np.all(np.diag(state) == player)) or  # Check main diagonal
             (row + col == self.row_count - 1 and np.all(np.diag(np.fliplr(state)) == player)))  # Check anti-diagonal
        )
