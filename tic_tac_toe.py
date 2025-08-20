import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.row_count * self.col_count
    
    def get_initial_state(self):
        return np.zeros((self.row_count, self.col_count))
    
    def get_next_state(self, state, action, player):
        next_state = state.copy()
        row = action // self.col_count
        col = action % self.col_count
        next_state[row, col] = player
        return next_state

    def get_valid_actions(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        row = action // self.col_count
        col = action % self.col_count
        player = state[row, col]

        return (
            (np.all(state[row, :] == player) or
             np.all(state[:, col] == player) or
             (row == col and np.all(np.diag(state) == player)) or
             (row + col == self.row_count - 1 and np.all(np.diag(np.fliplr(state)) == player)))
        )
    
    def is_terminal(self, state):
        return 1, self.check_win(state, np.argmax(state)) or 0, np.all(state != 0)

    def get_opponent(self, player):
        return -player
