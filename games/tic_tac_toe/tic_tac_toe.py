import numpy as np

class TicTacToe:
    def __init__(self):
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
        row = action // self.col_count
        col = action % self.col_count
        next_state[row, col] = player
        return next_state

    def get_valid_actions(self, state):
        # Return valid actions (empty cells) as a binary mask
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        # Check if the last action resulted in a win
        if action == None:
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
    
    def is_terminal(self, state, action):
        # Determine if the game is over (win, draw, or ongoing)
        if self.check_win(state, action):
            return 1, True  # Win
        elif np.all(state != 0):
            return 0, True  # Draw
        return -1, False  # Game ongoing

    def get_opponent(self, player):
        # Get opponent's player value
        return -player
    
    def get_opponent_val(self, val):
        # Get opponent's perspective value
        return -val
    
    def change_perspective(self, state, player):
        # Adjust board perspective based on the current player
        return state * player
    
    def get_encoded_state(self, state):
        encoded = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded = np.swapaxes(encoded, 0, 1) # (batch, channels, rows, cols)
        return encoded
