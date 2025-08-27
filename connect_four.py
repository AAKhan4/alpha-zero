import numpy as np

class ConnectFour:
    def __init__(self):
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
        # Apply action to the board and return the updated state
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_actions(self, state):
        # Return valid actions (empty cells) as a binary mask
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        # Check if the last action resulted in a win
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        player = state[row, action]

        def count_direction(delta_row, delta_col):
            for i in range(1, self.win_length):
                r = row + delta_row * i
                c = action + delta_col * i
                if r < 0 or r >= self.row_count or c < 0 or c >= self.col_count or state[r, c] != player:
                    return i - 1
                return self.win_length - 1
        
        return (
            count_direction(1, 0) + count_direction(-1, 0) + 1 >= self.win_length or  # Vertical
            count_direction(0, 1) + count_direction(0, -1) + 1 >= self.win_length or  # Horizontal
            count_direction(1, 1) + count_direction(-1, -1) + 1 >= self.win_length or  # Main diagonal
            count_direction(1, -1) + count_direction(-1, 1) + 1 >= self.win_length  # Anti-diagonal
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
