import numpy as np

class BaseGame:
    def __init__(self):
        # Initialize board dimensions and action space size
        self.row_count = None
        self.col_count = None
        self.action_size = None

    def __repr__(self):
        raise NotImplementedError

    def get_initial_state(self):
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        raise NotImplementedError

    def get_valid_actions(self, state: np.ndarray) -> list[int]:
        raise NotImplementedError

    def check_win(self, state: np.ndarray, action: int) -> bool:
        raise NotImplementedError

    def is_terminal(self, state: np.ndarray, action: int) -> tuple[int, bool]:
        # Determine if the game is over (win, draw, or ongoing)
        if self.check_win(state, action):
            return 1, True  # Win
        elif np.all(state != 0):
            return 0, True  # Draw
        return -1, False  # Game ongoing

    def get_opponent(self, player: int) -> int:
        # Get opponent's player value
        return -player

    def get_opponent_val(self, val: int) -> int:
        # Get opponent's perspective value
        return -val

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        # Adjust board perspective based on the current player
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        encoded = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded = np.swapaxes(encoded, 0, 1)  # (batch, channels, rows, cols)
        return encoded