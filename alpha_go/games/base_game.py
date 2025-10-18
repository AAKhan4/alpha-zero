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

    def get_next_state(self, game_info: dict, action: int) -> dict:
        raise NotImplementedError

    def get_valid_actions(self, game_info: dict) -> list[int]:
        raise NotImplementedError

    def is_valid_action(self, game_info: dict, action: int) -> bool:
        raise NotImplementedError

    def check_win(self, game_info: dict) -> bool | None:
        raise NotImplementedError

    def is_terminal(self, game_info: dict) -> tuple[int, bool]:
        # Determine if the game is over (win, draw, or ongoing)
        score = self.check_win(game_info)
        if not score:
            return 0, False  # Game ongoing
        return score, True  # Game over

    def change_perspective(self, game_info: dict) -> dict:
        raise NotImplementedError

    def get_opponent(self, player: int) -> int:
        # Get opponent's player value
        return -player

    def get_opponent_val(self, val: int) -> int:
        # Get opponent's perspective value
        return -val

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        encoded = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded = np.swapaxes(encoded, 0, 1)  # (batch, channels, rows, cols)
        return encoded
    
class GameState:
    def __init__(self, game: BaseGame, player: int = 1):
        self.game = game
        self.board = game.get_initial_state()
        self.player = player

    def get_info(self) -> dict:
        return {
            "state": self.game.change_perspective(self.board, self.player),
            "player": self.player
        }
    
    def update(self, game_info: dict) -> None:
        self.board = game_info["board"]
        self.player = game_info["player"]