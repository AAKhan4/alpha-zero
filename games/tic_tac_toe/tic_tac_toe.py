import numpy as np
from games.base_game import BaseGame, GameState

class TicTacToe(BaseGame):
    def __init__(self):
        super().__init__()
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.row_count * self.col_count
    
    def __repr__(self):
        return "TicTacToe"
    
    def get_valid_actions(self, game_info: dict) -> np.ndarray:
        board = game_info["board"]
        return (board.flatten() == 0).astype(int)

    def is_valid_action(self, game_info: dict, action: int) -> bool:
        valid_actions = self.get_valid_actions(game_info)
        return valid_actions[action] == 1
    
    def get_next_state(self, game_info: dict, action: int) -> dict:
        game_info = game_info.copy()
        new_state = game_info["board"].copy()
        row, col = divmod(action, self.col_count)
        new_state[row, col] = 1
        game_info["board"] = new_state
        return game_info
    
    def check_win(self, game_info: dict) -> int | None:
        board = game_info["board"]
        if np.any(board == 0):
            return None  # Game is still ongoing

        # Check rows and columns
        for i in range(3):
            if abs(np.sum(board[i, :])) == 3:
                return board[i, 0]
            if abs(np.sum(board[:, i])) == 3:
                return board[0, i]
        # Check diagonals
        diag1 = np.sum(board[i, i] for i in range(3))
        diag2 = np.sum(board[i, 2 - i] for i in range(3))
        if abs(diag1) == 3:
            return board[0, 0]
        if abs(diag2) == 3:
            return board[0, 2]
        return 0  # Draw
    
    def change_perspective(self, game_info):
        game_info = game_info.copy()
        game_info["board"] = -1 * game_info["board"]
        game_info["player"] *= -1
        return game_info
    
    def get_state_type(self):
        return TicTacToeState

class TicTacToeState(GameState):
    def __init__(self, game: BaseGame, player: int = 1):
        super().__init__(game, player)