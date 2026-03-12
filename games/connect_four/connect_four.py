import numpy as np
from games.base_game import GameState, BaseGame

class ConnectFour(BaseGame):
    def __init__(self):
        super().__init__()
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.col_count  # Actions correspond to columns

    def __repr__(self):
        return "ConnectFour"
    
    def get_valid_actions(self, game_info) -> np.ndarray:
        state: np.ndarray = game_info["board"]
        valid_actions = np.where(state[0, :] == 0, 1, 0)  # Valid if the top row of the column is empty
        return valid_actions
    
    def is_valid_action(self, game_info: dict, action: int) -> bool:
        valid_actions = self.get_valid_actions(game_info)
        return valid_actions[action] == 1
    
    def get_next_open_row(self, state: np.ndarray, col: int) -> int | None:
        for r in range(self.row_count - 1, -1, -1):
            if state[r, col] == 0:
                return r
        return None

    def get_next_state(self, game_info, action) -> dict:
        game_info = game_info.copy()
        state: np.ndarray = game_info["board"]
        row = self.get_next_open_row(state, action)
        if row is None:
            return game_info # Invalid action, return unchanged state
        
        new_state = state.copy()
        new_state[row, action] = 1  # Place the piece for the current player
        game_info["board"] = new_state
        return {
            "board": new_state * -1,  # Flip the board for the opponent's perspective
            "player": -game_info["player"]  # Switch player perspective
        }
    
    def check_win(self, game_info) -> int | None:
        state: np.ndarray = game_info["board"]
        # Check horizontal locations for win
        for c in range(self.col_count - 3):
            for r in range(self.row_count):
                if abs(np.sum(state[r, c:c+4])) == 4:
                    return state[r, c]
        
        # Check vertical locations for win
        for c in range(self.col_count):
            for r in range(self.row_count - 3):
                if abs(np.sum(state[r:r+4, c])) == 4:
                    return state[r, c]
        
        # Check positively sloped diagonals
        for c in range(self.col_count - 3):
            for r in range(self.row_count - 3):
                if abs(np.sum([state[r+i, c+i] for i in range(4)])) == 4:
                    return state[r, c]
        
        # Check negatively sloped diagonals
        for c in range(self.col_count - 3):
            for r in range(3, self.row_count):
                if abs(np.sum([state[r-i, c+i] for i in range(4)])) == 4:
                    return state[r, c]
        
        if np.all(state != 0):
            return 0  # Draw
        
        return None  # Game is still ongoing
    
    def get_state_type(self):
        return ConnectFourState


class ConnectFourState(GameState):
    def __init__(self, game: BaseGame, player: int = 1):
        super().__init__(game, player)