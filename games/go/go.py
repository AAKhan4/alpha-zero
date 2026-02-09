import numpy as np
from games.base_game import BaseGame, GameState
from sgfmill import boards

class Go(BaseGame):
    def __init__(self, board_size=9, komi=6.5, max_game_length=70):
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = board_size * board_size + 1  # +1 for the pass move
        self.komi = komi
        self.empty_board = boards.Board(board_size)
        self.colour_mapping = {1: 'b', -1: 'w', 0: None}
        self.max_game_length = max_game_length
        self.can_pass = True  # Go allows a "pass" action

    def __repr__(self):
        return "Go"

    def get_initial_state(self) -> np.ndarray:
        '''Returns the initial state of the Go board as a numpy array.'''
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count), dtype=np.int8)

    def get_neighbors(self, state: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        '''Returns the neighbors' indices and their values for a given position (r, c) on the board.'''
        offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        neighbors_idx = offsets + np.array([r, c])

        valid_idx = (neighbors_idx[:,0] >= 0) & (neighbors_idx[:,0] < self.row_count) & \
                    (neighbors_idx[:,1] >= 0) & (neighbors_idx[:,1] < self.col_count)
        neighbors_idx = neighbors_idx[valid_idx]

        return neighbors_idx, np.array([state[x, y] for x, y in neighbors_idx])

    def count_liberties(self, state: np.ndarray, r: int, c: int) -> tuple[list[list[int]], set[tuple[int, int]]]:
        '''Counts the liberties of the group of stones connected to the stone at (r, c).'''
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        stack = [(r, c)]
        group = [[r, c]]
        player = state[r, c]
        liberties = set()

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True

            neighbors_idx, neighbors_val = self.get_neighbors(state, x, y)
            
            unmarked_neighbors = np.where(neighbors_val == 0)[0]
            unmarked_neighbors = neighbors_idx[unmarked_neighbors]
            liberties.update((nx, ny) for nx, ny in unmarked_neighbors)

            adj_player_stones = np.where(neighbors_val == player)[0]
            adj_player_stones = neighbors_idx[adj_player_stones] # Get coordinates of adjacent same-color stones
            unvisited = ~np.isin(adj_player_stones, visited).all(axis=1)
            adj_player_stones = adj_player_stones[unvisited]

            for nx, ny in adj_player_stones:
                stack.append((nx, ny))
                group.append([nx, ny])

        return group, liberties

    def is_not_suicide(self, state: np.ndarray, r: int, c: int, player: int) -> bool:
        state = state.copy()
        state[r, c] = player
        _, liberties = self.count_liberties(state, r, c)
        return len(liberties) > 0

    def is_not_ko(self, game_info: dict, r: int, c: int) -> bool:
        if game_info["ko_position"] == (r, c):
            return False
        return True

    def get_valid_actions(self, game_info: dict) -> np.ndarray:
        '''Returns a numpy array indicating valid actions on the board.'''
        state: np.ndarray = game_info["board"]
        [r, c] = np.where(state == 0)
        mask = np.zeros(self.action_size, dtype=np.int8)
        mask[(r * self.col_count) + c] = 1
        mask[-1] = 1

        return mask

    def is_valid_action(self, game_info: dict, action: int) -> bool:
        '''Checks if a given action is valid based on the current game state.'''
        if (action == self.action_size-1) or (action < 0):  # Pass move or resignation
            return True
        r, c = divmod(action, self.col_count)
        return (game_info["board"][r, c] == 0) and self.is_not_suicide(game_info["board"], r, c, game_info["player"]) and self.is_not_ko(game_info, r, c)
    
    def apply_move(self, state: np.ndarray, r: int, c: int, player: int) -> tuple[np.ndarray, list[list[int]]]:
        state = state.copy()
        state[r, c] = player
        captured = []

        for nr, nc in self.get_neighbors(state, r, c)[0]:
            if state[nr, nc] == -player:
                group, liberties = self.count_liberties(state, nr, nc)
                if len(liberties) == 0:
                    for gx, gy in group:
                        state[gx, gy] = 0
                    captured.extend(group)

        return state, captured
    
    def get_next_state(self, game_info: dict, action: int) -> dict:
        '''Returns the next game state after applying the given action.'''
        state: np.ndarray = game_info["board"]
        last_moves: dict = game_info["last_moves"].copy()
        player: int = game_info["player"]
    
        # Apply the action to the board if it's not a pass or resignation
        if action != self.action_size - 1 and action >= 0:
            r,c = divmod(action, self.col_count)
            new_state, captured = self.apply_move(state, r, c, player)

            if len(captured) == 1:
                gx, gy = captured[0]
                _, liberties = self.count_liberties(new_state, r, c)
                if len(liberties) == 1:
                    ko_pos = (gx, gy)
                else:
                    ko_pos = None
            else:
                ko_pos = None

        # Update last moves
        last_moves[str(player)] = action
        game_info["action_count"] += 1
        if action == self.action_size - 1:  # Pass move
            return game_info
        if action < 0: # Resignation
            last_moves[str(player)] = self.action_size - 1  # Mark resignation as double pass for consistency
            game_info["action_count"] += 1
            last_moves[str(-player)] = self.action_size - 1
            return game_info

        game_info["board"] = new_state
        game_info["last_moves"] = last_moves
        game_info["ko_position"] = ko_pos

        return game_info

    def check_win(self, game_info: dict) -> int | None:
        '''Calculates the score to determine the winner of the game.'''

        board: np.ndarray = game_info["board"]
        game_board = boards.Board(self.row_count)
        for r in range(self.row_count):
            for c in range(self.col_count):
                stone = self.colour_mapping[board[r, c]]
                if stone is not None:
                    game_board.play(r, c, stone)

        player: int = game_info["player"]
        score = (game_board.area_score() - self.komi) * player  # Calc score based on perspective

        return score

    def is_terminal(self, game_info: dict) -> tuple[int | None, bool]:
        '''Determines if the game has reached a terminal state.'''

        last_moves: dict = game_info["last_moves"]

        if (not all(move == self.action_size - 1 for move in last_moves.values())) and game_info["action_count"] < self.max_game_length:
            return 0, False  # Game ongoing

        score = self.check_win(game_info)
        if score is None:
            return None, True  # Resignation 
        return score, True  # Game over

    def change_perspective(self, game_state: dict) -> dict:
        '''Changes the perspective of the game state to the current player.'''

        # Adjust board perspective based on the current player
        game_info = game_state.copy()
        len_game = game_info["action_count"]
        game_info["player"] = 1 if len_game % 2 == 0 else -1
        game_info["board"] = game_info["board"] * game_info["player"]
        return game_info

    def get_state_type(self):
        return GoState

class GoState(GameState):
    def __init__(self, game: Go, player=1):
        super().__init__(game=game, player=player)
        self.player: int = player
        self.last_moves: dict = {
            "1": None,
            "-1": None
        }
        self.action_count: int = 0
        self.ko_position: tuple[int, int] | None = None
        self.board = game.get_initial_state()
    
    def get_info(self):
        '''Returns a dictionary containing the current state information.'''
        return {
            "board": self.board,
            "player": self.player,
            "last_moves": self.last_moves,
            "action_count": self.action_count,
            "ko_position": self.ko_position
        }
    
    def update(self, game_info: dict):
        '''Updates the current state with the provided game information.'''
        self.player = game_info["player"]
        self.last_moves = game_info["last_moves"]
        self.ko_position = game_info["ko_position"]
        self.action_count = game_info["action_count"]
        self.board = game_info["board"]