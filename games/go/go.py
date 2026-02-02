import numpy as np
from games.base_game import BaseGame, GameState
from sgfmill import boards

class Go(BaseGame):
    def __init__(self, board_size=9, komi=6.5):
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = board_size * board_size + 1  # +1 for the pass move
        self.komi = komi
        self.empty_board = boards.Board(board_size)

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

    def detect_suicide_moves(self, state: np.ndarray) -> np.ndarray:
        '''Detects suicide moves on the board.'''
        suicide_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)

        possible_moves = np.where(state == 0)

        # Check each possible move for suicide
        for r, c in zip(*possible_moves):
            _, neighbors = self.get_neighbors(state, r, c)
            if not any(neighbors == -1):
                continue

            temp_state = np.copy(state)
            temp_state[r, c] = 1
            _, liberties = self.count_liberties(temp_state, r, c)
            if not liberties:
                suicide_moves[r, c] = 1

        return suicide_moves

    def detect_ko(self, state: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
        '''Detects ko moves on the board.'''
        ko_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        if np.sum(np.where(prev_state != 0)) < 2:
            return ko_moves

        possible_ko_moves = np.where((state == 0) & (prev_state == 1))

        # Check each possible ko move
        for r, c in zip(*possible_ko_moves):
            temp_state = np.copy(state)
            temp_state[r, c] = 1
            if np.array_equal(temp_state, prev_state):
                ko_moves[r, c] = 1
        return ko_moves

    def get_valid_actions(self, game_info: dict) -> np.ndarray:
        '''Returns a numpy array indicating valid actions on the board.'''
        state: np.ndarray = game_info["board"]
        prev_state: np.ndarray = game_info["prev_state"]
        suicide_moves = self.detect_suicide_moves(state)
        ko_moves = self.detect_ko(state, prev_state)
        valid_actions = np.zeros(self.action_size, dtype=np.int8)

        valid_actions[:-1] = (state.reshape(-1) == 0) & (suicide_moves.reshape(-1) == 0) & (ko_moves.reshape(-1) == 0)
        valid_actions[-1] = 1  # Pass is always valid
        return valid_actions

    def is_valid_action(self, game_info: dict, action: int) -> bool:
        '''Checks if a given action is valid based on the current game state.'''
        if action == self.row_count * self.col_count or action < 0:  # Pass move or resignation
            return True
        valid_actions = self.get_valid_actions(game_info)
        return valid_actions[action] == 1
    
    def get_next_state(self, game_state: dict, action: int) -> dict:
        '''Returns the next game state after applying the given action.'''

        game_info = game_state.copy()
        state: np.ndarray = game_info["board"].copy()
        board: boards.Board = game_info["game_board"].copy()
        last_moves: dict = game_info["last_moves"].copy()
        player: int = game_info["player"]

        # Apply the action to the board if it's not a pass or resignation
        if action != self.action_size - 1 and action >= 0:
            row, col = divmod(action, self.col_count)
            board.play(row, col, 'b' if player == 1 else 'w')

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

        game_info["last_moves"] = last_moves
        game_info["game_board"] = board

        # Update the board state
        mapping = {'b': 1, 'w': -1, None: 0}
        new_state = np.array([[mapping[c] for c in row] for row in board.board], dtype=np.int8)
        new_state *= player

        game_info["prev_state"] = state
        game_info["board"] = new_state

        return game_info

    def check_win(self, game_info: dict) -> int | None:
        '''Calculates the score to determine the winner of the game.'''

        board: boards.Board = game_info["game_board"]
        player: int = game_info["player"]

        score = board.area_score(komi=self.komi) * player # Calc score based on perspective

        return score

    def is_terminal(self, game_info: dict) -> tuple[int | None, bool]:
        '''Determines if the game has reached a terminal state.'''

        last_moves: dict = game_info["last_moves"]

        if not all(move == self.action_size - 1 for move in last_moves.values()):
            return 0, False  # Game ongoing

        score = self.check_win(game_info)
        if score is None:
            return None, False  # Resignation
        return score, True  # Game over

    def change_perspective(self, game_state: dict) -> dict:
        '''Changes the perspective of the game state to the current player.'''

        # Adjust board perspective based on the current player
        game_info = game_state.copy()
        len_game = game_info["action_count"]
        game_info["player"] = 1 if len_game % 2 == 0 else -1
        game_info["board"] = game_info["board"] * game_info["player"]
        game_info["prev_state"] = game_info["prev_state"] * game_info["player"]
        return game_info

    def get_state_type(self):
        return GoState

class GoState(GameState):
    def __init__(self, game: Go, player=1):
        super().__init__(game=game, player=player)
        self.board: np.ndarray = game.get_initial_state()
        self.player: int = player
        self.last_moves: dict = {
            "1": None,
            "-1": None
        }
        self.prev_state: np.ndarray = np.zeros((game.row_count, game.col_count), dtype=np.int8)
        self.game_board = boards.Board(game.row_count)
        self.action_count: int = 0
    
    def get_info(self):
        '''Returns a dictionary containing the current state information.'''
        return {
            "board": self.board.copy(),
            "player": self.player,
            "last_moves": self.last_moves.copy(),
            "prev_state": self.prev_state.copy(),
            "game_board": self.game_board.copy(),
            "action_count": self.action_count
        }
    
    def update(self, game_info: dict):
        '''Updates the current state with the provided game information.'''
        self.board = game_info["board"]
        self.player = game_info["player"]
        self.last_moves = game_info["last_moves"]
        self.prev_state = game_info["prev_state"]
        self.game_board = game_info["game_board"]
        self.action_count = game_info["action_count"]