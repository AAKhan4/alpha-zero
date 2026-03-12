import numpy as np
import sgfmill
from games.base_game import BaseGame, GameState
from sgfmill import boards

class Go(BaseGame):
    def __init__(self, board_size=9, komi=2.5, max_game_length=70):
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = (board_size * board_size) + 1  # +1 for the pass move
        self.komi = komi
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

    def get_surrounding_players(self, board: np.ndarray, group: list[tuple[int, int]]) -> set[int]:
        '''Determines which players surround a group of stones.'''
        surrounding_players = set()

        for r, c in group:
            _, neighbors_val = self.get_neighbors(board, r, c)
            for val in neighbors_val:
                if val != 0:
                    surrounding_players.add(val)

        return surrounding_players
    
    def get_region(self, game_info: dict, row: int, col: int) -> tuple[set[tuple[int, int]], set[int], int]:
        '''Returns the connected region of empty spaces and the players that surround it.'''
        # Based on algorithm from SGFMill
        spaces = set()
        neighbouring_colours = set()
        to_handle = set()
        to_handle.add((row, col))
        while to_handle:
            point = to_handle.pop()
            spaces.add(point)
            r, c = point
            neighbour_idx, neighbours = self.get_neighbors(game_info["board"], r, c)
            for idx, neighbour in zip(neighbour_idx, neighbours):
                r1, c1 = idx
                if neighbour is None:
                    if (r1, c1) not in spaces:
                        to_handle.add((r1, c1))
                else:
                    neighbouring_colours.add(neighbour)
        players = neighbouring_colours if neighbouring_colours else set()
        return spaces, players, len(spaces)

    def count_liberties(self, state: np.ndarray, r: int, c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        '''Counts the liberties of the group of stones connected to the stone at (r, c).'''
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        stack = [(r, c)]
        group = set()
        player = state[r, c]
        liberties = set()

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            group.add((x, y))

            neighbors_idx, neighbors_val = self.get_neighbors(state, x, y)
            for (nx, ny), val in zip(neighbors_idx, neighbors_val):
                if val == 0:
                    liberties.add((nx, ny))
                elif val == player and not visited[nx, ny]:
                    stack.append((nx, ny))

        return group, liberties
    
    def has_liberties(self, state: np.ndarray, r: int, c: int) -> bool:
        '''Checks if the stone at (r, c) has any liberties.'''
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        stack = [(r, c)]
        group = set()
        player = state[r, c]

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            group.add((x, y))

            neighbors_idx, neighbors_val = self.get_neighbors(state, x, y)
            for (nx, ny), val in zip(neighbors_idx, neighbors_val):
                if val == 0:
                    return True  # Found a liberty, no need to continue
                elif val == player and not visited[nx, ny]:
                    stack.append((nx, ny))
        return False  # No liberties found

    def is_not_suicide(self, state: np.ndarray, r: int, c: int) -> bool:
        idx, neighbors = self.get_neighbors(state, r, c)
        if np.any(neighbors == 0):
            return True
        state[r, c] = 1
        not_suicide = self.has_liberties(state, r, c)

        # Check if the move captures any opponent groups
        if not not_suicide:
            opponent = -1
            for (nr, nc), val in zip(idx, neighbors):
                if val == opponent:
                    if self.has_liberties(state, nr, nc):
                        state[r, c] = 0
                        return True

        state[r, c] = 0
        return not_suicide

    def is_not_ko(self, game_info: dict, r: int, c: int) -> bool:
        if not game_info["ko_position"]:
            return True
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
        return self.is_not_suicide(game_info["board"], r, c) and self.is_not_ko(game_info, r, c)
    
    def apply_move(self, state: np.ndarray, r: int, c: int) -> tuple[np.ndarray, list[list[int]]]:
        state = state.copy()
        if ((r * self.col_count) + c) == self.action_size - 1:  # Pass move
            return state, []
        state[r, c] = 1
        captured = []

        for nr, nc in self.get_neighbors(state, r, c)[0]:
            if state[nr, nc] == -1:
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
        if action >= 0:
            r,c = divmod(action, self.col_count)
            new_state, captured = self.apply_move(state, r, c)

            if len(captured) == 1:
                gx, gy = captured[0]
                _, liberties = self.count_liberties(new_state, r, c)
                if len(liberties) == 1:
                    ko_pos = (gx, gy)
                else:
                    ko_pos = None
            else:
                ko_pos = None
        new_state = new_state * -1  # Flip the board for the opponent's perspective

        # Update last moves
        last_moves[str(player)] = action
        if action < 0: # Resignation
            last_moves[str(player)] = self.action_size - 1  # Mark resignation as double pass for consistency
            last_moves[str(-player)] = self.action_size - 1
            return game_info

        return {
            "board": new_state.copy(),
            "player": -player,
            "last_moves": last_moves.copy(),
            "action_count": game_info["action_count"] + 1,
            "ko_position": ko_pos
        }

    def check_win(self, game_info: dict) -> int | None:
        '''
        Calculates the score to determine the winner of the game.
        Returns score from perspective of current player.
        '''
        score = self.calc_score(game_info) - (game_info["player"] * self.komi)  # Calc score based on perspective

        return score

    def calc_score(self, game_info: dict) -> int:
        '''Calculates the score of the game from the perspective of the current player.'''
        # Based on algorithm from SGFMill
        score = 0
        handled = set()
        for row in range(self.row_count):
            for col in range(self.col_count):
                if game_info["board"][row, col] != 0:
                    score += game_info["board"][row, col]  # Add points for occupied positions
                    continue
                point = (row, col)
                if point in handled:
                    continue
                region, players, size = self.get_region(game_info, row, col)
                for player in players:
                    score += player * size  # Add points for controlled empty regions
                handled.update(region)
        return score
    
    def is_terminal(self, game_info: dict) -> tuple[int | None, bool]:
        '''Determines if the game has reached a terminal state.'''

        last_moves: dict = game_info["last_moves"]

        if game_info["action_count"] < self.max_game_length and not all(move == self.action_size - 1 for move in last_moves.values()):
            return 0, False  # Game ends in a draw if both players pass or max game length is reached

        score = self.check_win(game_info)
        if score is None:
            return None, True  # Resignation 
        return score, True  # Game over

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