import numpy as np
from games.base_game import BaseGame, GameState

# TODO: FIX CAPTURE LOGIC

class Go(BaseGame):
    def __init__(self, board_size=9, komi=6.5):
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = board_size * board_size + 1  # +1 for the pass move
        self.komi = komi

    def __repr__(self):
        return "Go"

    def get_initial_state(self) -> np.ndarray:
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count), dtype=np.int8)

    def get_neighbors(self, state: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        neighbors_idx = offsets + np.array([r, c])

        valid_idx = (neighbors_idx[:,0] >= 0) & (neighbors_idx[:,0] < self.row_count) & \
                    (neighbors_idx[:,1] >= 0) & (neighbors_idx[:,1] < self.col_count)
        neighbors_idx = neighbors_idx[valid_idx]

        return neighbors_idx, np.array([state[x, y] for x, y in neighbors_idx])

    def count_liberties(self, state: np.ndarray, r: int, c: int) -> tuple[list[list[int]], set[tuple[int, int]]]:
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

    def remove_adj_dead_stones(self, state: np.ndarray, action: int, captures: int) -> tuple[np.ndarray, int]:
        new_state = np.copy(state)
        r, c = divmod(action, self.col_count)
        opponent = -1 * new_state[r, c]

        neighbors_idx, neighbors_val = self.get_neighbors(new_state, r, c)
        opponent_stones = np.where(neighbors_val == opponent)[0]
        opponent_stones = neighbors_idx[opponent_stones]  # Get coordinates of adjacent opponent stones
        for (nx, ny) in opponent_stones:
            group, liberties = self.count_liberties(new_state, nx, ny)
            group = np.array(group)
            if not liberties:
                continue
            new_state[group[:,0], group[:,1]] = 0
            captures += len(group)
        return new_state, captures

    def detect_suicide_moves(self, state: np.ndarray) -> np.ndarray:
        suicide_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)

        possible_moves = np.where(state == 0)

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

    def detect_ko(self, state: np.ndarray, last_2_boards: list[np.ndarray]) -> np.ndarray:
        ko_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        if len(last_2_boards) < 2:
            return ko_moves

        previous_board = last_2_boards[1]

        possible_ko_moves = np.where((state == 0) & (previous_board == 1))

        for r, c in zip(*possible_ko_moves):
            temp_state = np.copy(state)
            temp_state[r, c] = 1
            if np.array_equal(temp_state, previous_board):
                ko_moves[r, c] = 1
        return ko_moves

    def get_valid_actions(self, game_info: dict) -> np.ndarray:
        state: np.ndarray = game_info["board"]
        last_2_boards: list[np.ndarray] = game_info["last_2_boards"]
        suicide_moves = self.detect_suicide_moves(state)
        ko_moves = self.detect_ko(state, last_2_boards)
        valid_actions = np.zeros(self.action_size, dtype=np.int8)

        valid_actions[:-1] = (state.reshape(-1) == 0) & (suicide_moves.reshape(-1) == 0) & (ko_moves.reshape(-1) == 0)
        valid_actions[-1] = 1  # Pass is always valid
        return valid_actions

    def is_valid_action(self, game_info: dict, action: int) -> bool:
        state: np.ndarray = game_info["board"]
        last_2_boards: list[np.ndarray] = game_info["last_2_boards"]
        if action == self.row_count * self.col_count:  # Pass move
            return True
        valid_actions = self.get_valid_actions(state, last_2_boards)
        return valid_actions[action] == 1

    def get_next_state(self, game_state: dict, action: int) -> dict:
        game_info = game_state.copy()
        game_info["last_2_boards"] = game_info["last_2_boards"].copy()
        game_info["last_2_actions"] = game_info["last_2_actions"].copy()
        game_info["last_2_actions"].append(action)
        if len(game_info["last_2_actions"]) > 2:
            game_info["last_2_actions"].pop(0)

        if action == self.row_count * self.col_count or action < 0:  # Pass move or resignation
            return game_info
        row, col = divmod(action, self.col_count)

        new_state = game_info["board"].copy()
        new_state[row, col] = 1
        new_state, game_info["captures"] = self.remove_adj_dead_stones(new_state, action, game_info["captures"])

        game_info["last_2_boards"].append(new_state.copy())
        if len(game_info["last_2_boards"]) > 2:
            game_info["last_2_boards"].pop(0)
        game_info["board"] = new_state

        return game_info

    def calc_territory(self, state: np.ndarray) -> np.ndarray:
        territory = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        current_board = np.copy(state)

        def dfs(r, c):
            stack = [(r, c)]
            territory_cells = []
            bordering_colors = set()
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                visited[x, y] = True
                territory_cells.append([x, y])
                neighbors_idx, _ = self.get_neighbors(current_board, x, y)
                for nx, ny in neighbors_idx:
                    if visited[nx, ny]:
                        bordering_colors.add(current_board[nx, ny])
                    elif current_board[nx, ny] == 0:
                        stack.append((nx, ny))
            return territory_cells, bordering_colors

        empty_cells = np.where(current_board == 0)
        for r, c in zip(*empty_cells):
            if visited[r, c]:
                continue
            territory_cells, bordering_colors = dfs(r, c)
            if 1 in bordering_colors and -1 not in bordering_colors:
                territory[territory_cells[:,0], territory_cells[:,1]] = 1  # Black territory
            elif -1 in bordering_colors and 1 not in bordering_colors:
                territory[territory_cells[:,0], territory_cells[:,1]] = -1  # White territory

        return territory

    def remove_dead_stones_end(self, state: np.ndarray, captures: int) -> tuple[np.ndarray, int]:
        new_state = np.copy(state)
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        territory = self.calc_territory(state)
        possible_dead_stones = np.where((state != 0) & (territory.reshape(-1) == -state.reshape(-1)))

        for r, c in zip(*possible_dead_stones):
            if visited[r, c]:
                continue
            group, liberties = self.count_liberties(new_state, r, c)
            group = np.array(group)

            # Check for two eyes to determine if the group is alive
            eye_count = 0
            for x, y in liberties:
                _, neighbors_val = self.get_neighbors(new_state, x, y)
                if np.all(neighbors_val == state[r, c]):
                    eye_count += 1
            if eye_count >= 2:
                continue

            # If any liberty is in opponent's territory, the group is alive
            if liberties and not np.all(territory[liberties[:,0], liberties[:,1]] == -state[r, c]):
                continue
            new_state[group[:,0], group[:,1]] = 0
            visited[group[:,0], group[:,1]] = True
            captures += len(group) * state[r, c]

        return new_state, captures

    def check_win(self, game_info: dict) -> int | None:
        state: np.ndarray = game_info["board"]
        last_2_actions: list[int] = game_info["last_2_actions"]
        player: int = game_info["player"]
        captures: int = game_info["captures"]

        valid_actions = self.get_valid_actions(game_info)
        if len(last_2_actions) < 2:
            return -1, False
        if last_2_actions[-1] != self.row_count * self.col_count and \
           last_2_actions[-2] != self.row_count * self.col_count:
            return -1, False
        if np.any(valid_actions[:-1]):  # There are valid moves other than pass
            return -1, False

        state, captures = self.remove_dead_stones_end(state, captures)
        territory = self.calc_territory(state)
        score = (np.sum(territory) + captures) - (self.komi * player)
        return score

    def is_terminal(self, game_info: dict) -> tuple[int | None, bool]:
        last_2_actions: list[int] = game_info["last_2_actions"]

        if last_2_actions[-1] < 0:  # Resignation
            return None, True

        score = self.check_win(game_info)
        if not score:
            return 0, False  # Game ongoing
        return score, True  # Game over

    def change_perspective(self, game_state: dict) -> dict:
        # Adjust board perspective based on the current player
        game_info = game_state.copy()
        game_info["last_2_boards"] = game_info["last_2_boards"].copy()
        game_info["last_2_actions"] = game_info["last_2_actions"].copy()
        game_info["board"] = -1 * game_info["board"].copy()
        game_info["player"] *= -1
        game_info["captures"] *= -1
        return game_info


class GoState(GameState):
    def __init__(self, game: Go, last_2_actions=None, last_2_boards=None, captures=0, player=1):
        super().__init__(game=game, player=player)
        self.last_2_actions = last_2_actions or []
        self.last_2_boards = last_2_boards or []
        self.captures = captures
    
    def get_info(self):
        return {
            "board": self.board,
            "player": self.player,
            "last_2_actions": self.last_2_actions.copy(),
            "last_2_boards": self.last_2_boards.copy(),
            "captures": self.captures
        }
    
    def update(self, game_info: dict):
        self.board = game_info["board"]
        self.player = game_info["player"]
        self.last_2_actions = game_info["last_2_actions"]
        self.last_2_boards = game_info["last_2_boards"]
        self.captures = game_info["captures"]
