import numpy as np
from games.base_game import BaseGame

class Go(BaseGame):
    def __init__(self, board_size=9, komi=6.5):
        super().__init__()
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = board_size * board_size + 1  # +1 for the pass move

        self.last_action = None  # To track consecutive passes
        self.last_2_boards = []  # To track the last two board states for Ko rule
        self.score = -1 * komi  # Initial score considering komi
        self.captures = 0

    def __repr__(self):
        return "Go"

    def get_initial_state(self) -> np.ndarray:
        # Create an empty board (all zeros)
        return np.zeros((self.row_count, self.col_count), dtype=np.int8)

    def get_neighbors(self, state: np.ndarray, r: int, c: int) -> dict:
        neighbors = {}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = r + dx, c + dy
            if 0 <= nx < self.row_count and 0 <= ny < self.col_count:
                neighbors[(nx, ny)] = state[nx, ny]
        return neighbors

    def count_liberties(self, state: np.ndarray, r: int, c: int) -> tuple[list, int]:
        visited = set()
        stack = [(r, c)]
        group = [(r, c)]
        player = state[r, c]
        liberty_count = 0

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.row_count and 0 <= ny < self.col_count:
                    if state[nx, ny] == 0:
                        liberty_count += 1
                    elif state[nx, ny] == player and (nx, ny) not in visited:
                        group.append((nx, ny))
                        stack.append((nx, ny))
        return group, liberty_count

    def remove_adj_dead_stones(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = np.copy(state)
        r, c = divmod(action, self.col_count)
        opponent = -1 * new_state[r, c]

        neighbors = self.get_neighbors(new_state, r, c)
        for (nx, ny), val in neighbors.items():
            if val == opponent:
                group, liberties = self.count_liberties(new_state, nx, ny)
                if liberties == 0:
                    for x, y in group:
                        new_state[x, y] = 0
                    self.captures += opponent * len(group)
        return new_state

    def detect_suicide_moves(self, state: np.ndarray, player: int) -> np.ndarray:
        suicide_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)

        for r in range(self.row_count):
            for c in range(self.col_count):
                if state[r, c] != 0:
                    continue
                neighbors = self.get_neighbors(state, r, c)
                if not all(neighbors.values() != 0):
                    continue

                temp_state = np.copy(state)
                temp_state[r, c] = player
                if self.count_liberties(temp_state, r, c) == 0:
                    suicide_moves[r, c] = 1
        return suicide_moves

    def detect_ko(self, state: np.ndarray, player: int) -> np.ndarray:
        ko_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        if len(self.last_2_boards) < 2:
            return ko_moves

        current_board = state
        previous_board = self.last_2_boards[1]

        for r in range(self.row_count):
            for c in range(self.col_count):
                if current_board[r, c] != 0:  # Only consider empty positions
                    continue
                temp_state = np.copy(current_board)
                temp_state[r, c] = player
                if np.array_equal(temp_state, previous_board):
                    ko_moves[r, c] = 1

        return ko_moves

    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        player = -1 * state[divmod(self.last_action, self.col_count)]
        suicide_moves = self.detect_suicide_moves(state, player)
        ko_moves = self.detect_ko(state, player)
        valid_actions = np.zeros(self.action_size, dtype=np.int8)

        return (valid_actions and not suicide_moves and not ko_moves).astype(np.int8)

    def is_valid_action(self, state: np.ndarray, action: int) -> bool:
        valid_actions = self.get_valid_actions(state)
        return valid_actions[action] == 1

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        self.last_action = action
        if action == self.row_count * self.col_count:  # Pass move
            return state
        row, col = divmod(action, self.col_count)

        new_state = state.copy()
        new_state[row, col] = player
        new_state = self.remove_adj_dead_stones(new_state, action)

        self.last_2_boards.append(state)
        if len(self.last_2_boards) > 2:
            self.last_2_boards.pop(0)

        return new_state
    
    def calc_score(self, state):
        black_territory = 0
        white_territory = 0
        visited = np.zeros((self.row_count, self.col_count), dtype=bool)
        current_board = np.copy(state)

        def dfs(r, c):
            stack = [(r, c)]
            territory = []
            bordering_colors = set()
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                visited[x, y] = True
                territory.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.row_count and 0 <= ny < self.col_count:
                        if visited[nx, ny]:
                            bordering_colors.add(current_board[nx, ny])
                        elif current_board[nx, ny] == 0:
                            stack.append((nx, ny))
            return territory, bordering_colors

        for r in range(self.row_count):
            for c in range(self.col_count):
                if not visited[r, c]:
                    if current_board[r, c] == 0:
                        territory, bordering_colors = dfs(r, c)
                        if 1 in bordering_colors:
                            white_territory += len(territory)
                        elif -1 in bordering_colors:
                            black_territory += len(territory)

        self.score += black_territory - white_territory + self.captures
        return self.score
    
    def check_win(self, state, action):
        pass_action = self.row_count * self.col_count
        if action != pass_action or self.last_action != pass_action:
            return False, 0
        
        final_score = self.calc_score(state)
        if final_score > 0:
            return True  # Black wins
        elif final_score < 0:
            return False  # White wins
        else:
            return None  # Draw
        
    def is_terminal(self, state: np.ndarray, action: int) -> tuple[int, bool]:
        pass
