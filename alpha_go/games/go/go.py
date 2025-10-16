import numpy as np
from games.base_game import BaseGame

class Go(BaseGame):
    def __init__(self, board_size=9, komi=6.5):
        super().__init__()
        self.row_count = board_size
        self.col_count = board_size
        self.action_size = board_size * board_size + 1  # +1 for the pass move

        self.last_3_actions = []  # To track consecutive passes
        self.last_2_boards = []  # To track the last two board states for Ko rule
        self.score = -1 * komi  # Initial score considering komi
        self.captures = 0

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

    def remove_adj_dead_stones(self, state: np.ndarray, action: int) -> np.ndarray:
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
            self.captures += len(group)
        return new_state

    def detect_suicide_moves(self, state: np.ndarray, player: int) -> np.ndarray:
        suicide_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)

        possible_moves = np.where(state == 0)

        for r, c in zip(*possible_moves):
            _, neighbors = self.get_neighbors(state, r, c)
            if not any(neighbors == -player):
                continue

            temp_state = np.copy(state)
            temp_state[r, c] = player
            _, liberties = self.count_liberties(temp_state, r, c)
            if not liberties:
                suicide_moves[r, c] = 1

        return suicide_moves

    def detect_ko(self, state: np.ndarray, player: int) -> np.ndarray:
        ko_moves = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        if len(self.last_2_boards) < 2:
            return ko_moves

        previous_board = self.last_2_boards[1]

        possible_ko_moves = np.where((state == 0) & (previous_board == player))

        for r, c in zip(*possible_ko_moves):
            temp_state = np.copy(state)
            temp_state[r, c] = player
            if np.array_equal(temp_state, previous_board):
                ko_moves[r, c] = 1
        return ko_moves

    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        player = -1 * state[divmod(self.last_3_actions[-1], self.col_count)]
        suicide_moves = self.detect_suicide_moves(state, player)
        ko_moves = self.detect_ko(state, player)
        valid_actions = np.zeros(self.action_size, dtype=np.int8)

        valid_actions[:-1] = (state.reshape(-1) == 0) & (suicide_moves.reshape(-1) == 0) & (ko_moves.reshape(-1) == 0)
        valid_actions[-1] = 1  # Pass is always valid
        return valid_actions

    def is_valid_action(self, state: np.ndarray, action: int) -> bool:
        if action == self.row_count * self.col_count:  # Pass move
            return True
        valid_actions = self.get_valid_actions(state)
        return valid_actions[action] == 1

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        self.last_3_actions.append(action)
        if len(self.last_3_actions) > 3:
            self.last_3_actions.pop(0)
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

    def remove_dead_stones_end(self, state: np.ndarray) -> np.ndarray:
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
            self.captures += len(group)

        return new_state

    def calc_score(self, state: np.ndarray) -> int:
        state = self.remove_dead_stones_end(state)
        territory = self.calc_territory(state)
        self.score += np.sum(territory) + self.captures
        return self.score
    
    def check_win(self, state, action):
        last_player = self.last_3_actions[0]
        last_player = state[divmod(last_player, self.col_count)]

    # def calc_score(self, state):
    #     black_territory = 0
    #     white_territory = 0
    #     visited = np.zeros((self.row_count, self.col_count), dtype=bool)
    #     current_board = np.copy(state)

    #     def dfs(r, c):
    #         stack = [(r, c)]
    #         territory = []
    #         bordering_colors = set()
    #         while stack:
    #             x, y = stack.pop()
    #             if visited[x, y]:
    #                 continue
    #             visited[x, y] = True
    #             territory.append((x, y))
    #             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #                 nx, ny = x + dx, y + dy
    #                 if 0 <= nx < self.row_count and 0 <= ny < self.col_count:
    #                     if visited[nx, ny]:
    #                         bordering_colors.add(current_board[nx, ny])
    #                     elif current_board[nx, ny] == 0:
    #                         stack.append((nx, ny))
    #         return territory, bordering_colors

    #     for r in range(self.row_count):
    #         for c in range(self.col_count):
    #             if not visited[r, c]:
    #                 if current_board[r, c] == 0:
    #                     territory, bordering_colors = dfs(r, c)
    #                     if 1 in bordering_colors:
    #                         white_territory += len(territory)
    #                     elif -1 in bordering_colors:
    #                         black_territory += len(territory)

    #     self.score += black_territory - white_territory + self.captures
    #     return self.score
    
    # def check_win(self, state, action):
    #     pass_action = self.row_count * self.col_count
    #     if action != pass_action or self.last_action != pass_action:
    #         return False, 0
        
    #     final_score = self.calc_score(state)
    #     if final_score > 0:
    #         return True  # Black wins
    #     elif final_score < 0:
    #         return False  # White wins
    #     else:
    #         return None  # Draw
        
    # def is_terminal(self, state: np.ndarray, action: int) -> tuple[int, bool]:
    #     pass
