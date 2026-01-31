import os
import numpy as np
from sgfmill import boards
from games.base_game import BaseGame


class DataProcessor:
    '''A class for processing raw data files.'''

    def __init__(self, game: BaseGame, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.game = game
        self.board = boards.Board(self.game.row_count)
        self.board_size = self.game.row_count * self.game.col_count

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

    def process_data(self):
        '''Processes raw data files and saves them in a structured format.'''
        raw_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.sgf')]

        all_states = []
        all_actions = []
        all_values = []

        for file_name in raw_files: # loops assuming all files are in main raw data directory
            file_path = os.path.join(self.raw_data_dir, file_name)
            with open(file_path, 'r') as f:
                content = f.read().strip()
                moves = content.split(';')[1:]  # Skip header info
                header = moves[0]
                moves = moves[1:]  # Actual moves start from index 1

                res_idx = header.find('RE[')
                result = header[res_idx + 3:res_idx + 5] if res_idx != -1 else 'Unknown'
                result_color = 'b' if result == 'B+' else 'w'

                b = self.board.copy()

                try:
                    for move in moves:
                        if not move:
                            continue
                        color = move[0].lower()
                        last_idx = move.rfind(']')
                        coords = move[2:last_idx]
                        if coords == '':
                            continue
                        col = ord(coords[0]) - ord('a')
                        row = ord(coords[1]) - ord('a')

                        try:
                            if col < self.game.col_count and row < self.game.row_count:
                                b.play(row, col, color)
                        except (IndexError, ValueError):
                            raise ValueError(f"Invalid move {move} in file {file_name}")

                        mapping = {'b': 1, 'w': -1, None: 0}
                        new_state = np.array([[mapping[c] for c in row] for row in b.board], dtype=np.int8).reshape((9, 9))
                        if color == 'w':
                            new_state *= -1  # Perspective of white

                        action = row * self.game.col_count + col
                        if action >= self.board_size:
                            action = self.board_size  # Pass move

                        transforms = self.get_all_transforms(new_state, action)

                        value = 0.6 if color == result_color else 0.5

                        for s, a in transforms:
                            all_states.append(self.game.get_encoded_state(s))
                            all_actions.append(a)
                            all_values.append(value)
                except ValueError:
                    print(f"Invalid move in file {file_name}. Skipping rest of game.")
                    continue

        for i in range(2):
            # Act final moves as double pass to end the game
            transforms = self.get_all_transforms(new_state, self.board_size)
            for s, a in transforms:
                all_states.append(self.game.get_encoded_state(s if i == 0 else -s))
                all_actions.append(a)
                all_values.append(0.5)  # Neutral value for pass

        states_array = np.array(all_states, dtype=np.int8)
        actions_array = np.array(all_actions, dtype=np.int8)
        values_array = np.array(all_values, dtype=np.float32)

        unique_states, inv_idx, counts = np.unique(states_array, axis=0, return_inverse=True, return_counts=True)
        policies = np.zeros((len(unique_states), self.game.action_size), dtype=np.float32)
        agr_values = np.zeros(len(unique_states), dtype=np.float32)

        for idx, (a, v) in zip(inv_idx, zip(actions_array, values_array)):
            policies[idx][a] += 1
            agr_values[idx] += v

        policies /= counts[:, None] + 1e-8  # Normalize policies
        agr_values /= counts + 1e-8  # Normalize values

        np.save(os.path.join(self.processed_data_dir, 'states.npy'), unique_states)
        np.save(os.path.join(self.processed_data_dir, 'policies.npy'), policies)
        np.save(os.path.join(self.processed_data_dir, 'values.npy'), agr_values)

        print(f"Processed {len(raw_files)} files with a total of {len(all_states)} samples.")
        print(f"Generated {len(unique_states)} unique states.")
        print(f"Average samples per unique state: {np.mean(counts):.2f}")

    def get_all_transforms(self, state: np.ndarray, action: int) -> list[tuple[np.ndarray, int]]:
        '''Generates all rotations and reflections of the given state and action.'''
        transforms = []
        board_dim = (self.game.row_count, self.game.col_count)

        for k in range(4):
            rotated_state = np.rot90(state.reshape(board_dim), k).flatten()
            row, col = divmod(action, self.game.col_count)
            
            for _ in range(k):
                row, col = col, self.game.row_count - 1 - row  # Rotate coordinates
            rotated_action = row * self.game.col_count + col if action < self.board_size else action
            transforms.append((rotated_state, rotated_action))

            flipped_state = np.fliplr(rotated_state.reshape(board_dim)).flatten()

            flipped_col = self.game.col_count - 1 - col
            flipped_action = row * self.game.col_count + flipped_col if action < self.board_size else action
            transforms.append((flipped_state, flipped_action))

        return transforms