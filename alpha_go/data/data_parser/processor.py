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
        raw_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.txt')]

        all_states = []
        all_actions = []

        for file_name in raw_files: # loops assuming all files are in main raw data directory
            file_path = os.path.join(self.raw_data_dir, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    if ';' not in line and '[' not in line:
                        continue
                    b = self.board.copy()

                    try:
                        for move in line.strip().split(';'):
                            if not move:
                                continue
                            color = move[0].lower()
                            coords = move[2:-1]
                            if coords == '':
                                continue
                            col = ord(coords[0]) - ord('a')
                            row = ord(coords[1]) - ord('a')
                            try:
                                b.play(row, col, color)
                            except ValueError:
                                raise ValueError(f"Invalid move {move} in file {file_name}")

                            mapping = {'b': 1, 'w': -1, None: 0}
                            new_state = np.array([[mapping[c] for c in row] for row in b.board], dtype=np.int8)
                            if color == 'w':
                                new_state *= -1  # Perspective of white
                            all_states.append(self.game.get_encoded_state(new_state))
                            action = row * self.board_size + col
                            all_actions.append(action)
                    except ValueError:
                        print(f"Invalid move in file {file_name}. Skipping rest of game.")
                        continue

        states_array = np.array(all_states, dtype=np.int8)
        actions_array = np.array(all_actions, dtype=np.int8)

        np.save(os.path.join(self.processed_data_dir, 'states.npy'), states_array)
        np.save(os.path.join(self.processed_data_dir, 'actions.npy'), actions_array)