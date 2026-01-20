import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    '''A dataset class for processed data.'''

    def __init__(self, data_dir: str, transform_func: callable =None, device: torch.device = None):
        self.data_dir = data_dir
        self.transform = transform_func
        self.device = device if device is not None else torch.device('cpu')

        state_file = os.path.join(data_dir, "states.npy")
        action_file = os.path.join(data_dir, "actions.npy")

        if not (os.path.exists(state_file) and os.path.exists(action_file)):
            raise FileNotFoundError("One or more data files are missing in the specified directory.")
        
        self.states = np.load(state_file, mmap_mode='r')
        self.actions = np.load(action_file, mmap_mode='r')
        if len(self.states) != len(self.actions):
            raise ValueError("The number of states and actions must be the same.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state, action = self.states[idx], self.actions[idx]

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)

        if self.transform:
            state, action = self.transform(state, action)

        return state, action