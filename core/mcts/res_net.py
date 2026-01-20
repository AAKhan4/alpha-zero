import torch
import torch.nn as nn
import torch.nn.functional as F

from games.base_game import BaseGame

class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ResNet(nn.Module):
    def __init__(self, game: BaseGame, num_resBlocks: int, num_channels: int, device: torch.device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.col_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.col_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value