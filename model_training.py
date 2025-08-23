import torch

from alpha_zero import AlphaZero
import res_net
import tic_tac_toe


class ModelTrainer:
    def __init__(self, game=None, args=None):
        if game is None:
            game = tic_tac_toe.TicTacToe()

        if args is None:
            args = {
                "num_searches": 60,
                "c": 2,
                "num_iterations": 3,
                "num_self_play": 500,
                "num_epochs": 4,
                "batch_size": 64,
                "temperature": 1.25,
                "epsilon": 0.25,
                "alpha": 0.3
            }

        self.run(game, args)

    def run(self, game, args, model=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model if model else res_net.ResNet(game, 4, 64, device)  # Initialize the neural network model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer

        alpha_zero = AlphaZero(model, optimizer, game, args)
        alpha_zero.learn()

ModelTrainer()  # Start training the model