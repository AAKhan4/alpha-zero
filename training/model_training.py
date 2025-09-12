import torch

from core.alpha_zero import AlphaZero
from core.mcts.res_net import ResNet
from games.game_select import GameSelection


class ModelTrainer:
    def __init__(self, game=None, args=None):
            
        game = game if game else GameSelection().pick_game()

        args = args if args else GameSelection().get_args(game)

        print(f"\nTraining on {game} with args: {args}\n")
        self.run(game, args)

    def run(self, game, args, model=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        model = model if model else ResNet(game, args["res_blocks"], args["channels"], device)  # Initialize the neural network model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer

        alpha_zero = AlphaZero(model, optimizer, game, args)
        alpha_zero.learn()


ModelTrainer()  # Start training the model