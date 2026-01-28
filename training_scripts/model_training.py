import multiprocessing
import os
import argparse
from time import time
import torch

from core.alpha_zero import AlphaZero
from core.mcts.res_net import ResNet
from games.base_game import BaseGame
from games.go.go import Go
from games.tic_tac_toe.tic_tac_toe import TicTacToe
from games.connect_four.connect_four import ConnectFour
from training_scripts.training_args import TrainingArgsBuilder


class ModelTrainer:
    def __init__(self, game: BaseGame = None, args=None, model_dir: str = None, data_dir: str = None, flag: str = "rl"):

        game = game if game else TicTacToe()

        args_builder = TrainingArgsBuilder(game)
        args = args if args else args_builder.build_args(game)

        print(f"\nTraining on {game} with args: {args}\n")

        start_time = time()
        self.run(game, args, model_dir=model_dir, data_dir=data_dir, flag=flag)
        end_time = time()

        time_taken = end_time - start_time

        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTraining completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}s")

    def run(self, game: BaseGame, args: dict, model_dir: str = None, data_dir: str = None, flag: str = "rl"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        model = ResNet(game, args["res_blocks"], args["channels"], device)  # Initialize the neural network model
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])  # Adam optimizer

        if model_dir:
            model_path = None
            optimizer_path = None

            for f in os.listdir(model_dir):
                if f.startswith("model_") and f.endswith(".pth"):
                    model_path = max(model_path, key=lambda x: int(x[6])) if model_path else os.path.join(model_dir, f)
                elif f.startswith("optimizer_") and f.endswith(".pth"):
                    optimizer_path = max(optimizer_path, key=lambda x: int(x[10])) if optimizer_path else os.path.join(model_dir, f)
                
            if model_path and optimizer_path:
                model.load_state_dict(torch.load(os.path.join(model_dir, model_path)))
                optimizer.load_state_dict(torch.load(os.path.join(model_dir, optimizer_path)))
                print(f"Loaded model from {model_path} and optimizer from {optimizer_path}\n")
            else:
                raise FileNotFoundError("No valid model or optimizer files found in the specified directory.\n")
            
        if data_dir is None or not os.access(data_dir, os.R_OK):
            raise FileNotFoundError(f"Data directory {data_dir} not found or inaccessible.\n")

        alpha_zero = AlphaZero(model, optimizer, game, args)
        if flag == "sl":
            alpha_zero.supervised_learning(data_dir)
        else:
            alpha_zero.reinforcement_learning(flag=flag, pretraining_dir=data_dir)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="AlphaZero Reinforcement Learning Training")
    parser.add_argument(
        "--game",
        type=str,
        choices=["go", "tic_tac_toe", "connect_four"],
        default="tic_tac_toe",
        help="The game to train on (default: tic_tac_toe)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Directory of a pre-trained model to continue training from (default: None)",
    )
    parser.add_argument(
        "--flag",
        type=str,
        choices=["rl", "sl", "sl+rl"],
        default="rl",
        help="Reinforcement learning vs Supervised learning (default: reinforcement learning)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/processed_data/go/9x9",
        help="Data directory for supervised learning (default: ./data/processed_data/go/9x9)",
    )
    args = parser.parse_args()
    game_map = {
        "go": Go,
        "tic_tac_toe": TicTacToe,
        "connect_four": ConnectFour,
    }
    game_instance = game_map[args.game]()
    ModelTrainer(game=game_instance, model_dir=args.model, flag=args.flag, data_dir=args.data) # Start training the model