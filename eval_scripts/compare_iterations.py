import multiprocessing
import os
import argparse
from time import time
import torch
import numpy as np
from tqdm import tqdm

from core.mcts.res_net import ResNet
from games.base_game import BaseGame, GameState
from games.go.go import Go, GoState
from games.tic_tac_toe.tic_tac_toe import TicTacToe, TicTacToeState
from games.connect_four.connect_four import ConnectFour, ConnectFourState
from training_scripts.training_args import TrainingArgsBuilder

class IterationCompare():
    def __init__(self, game: BaseGame = None, args=None, model_type: str = None, num_games: int = 100, opponent: str = "rand"):
        game = game if game else Go()

        args_builder = TrainingArgsBuilder(game)
        args = args if args else args_builder.build_args(game)

        print(f"\nComparing all iterations of {model_type} against {opponent} on {game}\n")

        start_time = time()
        self.run(game, args, model_type=model_type, num_games=num_games, opponent=opponent)
        end_time = time()

        time_taken = end_time - start_time

        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nComparison completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}s")

    def run(self, game: BaseGame, args: dict, model_type: str = None, num_games: int = 100, opponent: str = "rand") -> None:
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        model_paths = self.get_model_paths(model_type, game)

        for model_path in model_paths:
            model_name = os.path.basename(model_path).split(".")[0]
            print(f"Comparing {model_name} against {opponent}...")

            results = self.play_games(game, model_path, num_games=num_games, args=args, opponent=opponent)

            win_rate = results[1] / sum(results.values()) * 100 if sum(results.values()) > 0 else 0
            print(f"{model_name} Win Rate: {win_rate:.2f}%")
            self.save_results(game, model_name, results)

    def get_model_paths(self, model_type: str, game: BaseGame) -> list:
        model_dir = os.path.join("./models", str(game), model_type)
        model_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.startswith("model_") and f.endswith(".pth")
        ]
        if not model_files:
            raise FileNotFoundError(f"No models found for type {model_type} in {model_dir}")
        return sorted(model_files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    def get_latest_sl_model_path(self, game: BaseGame) -> str:
        model_dir = os.path.join("./models", str(game), "sl")
        model_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.startswith("model_") and f.endswith(".pth")
        ]
        if not model_files:
            raise FileNotFoundError(f"No SL models found in {model_dir}")
        return sorted(model_files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))[-1]

    def play_games(self, game: BaseGame, model_path: str, num_games: int = 100, args: dict = None, opponent: str = "rand") -> dict:
        results = {1: 0, -1: 0, 0: 0}  # Initialize win/draw counters

        worker_args = [
            {
                'model_path': model_path,
                'game': game,
                'flip': (i % 2 == 1),  # Alternate starting positions
                'opponent': opponent
            }
            for i in range(num_games)
        ]

        with multiprocessing.Pool(processes=args["num_workers"]) as pool:
            with tqdm(total=num_games, desc=f"Playing games for {os.path.basename(model_path)}") as pbar:
                for result in pool.imap_unordered(self.game_loop_worker, worker_args):
                    if result > 0:
                        results[1] += 1
                    elif result < 0:
                        results[-1] += 1
                    else:
                        results[0] += 1
                    pbar.update(1)

        return results

    def save_results(self, game: BaseGame, model_name: str, results: dict) -> None:
        results_dir = os.path.join("./evaluation", "compare_iterations", str(game), model_name)
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "compare_iterations_results.txt")
        win_rate = results[1] / sum(results.values()) * 100 if sum(results.values()) > 0 else 0
        with open(result_file, "a") as f:
            f.write(f"{win_rate:.2f}%\n")
        print(f"Results saved to {result_file}\n")

    def game_loop_worker(self, worker_args) -> int:
        '''Worker function for playing a single game.'''
        game: BaseGame = worker_args['game']
        model_1 = self.load_model(worker_args['model_path'], game) if worker_args['model_path'] else None
        model_2 = self.load_model(self.get_latest_sl_model_path(game), game) if worker_args['opponent'] == "sl" else None
        flip_res = 1  # Used to flip the result when models are swapped
        if worker_args['flip']:
            model_1, model_2 = model_2, model_1  # Swap models for changing starting positions
            flip_res = -1  # Model 2 starts first

        game_state: GameState = game.get_state_type()(game)  # Initialize the game state

        terminal = False
        while not terminal:  # Max 70 turns to prevent infinite loops
            game_info = game_state.get_info()

            if game_info["player"] == 1:
                action = self.get_model_action(game, model_1, game_info)
            else:
                action = self.get_model_action(game, model_2, game_info)

            game_info = game.get_next_state(game_info, action)
            game_state.update(game_info)

            val, terminal = game.is_terminal(game_info)
            if terminal:
                return val * game_info["player"] * flip_res  # Return result from the perspective of model_1

    def load_model(self, model_path: str, game: BaseGame) -> ResNet:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = TrainingArgsBuilder(game).build_args(game)
        model = ResNet(game, args["res_blocks"], args["channels"], device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def get_valid_action(self, game: BaseGame, game_info: dict, policy: np.ndarray, az: bool = False) -> int:
        '''Get a random valid action from the game info.'''
        valid_actions = game.get_valid_actions(game_info)
        policy = policy * valid_actions  # Zero out invalid actions & sharpen the distribution
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1
        action = None
        while action is None:
            candidate = np.argmax(policy)  # Sample action based on policy
            if game.is_valid_action(game_info, candidate):
                action = candidate
            else:
                policy[candidate] = 0  # Zero out invalid action and renormalize
                policy /= np.sum(policy) if np.sum(policy) > 0 else 1
        return action

    def get_random_action(self, game: BaseGame, game_info: dict) -> int:
        '''Get a random valid action from the game info.'''
        valid_actions = game.get_valid_actions(game_info)
        policy = valid_actions / np.sum(valid_actions)
        action = None
        while action is None:
            candidate = np.random.choice(len(policy), p=policy)  # Sample action based on policy
            if game.is_valid_action(game_info, candidate):
                action = candidate
            else:
                policy[candidate] = 0  # Zero out invalid action and renormalize
                policy /= np.sum(policy) if np.sum(policy) > 0 else 1
        return action

    def get_model_action(self, game: BaseGame, model: ResNet, game_info: dict) -> int:
        '''Get the action from the model based on the current game state.'''
        az = False
        if model is None:
            action = self.get_random_action(game, game_info)
        else:
            policy, _ = model(
                torch.tensor(game.get_encoded_state(game_info["board"]), device=model.device).unsqueeze(0)
            )
            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
            action = self.get_valid_action(game, game_info, policy, az)

        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all iterations of a model type against random.")
    parser.add_argument("--game", type=str, choices=["tic_tac_toe", "connect_four", "go"], default="go", help="The game to use for comparison.")
    parser.add_argument("--model", type=str, choices=["sl", "rl", "sl+rl"], required=True, help="The model type to compare (e.g., 'rl').")
    parser.add_argument("--vs", type=str, choices=["sl", "rand"], default="rand", help="The opponent to play against.")
    parser.add_argument("--num_games", type=int, default=500, help="Number of games to play per model iteration.")
    args = parser.parse_args()

    game_map = {
        "tic_tac_toe": TicTacToe,
        "connect_four": ConnectFour,
        "go": Go
    }

    selected_game = game_map[args.game]()

    IterationCompare(game=selected_game, model_type=args.model, num_games=args.num_games, opponent=args.vs)