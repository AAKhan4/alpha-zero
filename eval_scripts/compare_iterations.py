import multiprocessing
import os
import argparse
from time import time
import torch
import numpy as np

from core.mcts.res_net import ResNet
from games.base_game import BaseGame, GameState
from games.go.go import Go, GoState
from games.tic_tac_toe.tic_tac_toe import TicTacToe, TicTacToeState
from games.connect_four.connect_four import ConnectFour, ConnectFourState
from training_scripts.training_args import TrainingArgsBuilder

class IterationCompare():
    def __init__(self, game: BaseGame = None, args=None, model: str = None, num_games: int = 100, vs_sl: bool = False):

        game = game if game else Go()

        args_builder = TrainingArgsBuilder(game)
        args = args if args else args_builder.build_args(game)

        print(f"\nComparing iterations {model} on {game} with args: {args}\n")

        start_time = time()
        self.run(game, args, model=model, num_games=num_games, vs_sl=vs_sl)
        end_time = time()

        time_taken = end_time - start_time

        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nComparison completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}s")

    def run(self, game: BaseGame, args: dict, model: str = None, num_games: int = 100, vs_sl: bool = False):
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        models = self.get_models(model, game, args) # Load models for comparison

        if models is None:
            raise ValueError("At least one model must be specified for comparison.")

        results = self.play_games(game, models, args, num_games=num_games, vs_sl=vs_sl)
        
        for idx, i in enumerate([1, 5, 10, 15, 20, 25]):
            win_rate = results[i][1] / num_games * 100
            print(f"Win Rate of iteration {idx} : {win_rate:.2f}%\n")
            self.save_results(num_games, i, model, results[idx])
    
    def save_results(self, num_games: int, model_idx: int, model_type: str, results: dict):
        results_dir = os.path.join("./evaluation/compare_iterations", model_type)
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f"iteration_{model_idx}_results.txt")
        with open(result_file, "w") as f:
            f.write(f"Results of iteration {model_idx}:\n")
            f.write(f"Wins: {results[1]}\n")
            f.write(f"Losses: {results[-1]}\n")
            f.write(f"Draws: {results[0]}\n")
            f.write(f"Win Rate: {results[1] / num_games * 100:.2f}%\n")
        print(f"Results saved to {result_file}\n")

    def get_models(self, model_type: str, game: BaseGame, args: dict) -> list[ResNet]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        if model_type is None:
            raise ValueError("Model parameter cannot be None when loading models.")
        else:
            # Load specified model
            model_path = os.path.join("./models", str(game), model_type)
            for f in os.listdir(model_path):
                if f.startswith("model_") and f.endswith(".pth") and (int(f[6]) == 1 or int(f[6]) % 5 == 0):
                    model = ResNet(game, args['res_blocks'], args['channels'], device=device)
                    model.load_state_dict(torch.load(os.path.join(model_path, f), map_location=device))
                    model.eval()
                    models.append(model)

        return models
    
    def get_sl_model(self, game: BaseGame, args: dict) -> ResNet:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(game, args['res_blocks'], args['channels'], device=device)
        sl_model_path = os.path.join("./models", str(game), "sl", "model_1.pth")
        
        model_files = [f for f in os.listdir(sl_model_path) if f.startswith("model_") and f.endswith(".pth")]

        if model_files:
            latest_model = max(model_files, key=lambda x: x[6])
            model.load_state_dict(torch.load(os.path.join(sl_model_path, latest_model)))
            model.eval()
            print(f"Loaded model: {latest_model}")
        else:
            raise FileNotFoundError("No valid model file found for the first model.")
        return model
    
    def get_model_action(self, game: BaseGame, model: ResNet, game_info: dict):
        policy, _ = model(
            torch.tensor(game.get_encoded_state(game_info["board"]), device=model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
        valid_actions = game.get_valid_actions(game_info)
        policy *= valid_actions
        policy = policy ** 5  # Boost probabilities to favor higher ones
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1
        action = np.random.choice(len(policy), p=policy)  # Sample action based on policy
        return action
    
    def get_random_action(self, game: BaseGame, game_info: dict):
        valid_actions = game.get_valid_actions(game_info)
        action = np.random.choice(np.where(valid_actions == 1)[0])
        return action
    
    def handle_non_terminal(self, game: BaseGame, game_state: GameState):
        if not terminal:
            game_info = game_state.get_info()
            
            for _ in range(2):  # Both players pass to end the game
                game_info = game.get_next_state(game_info, game.row_count * game.col_count)  # Pass action
                game_state.update(game_info)
                val, terminal = game.is_terminal(game_info)
                if terminal:
                    return val
        return 0  # Count draw if maxed moves & error reaching terminal state
    
    def play_games(self, game: BaseGame, models: list[ResNet], args: dict, num_games: int = 100, vs_sl: bool = False) -> dict:
        sl_model = None
        if vs_sl:
            sl_model = self.get_sl_model(game, args)
            models.append(sl_model)
        
        with multiprocessing.Pool(processes=len(models)) as pool:
            tasks = []
            for i, model in enumerate(models):
                for j in range(num_games):
                    flip_res = (j % 2 == 1)  # Alternate who goes first
                    tasks.append(pool.apply_async(self.game_loop_worker, args=(game, model, sl_model if vs_sl else None, flip_res, i)))
            
            results = {i: {1: 0, -1: 0, 0: 0} for i in range(len(models))}

            for task in tasks:
                idx, val = task.get()
                if val > 0:
                    results[idx][1] += 1
                elif val < 0:
                    results[idx][-1] += 1  # Loss for this model
                else:
                    results[idx][0] += 1  # Draw
        return results

    
    def game_loop_worker(self, game: BaseGame, model: ResNet, sl_model: ResNet = None, flip_res: bool = False, idx: int = 0):
        state_map = {TicTacToe: TicTacToeState,
                     ConnectFour: ConnectFourState,
                     Go: GoState}
        game_state: GameState = state_map[type(game)]()

        for _ in range(80): # Max moves to prevent infinite loops or long stalling games
            game_info = game_state.get_info()
            if (game_info["perspective"] == 1) ^ flip_res:
                action = self.get_model_action(game, model, game_info)
            else:
                if sl_model:
                    action = self.get_model_action(game, sl_model, game_info)
                else:
                    action = self.get_random_action(game, game_info)

            game_info = game.get_next_state(game_info, action)
            game_state.update(game_info)
            val, terminal = game.is_terminal(game_info)

            if terminal:
                val = val if not flip_res else -val
                return idx, val
        
        val = self.handle_non_terminal(game, game_state)
        val = val if not flip_res else -val
        return idx, val
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different iterations of trained models.")
    parser.add_argument("--game", type=str, choices=["tic_tac_toe", "connect_four", "go"], default="go", help="The game to use for comparison.")
    parser.add_argument("--model", type=str, choices=["rl", "sl+rl"], required=True, help="The model iteration identifier to load (e.g., 'iteration_10').")
    parser.add_argument("--num_games", type=int, default=30, help="Number of games to play for comparison.")
    parser.add_argument("--vs_sl", action="store_true", help="Whether to compare against a supervised learning model.")
    
    args = parser.parse_args()

    game_map = {
        "tic_tac_toe": TicTacToe,
        "connect_four": ConnectFour,
        "go": Go
    }

    game_instance = game_map[args.game]()

    IterationCompare(game=game_instance, model=args.model, num_games=args.num_games, vs_sl=args.vs_sl)