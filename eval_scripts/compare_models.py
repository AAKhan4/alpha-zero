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

class ModelCompare():
    def __init__(self, game: BaseGame = None, args=None, model_1: str = None, model_2: str = None, num_games: int = 100):

        game = game if game else Go()

        args_builder = TrainingArgsBuilder(game)
        args = args if args else args_builder.build_args(game)

        print(f"\nComparing models {model_1} vs {model_2} on {game} with args: {args}\n")

        start_time = time()
        self.run(game, args, model_1=model_1, model_2=model_2, num_games=num_games)
        end_time = time()

        time_taken = end_time - start_time

        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nComparison completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}s")

    def run(self, game: BaseGame, args: dict, m_1: str = None, m_2: str = None, num_games: int = 100):
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        m_vs_rand = None
        model_1 = self.get_model(m_1, game, args) if m_1 != "rand" else None
        model_2 = self.get_model(m_2, game, args) if m_2 != "rand" else None

        if model_1 is None and model_2 is None:
            raise ValueError("At least one model must be specified for comparison.")

        for m in [model_1, model_2]:
            if m:
                m.eval()  # Set model to evaluation mode
            else:
                m_vs_rand = model_1 if model_2 is None else model_2

        if m_vs_rand:
            results = self.play_games_vs_rand(game, m_vs_rand, args, num_games=num_games)
        else:
            results = self.play_games(game, model_1, model_2, args, num_games=num_games)
        
        self.save_results(m_1, m_2, results)
    
    def play_games_vs_rand(self, game: BaseGame, model_1: ResNet, args: dict, num_games: int = 100):
        results = {1: 0, -1: 0, 0: 0}  # Initialize win/draw counters

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            tasks = []
            for i in range(num_games):
                move_first = (i % 2 == 0)
                tasks.append(pool.apply_async(self.game_vs_rand_worker, args=(game, model_1, args, move_first)))

            for task in tasks:
                result = task.get()
                if result > 0:
                    results[1] += 1
                elif result < 0:
                    results[-1] += 1
                else:
                    results[0] += 1

        print(f"Results vs Random:\nModel Wins: {results[1]}, Random Wins: {results[-1]}, Draws: {results[0]}\n")
        return results
    
    def play_games(self, game: BaseGame, model_1: ResNet, model_2: ResNet, args: dict, num_games: int = 100):
        results = {1: 0, -1: 0, 0: 0}  # Initialize win/draw counters

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            tasks = []
            for _ in range(num_games//2):
                tasks.append(pool.apply_async(self.game_loop_worker, args=(game, model_1, model_2, False)))
                tasks.append(pool.apply_async(self.game_loop_worker, args=(game, model_2, model_1, True)))

            for task in tasks:
                result = task.get()
                if result > 0:
                    results[1] += 1
                elif result < 0:
                    results[-1] += 1
                else:
                    results[0] += 1

        return results

    def get_model(self, m: str, game: BaseGame, args: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ResNet(game, args["res_blocks"], args["channels"], device)

        model_path = os.path.join("models", str(game), m)

        model_files = [f for f in os.listdir(model_path) if f.startswith("model_") and f.endswith(".pth")]

        if model_files:
            latest_model = max(model_files, key=lambda x: x[6])
            model.load_state_dict(torch.load(os.path.join(model_path, latest_model)))
            print(f"Loaded model: {latest_model}")
        else:
            raise FileNotFoundError("No valid model file found for the first model.")

        model.eval()  # Set model to evaluation mode

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
    
    def save_results(self, model_1: str, model_2: str, results: dict):
        results_dir = "./evaluation/compare_models"
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f"{model_1}_vs_{model_2}_results.txt")
        with open(result_file, "w") as f:
            f.write(f"Results of {model_1} vs {model_2}:\n")
            f.write(f"Model 1 Wins: {results[1]}\n")
            f.write(f"Model 2 Wins: {results[-1]}\n")
            f.write(f"Draws: {results[0]}\n")
            f.write(f"Win Rate of {model_1} over {model_2}: {results[1] / sum(results.values()) * 100:.2f}%\n")
        print(f"Results saved to {result_file}\n")
    
    
    def game_vs_rand_worker(self, game: BaseGame, model_1: ResNet, move_first: bool = False):
        state_map = {TicTacToe: TicTacToeState,
                        ConnectFour: ConnectFourState,
                        Go: GoState}
        game_state: GameState = state_map[type(game)]()

        for _ in range(80): # Max moves to prevent infinite loops or long stalling games
            game_info = game_state.get_info()
            if game_info["perspective"] == 1:
                action = self.get_model_action(game, model_1, game_info)
            else:
                valid_actions = game.get_valid_actions(game_info)
                action = np.random.choice(np.where(valid_actions == 1)[0])  # Random valid action

            game_info = game.get_next_state(game_info, action)
            game_state.update(game_info)
            val, terminal = game.is_terminal(game_info)

            if terminal:
                return val if move_first else -val
        
        val = self.handle_non_terminal(game, game_state)
        return val if move_first else -val


    def game_loop_worker(self, game: BaseGame, model_1: ResNet, model_2: ResNet, flip_res: bool = False):
        state_map = {TicTacToe: TicTacToeState,
                     ConnectFour: ConnectFourState,
                     Go: GoState}
        game_state: GameState = state_map[type(game)]()

        for _ in range(80): # Max moves to prevent infinite loops or long stalling games
            game_info = game_state.get_info()
            if game_info["perspective"] == 1:
                action = self.get_model_action(game, model_1, game_info)
            else:
                action = self.get_model_action(game, model_2, game_info)

            game_info = game.get_next_state(game_info, action)
            game_state.update(game_info)
            val, terminal = game.is_terminal(game_info)

            if terminal:
                return val if not flip_res else -val
        
        val = self.handle_non_terminal(game, game_state)
        return val if not flip_res else -val
    

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Compare two trained models using self-play.")
    parser.add_argument("--game", type=str, choices=["tic_tac_toe", "connect_four", "go"], default="go", help="The game to use for comparison.")
    parser.add_argument("--model_1", type=str, choices=["rand", "sl", "rl", "sl+rl"], required=True, help="The identifier for the first model (or 'rand' for random).")
    parser.add_argument("--model_2", type=str, choices=["rand", "sl", "rl", "sl+rl"], default="rand", help="The identifier for the second model (or 'rand' for random).")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play for comparison.")
    args = parser.parse_args()

    if args.model_1 == args.model_2:
        raise ValueError("Model 1 and Model 2 must be different for comparison.")
    
    if args.model_1 == "rand":
        args.model_1, args.model_2 = args.model_2, args.model_1  # Ensure model_1 is not random for consistency

    game_map = {
        "tic_tac_toe": TicTacToe,
        "connect_four": ConnectFour,
        "go": Go
    }

    selected_game = game_map[args.game]()

    ModelCompare(game=selected_game, model_1=args.model_1, model_2=args.model_2, num_games=args.num_games)
        