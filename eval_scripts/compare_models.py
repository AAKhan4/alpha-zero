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

class ModelCompare():
    def __init__(self, game: BaseGame = None, args=None, model_1: str = None, model_2: str = None, num_games: int = 100):

        game = game if game else Go()

        args_builder = TrainingArgsBuilder(game)
        args = args if args else args_builder.build_args(game)

        print(f"\nComparing models {model_1} vs {model_2} on {game}\n")

        start_time = time()
        self.run(game, args, m_1=model_1, m_2=model_2, num_games=num_games)
        end_time = time()

        time_taken = end_time - start_time

        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nComparison completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}s")

    def run(self, game: BaseGame, args: dict, m_1: str = None, m_2: str = None, num_games: int = 100) -> None:
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

        results = self.play_games(game, m_1, m_2, num_games=num_games, args=args)
        
        self.save_results(game, m_1, m_2, results)
        total_games = sum(results.values())
        if total_games > 0:
            winrate = results[1] / total_games * 100
        else:
            winrate = 0
        print(f"Winrate of {m_1} over {m_2}: {winrate:.2f}%\n")
    

    def play_games(self, game: BaseGame, model_1: str, model_2: str = None, num_games: int = 100, args: dict = None) -> dict:
        '''Plays a series of games between two models and returns the results.'''
        results = {1: 0, -1: 0, 0: 0}  # Initialize win/draw counters

        print(f"Starting play_games with {num_games} games...")
        print(f"Game: {type(game).__name__}, Model 1: {model_1}, Model 2: {model_2}")

        worker_args = [
            {
                'model_1': model_1,
                'model_2': model_2,
                'game': game,
                'flip': (i % 2 == 1)  # Flip models for odd iterations to alternate starting positions
            }
            for i in range(num_games) # Create argument list for each game to be played in parallel
        ]

        with multiprocessing.Pool(processes=args["num_workers"]) as pool:
            with tqdm(total=num_games, desc="Playing games") as pbar:
                for result in pool.imap_unordered(game_loop_worker, worker_args):
                    if result > 0:
                        results[1] += 1
                    elif result < 0:
                        results[-1] += 1
                    else:
                        results[0] += 1
                    pbar.update(1)

        return results
    
    def save_results(self, game: BaseGame, model_1: str, model_2: str, results: dict) -> None:
        results_dir = f"./evaluation/compare_models/{game}"
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f"{model_1}_vs_{model_2}_results.txt")
        with open(result_file, "w") as f:
            f.write(f"Results of {model_1} vs {model_2}:\n")
            f.write(f"Model 1 Wins: {results[1]}\n")
            f.write(f"Model 2 Wins: {results[-1]}\n")
            f.write(f"Draws: {results[0]}\n")
            total_games = sum(results.values())
            if total_games > 0:
                win_rate = results[1] / total_games * 100
            else:
                win_rate = 0
            f.write(f"Win Rate of {model_1} over {model_2}: {win_rate:.2f}%\n")
        print(f"Results saved to {result_file}\n")


def get_model(m: str, game: BaseGame, args: dict) -> ResNet:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if m == "rand":
            return None  # No model, will use random actions

        model = ResNet(game, args["res_blocks"], args["channels"], device=device)
        model_path = os.path.join("models", str(game), m)
        model_files = [f for f in os.listdir(model_path) if f.startswith("model_") and f.endswith(".pth")]

        if model_files:
            latest_model = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))  # Get the model with the highest iteration number
            model.load_state_dict(torch.load(os.path.join(model_path, latest_model)))
        else:
            raise FileNotFoundError("No valid model file found for the first model.")

        model.eval()  # Set model to evaluation mode

        return model


def game_loop_worker(worker_args) -> int:
    '''Worker function for playing a single game.'''
    game: BaseGame = worker_args['game']
    model_1 = get_model(worker_args['model_1'], game, TrainingArgsBuilder(game).build_args(game)) if worker_args['model_1'] else None
    model_2 = get_model(worker_args['model_2'], game, TrainingArgsBuilder(game).build_args(game)) if worker_args['model_2'] else None
    flip_res = 1  # Used to flip the result when models are swapped
    if worker_args['flip']:
        model_1, model_2 = model_2, model_1  # Swap models for changing starting positions
        flip_res = -1  # Model 2 starts first

    state_map = {TicTacToe: TicTacToeState,
                 ConnectFour: ConnectFourState,
                 Go: GoState}
    game_state: GameState = state_map[type(game)](game)  # Initialize game state based on the game type

    terminal = False
    while not terminal:  # Max 70 turns to prevent infinite loops
        game_info = game_state.get_info()

        if game_info["player"] == 1:
            action = get_model_action(game, model_1, game_info)
        else:
            action = get_model_action(game, model_2, game_info)

        game_info = game.get_next_state(game_info, action)
        game_info = game.change_perspective(game_info)  # Switch perspective for the next turn
        game_state.update(game_info)

        val, terminal = game.is_terminal(game_info)
        if terminal:
            return val * game_info["player"] * flip_res  # Return result from the perspective of model_1
        

def get_valid_action(game: BaseGame, game_info: dict, policy: np.ndarray, az: bool = False) -> int:
    '''Get a random valid action from the game info.'''
    valid_actions = game.get_valid_actions(game_info)
    policy = policy * valid_actions  # Zero out invalid actions & sharpen the distribution
    policy /= np.sum(policy) if np.sum(policy) > 0 else 1
    action = None
    while action is None:
        candidate = np.random.choice(len(policy), p=policy)  # Sample action based on policy
        if game.is_valid_action(game_info, candidate):
            action = candidate
        else:
            policy[candidate] = 0  # Zero out invalid action and renormalize
            policy /= np.sum(policy) if np.sum(policy) > 0 else 1
    return action

def get_model_action(game: BaseGame, model: ResNet, game_info: dict) -> int:
    '''Get the action from the model based on the current game state.'''
    az = False
    if model is None:
        policy = np.ones(game.action_size) / game.action_size  # Uniform random policy
    else:
        policy, _ = model(
            torch.tensor(game.get_encoded_state(game_info["board"]), device=model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()

    action = get_valid_action(game, game_info, policy, az)
    return action
    

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Compare two trained models using self-play.")
    parser.add_argument("--game", type=str, choices=["tic_tac_toe", "connect_four", "go"], default="go", help="The game to use for comparison.")
    parser.add_argument("--model_1", type=str, choices=["rand", "sl", "rl", "sl+rl"], required=True, help="The identifier for the first model (or 'rand' for random).")
    parser.add_argument("--model_2", type=str, choices=["rand", "sl", "rl", "sl+rl"], default="rand", help="The identifier for the second model (or 'rand' for random).")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to play for comparison.")
    args = parser.parse_args()

    game_map = {
        "tic_tac_toe": TicTacToe,
        "connect_four": ConnectFour,
        "go": Go
    }

    selected_game = game_map[args.game]()

    ModelCompare(game=selected_game, model_1=args.model_1, model_2=args.model_2, num_games=args.num_games)
        