import math
import os
import random
import numpy as np
from core.mcts.mcts import MCTS
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from core.spg import SPG
from games.go import Go, GoState
import multiprocessing

from core.mcts.res_net import ResNet


class AlphaZero:
    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, game: Go, args: Dict):
        # Initialize AlphaZero with model, optimizer, game, and configuration arguments
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def self_play(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        # Perform self-play to generate training data
        ret_mem = []  # Memory to store game data
        games = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]  # Parallel games

        while games:
            # Conduct MCTS searches for all active games
            self.mcts.search(games)

            for i in range(len(games) - 1, -1, -1):
                spg = games[i]
                game_info = spg.game_state
                mcts_probs = self.calc_mcts_probs(spg)  # Compute MCTS probabilities
                spg.mem.append((spg.root.state, mcts_probs, spg.game_state.perspective))  # Store state, probs, player

                action = self.sample_action(mcts_probs, spg.game_state.state)  # Sample action based on MCTS probabilities
                game_info = self.game.get_next_state(spg.game_state, action)
                val, terminal = self.game.is_terminal(game_info, action)
                val /= abs(val) if val != 0 else 1  # Normalize value

                if terminal:
                    # Backpropagate results and remove finished games
                    self.backpropagate(spg, val, game_info["perspective"], ret_mem)  # CHECK IF CORRECT
                    games.pop(i)

                game_info = self.game.change_perspective(game_info)
                spg.game_state.update(game_info)
        return ret_mem

    def train(self, mem: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
        # Train the model using the generated memory
        random.shuffle(mem)  # Shuffle memory for training
        batch_losses = []

        for i in range(0, len(mem), self.args["batch_size"]):
            # Process batches of training data
            batch = mem[i:i + self.args["batch_size"]]
            state, pol_targets, val_targets = zip(*batch)
            state, pol_targets, val_targets = self.prepare_batch(state, pol_targets, val_targets)

            # Forward pass and compute loss
            out_pol, out_val = self.model(state)
            loss = self.calc_loss(out_pol, pol_targets, out_val, val_targets)
            batch_losses.append(loss.item())

            # Backward pass and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Return average loss for the epoch
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def learn(self):
        # Main learning loop for AlphaZero
        for i in range(self.args["num_iterations"]):
            mem = []  # Memory for self-play data
            self.model.eval()  # Set model to evaluation mode

            print(f"Model {i+1}\n")

            # Prepare arguments for worker processes running self-play
            sp_args = {
                "model_dict": self.model.state_dict(),
                "optimizer_dict": self.optimizer.state_dict(),
                "game": self.game,
                "args": self.args
            }

            # Use multiprocessing to perform self-play in parallel
            num_batches = max(self.args["num_workers"], math.ceil(self.args["num_self_play"] / self.args["max_parallel_games"]))
            games_per_batch = self.args["num_self_play"] // num_batches
            extra_games = self.args["num_self_play"] % num_batches

            batch_args = []
            for b in range(num_batches):
                batch_size = games_per_batch + (1 if b < extra_games else 0)
                if batch_size > 0:
                    sp_args_batch = sp_args.copy()
                    sp_args_batch["args"] = sp_args["args"].copy()
                    sp_args_batch["args"]["num_parallel_games"] = min(batch_size, self.args["max_parallel_games"])
                    batch_args.append(sp_args_batch)

            with torch.no_grad():
                with multiprocessing.Pool(processes=self.args["num_workers"]) as pool:
                    results = []
                    with tqdm(total=len(batch_args), desc="Self-play") as pbar:
                        for batch in pool.imap_unordered(self_play_worker, batch_args):
                            # Process self-play results
                            results.append(batch)
                            pbar.update(1)
                    mem = [item for sublist in results for item in sublist]

            self.model.train()  # Set model to training mode
            epoch_losses = []
            for _ in tqdm(range(self.args["num_epochs"]), desc="Training"):
                avg_loss = self.train(mem)  # Train on self-play data
                epoch_losses.append(avg_loss)

            print(f"Current Loss: {avg_loss}\n")

            # Save model and training losses
            self.save_model(i)
            self.save_losses(i, epoch_losses)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory

    def calc_mcts_probs(self, spg: SPG) -> np.ndarray:
        # Compute MCTS probabilities for actions
        mcts_probs = np.zeros(self.game.action_size)
        for child in spg.root.children:
            mcts_probs[child.action] = child.visit_count
        return mcts_probs / np.sum(mcts_probs)

    def sample_action(self, mcts_probs: np.ndarray, state: np.ndarray) -> int:
        temp = self.args["init_temperature"]
        if temp > 0.1:
            num_moves = np.sum(state != 0)
            temp = temp - self.args["temp_decay"] * (num_moves // self.args["temp_threshold"])
        temp = max(temp, 0.1)  # Ensure temperature doesn't go below 0.1

        # Sample an action based on MCTS probabilities and temperature
        action_probs = mcts_probs ** (1 / temp)
        action_probs /= np.sum(action_probs) if np.sum(action_probs) > 0 else 1
        return np.random.choice(self.game.action_size, p=action_probs)

    def backpropagate(self, spg: SPG, val: float, player: int, ret_mem: List) -> None:
        # Backpropagate game results to update memory
        for hist_state, hist_probs, hist_player in spg.mem:
            hist_outcome = val if hist_player == player else self.game.get_opponent_val(val)
            ret_mem.append((
                self.game.get_encoded_state(hist_state),
                hist_probs,
                hist_outcome
            ))

    def prepare_batch(self, state: np.ndarray, pol_targets: np.ndarray, val_targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prepare batch data for training
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
        pol_targets = torch.tensor(np.array(pol_targets), dtype=torch.float32, device=self.model.device)
        val_targets = torch.tensor(np.array(val_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
        return state, pol_targets, val_targets

    def calc_loss(self, out_pol: torch.Tensor, pol_targets: torch.Tensor, out_val: torch.Tensor, val_targets: torch.Tensor) -> torch.Tensor:
        # Compute combined policy and value loss
        policy_loss = F.kl_div(torch.log_softmax(out_pol, dim=1), pol_targets, reduction="batchmean")
        value_loss = F.mse_loss(out_val, val_targets)
        return (policy_loss + value_loss)

    def save_model(self, iteration: int) -> None:
        # Save model and optimizer state to disk
        model_dir = os.path.join("./models", f"{self.game}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{iteration}.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, f"optimizer_{iteration}.pth"))

    def save_losses(self, iteration: int, epoch_losses: List[float]) -> None:
        # Save training losses to a file
        loss_file = os.path.join("./models", f"{self.game}", f"loss_{iteration}.txt")
        with open(loss_file, "w") as f:
            f.writelines(f"{loss}\n" for loss in epoch_losses)


# Worker function for parallel self-play
def self_play_worker(args: dict) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create AlphaZero instance and perform self-play
        model = ResNet(args["game"], args["args"]["res_blocks"], args["args"]["channels"], device)
        model.load_state_dict(args["model_dict"])
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["args"]["lr"], weight_decay=args["args"]["weight_decay"])
        optimizer.load_state_dict(args["optimizer_dict"])
        game = args["game"].__class__()  # Create a new object of the same type as args["game"]
        az = AlphaZero(model, optimizer, game, args["args"])

        # Call self_play() and return the result
        result = az.self_play()  # Ensure result is on CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        return result
