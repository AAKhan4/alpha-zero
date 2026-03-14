import math
import os
import random
import numpy as np
from core.mcts.mcts import MCTS
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from core.mcts.node import Node
from core.spg import SPG
from games.base_game import BaseGame
import multiprocessing

from core.mcts.res_net import ResNet

REPLAY_BUFFER_SIZE = 4000  # Maximum size of the replay buffer to store self-play data


class AlphaZero:
    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, game: BaseGame, args: Dict):
        """Initialize AlphaZero with model, optimizer, game, and configuration arguments"""
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.replay_buffer_size = args.get("replay_buffer_size", REPLAY_BUFFER_SIZE)

    def pretrain(self, data: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
        """Pretrain the model using human-gameplay data"""
        random.shuffle(data)  # Shuffle data for training

        batch_losses = []
        for i in range(0, len(data), self.args["batch_size"]):
            # Process batches of training data
            batch = data[i:i + self.args["batch_size"]]
            state, action = zip(*batch)
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            state = state.reshape((state.size(0), 3, self.game.row_count, self.game.col_count))  # Reshape for Go 9x9

            # Forward pass and compute loss
            out_pol, _ = self.model(state)
            
            out_policy = torch.softmax(out_pol, dim=1).detach().cpu().numpy()
            target_policy = np.zeros_like(out_policy)
            for idx, act in enumerate(action):
                target_policy[idx][act] = 0.85  # Set target probability for the correct action
                others = np.where(np.arange(out_policy.shape[1]) != act)[0]
                target_policy[idx][others] = 0.15 / len(others)  # Distribute remaining probability among all actions
            target_pol = torch.tensor(target_policy, dtype=torch.float32, device=self.model.device)


            loss = F.kl_div(torch.log_softmax(out_pol, dim=1), target_pol, reduction="batchmean")  # KL divergence for policy loss
            batch_losses.append(loss.item())
            # Backward pass and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
        
        # Return average loss for the epoch
        return float(np.mean(batch_losses)) if batch_losses else 0.0
    
    def self_play(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Perform self-play to generate training data"""
        ret_mem = []  # Memory to store game data
        games = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]  # Parallel games
        while games:
            # Conduct MCTS searches for all active games
            self.mcts.search(games)

            for i in range(len(games))[::-1]:
                spg = games[i]
                game_info = spg.root.game_state.get_info()
                
                val, terminal = self.game.is_terminal(game_info)
                val /= abs(val) if val != 0 else 1  # Normalize value to [-1, 1]
                if terminal:
                    # Backpropagate results and remove finished games
                    self.backpropagate(spg, game_info["player"], val, ret_mem)
                    games.pop(i)
                    continue
                
                mcts_probs = self.calc_mcts_probs(spg)  # Compute MCTS probabilities
                board_state = game_info["board"]  # Use original shape for compatibility with TicTacToe and ConnectFour
                spg.mem.append((board_state, mcts_probs, game_info["player"]))  # Store state, probs, player for training

                action = self.sample_action(mcts_probs, game_info)  # Sample action based on MCTS probabilities
                new_root = spg.root.children.get(action)  # Move down the tree to the chosen action
                spg.root = None
                if new_root is None:
                    raise ValueError("Failed to move down the tree to the chosen action. Action: {}, State info: {}".format(action, game_info))
                spg.node = None
                new_root.make_root() # Set the new root for the next iteration
                spg.root = new_root  # Set new root for the next iteration

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory

        return ret_mem

    def train(self, mem: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
        """Train the model using the generated memory from self-play"""
        random.shuffle(mem)  # Shuffle memory for training
        batch_losses = []

        for i in range(0, len(mem), self.args["batch_size"]):
            # Process batches of training data
            batch = mem[i:i + self.args["batch_size"]]
            states, pol_targets, val_targets = zip(*batch)
            states, pol_targets, val_targets = np.stack(states), np.stack(pol_targets), np.array(val_targets)
            states, pol_targets, val_targets = self.prepare_batch(states, pol_targets, val_targets)

            # Forward pass and compute loss
            out_pol, out_val = self.model(states)
            loss = self.calc_loss(out_pol, pol_targets, out_val, val_targets)
            batch_losses.append(loss.item())

            # Backward pass and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory

        # Return average loss for the epoch
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def supervised_learning(self, pretraining_dir: str) -> None:
        """Supervised learning from pretraining data"""
        pretraining_data = self.prepare_pretraining_data(pretraining_dir)

        pretraining_losses = []

        self.model.train()
        for i in tqdm(range(self.args["pretraining_epochs"]), desc="Supervised Training"):
            avg_loss = self.pretrain(pretraining_data)
            pretraining_losses.append(avg_loss)
            self.save_model(i, flag="sl")
            self.save_losses(i, pretraining_losses, flag="sl")

        print(f"Supervised Training Final Loss: {avg_loss}\n")


    def reinforcement_learning(self, flag: str = "rl", pretraining_dir: str = None) -> None:
        """AlphaZero reinforcement learning loop"""
        if flag != "rl" and pretraining_dir:
            self.supervised_learning(pretraining_dir)

        replay_buffer = []  # Buffer to store self-play data for training
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

            print(f"Starting self-play with {len(batch_args)} batches of games...")

            with torch.no_grad():
                with multiprocessing.Pool(processes=self.args["num_workers"]) as pool:
                    results = []
                    with tqdm(total=len(batch_args), desc="Self-play") as pbar:
                        for batch in pool.imap_unordered(self_play_worker, batch_args):
                            # Process self-play results
                            results.append(batch)
                            pbar.update(1)
                    mem = [item for sublist in results for item in sublist]

            replay_buffer.extend(mem)  # Add new self-play data to the replay buffer
            if len(replay_buffer) > self.replay_buffer_size:  # Limit replay buffer size
                replay_buffer = replay_buffer[-self.replay_buffer_size:]
            print(f"Self-play completed with {len(mem)} samples. Training on {len(replay_buffer)} samples.\n")

            self.model.train()  # Set model to training mode
            epoch_losses = []
            for _ in tqdm(range(self.args["num_epochs"]), desc="Training"):
                avg_loss = self.train(replay_buffer)  # Train on self-play data
                epoch_losses.append(avg_loss)

            print(f"Current Loss: {avg_loss}\n")

            # Save model and training losses
            self.save_model(i, flag=flag)
            self.save_losses(i, epoch_losses, flag=flag)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory

    def calc_mcts_probs(self, spg: SPG) -> np.ndarray:
        """Compute MCTS probabilities for actions"""
        temp = self.get_temp(spg.root.game_state.get_info())  # Get temperature based on game progress

        mcts_probs = np.zeros(self.game.action_size)
        if not spg.root.children:
            raise ValueError("MCTS failed to expand any children for the current game state. State info: {}".format(spg.root.game_state.get_info()))

        for action, child in spg.root.children.items():
            mcts_probs[action] = child.visit_count + 1  # Use visit counts as probabilities

        # Normalize probabilities
        total_visits = np.sum(mcts_probs)
        mcts_probs = (mcts_probs / total_visits)

        # Apply temperature to control exploration
        mcts_probs = np.power(mcts_probs, 1 / temp)  # Apply temperature
        mcts_probs /= np.sum(mcts_probs)  # Re-normalize after applying temperature

        return mcts_probs
    
    def get_temp(self, game_info: dict) -> float:
        """Calculate temperature based on game progress for action selection"""
        temp = self.args["init_temperature"]
        if temp > self.args["temp_floor"]:
            num_moves = np.sum(game_info["board"] != 0) if not game_info.get("action_count") else game_info["action_count"]
            temp = temp - (self.args["temp_decay"] * (num_moves // self.args["temp_threshold"]))
        return max(temp, self.args["temp_floor"])  # Ensure temperature doesn't go below temp_floor

    def sample_action(self, mcts_probs: np.ndarray, game_info: dict) -> int:
        """Sample an action based on MCTS probabilities and temperature"""

        # Sample an action based on MCTS probabilities and temperature
        if np.any(np.isnan(mcts_probs)):
            raise ValueError("MCTS probabilities contain NaN values: current probs: {}".format(mcts_probs))
        return np.random.choice(self.game.action_size, p=mcts_probs)

    def backpropagate(self, spg: SPG, player: int, val: float, ret_mem: List) -> None:
        """Backpropagate game results to update memory"""
        for hist_state, hist_prob, hist_player in spg.mem:
            out = val if hist_player == player else self.game.get_opponent_val(val)
            ret_mem.append((hist_state, hist_prob, out))  # Store state, MCTS probabilities, and value from player's perspective
            # print("Backprop: state: \n{}, prob: {}, player: {}, val: {}".format(hist_state, hist_prob, hist_player, out))

    def prepare_batch(self, state: np.ndarray, pol_targets: np.ndarray, val_targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for training"""
        state = torch.tensor(self.game.get_encoded_state(state), dtype=torch.float32, device=self.model.device)
        pol_targets = torch.tensor(pol_targets, dtype=torch.float32, device=self.model.device)
        val_targets = torch.tensor(val_targets.reshape(-1, 1), dtype=torch.float32, device=self.model.device)
        return state, pol_targets, val_targets

    def calc_loss(self, out_pol: torch.Tensor, pol_targets: torch.Tensor, out_val: torch.Tensor, val_targets: torch.Tensor) -> torch.Tensor:
        """Compute combined policy and value loss"""
        pol_targets /= pol_targets.sum(dim=1, keepdim=True)  # Normalize policy targets
        policy_loss = F.kl_div(torch.log_softmax(out_pol, dim=1), pol_targets, reduction="batchmean")  # KL divergence for policy loss
        value_loss = F.mse_loss(out_val, val_targets)
        total_loss = policy_loss + value_loss  # Combine policy and value losses

        return total_loss

    def save_model(self, iteration: int, flag: str = None) -> None:
        """Save model and optimizer state to disk"""
        model_dir = os.path.join("./models", f"{self.game}", f"{flag}" if flag else "")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{iteration}.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, f"optimizer_{iteration}.pth"))

    def save_losses(self, iteration: int, epoch_losses: List[float], flag: str = None) -> None:
        """Save training losses to a file"""
        loss_file = os.path.join("./models", f"{self.game}", f"{flag}" if flag else "", "losses", f"loss_{iteration}.txt")
        os.makedirs(os.path.dirname(loss_file), exist_ok=True)
        with open(loss_file, "w") as f:
            f.writelines(f"{loss}\n" for loss in epoch_losses)
    
    def prepare_pretraining_data(self, pretraining_dir: str) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Prepare pretraining data from the specified directory"""
        states = np.load(os.path.join(pretraining_dir, "states.npy"))
        actions = np.load(os.path.join(pretraining_dir, "actions.npy"))

        pretraining_data = list(zip(states, actions))
        return pretraining_data


def self_play_worker(args: dict) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Worker function for self-play in multiprocessing"""
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
