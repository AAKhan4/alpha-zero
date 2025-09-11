import os
import random
import numpy as np
from core.mcts.mcts import MCTS
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple

class AlphaZero:
    def __init__(self, model, optimizer, game, args: Dict):
        # Initialize AlphaZero with model, optimizer, game, and configuration arguments
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def self_play(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        # Perform self-play to generate training data
        ret_mem = []  # Memory to store game data
        player = 1  # Start with player 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]  # Parallel games

        while spGames:
            # Collect states and perform MCTS for all games
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            self.mcts.parallel_search(neutral_states, spGames)

            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]
                mcts_probs = self.calc_mcts_probs(spg)  # Compute MCTS probabilities
                spg.mem.append((spg.root.state, mcts_probs, player))

                action = self.sample_action(mcts_probs)  # Sample action based on MCTS probabilities
                spg.state = self.game.get_next_state(spg.state, action, player)
                val, terminal = self.game.is_terminal(spg.state, action)

                if terminal:
                    # Backpropagate results and remove finished games
                    self.backpropagate(spg, val, player, ret_mem)
                    spGames.pop(i)

            player = self.game.get_opponent(player)  # Switch player
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
            with torch.no_grad():
                for _ in tqdm(range(self.args["num_self_play"] // self.args["num_parallel_games"])):
                    mem.extend(self.self_play())  # Generate self-play data

            self.model.train()  # Set model to training mode
            epoch_losses = []
            for _ in tqdm(range(self.args["num_epochs"])):
                avg_loss = self.train(mem)  # Train on self-play data
                epoch_losses.append(avg_loss)

            # Save model and training losses
            self.save_model(i)
            self.save_losses(i, epoch_losses)

    def calc_mcts_probs(self, spg) -> np.ndarray:
        # Compute MCTS probabilities for actions
        mcts_probs = np.zeros(self.game.action_size)
        for child in spg.root.children:
            mcts_probs[child.action] = child.visit_count
        return mcts_probs / np.sum(mcts_probs)

    def sample_action(self, mcts_probs: np.ndarray) -> int:
        # Sample an action based on MCTS probabilities and temperature
        action_probs = mcts_probs ** (1 / self.args["temperature"])
        action_probs /= np.sum(action_probs) if np.sum(action_probs) > 0 else 1
        return np.random.choice(self.game.action_size, p=action_probs)

    def backpropagate(self, spg, val: float, player: int, ret_mem: List):
        # Backpropagate game results to update memory
        for hist_state, hist_probs, hist_player in spg.mem:
            hist_outcome = val if hist_player == player else self.game.get_opponent_val(val)
            ret_mem.append((
                self.game.get_encoded_state(hist_state),
                hist_probs,
                hist_outcome
            ))

    def prepare_batch(self, state, pol_targets, val_targets):
        # Prepare batch data for training
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
        pol_targets = torch.tensor(np.array(pol_targets), dtype=torch.float32, device=self.model.device)
        val_targets = torch.tensor(np.array(val_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
        return state, pol_targets, val_targets

    def calc_loss(self, out_pol, pol_targets, out_val, val_targets):
        # Compute combined policy and value loss
        policy_loss = F.kl_div(torch.log_softmax(out_pol, dim=1), pol_targets, reduction="batchmean")
        value_loss = F.mse_loss(out_val, val_targets)
        return policy_loss + value_loss

    def save_model(self, iteration: int):
        # Save model and optimizer state to disk
        model_dir = os.path.join("./models", f"{self.game}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{iteration}.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, f"optimizer_{iteration}.pth"))

    def save_losses(self, iteration: int, epoch_losses: List[float]):
        # Save training losses to a file
        loss_file = os.path.join("./models", f"{self.game}", f"loss_{iteration}.txt")
        with open(loss_file, "w") as f:
            f.writelines(f"{loss}\n" for loss in epoch_losses)

class SPG:
    def __init__(self, game):
        # Initialize a self-play game instance
        self.state = game.get_initial_state()  # Initial game state
        self.mem = []  # Memory for storing game history
        self.root, self.node = None, None  # MCTS root and current node
