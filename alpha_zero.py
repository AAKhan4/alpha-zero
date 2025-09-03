import os
import random
import numpy as np
import mcts
import torch
import torch.nn.functional as F
from tqdm import tqdm

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts.MCTS(game, args, model)  # Initialize Monte Carlo Tree Search

    def self_play(self):
        ret_mem = []  # Memory to store game data for training
        player = 1  # Start with player 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]  # Initialize parallel games

        while len(spGames) > 0:
            # Stack states of all games for batch processing
            states = np.stack([spg.state for spg in spGames])

            # Convert states to the perspective of the current player
            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)  # Perform MCTS for all games

            for i in range(len(spGames))[::-1]:  # Iterate in reverse to safely remove finished games
                spg = spGames[i]

                # Compute action probabilities based on visit counts
                mcts_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    mcts_probs[child.action] = child.visit_count
                mcts_probs /= np.sum(mcts_probs)  # Normalize probabilities

                # Store state, action probabilities, and player in memory
                spg.mem.append((spg.root.state, mcts_probs, player))

                # Apply temperature to action probabilities for exploration
                action_probs = mcts_probs ** (1 / self.args["temperature"])
                action_probs /= np.sum(action_probs) if np.sum(action_probs) > 0 else 1
                action = np.random.choice(self.game.action_size, p=action_probs)  # Sample action

                # Update game state based on the chosen action
                spg.state = self.game.get_next_state(spg.state, action, player)

                # Check if the game has reached a terminal state
                val, terminal = self.game.is_terminal(spg.state, action)

                if terminal:
                    # Backpropagate the outcome to all states in the game's memory
                    for hist_state, hist_probs, hist_player in spg.mem:
                        hist_outcome = val if hist_player == player else self.game.get_opponent_val(val)
                        ret_mem.append((
                            self.game.get_encoded_state(hist_state),  # Encode state for training
                            hist_probs,  # Action probabilities
                            hist_outcome  # Game outcome
                        ))

                    del spGames[i]  # Remove finished game
            
            # Switch to the opponent player
            player = self.game.get_opponent(player)

        return ret_mem

    def train(self, mem):
        random.shuffle(mem)  # Shuffle memory for training
        for i in range(0, len(mem), self.args["batch_size"]):
            # Create batches of training data
            batch = mem[i:min(len(mem), i + self.args["batch_size"])]
            state, pol_targets, val_targets = zip(*batch)

            # Convert data to NumPy arrays and reshape value targets
            state, pol_targets, val_targets = np.array(state), np.array(pol_targets), np.array(val_targets).reshape(-1, 1)

            # Convert data to PyTorch tensors
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            pol_targets = torch.tensor(pol_targets, dtype=torch.float32, device=self.model.device)
            val_targets = torch.tensor(val_targets, dtype=torch.float32, device=self.model.device)

            # Forward pass through the model
            out_pol, out_val = self.model(state)

            # Compute loss: KL divergence for policy and MSE for value
            loss = F.kl_div(out_pol, pol_targets, reduction="batchmean") + F.mse_loss(out_val, val_targets)

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for i in range(self.args["num_iterations"]):
            mem = []  # Initialize memory for this iteration

            self.model.eval()  # Set model to evaluation mode
            for _ in tqdm(range(self.args["num_self_play"] // self.args["num_parallel_games"])):
                mem.extend(self.self_play())  # Collect self-play data

            self.model.train()  # Set model to training mode
            for _ in tqdm(range(self.args["num_epochs"])):
                self.train(mem)  # Train the model on collected data

            # Save model and optimizer states
            os.makedirs("./models", exist_ok=True)
            torch.save(self.model.state_dict(), f"./models/{self.game}_model_{i}.pth")
            torch.save(self.optimizer.state_dict(), f"./models/{self.game}_optimizer_{i}.pth")

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()  # Initialize game state
        self.mem = []  # Memory to store game history
        self.root, self.node = None, None  # Root and current node for MCTS
