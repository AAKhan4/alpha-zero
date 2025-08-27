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
        self.mcts = mcts.MCTS(game, args, model)

    def self_play(self):
        ret_mem = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i  in range(len(spGames))[::-1]:
                spg = spGames[i]

                # Compute action probabilities based on visit counts
                mcts_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    mcts_probs[child.action] = child.visit_count

                mcts_probs /= np.sum(mcts_probs)

                spg.mem.append((spg.root.state, mcts_probs, player))

                action_probs = mcts_probs ** (1 / self.args["temperature"])
                action_probs /= np.sum(action_probs) if np.sum(action_probs) > 0 else 1
                action = np.random.choice(self.game.action_size, p=action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                val, terminal = self.game.is_terminal(spg.state, action)

                if terminal:
                    for hist_state, hist_probs, hist_player in spg.mem:
                        hist_outcome = val if hist_player == player else self.game.get_opponent_val(val)
                        ret_mem.append((
                            self.game.get_encoded_state(hist_state),
                            hist_probs,
                            hist_outcome
                        ))

                    del spGames[i]
            
            player = self.game.get_opponent(player)

        return ret_mem

    def train(self, mem):
        random.shuffle(mem)
        for i in range(0, len(mem), self.args["batch_size"]):
            batch = mem[i:min(len(mem), i + self.args["batch_size"])]
            state, pol_targets, val_targets = zip(*batch)

            state, pol_targets, val_targets = np.array(state), np.array(pol_targets), np.array(val_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            pol_targets = torch.tensor(pol_targets, dtype=torch.float32, device=self.model.device)
            val_targets = torch.tensor(val_targets, dtype=torch.float32, device=self.model.device)

            out_pol, out_val = self.model(state)

            loss = F.kl_div(out_pol, pol_targets) + F.mse_loss(out_val, val_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for i in range(self.args["num_iterations"]):
            mem = []

            self.model.eval()
            for _ in tqdm(range(self.args["num_self_play"] // self.args["num_parallel_games"])):
                mem.extend(self.self_play())

            self.model.train()
            for _ in tqdm(range(self.args["num_epochs"])):
                self.train(mem)

            os.makedirs("./models", exist_ok=True)

            torch.save(self.model.state_dict(), f"./models/{self.game}_model_{i}.pth")
            torch.save(self.optimizer.state_dict(), f"./models/{self.game}_optimizer_{i}.pth")

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.mem = []
        self.root, self.node = None, None
