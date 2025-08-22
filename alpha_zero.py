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
        mem = []
        player = 1
        state = self.game.get_initial_state()

        while 1:
            neutral_state = self.game.change_perspective(state, player)
            mcts_probs = self.mcts.search(neutral_state)

            mem.append((neutral_state, mcts_probs, player))

            action_probs = mcts_probs ** (1 / self.args["temperature"])
            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            val, terminal = self.game.is_terminal(state, action)

            if terminal:
                returnMem = []
                for hist_state, hist_probs, hist_player in mem:
                    hist_outcome = val if hist_player == player else self.game.get_opponent_val(val)
                    returnMem.append((
                        self.game.get_encoded_state(hist_state),
                        hist_probs,
                        hist_outcome
                    ))

                return returnMem

    def train(self, mem):
        random.shuffle(mem)
        for i in range(0, len(mem), self.args["batch_size"]):
            batch = mem[i:min(len(mem) - 1, i + self.args["batch_size"])]
            state, pol_targets, val_targets = zip(*batch)

            state, pol_targets, val_targets = np.array(state), np.array(pol_targets), np.array(val_targets).reshape(-1, 1)

            state, pol_targets, val_targets = (
                torch.tensor(state, dtype=torch.float32, device=self.model.device),
                torch.tensor(pol_targets, dtype=torch.float32, device=self.model.device),
                torch.tensor(val_targets, dtype=torch.float32, device=self.model.device)
            )

            out_pol, out_val = self.model(state)

            loss = F.cross_entropy(out_pol, pol_targets) + F.mse_loss(out_val, val_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for i in range(self.args["num_iterations"]):
            mem = []

            self.model.eval()
            for _ in tqdm(range(self.args["num_self_play"])):
                mem.extend(self.self_play())

            self.model.train()
            for _ in tqdm(range(self.args["num_epochs"])):
                self.train(mem)

            os.makedirs("./models", exist_ok=True)

            torch.save(self.model.state_dict(), f"./models/model_{i}.pth")
            torch.save(self.optimizer.state_dict(), f"./models/optimizer_{i}.pth")
