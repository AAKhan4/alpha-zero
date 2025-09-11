import matplotlib.pyplot as plt

from games.tic_tac_toe.tic_tac_toe import TicTacToe

from core.mcts.res_net import ResNet

import torch

import numpy as np

tictactoe = TicTacToe()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)
state = tictactoe.get_next_state(state, 6, 1)
state = tictactoe.get_next_state(state, 8, 1)


encoded_state = tictactoe.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(tictactoe, 4, 64, device=device)
model.load_state_dict(torch.load('./models/model_2.pth', map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, dim=1).squeeze(0).to(device).detach().cpu().numpy()
policy *= tictactoe.get_valid_actions(state)
policy /= np.sum(policy) if np.sum(policy) > 0 else 1

print(value)

print(state)
print(tensor_state)

print(policy)
print()
plt.bar(range(tictactoe.action_size), policy)
plt.xlabel("Actions")
plt.ylabel("Probabilities")
plt.title("Action Probabilities")
plt.show()
