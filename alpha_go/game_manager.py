import torch
import numpy as np
from core.mcts.res_net import ResNet
from games.go import Go, GoState
from training.training_args import TrainingArgsBuilder
import os

# Initialize the game
game = Go(board_size=9, komi=6.5)
game_state = GoState(game=game)
args = TrainingArgsBuilder(game).args

# Declare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, args["res_blocks"], args["channels"], device)  # Initialize the neural network model

loss = float('inf')

with open(f"./models/{game}/loss_{args['num_iterations']-1}.txt", "r") as file:
    losses = file.readlines()
    loss = float(losses[-1].strip()) if losses else float('inf')

print(f"\nLoading model {args['num_iterations']} with loss {loss}\n")
model.load_state_dict(torch.load(f"./models/{game}/model_{args['num_iterations']-1}.pth", map_location=device))  # Load the trained model
model.eval()  # Set model to evaluation mode

while True:
    print_state = np.where(game_state.state == 1, "| X |", np.where(game_state.state == -1, "| O |", "|   |"))
    print(print_state)  # Display the current game state
    game_info = game_state.get_info()

    if game_info["perspective"] == 1:
        # Player 1's turn (human input)
        print("\nPlayer 1's turn (X)")
        valid_actions = game.get_valid_actions(game_info)  # Get valid moves
        print([i for i in range(len(valid_actions)) if valid_actions[i] == 1])  # Show valid actions
        action = int(input(f"Player {game_info['perspective']}, enter your action (0-8): "))

        if action != -1 and not game.is_valid_action(game_info, action):
            print("Invalid action. Try again.")
            continue
    else:
        with torch.no_grad():
            policy, val = model(
                torch.tensor(game.get_encoded_state(game_info["state"]), device=model.device).unsqueeze(0)
            )

        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
        valid_actions = game.get_valid_actions(game_info)
        policy *= valid_actions
        policy = policy ** 5  # Boost probabilities to favor higher ones
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1

        action = np.random.choice(len(policy), p=policy)  # Sample action based on policy
        print(f"\nPlayer {game_info['perspective']}'s turn (O)\n")
        print(f"AI chose action: {action}\n")

    # Update the game state based on the chosen action
    game_info = game.get_next_state(game_info, action)
    val, terminal = game.is_terminal(game_info["state"], game_info["last_2_actions"], game_info["perspective"], game_info["captures"])  # Check if the game is over

    if terminal:
        temp = game.change_perspective(game_info) if game_info["perspective"] == -1 else game_info
        print_state = np.where(temp["state"] == 1, "| X |", np.where(temp["state"] == -1, "| O |", "|   |"))
        print(print_state)  # Display the final state
        if val == 0:
            print("It's a draw!")  # Declare draw
        elif val is None:
            print(f"Player {temp['perspective']} wins by resignation!")  # Declare resignation win
        else:
            print(f"Player {game_info['perspective'] * val / abs(val)} wins with score {abs(val)}!")  # Declare win with score
        break

    # Switch to the other player
    game_info = game.change_perspective(game_info)
    game_state.update(game_info)
