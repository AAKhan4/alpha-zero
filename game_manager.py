import torch
import tic_tac_toe
import numpy as np
import res_net
import matplotlib.pyplot as plt


# Initialize the game
game = tic_tac_toe.TicTacToe()

# Declare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

player = 1  # Player 1 starts
state = game.get_initial_state()  # Get the initial game state

# MCTS configuration
args = {
        "num_searches": 60,  # Number of searches for MCTS
        "c": 2,  # Exploration constant
        "num_iterations": 3,  # Number of training iterations
        "num_self_play": 100,  # Number of self-play games
        "num_epochs": 4,  # Number of training epochs
        "batch_size": 64,  # Batch size for training
        "temperature": 1.25,  # Temperature for action selection
        "epsilon": 0.25,  # Epsilon for exploration
        "alpha": 0.3  # Alpha for Dirichlet noise
    }

model = res_net.ResNet(game, 4, 64, device)  # Initialize the neural network model

model.load_state_dict(torch.load(f"./models/model_{args['num_iterations'] - 1}.pth", map_location=device))  # Load the trained model
model.eval()  # Set model to evaluation mode

while 1:
    print(state)  # Display the current game state

    if player == 1:
        # Player 1's turn (human input)
        print("\nPlayer 1's turn (X)")
        valid_actions = game.get_valid_actions(state)  # Get valid moves
        print(valid_actions)
        action = int(input(f"Player {player}, enter your action (0-8): "))

        if valid_actions[action] == 0:  # Check for invalid moves
            print("Invalid action. Try again.")
            continue
    else:
        # Player 2's turn (AI using MCTS)
        print("\nPlayer 2's turn (O)")
        encoded_state = game.get_encoded_state(state)  # Get the state from the opponent's perspective
        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        policy, val = model(tensor_state)
        policy = torch.softmax(policy, dim=1).squeeze(0).detach().cpu().numpy()
        policy *= valid_actions
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1
        val = val.item()

        action = np.argmax(policy)  # Choose the best move

    # Update the game state based on the chosen action
    state = game.get_next_state(state, action, player)
    val, terminal = game.is_terminal(state, action)  # Check if the game is over

    if terminal:
        print(state)  # Display the final state
        if val == 1:
            print(f"Player {player} wins!")  # Declare winner
        elif val == 0:
            print("It's a draw!")  # Declare draw
        else:
            print("Game is still ongoing.")  # Shouldn't occur here
        break

    # Switch to the other player
    player = game.get_opponent(player)
