import torch
import numpy as np
from core.mcts.res_net import ResNet
from games.game_select import GameSelection

# Initialize the game
game = GameSelection().pick_game()
args = GameSelection().get_args(game)

# Declare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

player = 1  # Player 1 starts
state = game.get_initial_state()  # Get the initial game state


model = ResNet(game, args["res_blocks"], args["channels"], device)  # Initialize the neural network model

model.load_state_dict(torch.load(f"./models/{game}/model_{args['num_iterations'] - 1}.pth", map_location=device))  # Load the trained model
model.eval()  # Set model to evaluation mode

while True:
    print(state)  # Display the current game state

    if player == 1:
        # Player 1's turn (human input)
        print("\nPlayer 1's turn (X)")
        valid_actions = game.get_valid_actions(state)  # Get valid moves
        print([i for i in range(len(valid_actions)) if valid_actions[i] == 1])  # Show valid actions
        action = int(input(f"Player {player}, enter your action (0-8): "))

        if valid_actions[action] == 0:  # Check for invalid moves
            print("Invalid action. Try again.")
            continue
    else:
        policy, val = model(
                    torch.tensor(game.get_encoded_state(state), device=model.device).unsqueeze(0)
                )

        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
        valid_actions = game.get_valid_actions(state).astype(bool)
        policy *= valid_actions
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1

        action = np.argmax(policy)  # Choose the best move based on policy

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
