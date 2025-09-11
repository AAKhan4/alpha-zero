import torch
import numpy as np
from core.mcts.res_net import ResNet
from core.mcts.mcts import MCTS
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

monte_carlo = MCTS(game, args, model)  # Initialize MCTS

while 1:
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
        # Player 2's turn (AI using MCTS)
        print("\nPlayer 2's turn (O)")
        mcts_probs = monte_carlo.search(state)
        action = np.argmax(mcts_probs)  # Choose the best move

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
