import matplotlib.pyplot as plt
from training_scripts.training_args import TrainingArgsBuilder
from games.tic_tac_toe.tic_tac_toe import TicTacToe
from games.go.go import Go
from games.connect_four.connect_four import ConnectFour

import os
import argparse

def plot_losses(game: str, flag: str):
    game = Go() if game.lower() == "go" else TicTacToe() if game.lower() == "tic_tac_toe" else ConnectFour()
    flag = "sl" if flag.lower() == "sl" else "rl"
    args_builder = TrainingArgsBuilder(game)
    args = args_builder.build_args(game)


    # Directory containing the loss files for all iterations
    loss_dir = f"./models/{game}/{flag}/losses"

    # Collect losses from all iterations
    all_losses = []
    for iteration in range(args['num_iterations']):
        loss_file = os.path.join(loss_dir, f"loss_{iteration}.txt")
        if os.path.exists(loss_file):
            with open(loss_file, "r") as file:
                losses = [float(line.strip()) for line in file]
                all_losses.append((iteration, losses))

    # Plot the losses for all iterations
    for iteration, losses in all_losses:
        plt.plot(losses, label=f"Iteration {iteration}")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.title("Mean Model Loss Over Epochs for All Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True, help="The game to plot losses for (e.g., 'go' or 'tic_tac_toe').")
    parser.add_argument("--flag", type=str, required=True, help="The training flag (e.g., 'sl' or 'rl').")

    args = parser.parse_args()
    plot_losses(game=args.game, flag=args.flag)