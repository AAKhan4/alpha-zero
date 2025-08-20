import tic_tac_toe
import mcts
import numpy as np

game = tic_tac_toe.TicTacToe()

player = 1
state = game.get_initial_state()

args = {
    "num_searches": 1000,
    "c": 1.41
}

monte_carlo = mcts.MCTS(game, args)

while 1:
    print(state)

    if player == 1:
        print("\nPlayer 1's turn (X)")
        valid_actions = game.get_valid_actions(state)
        print(valid_actions)
        action = int(input(f"Player {player}, enter your action (0-8): "))

        if valid_actions[action] == 0:
            print("Invalid action. Try again.")
            continue
    else:
        print("\nPlayer 2's turn (O)")
        neutral_state = game.change_perspective(state, player)
        mcts_probs = monte_carlo.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)
    val, terminal = game.is_terminal(state, action)

    if terminal:
        print(state)
        if val == 1:
            print(f"Player {player} wins!")
        elif val == 0:
            print("It's a draw!")
        else:
            print("Game is still ongoing.")
        break

    player = game.get_opponent(player)
