import tic_tac_toe

game = tic_tac_toe.TicTacToe()

player = 1
state = game.get_initial_state()

while 1:
    print(state)
    valid_actions = game.get_valid_actions(state)
    print(valid_actions)
    action = int(input(f"Player {player}, enter your action (0-8): "))

    if valid_actions[action] == 0:
        print("Invalid action. Try again.")
        continue

    state = game.get_next_state(state, action, player)
    val, terminal = game.is_terminal(state, action)

    if terminal:
        if val == 1:
            print(f"Player {player} wins!")
        elif val == 0:
            print("It's a draw!")
        else:
            print("Game is still ongoing.")
        break

    player = game.get_opponent(player)
