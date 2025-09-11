import tic_tac_toe
import connect_four

class GameSelection:
    def pick_game(self):
        print("Select Game?")
        print("0- tic-tac-toe")
        print("1- connect-four")

        if int(input("Enter choice: ")) == 1:
            return connect_four.ConnectFour()
        else:
            return tic_tac_toe.TicTacToe()
        
    def get_args(self, game):
        if isinstance(game, connect_four.ConnectFour):
            args = {
                "num_searches": 600,
                "c": 2,
                "num_iterations": 8,
                "num_self_play": 500,
                "num_parallel_games": 100,
                "num_epochs": 4,
                "batch_size": 128,
                "temperature": 1.25,
                "epsilon": 0.25,
                "alpha": 0.3
            }
        elif isinstance(game, tic_tac_toe.TicTacToe):
            args = {
                "num_searches": 60,
                "c": 2,
                "num_iterations": 3,
                "num_self_play": 500,
                "num_parallel_games": 100,
                "num_epochs": 4,
                "batch_size": 64,
                "temperature": 1.25,
                "epsilon": 0.25,
                "alpha": 0.3
            }

        return args