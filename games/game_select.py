from games.tic_tac_toe.tic_tac_toe import TicTacToe
from games.connect_four.connect_four import ConnectFour
from training.training_args import TrainingArgsBuilder

class GameSelection:
    def pick_game(self):
        print("Select Game?")
        print("0- tic-tac-toe")
        print("1- connect-four")

        if int(input("Enter choice: ")) == 1:
            return ConnectFour()
        else:
            return TicTacToe()
        
    def get_args(self, game):
        return TrainingArgsBuilder(game).args
