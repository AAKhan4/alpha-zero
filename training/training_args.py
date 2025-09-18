from games.base_game import BaseGame


class TrainingArgsBuilder:
    def __init__(self, game: BaseGame):
        self.args = {
            "num_searches": 100, # Number of MCTS simulations per move
            "c": 1.1, # Exploration constant for MCTS
            "num_iterations": 10, # Number of training iterations
            "num_self_play": 200, # Number of self-play games per iteration
            "max_parallel_games": 100, # Max parallel games during self-play
            "num_epochs": 10, # Training epochs per iteration
            "batch_size": 64, # Mini-batch size for training
            "init_temperature": 1.0, # Initial temperature for action selection
            "temp_threshold": 10, # Moves before temperature decay
            "temp_decay": 0.3, # Temperature decay rate
            "epsilon": 0.25, # Exploration noise weight
            "alpha": 0.4, # Dirichlet noise parameter
            "res_blocks": 4, # Number of residual blocks in the neural network
            "channels": 32, # Number of channels in the neural network
            "num_workers": 5, # Number of parallel worker processes
            "lr": 0.001, # Learning rate for the optimizer
            "weight_decay": 1e-4 # Weight decay for the optimizer
        }
        self.build_args(game)

    def build_args(self, game: BaseGame) -> dict:
        if game.__class__.__name__ == "ConnectFour":
            self.args.update({
                "num_searches": 400,
                "num_self_play": 200,
                "c": 1.5,
                "num_iterations": 12,
                "init_temperature": 1.1,
                "batch_size": 128,
                "res_blocks": 8,
                "channels": 64
            })
        elif game.__class__.__name__ == "TicTacToe":
            pass  # Use default args
        else:
            raise ValueError("Unsupported game type")
        return self.args
