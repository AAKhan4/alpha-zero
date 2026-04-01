from games.base_game import BaseGame


class TrainingArgsBuilder:
    def __init__(self, game: BaseGame):
        self.args = {
            "num_searches": 30, # Number of MCTS simulations per move
            "c": 0.8, # Exploration constant for MCTS
            "num_iterations": 10, # Number of training iterations
            "num_self_play": 100, # Number of self-play games per iteration
            "max_parallel_games": 128, # Max parallel games during self-play
            "num_epochs": 7, # Training epochs per iteration
            "batch_size": 16, # Mini-batch size for training
            "init_temperature": 1.0, # Initial temperature for action selection
            "temp_threshold": 5, # Moves before temperature decay
            "temp_decay": 0.9, # Temperature decay rate
            "temp_floor": 0.01, # Minimum temperature after decay
            "epsilon": 0.25, # Exploration noise weight
            "alpha": 0.3, # Dirichlet noise parameter
            "res_blocks": 2, # Number of residual blocks in the neural network
            "channels": 16, # Number of channels in the neural network
            "num_workers": 10, # Number of parallel worker processes
            "lr": 3e-3, # Learning rate for the optimizer
            "weight_decay": 1e-4, # Weight decay for the optimizer
            "pretraining_epochs": 0 # Number of epochs for pretraining
        }
        self.build_args(game)

    def build_args(self, game: BaseGame) -> dict:
        if game.__class__.__name__ == "ConnectFour":
            self.args.update({
                "num_searches": 100,
                "num_self_play": 200,
                "c": 1.2,
                "num_iterations": 10,
                "num_epochs": 8,
                "batch_size": 64,
                "temp_threshold": 10,
                "temp_decay": 0.6,
                "temp_floor": 0.1,
                "res_blocks": 8,
                "channels": 128,
                "lr": 3e-4,
                "replay_buffer_size": 10000
            })
        elif game.__class__.__name__ == "Go":
            self.args.update({
                "num_searches": 100,
                "num_self_play": 400,
                "c": 1.9,
                "num_iterations": 30,
                "num_epochs": 8,
                "batch_size": 128,
                "init_temperature": 1.2,
                "temp_threshold": 12,
                "temp_decay": 0.4,
                "temp_floor": 0.4,
                "res_blocks": 8,
                "channels": 64,
                "alpha": 0.05,
                "epsilon": 0.20,
                "lr": 1e-4,
                "pretraining_epochs": 8,
                "replay_buffer_size": 200000
            })
        elif game.__class__.__name__ == "TicTacToe":
            pass  # Use default args
        else:
            raise ValueError("Unsupported game type")
        return self.args
