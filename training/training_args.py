class TrainingArgsBuilder:
    def __init__(self, game):
        self.args = {
            "num_searches": 90,
            "c": 2,
            "num_iterations": 5,
            "num_self_play": 500,
            "num_parallel_games": 100,
            "num_epochs": 4,
            "batch_size": 64,
            "temperature": 1.2,
            "epsilon": 0.25,
            "alpha": 0.3,
            "res_blocks": 4,
            "channels": 64,
            "num_workers": 5,
            "lr": 0.001,
            "weight_decay": 1e-4
        }
        self.build_args(game)

    def build_args(self, game):
        if game.__class__.__name__ == "ConnectFour":
            self.args.update({
                "num_searches": 600,
                "num_iterations": 10,
                "num_epochs": 4,
                "batch_size": 128,
                "res_blocks": 9,
                "channels": 128
            })
        elif game.__class__.__name__ == "TicTacToe":
            pass  # Use default args
        else:
            raise ValueError("Unsupported game type")
        return self.args
