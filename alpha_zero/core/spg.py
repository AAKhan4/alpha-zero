from core.mcts.node import Node
from games.base_game import BaseGame


class SPG:
    def __init__(self, game: BaseGame):
        # Initialize a self-play game instance
        self.state = game.get_initial_state()  # Initial game state
        self.mem = []  # Memory for storing game history
        self.root: Node = None  # MCTS root node
        self.node: Node = None  # Current MCTS node