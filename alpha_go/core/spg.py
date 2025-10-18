from core.mcts.node import Node
from games.go import Go, GoState


class SPG:
    def __init__(self, game: Go):
        # Initialize a self-play game instance
        self.game_state: GoState = GoState(game=game)  # Current game state
        self.mem = []  # Memory for storing game history
        self.root: Node = None  # MCTS root node
        self.node: Node = None  # Current MCTS node