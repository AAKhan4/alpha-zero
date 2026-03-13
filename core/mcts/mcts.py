import numpy as np
import torch

from core.mcts.node import Node
from core.spg import SPG
from core.mcts.res_net import ResNet
from games.base_game import BaseGame


# Implements the Monte Carlo Tree Search (MCTS) algorithm
class MCTS:
    def __init__(self, game: BaseGame, args: dict, model: ResNet):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant, number of searches)
        self.model = model  # Neural network model for policy and value predictions

    # Performs MCTS for multiple self-play games in parallel
    @torch.no_grad()
    def search(self, games: list[SPG]) -> None:
        '''Get initial policy and value predictions from the model'''

        # Initialize root nodes for all parallel games if not already initialized
        for i, game in enumerate(games):
            if not game.root:
                game.root = Node(self.game, self.args)  # Create root node with the initial game state

        # Perform the specified number of MCTS searches
        for _ in range(self.args["num_searches"]):
            expandable_nodes: list[Node] = []
            for game in games:
                node = game.root  # Start from the root node

                # Selection: Traverse the tree to find a node to expand
                while node.is_fully_expanded():
                    node = node.select()  # Select the best child node

                # Check if the selected node is terminal
                val, terminal = self.game.is_terminal(node.rebuild_state().get_info())
                val /= abs(val) if val != 0 else 1  # Normalize terminal value to [-1, 1]

                if terminal:
                    node.backpropagate(val)
                else:
                    expandable_nodes.append(node)

            # Collect all nodes that can be expanded
            if expandable_nodes:
                # Get states for all expandable nodes
                states = np.stack([node.rebuild_state().get_info()["board"] for node in expandable_nodes])
                # Get policy and value predictions for these states
                policy, val = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()  # Apply softmax to policy
                val = val.cpu().numpy()  # Get value predictions as numpy array

                # Expand and backpropagate for all expandable nodes
                for i, node in enumerate(expandable_nodes):
                    node.expand(policy[i])  # Expand the node with the new policy
                    node.backpropagate(val[i])  # Backpropagate the value
            else:
                # Skip this iteration if there are no expandable nodes
                continue
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory after each search iteration to prevent memory overflow
