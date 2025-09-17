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
    def search(self, states: np.ndarray, games: list[SPG]) -> None:
        # Get initial policy and value predictions from the model
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()  # Apply softmax to get probabilities

        # Add Dirichlet noise for exploration
        policy = (1 - self.args["epsilon"]) * policy + self.args["epsilon"] * np.random.dirichlet(
            [self.args["alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        # Mask invalid actions and normalize probabilities for all states
        valid_actions = np.stack([self.game.get_valid_actions(state) for state in states])  # Batch valid actions
        policy *= valid_actions  # Mask invalid actions for all states
        policy /= np.sum(policy, axis=1, keepdims=True)  # Normalize probabilities across actions

        # Initialize root nodes for all parallel games
        for i, game in enumerate(games):
            game.root = Node(self.game, self.args, states[i], visit_count=0)
            game.root.expand(policy[i])

        # Perform the specified number of MCTS searches
        for _ in range(self.args["num_searches"]):
            expandable_nodes: list[Node] = []
            for game in games:
                game.node = None  # Reset the expandable node
                node = game.root  # Start from the root node

                # Selection: Traverse the tree to find a node to expand
                while node.is_fully_expanded():
                    node = node.select()  # Select the best child node

                # Check if the selected node is terminal
                val, terminal = self.game.is_terminal(node.state, node.action)

                if terminal:
                    # If terminal, backpropagate the result
                    val = self.game.get_opponent_val(val)  # Flip value for the opponent's perspective
                    node.backpropagate(val)
                else:
                    expandable_nodes.append(node)

            # Collect all nodes that can be expanded
            if expandable_nodes:
                # Get states for all expandable nodes
                states = np.stack([node.state for node in expandable_nodes])
                # Get policy and value predictions for these states
                policy, val = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()  # Apply softmax to policy
                val = val.cpu().numpy()  # Convert value tensor to numpy


            # Mask invalid actions and normalize probabilities for all states
            valid_actions = np.stack([self.game.get_valid_actions(state) for state in states])  # Batch valid actions
            policy *= valid_actions  # Mask invalid actions for all states
            policy /= np.sum(policy, axis=1, keepdims=True)  # Normalize probabilities across actions

            # Expand and backpropagate for all expandable nodes
            for i, node in enumerate(expandable_nodes):
                node.expand(policy[i])  # Expand the node with the new policy
                node.backpropagate(val[i])  # Backpropagate the value
