import numpy as np
import torch

from core.alpha_zero import SPG
from core.mcts.res_net import ResNet
from games.base_game import BaseGame

# Represents a node in the Monte Carlo Tree Search (MCTS) tree
class Node:
    def __init__(self, game: BaseGame, args: dict, state: np.ndarray, parent: 'Node' = None, action: int = None, prior: float = 0, visit_count: int = 0):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant)
        self.state = state  # Current game state at this node
        self.parent = parent  # Parent node in the tree
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability of selecting this action
        self.children: list[Node] = []  # List of child nodes
        self.visit_count = visit_count  # Number of times this node was visited
        self.value_sum = 0.0  # Cumulative value from simulations

    # Checks if all valid actions have been expanded into child nodes
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    # Selects the child node with the highest Upper Confidence Bound (UCB) score
    def select(self) -> 'Node':
        # Use max with a key function to find the child with the highest UCB score
        return max(self.children, key=self.get_ucb)

    # Calculates the UCB score for a given child node
    def get_ucb(self, child: 'Node') -> float:
        # Q-value: normalized value of the node (scaled to [-1, 1])
        q = (1 - ((child.value_sum / child.visit_count) + 1) / 2) if child.visit_count > 0 else 0
        # UCB formula: Q + exploration term
        return q + self.args['c'] * (np.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    # Expands the node by creating child nodes for valid actions based on the policy
    def expand(self, policy: np.ndarray) -> None:
        for action, prob in enumerate(policy):
            if prob > 0.0:  # Only expand actions with non-zero probability
                child_state = self.state.copy()  # Copy the current state
                child_state = self.game.get_next_state(child_state, action, 1)  # Apply the action
                child_state = self.game.change_perspective(child_state, player=-1)  # Switch perspective

                # Create a new child node
                child = Node(self.game, self.args, child_state, parent=self, action=action, prior=prob)
                self.children.append(child)

    # Updates the node and its ancestors with the result of a simulation
    def backpropagate(self, value: float) -> None:
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add the simulation value to the cumulative sum

        # Flip the value for the opponent's perspective
        value = self.game.get_opponent_val(value)

        # Recursively backpropagate to the parent node
        if self.parent is not None:
            self.parent.backpropagate(value)

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
