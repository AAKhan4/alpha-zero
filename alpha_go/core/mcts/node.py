# Represents a node in the Monte Carlo Tree Search (MCTS) tree
import numpy as np
from games.go import Go


class Node:
    def __init__(self, game: Go, args: dict, state: dict, parent: 'Node' = None, action: int = None, prior: float = 0, visit_count: int = 0):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant)
        self.info = state  # Current game state at this node
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
        q = (child.value_sum / child.visit_count) if child.visit_count > 0 else 0
        # UCB formula: Q + exploration term
        return q + self.args['c'] * (np.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    # Expands the node by creating child nodes for valid actions based on the policy
    def expand(self, policy: np.ndarray):
        for action, prob in enumerate(policy):
            if prob > 0.0:  # Only expand actions with non-zero probability
                child_state = self.game.get_next_state(self.info, action)  # Apply the action
                child_state = self.game.change_perspective(child_state)  # Switch perspective

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