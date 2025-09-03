import numpy as np
import torch

# Represents a node in the Monte Carlo Tree Search (MCTS) tree
class Node:
    def __init__(self, game, args, state, parent=None, action=None, prior=0, visit_count=0):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant)
        self.state = state  # Current game state at this node
        self.parent = parent  # Parent node in the tree
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability of selecting this action
        self.children = []  # List of child nodes
        self.visit_count = visit_count  # Number of times this node was visited
        self.value_sum = 0.0  # Cumulative value from simulations

    # Checks if all valid actions have been expanded into child nodes
    def is_fully_expanded(self):
        return len(self.children) > 0

    # Selects the child node with the highest Upper Confidence Bound (UCB) score
    def select(self):
        best_child = None
        best_ucb = -np.inf  # Initialize with negative infinity

        for child in self.children:
            ucb = self.get_ucb(child)  # Calculate UCB score for each child
            if ucb > best_ucb:  # Update the best child if a higher UCB is found
                best_ucb = ucb
                best_child = child

        return best_child

    # Calculates the UCB score for a given child node
    def get_ucb(self, child):
        # Q-value: normalized value of the node (scaled to [-1, 1])
        q = (1 - ((child.value_sum / child.visit_count) + 1) / 2) if child.visit_count > 0 else 0
        # UCB formula: Q + exploration term
        return q + self.args['c'] * (np.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    # Expands the node by creating child nodes for valid actions based on the policy
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0.0:  # Only expand actions with non-zero probability
                child_state = self.state.copy()  # Copy the current state
                child_state = self.game.get_next_state(child_state, action, 1)  # Apply the action
                child_state = self.game.change_perspective(child_state, player=-1)  # Switch perspective

                # Create a new child node
                child = Node(self.game, self.args, child_state, parent=self, action=action, prior=prob)
                self.children.append(child)

    # Updates the node and its ancestors with the result of a simulation
    def backpropagate(self, value):
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add the simulation value to the cumulative sum

        # Flip the value for the opponent's perspective
        value = self.game.get_opponent_val(value)

        # Recursively backpropagate to the parent node
        if self.parent is not None:
            self.parent.backpropagate(value)

# Implements the Monte Carlo Tree Search (MCTS) algorithm
class MCTS:
    def __init__(self, game, args, model):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant, number of searches)
        self.model = model  # Neural network model for policy and value predictions

    # Performs MCTS search and returns action probabilities
    @torch.no_grad()  # Disable gradient computation for efficiency
    def search(self, states, spGames):
        # Get initial policy and value predictions from the model
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()  # Apply softmax to get probabilities

        # Add Dirichlet noise for exploration
        policy = (1 - self.args["epsilon"]) * policy + self.args["epsilon"] * np.random.dirichlet(
            [self.args["alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        # Initialize the root nodes for all search games
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_actions = self.game.get_valid_actions(states[i])  # Mask invalid actions
            spg_policy *= valid_actions  # Apply mask
            spg_policy /= np.sum(spg_policy) if np.sum(spg_policy) > 0 else 1  # Normalize probabilities
            spg.root = Node(self.game, self.args, states[i], visit_count=1)  # Create root node
            spg.root.expand(spg_policy)  # Expand root node with initial policy

        # Perform the specified number of MCTS searches
        for _ in range(self.args["num_searches"]):
            for spg in spGames:
                spg.node = None  # Reset the expandable node
                node = spg.root  # Start from the root node

                # Selection: Traverse the tree to find a node to expand
                while node.is_fully_expanded():
                    node = node.select()  # Select the best child node

                # Check if the selected node is terminal
                val, terminal = self.game.is_terminal(node.state, node.action)
                val = self.game.get_opponent_val(val)  # Flip value for the opponent's perspective

                if terminal:
                    # If terminal, backpropagate the result
                    node.backpropagate(val)
                else:
                    # If not terminal, mark the node for expansion
                    spg.node = node

            # Collect all nodes that can be expanded
            expandable_spgs = [i for i in range(len(spGames)) if spGames[i].node is not None]
            if len(expandable_spgs) > 0:
                # Get states for all expandable nodes
                states = np.stack([spGames[i].node.state for i in expandable_spgs])
                # Get policy and value predictions for these states
                policy, val = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()  # Apply softmax to policy
                val = val.cpu().numpy()  # Convert value tensor to numpy

            # Expand and backpropagate for each expandable node
            for i, spg_idx in enumerate(expandable_spgs):
                node = spGames[spg_idx].node
                spg_policy, spg_val = policy[i], val[i]
                valid_actions = self.game.get_valid_actions(node.state).astype(bool)  # Mask invalid actions
                spg_policy = spg_policy * valid_actions  # Apply mask
                spg_policy /= np.sum(spg_policy) if np.sum(spg_policy) > 0 else 1  # Normalize probabilities

                # Expand the node with the new policy
                node.expand(spg_policy)
                # Backpropagate the value
                node.backpropagate(spg_val)
