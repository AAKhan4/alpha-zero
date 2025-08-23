import numpy as np
import torch

# Represents a node in the MCTS tree
class Node:
    def __init__(self, game, args, state, parent=None, action=None, prior=0, visit_count=0):
        self.game = game  # Game logic
        self.args = args  # MCTS parameters
        self.state = state  # Current game state
        self.parent = parent  # Parent node
        self.action = action  # Action leading to this node
        self.prior = prior  # Prior probability of this action

        self.children = []  # List of child nodes
        self.visit_count = visit_count  # Number of visits to this node
        self.value_sum = 0.0  # Sum of values from simulations

    # Checks if all valid actions have been expanded
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    # Selects the child node with the highest UCB score
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child
    
    # Calculates the Upper Confidence Bound (UCB) score for a child node
    def get_ucb(self, child):
        q = (1 - ((child.value_sum / child.visit_count) + 1) / 2) if child.visit_count > 0 else 0
        return q + self.args['c'] * (np.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    # Expands the node by creating a new child node for a random valid action
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                # Create a child node for the action
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, parent=self, action=action, prior=prob)
                self.children.append(child)

    # Updates the node and its ancestors with the simulation result
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        value = self.game.get_opponent_val(value)

        if self.parent is not None:
            self.parent.backpropagate(value)

# Implements the Monte Carlo Tree Search algorithm
class MCTS:
    def __init__(self, game, args, model):
        self.game = game  # Game logic
        self.args = args  # MCTS parameters
        self.model = model  # Neural network model

    # Performs MCTS search and returns action probabilities
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=0)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(root.state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

        policy = (1 - self.args["epsilon"]) * policy + self.args["epsilon"] * np.random.dirichlet([self.args["alpha"]] * self.game.action_size)
        
        valid_actions = self.game.get_valid_actions(root.state).astype(bool)
        policy = policy * valid_actions
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1

        root.expand(policy)

        for _ in range(self.args["num_searches"]):
            node = root

            # Selection: Traverse the tree to find a node to expand
            while node.is_fully_expanded():
                node = node.select()
            
            val, terminal = self.game.is_terminal(node.state, node.action)
            val = self.game.get_opponent_val(val)

            if not terminal:
                policy, val = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )

                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_actions = self.game.get_valid_actions(node.state).astype(bool)
                policy = policy * valid_actions
                policy /= np.sum(policy) if np.sum(policy) > 0 else 1

                val = val.item()

                # Expansion: Add new child nodes for all valid actions
                node.expand(policy)

            # Backpropagation: Update the tree with the simulation result
            node.backpropagate(val)
    
        # Compute action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs
