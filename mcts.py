import tic_tac_toe
import numpy as np

# Represents a node in the MCTS tree
class Node:
    def __init__(self, game, args, state, parent=None, action=None):
        self.game = game  # Game logic
        self.args = args  # MCTS parameters
        self.state = state  # Current game state
        self.parent = parent  # Parent node
        self.action = action  # Action leading to this node

        self.children = []  # List of child nodes
        self.expandable_actions = game.get_valid_actions(state)  # Actions that can be expanded
        self.visit_count = 0  # Number of visits to this node
        self.value_sum = 0.0  # Sum of values from simulations

    # Checks if all valid actions have been expanded
    def is_fully_expanded(self):
        return np.sum(self.expandable_actions) == 0 and len(self.children) > 0
    
    # Selects the child node with the highest UCB score
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = child.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child
    
    # Calculates the Upper Confidence Bound (UCB) score for a child node
    def get_ucb(self, child):
        q = (1 - ((child.value_sum / child.visit_count) + 1) / 2) if child.visit_count > 0 else 0
        return q + self.args['c'] * np.sqrt(np.log(self.visit_count) / child.visit_count)
    
    # Expands the node by creating a new child node for a random valid action
    def expand(self):
        if not self.expandable_actions.any():
            return None
        
        action = np.random.choice(np.where(self.expandable_actions == 1)[0])
        self.expandable_actions[action] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)

        child = Node(self.game, self.args, child_state, parent=self, action=action)
        self.children.append(child)
        return child

    # Simulates a random rollout from the current state until terminal state
    def simulate(self):
        val, terminal = self.game.is_terminal(self.state, self.action)
        val = self.game.get_opponent_val(val)

        if terminal:
            return val
        
        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            valid_actions = self.game.get_valid_actions(rollout_state)
            action = np.random.choice(np.where(valid_actions == 1)[0])

            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            val, terminal = self.game.is_terminal(rollout_state, action)

            if terminal:
                return self.game.get_opponent_val(val) if rollout_player == -1 else val
            
            rollout_player = self.game.get_opponent(rollout_player)

    # Updates the node and its ancestors with the simulation result
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        value = self.game.get_opponent_val(value)

        if self.parent is not None:
            self.parent.backpropagate(value)

# Implements the Monte Carlo Tree Search algorithm
class MCTS:
    def __init__(self, game, args):
        self.game = game  # Game logic
        self.args = args  # MCTS parameters

    # Performs MCTS search and returns action probabilities
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args["num_searches"]):
            node = root

            # Selection: Traverse the tree to find a node to expand
            while node.is_fully_expanded():
                node = node.select()
            
            val, terminal = self.game.is_terminal(node.state, node.action)
            val = self.game.get_opponent_val(val)

            if not terminal:
                # Expansion: Add a new child node
                node = node.expand()

                # Simulation: Perform a random rollout
                val = node.simulate()

            # Backpropagation: Update the tree with the simulation result
            node.backpropagate(val)
    
        # Compute action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs
