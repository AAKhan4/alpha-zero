import tic_tac_toe
import numpy as np

class Node:
    def __init__(self, game, args, state, parent=None, action=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.expandable_actions = game.get_valid_actions(state)
        self.visit_count = 0
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return np.sum(self.expandable_actions) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = child.get_ucb()
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child
    
    def get_ucb(self, child):
        q = (1 - ((child.value_sum / child.visit_count) + 1) / 2) if child.visit_count > 0 else 0
        return q + self.args.c * np.sqrt(np.log(self.visit_count) / child.visit_count)
    
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

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        value = self.game.get_opponent_val(value)

        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        # Perform the MCTS search
        root = Node(self.game, self.args, state)

        for search in range(self.args.num_searches):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()
            
            val, terminal = self.game.is_terminal(node.state, node.action)
            val = self.game.get_opponent_val(val)

            if not terminal:
                # Expansion
                node = node.expand()

                # Simulation
                val = node.simulate()

            # Backpropagation
            node.backpropagate(val)
    
        # Return visit_counts   
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs
