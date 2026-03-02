# Represents a node in the Monte Carlo Tree Search (MCTS) tree
import copy
import numpy as np
from games.base_game import BaseGame, GameState

MAX_BRANCHING_FACTOR = 10  # Maximum number of child nodes to expand per node if available valid actions exceed this number
MAX_BRANCHING_FACTOR = 20  # Maximum number of child nodes to expand per node
MAX_ROOT_EXPANSION = 30  # Maximum number of child nodes to expand for the root node

class Node:
    def __init__(self, game: BaseGame, args: dict, game_state: GameState, parent: 'Node' = None, action: int = None, prior: float = 0, visit_count: int = 0):
        self.game = game  # Game logic object
        self.args = args  # MCTS parameters (e.g., exploration constant)
        self.game_state: GameState = copy.deepcopy(game_state)  # Current game state at this node
        self.parent = parent  # Parent node in the tree
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability of selecting this action
        self.children: list[Node] = []  # List of child nodes
        self.visit_count = visit_count  # Number of times this node was visited
        self.value_sum = 0.0  # Cumulative value from simulations
        self.depth = parent.depth + 1 if parent else 0  # Depth of the node in the tree

    def is_fully_expanded(self) -> bool:
        '''Returns True if the node has any children.'''
        valid_actions = self.game.get_valid_actions(self.game_state.get_info()).sum()
        max_expansion = MAX_ROOT_EXPANSION if self.depth == 0 else MAX_BRANCHING_FACTOR
        return len(self.children) > 0 and len(self.children) >= min(max_expansion, valid_actions)  # Consider node fully expanded if it has at least 20 children or all valid actions are expanded

    def select(self) -> 'Node':
        '''Selects the child node with the highest UCB score.'''
        # Use max with a key function to find the child with the highest UCB score
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child: 'Node') -> float:
        '''Calculates the UCB score for a child node.'''
        # Q-value: normalized value of the node (scaled to [-1, 1])
        q = 1 if (child.visit_count == 0) else 1 - (child.value_sum / child.visit_count)
        # UCB formula: Q + exploration term
        return q + (self.args['c'] * child.prior * np.sqrt(self.visit_count + 1) / (1 + child.visit_count))

    def get_top_actions(self, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Returns the indices of the top-k actions based on the policy probabilities.'''
        k = max(MAX_BRANCHING_FACTOR, MAX_BRANCHING_FACTOR - self.depth)  # Limit to top-K actions for expansion
        if self.depth == 0:
            k = MAX_ROOT_EXPANSION # Expand more actions at the start for better exploration

        top_actions = np.argsort(policy)[::-1]  # Get indices of actions sorted in descending order
        if policy.size > k:
            top_actions = top_actions[:k]  # Get indices of actions sorted in descending order for top k
        if (self.game.can_pass) and not (self.game.action_size-1 in top_actions):
            policy[self.game.action_size-1] += 0.001  # Slightly increase the probability of the "pass" action if it's not already in the top actions
            top_actions[-1] = self.game.action_size-1  # Ensure the "pass" action is included in top actions
            policy = policy / np.sum(policy)  # Re-normalize the policy after adjustment
        
        return policy, top_actions

    def expand(self, policy: np.ndarray):
        '''Expands the node by creating child nodes for valid actions.'''
        # Optional: restrict expansion to top-K actions
        policy, top_actions = self.get_top_actions(policy)

        info = self.game_state.get_info()

        existing_children = set([child.action for child in self.children])  # Get actions of existing children to avoid duplicates
        for action in top_actions:
            prob = policy[action]
            if prob <= 0 or not self.game.is_valid_action(info, action) or action in existing_children:
                continue  # Skip actions with zero probability or already existing children
            # Apply move to get the next game state
            next_info = self.game.get_next_state(info, action)
            next_info = self.game.change_perspective(next_info)
            child_state = self.game.get_state_type()(game=self.game)
            child_state.update(next_info)
            child = Node(game=self.game, args=self.args, game_state=child_state, parent=self, action=action, prior=prob)

            self.children.append(child)
            existing_children.add(action)  # Add action to existing children set to prevent duplicates

    def backpropagate(self, value: float) -> None:
        '''Backpropagates the simulation result up the tree.'''
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add the simulation value to the cumulative sum

        # Recursively backpropagate to the parent node
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_val(value))  # Flip value for the opponent