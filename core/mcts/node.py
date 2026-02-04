# Represents a node in the Monte Carlo Tree Search (MCTS) tree
import copy
import numpy as np
from games.base_game import BaseGame, GameState


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

    def is_fully_expanded(self) -> bool:
        '''Returns True if the node has any children.'''
        return len(self.children) > 0

    def select(self) -> 'Node':
        '''Selects the child node with the highest UCB score.'''
        # Use max with a key function to find the child with the highest UCB score
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child: 'Node') -> float:
        '''Calculates the UCB score for a child node.'''
        # Q-value: normalized value of the node (scaled to [-1, 1])
        q = (child.value_sum / child.visit_count) if child.visit_count > 0 else 0
        # UCB formula: Q + exploration term
        return q + self.args['c'] * np.sqrt(np.log(self.visit_count + 1) / (child.visit_count + 1)) * child.prior

    def expand(self, policy: np.ndarray):
        '''Expands the node by creating child nodes for valid actions.'''
        # Optional: restrict expansion to top-K actions
        if policy.size > 20:
            top_k = 20
            top_actions = np.argsort(policy)[-top_k:]
        else:
            top_actions = np.nonzero(policy)[0]

        for action in top_actions:
            prob = policy[action]
            if prob <= 0:
                continue

            info = self.game_state.get_info()

            if not self.game.is_valid_action(info, action):
                continue

            # Apply move
            next_info = self.game.get_next_state(info, action)
            next_info = self.game.change_perspective(next_info)

            child_state = self.game.get_state_type()(game=self.game)
            child_state.update(next_info)

            child = Node(game=self.game, args=self.args, game_state=child_state, parent=self, action=action, prior=prob)

            self.children.append(child)

    def backpropagate(self, value: float) -> None:
        '''Backpropagates the simulation result up the tree.'''
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add the simulation value to the cumulative sum

        # Flip the value for the opponent's perspective
        value = self.game.get_opponent_val(value)

        # Recursively backpropagate to the parent node
        if self.parent is not None:
            self.parent.backpropagate(value)