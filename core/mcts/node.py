# Represents a node in the Monte Carlo Tree Search (MCTS) tree
import copy
import numpy as np
from games.base_game import BaseGame, GameState

MIN_BRANCHING_FACTOR = 8  # Minimum number of child nodes to expand per node if available valid actions exceed this number
MAX_BRANCHING_FACTOR = 12  # Maximum number of child nodes to expand per node
MAX_ROOT_EXPANSION = 16  # Maximum number of child nodes to expand for the root node
MAX_EXPANSION_TEMP = 1.2  # Max temperature for controlling expansion with get top actions
MIN_EXPANSION_TEMP = 0.5  # Min temperature for controlling expansion with get top actions
PASS_CAP_FACTOR = 500  # Factor to determine the cap for pass action probability based on action count

class Node:
    def __init__(self, game: BaseGame, args: dict, action: int = None, parent: 'Node' = None, prior: float = 0, visit_count: int = 0):
        self.game = game  # Game logic object
        self.c = args['c']  # MCTS parameters (e.g., exploration constant)
        self.alpha = args['alpha']  # Dirichlet noise parameter for exploration
        self.epsilon = args['epsilon']  # Exploration noise weight
        self.parent = parent  # Parent node in the tree
        self.game_state: GameState = None if parent else self.game.get_state_type()(self.game)  # Game state at this node, will be set when rebuilding the state from the root
        self.prior = prior  # Prior probability of selecting this action
        self.children: dict[int, Node] = {}  # Dictionary of child nodes
        self.visit_count = visit_count  # Number of times this node was visited
        self.value_sum = 0.0  # Cumulative value from simulations
        self.depth = parent.depth + 1 if parent else 0  # Depth of the node in the tree
        self.action = action  # Action taken to reach this node from its parent

    def is_fully_expanded(self) -> bool:
        '''Returns True if the node has any children.'''
        state = self.rebuild_state()  # Rebuild the game state by applying moves from the root to this node
        valid_actions = self.game.get_valid_actions(state.get_info()).sum()
        max_expansion = MAX_ROOT_EXPANSION if self.depth == 0 else max(MIN_BRANCHING_FACTOR, MAX_BRANCHING_FACTOR - self.depth)
        return len(self.children) > 0 and len(self.children) >= min(max_expansion, valid_actions)  # Consider node fully expanded if it has at least 20 children or all valid actions are expanded
    
    def rebuild_state(self, cache: bool = False) -> GameState:
        '''Rebuilds the game state by applying the moves from the root to this node.'''
        if self.game_state is not None:
            return self.game_state  # Return existing game state if already built

        actions = []
        temp = self
        while temp.parent is not None:
            actions.append(temp.action)  # Collect actions from this node up to the root
            temp = temp.parent
        state = copy.deepcopy(temp.game_state)  # Start with the game state at the root
        for action in actions[::-1]:  # Apply actions in reverse order to rebuild the state
            if action is not None:
                info = state.get_info()
                next_info = self.game.get_next_state(info, action)  # Get the next game state after applying the action
                state.update(next_info)  # Update the state with the new game information

        if cache:
            self.game_state = state  # Cache the rebuilt game state for future use

        return state
    
    def make_root(self):
        '''Transforms this node into the new root by detaching it from its parent and resetting depth.'''
        state = self.rebuild_state()  # Rebuild the game state for this node
        self.parent = None  # Detach from parent to save memory
        self.action = None  # Clear the action since this is now the root
        self.depth = 0  # Reset depth for the new root
        self.game_state = state  # Update the game state to the rebuilt state

    def select(self) -> 'Node':
        '''Selects the child node with the highest UCB score.'''
        # Use max with a key function to find the child with the highest UCB score
        return max(self.children.values(), key=self.get_ucb)

    def get_ucb(self, child: 'Node') -> float:
        '''Calculates the UCB score for a child node.'''
        # Q value is calculated as 1 - (value_sum / visit_count) to represent the win rate for the current player, with a default of 1 for unvisited nodes to encourage exploration
        q = 1 - (child.value_sum / child.visit_count) if child.visit_count > 0 else 1
        # UCB formula: Q + exploration term
        return q + (self.c * child.prior * np.sqrt(self.visit_count + 1) / (1 + child.visit_count))

    def get_top_actions(self, policy: np.ndarray, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        '''Returns the indices of the k actions based on the policy probabilities.'''
        k = max(MIN_BRANCHING_FACTOR, MAX_BRANCHING_FACTOR - self.depth)  # Limit to top-K actions for expansion
        if self.depth == 0:
            k = MAX_ROOT_EXPANSION  # Expand more actions at the start for better exploration
        
        temp = MAX_EXPANSION_TEMP - ((MAX_EXPANSION_TEMP - MIN_EXPANSION_TEMP) * (self.depth / 10))  # Decrease temperature with depth
        if self.game.can_pass:
            temp = MAX_EXPANSION_TEMP - ((MAX_EXPANSION_TEMP - MIN_EXPANSION_TEMP) * (state.action_count / self.game.max_game_length))  # Decrease temperature with action count

        valid_actions = self.game.get_valid_actions(state.get_info())
        policy *= valid_actions  # Mask invalid actions by setting their probabilities to zero
        policy = policy ** (1 / temp)  # Apply temperature to control exploration vs exploitation
        policy /= np.sum(policy) if np.sum(policy) > 0 else 1  # Re-normalize the policy after masking invalid actions

        # Probabilistically sample top-k actions based on their probabilities
        action_indices = np.arange(len(policy))
        top_actions = np.random.choice(
            action_indices, size=min(k, valid_actions.sum()), replace=False, p=policy
        )

        # Ensure pass action is included if it's valid and not in top actions
        if self.game.can_pass:
            pass_action = self.game.action_size - 1
            if valid_actions[pass_action] and pass_action not in top_actions:
                # Find the action with the lowest probability in top_actions and replace it with pass action
                min_idx = np.argmin([policy[action] for action in top_actions])
                top_actions[min_idx] = pass_action

        # Create an updated policy that only includes the selected top actions
        updated_pol = np.zeros_like(policy)
        updated_pol[top_actions] = policy[top_actions]
        updated_pol /= np.sum(updated_pol) if np.sum(updated_pol) > 0 else 1  # Re-normalize after selecting top actions

        return updated_pol, top_actions

    def create_child(self, info: dict, action: int, policy: float) -> 'Node':
        '''Creates a child node for the given action if it's valid.'''
        prob = policy[action]
        child = self.children.get(action)
        if child is not None:
            child.prior = prob  # Update prior if child already exists (e.g., from a previous expansion)
            return child  # Return existing child if it already exists
        if not self.game.is_valid_action(info, action):
            return None
        child = Node(game=self.game, args={'c': self.c, 'alpha': self.alpha, 'epsilon': self.epsilon}, action=action, parent=self, prior=prob)
        return child

    def prune_branch(self, top_actions: list[int]):
        '''Prunes child nodes that are not in the top actions list.'''
        for action in list(self.children.keys()):
            if action not in top_actions:
                del self.children[action]  # Remove child nodes that are not in the top actions list
    
    def add_dirichlet_noise(self, policy: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
        '''Adds Dirichlet noise to the given policy for exploration.'''
        if self.parent is not None:
            return policy  # Only add noise at the root node for better exploration
        noise = np.random.dirichlet([alpha] * self.game.action_size)  # Generate Dirichlet noise
        noisy_policy = (1 - epsilon) * policy + epsilon * noise  # Mix original policy with noise
        return noisy_policy

    def cap_pass_probability(self, policy: np.ndarray, state: GameState) -> np.ndarray:
        '''Caps the probability of the pass action to prevent it from dominating other actions.'''
        if not self.game.can_pass:
            return policy  # No pass action to cap
        pass_action = self.game.action_size - 1
        pass_cap = state.action_count / PASS_CAP_FACTOR if state.action_count < 60 else 0.15  # Allow higher pass probability
        if policy[pass_action] <= 0:
            policy[pass_action] = 0.001  # Assign a small probability to the pass action if it's zero to ensure it can be selected
        elif policy[pass_action] > pass_cap:
            policy[pass_action] = pass_cap  # Cap the pass action probability to prevent it from dominating other actions
        policy /= np.sum(policy)  # Re-normalize after adjusting pass action probability
        return policy

    def expand(self, policy: np.ndarray):
        '''Expands the node by creating child nodes for valid actions.'''
        state = self.rebuild_state(cache=True)  # Rebuild the game state by applying moves from the root to this node

        # Cap pass & add dirichlet noise to the policy for better exploration at the root node
        policy = self.cap_pass_probability(self.add_dirichlet_noise(policy, self.alpha, self.epsilon), state)

        # Restrict expansion to top-K actions
        policy, top_actions = self.get_top_actions(policy, state)
        if np.sum(policy) == 0:
            # If all actions have zero probability
            raise ValueError("All actions have zero probability after masking invalid actions. State info: {}".format(state.get_info()))
        
        self.prune_branch(top_actions)  # Prune branches that are not in the top actions list

        info = state.get_info()

        for i in range(len(top_actions)):
            action = top_actions[i]
            child = self.create_child(info, action, policy)
            if child is not None:
                self.children[action] = child # Add the new child node to the children dictionary

    def backpropagate(self, value: float) -> None:
        '''Backpropagates the simulation result up the tree.'''
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add the simulation value to the cumulative sum

        # Recursively backpropagate to the parent node
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_val(value))  # Flip value for the opponent