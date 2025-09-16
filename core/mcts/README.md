# AlphaZero MCTS & ResNet Implementation

This module implements Monte Carlo Tree Search (MCTS) and a Residual Neural Network (ResNet) for AlphaZero-style reinforcement learning on board games such as Tic Tac Toe and Connect Four.
<br>

## ResNet Architecture

- **Start Block:** Initial convolution and batch normalization.
- **Residual Blocks:** Stack of residual layers for deep feature extraction.
- **Policy Head:** Outputs action probabilities for all valid moves.
- **Value Head:** Outputs a scalar value representing the expected outcome.

The ResNet takes the encoded board state as input and predicts both the policy (move probabilities) and value (game outcome).
<br>

## Monte Carlo Tree Search (MCTS)

MCTS is used to explore possible moves and outcomes by building a search tree:

The `search` function performs the core MCTS operations for multiple parallel games. It begins by predicting the initial policy and value for the given states using the ResNet model. Dirichlet noise is added to the policy for exploration, and invalid actions are masked out. The function then initializes root nodes for all games and iteratively performs the following steps:

1. **Selection:** Traverse the tree from the root to find a node to expand, using UCB scores to select the most promising child nodes.
2. **Expansion:** Expand the selected node by predicting the policy and value for its state using the ResNet model. The policy is adjusted to account for valid actions.
3. **Simulation:** If the selected node is terminal, backpropagate the result immediately. Otherwise, add the node to the list of expandable nodes.
4. **Backpropagation:** Update the visit counts and values of nodes along the path from the expanded node back to the root.

This process is repeated for a specified number of searches, refining the policy and value estimates for each game.

The search results are used to guide self-play and training.
<br>
