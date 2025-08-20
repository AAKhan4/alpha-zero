# alpha-zero

## MCTS

### mcts.py

"""
Monte Carlo Tree Search (MCTS) implementation.

This module provides an algorithm for decision-making in games or simulations
using a tree-based search strategy. MCTS balances exploration and exploitation
by simulating random playouts and updating node statistics to guide future
searches. Key components include:

- Selection: Traverses the tree using a selection policy (e.g., UCB1) to find a promising node.
- Expansion: Adds new child nodes to the tree for unexplored actions.
- Simulation: Performs random playouts from the expanded node to estimate outcomes.
- Backpropagation: Updates node statistics based on simulation results.

Suitable for domains with large state spaces and uncertain outcomes.
"""