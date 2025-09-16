# MCTS & ResNet: AlphaZero Core

This directory contains the Monte Carlo Tree Search (MCTS) and Residual Neural Network (ResNet) components for AlphaZero-style reinforcement learning on board games like Tic Tac Toe and Connect Four.

<br>

## ResNet Architecture

The ResNet processes board states to predict move probabilities and game outcomes.

### Key Components:
- **Start Block:** Performs initial convolution and normalization.
- **Residual Blocks:** Extract deep features through stacked layers.
- **Policy Head:** Outputs probabilities for valid moves.
- **Value Head:** Predicts the expected game outcome.

<br>

## MCTS Process

MCTS builds a search tree to explore possible moves and outcomes.

### Steps:
1. **Selection:** Traverse the tree using UCB scores to find the best node.
2. **Expansion:** Add new nodes using ResNet predictions for policy and value.
3. **Simulation:** Backpropagate results for terminal nodes.
4. **Backpropagation:** Update visit counts and values along the path to the root.

This iterative process improves policy and value estimates, guiding self-play and training.

<br>

## Summary

The `mcts` module integrates ResNet predictions with MCTS to enable efficient decision-making and model improvement through self-play.
