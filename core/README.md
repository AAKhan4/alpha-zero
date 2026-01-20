# Core Directory: AlphaZero and MCTS

This directory contains the core components of an AlphaZero-style framework for training AI to play board games like Tic Tac Toe and Connect Four.

<br>

## AlphaZero

AlphaZero combines deep learning and Monte Carlo Tree Search (MCTS) to iteratively improve its model through self-play.

### Key Components:
- **Neural Network (ResNet):** Predicts move probabilities and game outcomes.
- **MCTS:** Guides decisions by simulating moves.
- **Self-Play:** Generates training data by playing against itself.
- **Training Loop:** Updates the model to minimize prediction errors.

<br>

### Features:
- **Self-Play:** Uses MCTS to generate games.
- **Training:** Optimizes the ResNet model with self-play data.
- **Evaluation:** Compares the current model with previous versions.

<br>

## MCTS Sub-Directory

The `mcts` sub-directory implements MCTS and the ResNet model.

### ResNet Architecture:
- **Start Block:** Initial convolution and normalization.
- **Residual Blocks:** Extract deep features.
- **Policy Head:** Outputs probabilities for valid moves.
- **Value Head:** Outputs the expected game outcome.

The ResNet processes the board state to predict both the policy (move probabilities) and value (game outcome).

<br>

### MCTS Process:
1. **Selection:** Traverse the tree using UCB scores to find the best node.
2. **Expansion:** Add new nodes using ResNet predictions for policy and value.
3. **Simulation:** Backpropagate results for terminal nodes.
4. **Backpropagation:** Update visit counts and values along the path to the root.

This iterative process refines the policy and value estimates, guiding self-play and training.

<br>

## Summary

The `core` directory integrates the `alpha_zero` module for managing training and the `mcts` sub-directory for search and prediction. Together, they enable AI to improve through self-play.
