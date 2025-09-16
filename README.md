
# AlphaZero: Self-Play Reinforcement Learning for Board Games

This project implements AlphaZero-style reinforcement learning for board games, including Tic Tac Toe and Connect Four. It combines Monte Carlo Tree Search (MCTS) and a deep Residual Neural Network (ResNet) to learn optimal strategies through self-play and training.

## Project Inspiration

Inspired by:
- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)

## Features Overview

- **AlphaZero (Core):** Orchestrates self-play, training, and model management. Uses batching and multiprocessing for efficient data generation & learning.
- **MCTS & ResNet:** Implements search and prediction for move selection and game outcome estimation.
- **Games:** Modular implementations for Tic Tac Toe and Connect Four, built on a shared `BaseGame` class for extensibility.
- **Training:** Modules for running, evaluating, and configuring model training.

## Project Structure

```
alpha-zero/
  core/           # AlphaZero - Core Implementation
    mcts/           # MCTS & ResNet
  games/          # Game logic for Tic Tac Toe & Connect Four
  training/       # Training, evaluation, and configuration
  models/         # Saved models and logs
```

## Key Concepts

- **Self-Play:** AlphaZero generates training data by playing games against itself using MCTS guided by the neural network.
- **Training:** The ResNet model is trained on self-play data to predict policy (move probabilities) and value (expected outcome).
- **MCTS:** Builds a search tree to explore moves, balancing exploration and exploitation.
- **Modularity:** Easily add new games, change model/training parameters or build up on the current implementation of AlphaZero.

## Usage

- Configure training parameters in `training/training_args.py`.
- Run `training/model_training.py` to start training.
- Visualise avg loss rates for models graphically with `training/eval_training.py`.
- Use `game_manager.py` to play against the trained model.
- Models and logs are saved in the `models/` directory.

## References

- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Residual Networks – GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/)
- [AlphaZero Explained – Nik Cheerla](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)