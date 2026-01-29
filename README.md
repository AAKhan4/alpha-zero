# AlphaZero-Style Reinforcement Learning Framework

An end-to-end reinforcement learning framework for two-player games, combining supervised pretraining from human-gameplay data with AlphaZero-style self-play and Monte Carlo Tree Search. Allows model to learn games when given rules & (optional) human game data.

## Motivation

I was introduced to the game of Go by a friend and quickly became very interested in the game due to its simple rules yet complex nature.
To learn the game & improve my skills I signed-up on various popular Go websites to play against bots & slowly learn the game, quickly realising that most of these platforms impose strict rate limits on gameplay vs bots. This led me to explore how such bots are trained & implemented. 

This project is a gap-year research-style exploration of modern reinforcement learning systems, focusing on self-play, Monte Carlo Tree Search, and neural network training pipelines.
The primary objective is to design, implement, and evaluate an end-to-end game agent that learns through a combination of human data and self-play. First implemented Tic-Tac-Toe & Connect Four before moving on to more complex game of Go.

Initially inspired by:
- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)

## Features Overview

- **AlphaZero (Core):** Orchestrates self-play & training loops + model updates. Uses batching and multiprocessing for efficient data generation & learning.
- **MCTS & ResNet:** Implements tree search and inference for move selection and game outcome estimation.
- **Games:** Modular implementations for Tic-Tac-Toe, Connect Four, & Go, built on a shared `BaseGame` class for extensibility.
- **Training:** Modules for running, evaluating, and configuring model training.

### System Overview

Human SGF Games
      ↓
Supervised Policy Pretraining
      ↓
Neural Network (Policy + Value)
      ↓
MCTS-Guided Self-Play
      ↓
Replay Buffer
      ↓
Iterative Training Loop

The system iteratively generates its own training data through self-play and improves without relying on external evaluation engines.

### Key Concepts

- **Supervised Pretraining:** The model policy (move probabilities) is trained on data extracted from human games samples.
  - Provides a better starting point for AlphaZero-style training for more complex games such as Go. Inspired by AlphaGo.
- **Self-Play:** AlphaZero generates training data by playing games against itself using MCTS guided by the neural network.
- **Training:** The ResNet model is trained on self-play data to predict policy (move probabilities) and value (expected outcome).
- **MCTS:** Builds a search tree to explore moves, balancing exploration and exploitation.
- **Flexibility:** Easily add new games implementing BaseGame, change model/training parameters or build up on the current implementation of AlphaZero.

### MCTS

Monte Carlo Tree Search (MCTS) is a search algorithm used to make decisions in games by simulating potential future moves and outcomes. In this implementation, MCTS is tightly integrated with the neural network model to guide self-play and improve decision-making. The key steps in the MCTS process are:

1. **Selection**: Starting from the root node, the algorithm traverses the tree by selecting child nodes with the highest Upper Confidence Bound (UCB) score. This balances exploration (trying less-visited nodes) and exploitation (choosing nodes with higher value estimates).

2. **Expansion**: When a node is selected that is not fully expanded, it is expanded by creating child nodes for all valid actions. The neural network provides a policy (probability distribution over actions) to guide this expansion.

3. **Simulation**: For expandable nodes, the neural network predicts the policy (move probabilities) and value (expected outcome) for the current state. This information is used to simulate the game further.

4. **Backpropagation**: The result of the simulation (value) is propagated back up the tree, updating the visit counts and value estimates of all nodes along the path.

5. **Self-Play**: The root node's child with the highest visit count is selected as the next move, and the process repeats for subsequent game states.

Key features of this implementation:
- **Neural Network Integration**: The ResNet model predicts both the policy (action probabilities) and value (expected outcome) for each state, enabling efficient and informed search.
- **Parallelization**: MCTS is performed for multiple self-play games in parallel, leveraging batching for efficiency.
- **Exploration Noise**: Dirichlet noise is added to the root node's policy during self-play to encourage exploration and prevent the model from overfitting to specific strategies.
- **Tree Representation**: Each node in the tree represents a game state, storing visit counts, value estimates, and prior probabilities for actions.

This implementation ensures that the model learns by iteratively improving its policy and value predictions through self-play, guided by MCTS.

## Project Repository Structure

```
alpha-zero/
  core/               # AlphaZero - Core Implementation
    mcts/             # MCTS & ResNet
  data/               # Human gameplay data for Go (raw data, data parser & processed data)
  games/              # Game logic for BaseGame + Tic-Tac-Toe, Connect Four & Go
  training_scripts/   # Scripts to run training, config & evaluation
  models/             # Saved models and logs (not added to GitHub)
```

## Usage (Training Models)

- Use `training_scripts/data_processing.py` to process raw human-gameplay data for supervised training.
- Configure training parameters in `training_scripts/training_args.py`.

- Run `training_scripts/model_training.py` to start training with `flag` for supervised learning, reinforcement learning, or both:
  - **Arguments for `model_training.py`:**
    - `--game`: Specifies the game to train on. Options include:
      - `go`: Train on the game of Go.
      - `tic_tac_toe`: Train on Tic-Tac-Toe (default).
      - `connect_four`: Train on Connect Four.
    - `--model`: Path to the directory containing a pre-trained model to continue training from. If not provided, training starts from scratch.
    - `--flag`: Specifies the training mode. Options include:
      - `rl`: Reinforcement learning (default).
      - `sl`: Supervised learning.
      - `sl+rl`: Supervised pretraining followed by reinforcement learning.
    - `--data`: Path to the directory containing training data. This is required for supervised learning (`sl` or `sl+rl`) and defaults to `./data/processed_data/go/9x9`.

- Visualise avg loss rates for models graphically with `training_scripts/eval_training.py`:
  - **Arguments for `eval_training.py`:**
    - `--game`: Specifies the game to evaluate. Options include:
      - `go`: Evaluate the game of Go.
      - `tic_tac_toe`: Evaluate Tic-Tac-Toe.
      - `connect_four`: Evaluate Connect Four.
    - `--flag`: Specifies the training mode to evaluate. Options include:
      - `rl`: Reinforcement learning.
      - `sl`: Supervised learning.

  This script reads the loss files from the `models/<game>/<flag>/losses` directory and plots the mean loss over epochs for all iterations.

- Models and logs are saved in the `models/` directory.

## Game Engine

All games built on shared `BaseGame` class, which provides a modular and extensible foundation for games. This design streamlines the implementation and facilitates easy addition of new games.

All games store state information (current board, player, etc.) in `GameState`.

### Go Engine & Rules

- Board size: 9×9
- Scoring: Chinese area scoring
- Ko: simple ko enforced
- Rule enforcement: sgfmill board
- Neural representation: NumPy array (3-plane encoding)

#### Design Decisions

- 9×9 board chosen to balance strategic depth with computational feasibility.
- Chinese area scoring used to simplify end-game evaluation.

## State Representation

Each board position is encoded using a 3-plane representation for model training:
- current player's stones
- opponent stones
- empty intersections

All states are normalised to the current player's perspective.

## Training Pipeline

Training proceeds in iterative cycles consisting of:
1. Optional supervised pretraining on human game data
2. MCTS-guided self-play to generate new game trajectories
3. Training the policy and value networks on accumulated self-play data
4. Repeating the process with the updated model

Evaluation of this pipeline is currently in progress.

## Current Status

- Supervised pretraining implemented and functional
- Self-play training loop implemented
- Models can self-play and be evaluated programmatically
- Evaluation experiments are currently being conducted

## Evaluation Plan (In Progress)

The system is evaluated across three training regimes:
- Supervised learning only (SL)
- Self-play reinforcement learning only (RL)
- Supervised pretraining followed by self-play (SL+RL)

Planned evaluation metrics include:
- Win rate against a fixed random baseline for all models
- Cross-play win rates between SL, RL, and SL+RL models
- Learning curves showing win rate vs training iteration
- Comparison of training time and convergence speed
- Time-to-threshold analysis (iterations/time required to reach a fixed win rate)

These evaluations are designed to compare both final performance and training efficiency.

## Preliminary Observations

Initial experiments suggest that supervised pretraining improves early self-play stability and accelerates convergence compared to reinforcement learning from scratch.

## Limitations & Future Work

- Experiments currently limited to 9×9 Go
- Limited human game dataset for pretraining
- Simple ko rule (no superko)
- Future work includes larger board sizes, extended evaluation, and league-based comparisons

## References

- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)
- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [AlphaZero Explained – Nik Cheerla](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [Sgfmill](https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/)
