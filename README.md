# AlphaZero-Style Reinforcement Learning Framework

An end-to-end reinforcement learning framework for two-player games, combining supervised pretraining from human-gameplay data with AlphaZero-style self-play and Monte Carlo Tree Search. Allows model to learn games when given rules & (optional) human game data.

## Motivation

I was introduced to the game of Go by a friend and quickly became very interested in the game due to its simple rules yet complex nature.
To learn the game & improve my skills, I signed up for various popular Go websites to play against bots & slowly learn, quickly realising that most of these platforms impose strict rate limits on gameplay vs bots. This led me to explore how such bots are trained & implemented. 

This project is a gap-year research-style exploration of modern reinforcement learning systems, focusing on self-play, Monte Carlo Tree Search, and neural network training pipelines.
The primary objective is to design, implement, and evaluate an end-to-end game agent that learns through a combination of human data and self-play. First implemented Tic-Tac-Toe & Connect Four before moving on to the more complex game of Go.

Initially inspired by:
- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)

## Features Overview

- **AlphaZero (Core):** Orchestrates self-play & training loops + model updates. Uses batching and multiprocessing for efficient data generation & learning.
- **MCTS & ResNet:** Implements tree search and inference for move selection and game outcome estimation.
- **Games:** Modular implementations for Tic-Tac-Toe, Connect Four, & Go, built on a shared `BaseGame` class for extensibility.
- **Training:** Modules for running, evaluating, and configuring model training.

### System Overview

Human SGF Games <br>
&nbsp;&nbsp;&nbsp;&nbsp; ↓ <br>
Supervised Policy Pretraining <br>
&nbsp;&nbsp;&nbsp;&nbsp; ↓ <br>
Neural Network (Policy + Value) <br>
&nbsp;&nbsp;&nbsp;&nbsp; ↓ <br>
MCTS-Guided Self-Play <br>
&nbsp;&nbsp;&nbsp;&nbsp; ↓ <br>
Replay Buffer <br>
&nbsp;&nbsp;&nbsp;&nbsp; ↓ <br>
Iterative Training Loop <br>

The system iteratively generates its own training data through self-play and improves without relying on external evaluation engines.

### Key Concepts

- **Supervised Pretraining:** The model policy (move probabilities) is trained on data extracted from human game samples.
  - Provides a better starting point for AlphaZero-style training for more complex games such as Go. Inspired by AlphaGo.
- **Self-Play:** AlphaZero generates training data by playing games against itself using MCTS guided by the neural network.
- **Training:** The ResNet model is trained on self-play data to predict policy (move probabilities) and value (expected outcome).
- **MCTS:** Builds a search tree to explore moves, balancing exploration and exploitation.
- **Flexibility:** Easily add new games implementing BaseGame, change model/training parameters or build up on the current implementation of AlphaZero.

### MCTS

Monte Carlo Tree Search (MCTS) is a search algorithm used to make decisions in games by simulating potential future moves and outcomes. In this implementation, MCTS is tightly integrated with the neural network model to guide self-play and improve decision-making. The key steps in the MCTS process are:

1. **Selection**: Starting from the root node, the algorithm traverses the tree by selecting child nodes with the highest Upper Confidence Bound (UCB) score. This balances exploration (trying less-visited nodes) and exploitation (choosing nodes with higher value estimates).

2. **Expansion**: When a node is selected that is not fully expanded, it is expanded by creating child nodes for all valid actions. The neural network provides a policy (probability distribution over actions) to guide this expansion.

3. **Simulation**: For expandable nodes, the neural network predicts the policy (move probabilities) and value (expected outcome) for the current state. This information is used to further simulate the game.

4. **Backpropagation**: The result of the simulation (value) is propagated back up the tree, updating the visit counts and value estimates of all nodes along the path.

5. **Self-Play**: The root node's child with the highest visit count is selected as the next move, and the process repeats for subsequent game states.

Key features & considerations of this implementation:
- **Neural Network Integration**: The ResNet model predicts both the policy (action probabilities) and value (expected outcome) for each state, enabling efficient and informed search.
- **Parallelisation**: MCTS is performed for multiple self-play games in parallel, leveraging batching for efficiency.
- **Exploration Noise**: Dirichlet noise is added to the root node's policy during self-play to encourage exploration and prevent the model from overfitting to specific strategies.
- **Tree Representation**: Each node in the tree represents a game state, storing visit counts, value estimates, and prior probabilities for actions.
- **Sub-tree Reuse**: After selecting an action during self-play, the corresponding child node becomes the new root for subsequent MCTS iterations, preserving prior search statistics and reducing redundant computation.
- **Limited Expansion**: Expands top-K actions by policy probability, with K decreasing at deeper depths (root: 30 actions, deeper nodes: 10), reducing computation for large action spaces.

This implementation ensures that the model learns by iteratively improving its policy and value predictions through self-play, guided by MCTS.

## Project Repository Structure

```
alpha-zero/
  core/               # AlphaZero - Core Implementation
    mcts/             # MCTS & ResNet
  data/               # Human gameplay data for Go (raw data, data parser & processed data)
  games/              # Game logic for BaseGame + Tic-Tac-Toe, Connect Four & Go
  training_scripts/   # Scripts to run model training & config - saces models in ./models
  eval_scripts/       # Scripts to run evaluation of models - saves results in ./evaluation
  models/             # Saved models and logs (not added to GitHub)
  evaluation/         # Saved data from models evaluation (not added to GitHub)
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

All games are built on a shared `BaseGame` class that provides a modular, extensible foundation. This design streamlines implementation and makes it easy to add new games.

All games store state information (e.g., the current board, player, etc.) in `GameState`.

### Go Engine & Rules

- Board size: 9×9
- Scoring: Chinese area scoring
- Ko: simple ko enforced
- Rule enforcement: sgfmill board
- Neural representation: NumPy array (3-plane encoding)

#### Design Decisions

- 9×9 board chosen to balance strategic depth with computational feasibility.
- Chinese area scoring is used to simplify end-game evaluation.

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

- Supervised pretraining implemented and functional.
- Self-play training loop implemented.
- Models can self-play and be evaluated programmatically.
- Evaluation experiments are currently being conducted.

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

## Evaluation

### Overview

The evaluation of performance and training dynamics of the implemented AlphaZero-inspired framework were conducted under three training regimes:
- Supervised Learning (SL)
- Reinforcement Learning (RL)
- Hybrid Supervised + Reinforcement Learning (SL + RL)

All experiments were conducted for the 9x9 variant of the game Go, using identical neural network architectures to ensure that performance differences arise primarily from training methodology rather than model capacity.

Evaluation focuses on various key aspects of training behaviour and model performance.

### Training Dynamics

The training process is analysed with respect to several important characteristics.

#### Exploration/Exploitation Trade-offs

For RL models, training relies on MCTS to guide self-play. The balance between exploration and exploitation is controlled through parameters such as:
- exploration constant: C
- Dirichlet noise parameters: aplha (α) and epsilon (ϵ)
- temperation scheduling during action sampling

These parameters significantly influence the diversity and quality of states encountered during training and therefore the stability of policy learning.

The experiments investigate how different exploration settings affect:
- diversity of generated states
- stability of policy improvements
- robustness against stochastic opponents

#### Convergence Behaviour

Convergence is evaluated by monitoring how model performance evolves over training iterations.

Key indicator of convergence include:
- improvement in win rate over training iterations
- stability of policy predictions
- reduction in training loss
- consistency of MCTS search results

Due to the limited compute budget, convergence in primarily evaluated through model performance trends rather than theoretical optimality.

#### Performance Evaluation

Model strength is measured through competitive play against baseline opponents.

Two main baselines are used as follows.

##### Performance Vs Random Model

The main evaluation metric is win rate against a uniformly random policy agent.

This benchmark provides a useful measure of:
- general game understanding
- robustness to unpredictable play
- ability to capitalise on weak opponent moves

Random opponents often produce highly irregular board states. As a result, strong performance against random play indicates that the model has learnt generalisable strategy patterns rather than memorising deterministic move sequences.

##### Performance Vs SL Model

For RL models, additional evaluation compares performance against SL model trained on professional games data.

This comparison helps measure:
- the extent to which self-play improves beyond imitation learning
- the effectiveness of reinforcement learning in refining policy quality
- the ability of self-play to discover strategies not present in the supervised dataset

#### Training Efficiency

In addition to raw performance, training efficiency is considered.

Important factors include:
- total training time
- number of self-play games required
- number of MCTS searches per move
- rate of improvement across training iterations

Because the system was developed under limited computational resources, careful parameter tuning was required to maximise learning efficiency while maintaining feasible training times.

#### Practical Constraints

A key limitation of the experiments is the available compute environment.

Training was conducted on consumer hardware with restricted:
- RAM capacity
- CPU parallelism
- overall training time budget

These constraints influenced several design decisions, including:
- limiting number of MCTS simulations per move
- restricting replay buffer size
- using relatively small neural network architectures
- balancing number of self-play games per iteration with search depth

Despite these constraints, the experiments aim to demonstrate that AphaZero-style training remains effective at smaller scales when parameters are carefully tuned.

### Common Training Configuration

All experiments share a common neural network architecture and core training framework. This ensures that differences in performance arise primarily from the training method rather than architectural variation.

#### Neural Network Architecture

The policy and value networks use a residual convolutional architecture similar to the one descibed in AlphaZero.

Architecture summary:
|Component      |Specification            |
|---|---|
|Backbone       |Residual NN              |
|Residual Blocks|8                        |
|Channels       |64                       |
|Outputs        |Policy head + Value head |

The policy head outputs a policy distribution over all legal actions, while the value head predicts the game outcome from the current player's perspective.

This architecture represents a compromise between model capacity and computational efficiency, allowing training to remain feasible under limited hardware resources.

#### State Representation

Game states are encoded as multi-channel tensors representing board occupancy.

Encoding includes separate channels for:
- player stones
- empty spaces
- opponent stones

States are always represented from the perspective of the current player to play the next move. This ensures that the network learns symmetric strategic patterns independent of player colour.

#### Optimisation

All models are trained using gradient-based optimisation with following parameters:
|Parameter      |Value                    |
|---|---|
|Optimiser       |Adam              |
|Learning Rate  |8e-5                        |
|Weight Decay       |1e-4                   |
|Batch Size        |128 |

Loss functions combine policy and value objectives:
- Policy loss: KL-divergence between predicted policy and target distribution
- Value loss: mean squared error between predicted value and game outcome

#### Replay Buffer

For RL models, training data generated through self-play is stored in a replay-buffer.

|Parameter      |Value                    |
|---|---|
|Replay Buffer Size       |120000 samples |

This buffer size provides a balance between:
- retaining sufficient training diversity
- avoiding excessive memory consumption
- preventing early training data from dominating later

#### Hardware Environment

Experiments were conducted on a limited individual system with constrained memory and processing resources.

To maintain stable training under these conditions:
- the number of parallel self-play workers was limited
- MCTS search counts were restricted
- branching factor for MCTS node expansion were capped

These considerations significantly influenced the final training configuration along with models learning behaviour.

### Supervised Learning (SL)

#### Overview

The SL stage trains the policy network using a dataset of professional-level move selections. The objective of this stage is to learn strong prior move probabilities that reflect human or expert play patterns. These learned priors can then be evaluated directly or used to initialise reinforcement learning (RL) through self-play.

Unlike RL approaches such as AlphaZero, SL relies entirely on existing gameplay data rather than exploration-driven self-play. As a result, the model learns to imitate observed moves but does not directly optimise for game outcomes.

This section evaluates how effectively SL training alone produces a competitive policy. and how well such a model generalises when facing opponents that produce non-expert/more diverse game states.

#### Dataset and Preprocessing

The supervised dataset contains 170,000+ state-action pairs.

Each sample consists of:
- a board state from current player perspective
- the move selected in professional game in position

Game states were converted into the same tensor representation used during RL. The representation includes separate channels indicating the location of player stones, empty spaces, and opponent stones.

To improve state-action diversity in dataset and model generalisation, two data augmentation steps were taken.

- **State-Action Transformations**:
  Each board state and action pair is augmented using all possible rotations (0°, 90°, 180°, 270°) and horizontal reflections. For each transformation, both the board tensor and the action index are adjusted to match the new orientation. This increases dataset diversity and helps the model generalise to symmetric positions.

  Specifically, for each original state-action pair:
  - Four rotated versions are generated (rotating the board and action coordinates)
  - Each rotated version is also horizontally flipped, producing eight total variants per original pair
  - Action indices are recalculated to correspond to the transformed board

  This augmentation ensures the model learns from a wider range of spatial patterns and reduces overfitting to specific board layouts.

- **Action Perturbation**:
  To further increase dataset diversity and robustness, action perturbation is applied to a subset of state-action pairs. With a probability of 12%, the original action is perturbed to a nearby valid move within a radius of up to two spaces from the original location. The perturbation process attempts up to 10 random offsets, ensuring the new action is within board bounds and valid according to game rules.

  Specifically:
  - For each original action (except pass moves), random row and column offsets are generated within a radius of 2.
  - If the resulting move is valid (not violating game rules and within board limits), it is used as a perturbed action.
  - The perturbed action and its corresponding board state are then augmented using the same set of rotations and reflections as above, producing additional variants.
  - If no valid perturbation is found after 10 attempts, no perturbed action is added for that sample.

  By exposing the model to a wider variety of move choices and board states, this augmentation helps it generalise beyond expert play and adapt to unconventional strategies. In games like Go, where overall shape and territory are key, nearby moves often preserve strategic patterns. This approach encourages the model to recognise and respond to broader spatial concepts, reducing overfitting to strictly professional moves.

#### Training Configuration

SL training uses the same neural network architecture and optimisation configuration described in the common training setup.

Key parameters used during SL training:
|Parameter      |Value                    |
|---|---|
|Epochs       |8              |
|Optimiser       |Adam              |
|Learning Rate  |1e-4                        |
|Weight Decay       |1e-4                   |
|Batch Size        |128 |

#### Policy Target Construction

Rather than training using purely deterministic target action, policy targets were smoothed to improve generalisation.

The target distribution assigns most probability mass to the move in dataset while retaining a small probability for other legal actions.

This smoothing reduces overfitting and encourages network to retain some uncertainty when evaluating unfamiliar states.

Without smoothing, the network quickly becomes overly deterministic, which negatively impacts robustness when encountering states that were not present in the supervised dataset.

#### Training Behaviour

##### Convergence Characteristics

During training, the SL model exhibits rapod loss reduction during early epochs, followed by gradual convergence.

In preliminary experiments, loss values typically plateau after 12-15 epochs. Training beyond this point produces little improvement in prediction accuracy but increases the tendency of model to become overly deterministic.

To prevent excessive overfitting to the dataset, training was therefore limited to a moderate number of epochs.

##### Policy Determinism

A notable property of purely supervised models iss their tendency to develop highly deterministic policies.

When trained for too many epochs, the model strongly prefers a single move in post positions. While this behaviour is desirable when the encountered states closely resemble the training data, it can be problematic when facing unpredictable opponents.

Random opponents frequently generate irregular board states that are unlikely to appear in expert gameplay datasets. In such situations, a deterministic policy may struggle to adapt and can make poor decisions due to a lack of alternative move exploration.

This behaviour highlights an important limitation of imitation-based training in complex games.

#### Performance Evaluation

##### Training Efficiency

SL training is computationally inexpensive compared to RL approaches. Because the dataset is static and no self-play generation is required, the training process completes quickly even on limited hardware.

For the experiments conducted in this project the metrics are as follows.

|Metric      |Value                    |
|---|---|
|Epochs       |8              |
|Sample Size       |170,136              |
|Batch Size        |128 |
|Training Time       |0h:2m:23s                   |

This short training time highlights one of the key advantages of the SL training regime. It can produce a reasonably competent policy in a fraction of the time required for RL methods.

However, the resulting policy is limited by the information contained within the dataset and cannot improve beyond the patterns present in the expert move distribution.

##### Performance Vs Random Model

To evaluate the practical strength of the SL model, games were played against a uniformly random opponent.

The random model selects from all legal actions with equal probability at each move. Although weak strategically, such opponent frequently generates irregular or suboptimal board states. This provides a useful test of whether the SL model can generalise beyond positions commonly seen in expert games.

For this evaluation, 1000 games are played between the SL and random models, with each alternating every game to play the first move.

The final SL model achieved the following results:
SL Wins: 609
Random Wins: 391
Draws: 0
Win Rate of SL over Random: 60.90%

This indicates that the model has learned a meaningful degree of game structure and is capable of exploiting many obvious mistakes made by the random opponent.

However, the win rate remains below what can typically by achieved with RL methods.

##### Performance Across Training Epochs

Performance was also measured across intermediate training checkpoints.

Interestingly, later stages of earlier checkpoints occasionally produced slightly higher win rates against the random model than the final training model.

For example, Epoch 6 produced a win rate of ~65% as compared to the final 60.90%

While the Epoch 6 model achieved a marginally higher win rate in this evaluation, the final Epoch 8 model was selected as the SL baseline for subsequent experiments.

This decision was made because the later model demonstrated more stable performance across repeated evaluation runs and over larger number of simulated games, whereas earlier checkpoints exhibited higher variance in results.

Earlier experiment showed that continuing training beyond 8 iterations, the SL performance against random model starts to eventually decay, leading SL winrate to dive far below 50%. This observation falls in line with earlier assumptions, as it is expected that excessive SL training leads to more deterministic play, making the model weaker and less robust against an opponent that often produces unexpected game states.

#### Strenghts and Limitations of SL

##### Strengths

Despite its limitations, SL provides several important benefits.

- **Training Time & Compute**
  Compared to RL, an SL training regime requires much less training time and computational resources to produce tangible results.

- **Strong Initial Policy**
  The Supervised model learns meaningful move priors that reflect human strategic knowledge. These priors provide a strong initialisation for RL.

- **Faster Early Learning**
  Self-play systems initialised with SL model are more likely to converge faster compared to those trained from scratch.

##### Limitations

SL alone is insufficient to produce a highly competitive Go model.

The following are the primary limitations observed during evaluation.

- **Limited Strategic Adaptation**
  Because training targets are fixed, the model cannot discover new strategies beyond those present in the dataset.

- **Poor Generalisation to Random Play**
  Random opponents generate positions that are rarely observed in professional-level games, exposing weakness in purely imitation-based policies.

- **Deterministic Policy Behaviour**
  Overtraining leads to extremely confident predictions, reducing flexibility when encountering unfamiliar games states.

#### Motivation for RL

These limitations motivate the transition to reinforcement learning through self-play.

Self-play allows agents to:
- explore previously unseen positions
- learn strategies not present in the dataset
- directly optimise for winning outcomes

Despite its shortcomings and imitation-based policies, SL serves as a strong baseline for human-like gameplay behaviour and is therefore used as a standard; RL will be evaluated against this model in addition to a random model.

### Limitations

- Experiments currently limited to 9×9 Go
- Limited computation and resource allocation for training
- Limited human game dataset for pretraining
- Simple ko rule (no superko)

## References

- [AlphaZero from Scratch – Machine Learning Tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&t=14473s)
- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [AlphaZero Explained – Nik Cheerla](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [Sgfmill](https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/)
