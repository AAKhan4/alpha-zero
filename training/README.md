# Training Modules: Overview

This directory contains the core training modules for the AlphaZero framework, enabling model training, evaluation, and configuration management.

<br>

## Model Training

`model_training.py`

Handles the main training loop for the AlphaZero model, integrating self-play, neural network updates, and data management.

### Key Features:
- Instantiates and runs training for the AlphaZero neural network model.
- Optimizes training for GPU or CPU environments.

Quick & easy way to initiate AlphaZero learning cycle.


<br>

### Eval Training

`eval_training.py`

Allows for simple evaluation of trained models by comparing their performance against previous versions.

#### Key Features:
- **Model Comparison:** Evaluates all model versions for game.
- **Performance Metrics:** Graphically presents average loss values in each epoch for all models.

<br>

### Training Args

`training_args.py`

Manages settings and hyperparameters for the training process, ensuring flexibility and reproducibility.

#### Key Features:
- **Hyperparameter Management:** Defines learning rates, batch sizes, and other training parameters.
- **Configuration Parsing:** Reads and validates user-defined settings.
- **Reproducibility:** Ensures consistent training runs with fixed seeds.
- **Flexibility:** Simplifies the process of adapting the framework to new games.

<br>

## Summary

The training modules provide a quick & easy way to train & evaluate AlphaZero models for specified games. Each module is designed to be modular and easy to integrate into the overall system.