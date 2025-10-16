# Connect Four & Tic Tac Toe: Game Implementation

This directory contains the core implementations for the Connect Four and Tic Tac Toe games, along with a test suite and a game selection utility.

<br>

## Game Implementations

The Connect Four and Tic Tac Toe modules are built on a shared `BaseGame` class, which provides a modular and extensible foundation for game development. This design streamlines the implementation and facilitates the addition of new games.

### Key Features:
- **Modular Architecture:** The `BaseGame` class encapsulates shared game logic, minimizing redundancy.
- **Game State Management:** Encodes and processes the board state.
- **Move Validation:** Ensures only valid moves are executed during gameplay.
- **Win and Draw Detection:** Identifies game outcomes based on predefined rules.
- **Turn-Based Gameplay:** Manages player turn & change in perspective.
- **Extensibility:** New games can be implemented by inheriting from `BaseGame` and defining specific rules and mechanics.

This structure ensures maintainability, scalability, and ease of integration for future enhancements.

<br>

## Test Suite

The test allows to select & examine model's next action probability for various game states graphically.

<br>

## Game Selection Utility

The `game_select` module provides an interface for choosing between Connect Four and Tic Tac Toe, and can be extended for more games.

### Features:
- **Game Initialization:** Sets up the selected game with its respective training parameters.
- **Dynamic Selection:** Allows users to select game for training or play during run-time.

<br>

## Summary

This directory encapsulates the logic for the board games, ensuring robust gameplay and modular design. Some tests & the game selection utility is also provided.