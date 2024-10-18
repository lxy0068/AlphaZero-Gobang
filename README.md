# AlphaZero-Gobang

An advanced AI implementation of the Gobang game (Five in a Row) using the AlphaZero algorithm. This project combines Monte Carlo Tree Search (MCTS) and a deep neural network to create an AI capable of learning from self-play to achieve competitive gameplay.

## Features

- Utilizes a combination of deep neural networks and MCTS to learn optimal moves.
- The AI agent trains through self-play to continuously improve its performance.
- Supports different Gobang board sizes.
- Includes scripts for training, testing, and evaluating the AI's performance.
- Uses PyQt5 and Pygame for an interactive board display.
- Visualization and testing tools to evaluate AI performance.

## Project Structure

- **alphazero/**: Contains the core implementation of the AlphaZero algorithm.
  - `__init__.py`: Initializes the `alphazero` module.
  - `alpha_zero_mcts.py`: Implements Monte Carlo Tree Search (MCTS) logic for decision-making.
  - `chess_board.py`: Manages the Gobang board, game rules, and move validation.
  - `node.py`: Defines the structure and operations for MCTS nodes.
  - `policy_value_net.py`: Defines the neural network architecture for predicting move probabilities and game outcomes.
  - `rollout_mcts.py`: Implements an alternative MCTS method using rollouts.
  - `self_play_dataset.py`: Handles data generation from self-play sessions and prepares data for training.
  - `train.py`: Orchestrates the self-play, training, and model update processes.

- **model/**: Stores trained models and checkpoints.
  - `best_policy_value_net.pth`: Contains the weights of the best-performing policy-value network model.

- **ai_test.py**: Provides scripts for testing the AI’s performance against predefined benchmarks or different versions of the model.

- **board_display.py**: Handles the rendering and visualization of the Gobang board, using libraries like `pygame` for a graphical display.

- **game.py**: Manages the overall game flow, including AI vs. AI matches, simulations, and gameplay logic.

- **humain_play.py**: Enables a human player to play against the trained AI through a user-friendly graphical interface.

- **train.py**: A main script that can be run to train the AlphaZero model through self-play and continuous learning.

## Prerequisites

- **Python**: Version 3.8 or higher is recommended for compatibility with deep learning libraries (The python version used in this project is 3.11).
- **CUDA**: For faster training (NVIDIA GPU) .

## Installation

1. Clone the Repository:

   ```
   git clone https://github.com/lxy0068/AlphaZero-Gobang.git
   cd AlphaZero-Gobang
   ```

2. Set Up a Virtual Environment:

   ```
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Required Libraries:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the AI

To train the AlphaZero model using self-play data, run the `train.py` script:

```bash
python train.py
```

- It sets up the training parameters such as learning rate, board size, batch size, number of MCTS iterations, and self-play games. These can be adjusted in the `get_train_config()` function within `train.py`.

- The `TrainModel` class is instantiated using the defined configuration. It initializes the policy-value network and the MCTS (Monte Carlo Tree Search) agent.

- The model generates training data through self-play, playing against itself to explore optimal moves. During each self-play game, the model records game states, action probabilities, and outcomes, which are used to train the policy and value network.

- After collecting sufficient self-play data, the model trains on mini-batches using a combination of policy and value loss. The training process continues iteratively, improving the model with each round of self-play data.

- Periodically, the trained model is evaluated against a stored best model. If the new model achieves a win rate above a certain threshold (e.g., 55%), it replaces the best model.

- The script saves the model weights, training logs, and self-play data to the `model/` and `log/` directories. 

### Play Against the AI

Use `humain_play.py` to challenge the trained AI:

```bash
python humain_play.py
```

This script launches a graphical interface where you can play Gobang against the AI. The game features:

- Click on the board to place your pieces, and watch the AI respond with its own moves.
- The board is updated in real-time, showing both human and AI moves.
- The script announces the winner at the end of each game and provides an option to start a new game.
- Click the "New Game" button after a match to reset the board and play again.

### Test the AI

Run `ai_test.py` to evaluate the AI's performance against different benchmarks. This script dynamically adjusts the parameters according to the win or loss of each game, providing a more intuitive display of the model's capabilities:

```bash
python ai_test.py
```

**Features of `ai_test.py`:**

- Sets up two instances of the AlphaZero model with random initial parameters.
- Adjusts the MCTS exploration constant (`c_puct`) and the number of iterations based on the outcomes of each game to balance exploration and exploitation.
- The script simulates multiple games between two models (Model 1 and Model 2) with varying parameter settings, tracking win rates and ties.
- Uses a real-time board display to visualize the moves during the game.


## Contributing

Contributions are welcome! Please open an issue for bug reports, feature requests, or submit pull requests for improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or feedback, please reach out to [xingyanliu10@gmail.com].

## Acknowledgements

This project is inspired by the original AlphaZero implementation by DeepMind. Special thanks to the open-source community for providing tools and libraries that made this project possible.
