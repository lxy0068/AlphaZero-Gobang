# coding: utf-8
from alphazero.train import TrainModel

def get_train_config():
    """
    Returns the training configuration for the AlphaZero model.

    Returns
    -------
    dict
        A dictionary containing the training parameters.
    """
    return {
        'lr': 1e-2,  # Learning rate
        'c_puct': 3,  # Exploration constant for MCTS
        'board_len': 9,  # Length of the board (number of rows/columns)
        'batch_size': 500,  # Batch size for training
        'is_use_gpu': True,  # Use GPU if available
        'n_test_games': 10,  # Number of games to play for testing
        'n_mcts_iters': 500,  # Number of MCTS iterations per move
        'n_self_plays': 4000,  # Number of self-play games for training
        'is_save_game': False,  # Whether to save game logs
        'n_feature_planes': 6,  # Number of feature planes for input representation
        'check_frequency': 100,  # Frequency of model evaluation during training
        'start_train_size': 500  # Minimum dataset size before training begins
    }

def initialize_model(config):
    """
    Initializes the AlphaZero model with the given training configuration.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.

    Returns
    -------
    TrainModel
        An instance of the TrainModel class initialized with the specified config.
    """
    return TrainModel(**config)

def main():
    """
    Main function to configure, initialize, and train the AlphaZero model.
    """
    train_config = get_train_config()

    train_model = initialize_model(train_config)

    train_model.train()

if __name__ == "__main__":
    main()