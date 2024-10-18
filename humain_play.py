from board_display import Board
from alphazero.train import TrainModel

def human_play():
    """
    Function to handle a game between a human player and an AI using the AlphaZero model.
    Includes displaying each move and a restart mechanism.
    """
    # Configuration settings for training the model
    train_config = {
        'lr': 1e-2,
        'c_puct': 3,
        'board_len': 9,
        'batch_size': 500,
        'is_use_gpu': True,
        'n_test_games': 10,
        'n_mcts_iters': 500,
        'n_self_plays': 4000,
        'is_save_game': False,
        'n_feature_planes': 6,
        'check_frequency': 100,
        'start_train_size': 500
    }

    # Initialize the training model with the specified configuration
    train_model = TrainModel(**train_config)
    train_model.policy_value_net.eval()
    train_model.chess_board.clear_board()
    train_model.mcts.set_self_play(False)

    # Initialize the game board
    board_len = 9
    board = Board(board_len)

    while True:
        play_game(train_model, board, board_len)

def play_game(train_model, board, board_len):
    """
    Manage a single game between the human and AI, with the option to restart after the game ends.

    Parameters
    ----------
    train_model : TrainModel
        The trained AlphaZero model used for AI decisions.
    board : Board
        The graphical board interface for displaying moves.
    board_len : int
        The length of the game board.
    """
    player = 0  # Player 0 (human) starts first, AI is player 1
    is_over = False
    winner = None

    # Reset the board for a new game
    board.new_game()
    train_model.chess_board.clear_board()
    move_count = 0  # Track the move count for display

    # Main game loop
    while not is_over:
        move_count += 1  # Increment the move count with each valid move

        if player == 0:
            # Human player's turn
            board.clear()
            a, b = board.get_click()  # Get the click position from the user
            x, y = board.change_pos(abs(a - 75), abs(b - 75))
            if 0 <= x < board_len and 0 <= y < board_len:
                move_successful = train_model.chess_board.do_action_((y, x))
                if move_successful:
                    print(f"Move {move_count}: Human moves to ({y}, {x})")
                    board.draw_circle(pos=(x * 50 + 75, y * 50 + 75), player=player)
                    board.update()
                    player = 1  # Switch to the AI's turn
                    is_over, winner = train_model.chess_board.is_game_over()

        elif player == 1:
            # AI's turn
            is_over, winner, action = train_model.do_mcts_action(train_model.mcts)
            x = action // board_len
            y = action % board_len
            print(f"Move {move_count}: AI moves to ({y}, {x})")
            board.draw_circle(pos=(y * 50 + 75, x * 50 + 75), player=player)
            board.update()
            player = 0  # Switch back to the human player's turn

    # Game has ended; handle the result
    if winner == 0:
        print("Game Over: AI wins!")
        board.display_message('AI Wins! Click "New Game" to play again.')
    elif winner == 1:
        print("Game Over: Human wins!")
        board.display_message('You Win! Click "New Game" to play again.')
    else:
        print("Game Over: It's a tie!")
        board.display_message('It\'s a Tie! Click "New Game" to play again.')

    # Wait for the user to click the "New Game" button before restarting
    board.handle_events()  # This will handle the "New Game" button click

if __name__ == '__main__':
    human_play()
