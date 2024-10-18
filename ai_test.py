from board_display import Board
from alphazero.train import TrainModel
from multiprocessing import Process, Queue, Manager
import random

def play_chess(q: Queue, q2: Queue):
    """
    Simulate games between two AlphaZero models with dynamic parameter adjustments.

    Parameters
    ----------
    q : Queue
        Queue for sending move data to the display process.
    q2 : Queue
        Queue for sending game results to the main process.
    """
    # Base training configuration
    base_train_config = {
        'lr': 1e-2,
        'board_len': 9,
        'batch_size': 500,
        'is_use_gpu': True,
        'n_test_games': 10,
        'n_self_plays': 4000,
        'is_save_game': False,
        'n_feature_planes': 6,
        'check_frequency': 100,
        'start_train_size': 500
    }

    print("Initializing AlphaZero models with differences...")

    # Randomize initial settings for each model
    c_puct_1 = random.uniform(2.5, 4.0)
    c_puct_2 = random.uniform(2.5, 4.0)
    n_mcts_iters_1 = random.randint(400, 600)
    n_mcts_iters_2 = random.randint(400, 600)

    # Configure each model
    train_config_1 = base_train_config.copy()
    train_config_1.update({'c_puct': c_puct_1, 'n_mcts_iters': n_mcts_iters_1})

    train_config_2 = base_train_config.copy()
    train_config_2.update({'c_puct': c_puct_2, 'n_mcts_iters': n_mcts_iters_2})

    # Initialize models
    train_model_1 = TrainModel(**train_config_1)
    train_model_2 = TrainModel(**train_config_2)
    train_model_1.policy_value_net.eval()
    train_model_2.policy_value_net.eval()

    train_model_1.chess_board.clear_board()
    train_model_2.chess_board.clear_board()
    train_model_1.mcts.set_self_play(False)
    train_model_2.mcts.set_self_play(False)

    board_len = 9
    print("Drawing board...")
    win_model_1 = 0
    win_model_2 = 0
    ties = 0

    print("Model 1 (Black) starts first")
    for i in range(10):
        player = 0  # Model 1 starts as Black
        train_model_1.chess_board.clear_board()
        train_model_2.chess_board.clear_board()
        is_over = False

        # Adjust parameters based on outcomes
        if ties > 0:
            # Increase differences on tie
            c_puct_diff = ties * 0.5
            iter_diff = ties * 50
        else:
            # Decrease differences on win
            c_puct_diff = -0.25
            iter_diff = -25

        # Adjust with bounds
        c_puct_1 = min(max(c_puct_1 + c_puct_diff, 1.5), 5.0)
        c_puct_2 = min(max(c_puct_2 - c_puct_diff, 1.5), 5.0)
        n_mcts_iters_1 = min(max(n_mcts_iters_1 + iter_diff, 200), 800)
        n_mcts_iters_2 = min(max(n_mcts_iters_2 - iter_diff, 200), 800)

        # Update MCTS parameters
        train_model_1.mcts.c_puct = c_puct_1
        train_model_1.mcts.n_iters = n_mcts_iters_1
        train_model_2.mcts.c_puct = c_puct_2
        train_model_2.mcts.n_iters = n_mcts_iters_2

        print(
            f"Game {i + 1}: Model 1 (c_puct={c_puct_1}, iters={n_mcts_iters_1}) vs Model 2 (c_puct={c_puct_2}, iters={n_mcts_iters_2})")

        while not is_over:
            if player == 0:
                # Model 1 move
                is_over, winner, action = train_model_1.do_mcts_action(train_model_1.mcts)
                train_model_2.chess_board.do_action(action)
                x, y = action // board_len, action % board_len
                print("Model 1 (Black) move:", x, y)
                q.put(((y * 50 + 75, x * 50 + 75), player), block=False)
                player = 1  # Switch to Model 2
            else:
                # Model 2 move
                is_over, winner, action = train_model_2.do_mcts_action(train_model_2.mcts)
                train_model_1.chess_board.do_action(action)
                x, y = action // board_len, action % board_len
                print("Model 2 (White) move:", x, y)
                q.put(((y * 50 + 75, x * 50 + 75), player), block=False)
                player = 0  # Switch to Model 1

        # Record results and reset for the next game
        if winner == 0:
            print(f"Game {i + 1}: Model 1 (Black) wins")
            q2.put(0)
            win_model_1 += 1
            ties = 0  # Reset ties on win
        elif winner == 1:
            print(f"Game {i + 1}: Model 2 (White) wins")
            q2.put(1)
            win_model_2 += 1
            ties = 0
        else:
            print(f"Game {i + 1}: It's a tie")
            q2.put(-1)
            ties += 1

    print(f"Model 1 (Black) win rate: {win_model_1 / 10 * 100}%")
    print(f"Model 2 (White) win rate: {win_model_2 / 10 * 100}%")
    print(f"Number of ties: {ties}")


if __name__ == '__main__':
    q = Manager().Queue(maxsize=-1)
    q2 = Manager().Queue(maxsize=-1)

    # Start the chess playing process
    p = Process(target=play_chess, args=(q, q2))
    p.start()

    # Display the game board
    board = Board(9)
    while True:
        board.clock.tick(60)
        board.quit()

        # Reset board after each game
        while not q2.empty():
            result = q2.get()
            while not q.empty():
                q.get()
            board.new_game()
            if result == -1:
                print("The game was a tie. Adjusting parameters for the next game...")
            elif result == 0:
                print("Model 1 won the game. Reducing parameter differences...")
            elif result == 1:
                print("Model 2 won the game. Reducing parameter differences...")

        # Draw moves as they come in
        while not q.empty():
            pos, player = q.get()
            board.draw_circle(pos=pos, player=player)
        board.update()