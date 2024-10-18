# coding: utf-8
import random
import numpy as np
from .chess_board import ChessBoard
from .node import Node


class RolloutMCTS:
    """
    Monte Carlo Tree Search (MCTS) using a random rollout policy.
    """

    def __init__(self, c_puct: float = 5, n_iters: int = 1000) -> None:
        """
        Initialize the RolloutMCTS with exploration parameters.

        Parameters
        ----------
        c_puct : float
            Exploration constant to balance exploration and exploitation.
        n_iters : int
            Number of iterations for the MCTS search.
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.root = Node(prior_prob=1, c_puct=c_puct, parent=None)

    def get_action(self, chess_board: ChessBoard) -> int:
        """
        Determine the next action based on the current board state.

        Parameters
        ----------
        chess_board : ChessBoard
            The current state of the chessboard.

        Returns
        -------
        action : int
            The chosen action for the next move.
        """
        # Perform MCTS iterations
        for _ in range(self.n_iters):
            board_copy = chess_board.copy()
            node = self.root

            # Traverse the tree until a leaf node is reached
            while not node.is_leaf_node():
                action, node = node.select()
                board_copy.do_action(action)

            # Expand leaf node if the game is not over
            is_over, winner = board_copy.is_game_over()
            if not is_over:
                node.expand(self.__default_policy(board_copy))

            # Perform a random rollout simulation
            value = self.__rollout(board_copy)

            # Backup the value through the tree
            node.backup(-value)

        # Select the action with the highest visit count
        action = max(self.root.children.items(), key=lambda i: i[1].N)[0]

        # Reset the root node for the next search
        self.root = Node(prior_prob=1, c_puct=self.c_puct)
        return action

    def __default_policy(self, chess_board: ChessBoard):
        """
        Generate action probabilities using a uniform random policy.

        Parameters
        ----------
        chess_board : ChessBoard
            The current state of the chessboard.

        Returns
        -------
        action_probs : Iterable[Tuple[int, float]]
            List of tuples containing (action, prior_prob) pairs for each available action.
        """
        n = len(chess_board.available_actions)
        probs = np.ones(n) / n  # Uniform probability distribution
        return zip(chess_board.available_actions, probs)

    def __rollout(self, board: ChessBoard) -> int:
        """
        Simulate a random game until the end to estimate the value of a state.

        Parameters
        ----------
        board : ChessBoard
            The current state of the chessboard.

        Returns
        -------
        value : int
            The value of the final state:
            * 1 if the current player wins,
            * -1 if the opponent wins,
            * 0 for a draw.
        """
        current_player = board.current_player

        # Randomly play out the game until it is over
        while True:
            is_over, winner = board.is_game_over()
            if is_over:
                break
            action = random.choice(board.available_actions)
            board.do_action(action)

        # Evaluate the outcome from the perspective of the current player
        if winner is not None:
            return 1 if winner == current_player else -1
        return 0