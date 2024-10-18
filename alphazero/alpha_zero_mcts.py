# coding: utf-8
from typing import Tuple, Union
import numpy as np
from .chess_board import ChessBoard
from .node import Node
from .policy_value_net import PolicyValueNet


class AlphaZeroMCTS:
    """
    Monte Carlo Tree Search (MCTS) based on a Policy-Value Network.
    """

    def __init__(self, policy_value_net: PolicyValueNet, c_puct: float = 4, n_iters: int = 1200, is_self_play: bool = False) -> None:
        """
        Initialize the MCTS with a policy-value network and exploration parameters.

        Parameters
        ----------
        policy_value_net : PolicyValueNet
            The policy-value network used for predicting action probabilities and value estimations.
        c_puct : float
            Exploration constant that balances exploration and exploitation.
        n_iters : int
            Number of iterations for tree search.
        is_self_play : bool
            Indicates whether the MCTS is in self-play mode.
        """
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.is_self_play = is_self_play
        self.root = Node(prior_prob=1, parent=None)

    def get_action(self, chess_board: ChessBoard) -> Union[Tuple[int, np.ndarray], int]:
        """
        Perform MCTS to get the best action for the current game state.

        Parameters
        ----------
        chess_board : ChessBoard
            The current state of the chessboard.

        Returns
        -------
        action : int
            The selected action based on MCTS.
        pi : np.ndarray, shape=(board_len^2,)
            Action probability distribution, returned only in self-play mode.
        """
        # Perform MCTS iterations to explore the game tree
        for _ in range(self.n_iters):
            board_copy = chess_board.copy()
            node = self.root

            # Traverse the tree until a leaf node is reached
            while not node.is_leaf_node():
                action, node = node.select()
                board_copy.do_action(action)

            # Expand the leaf node or backpropagate if the game is over
            is_over, winner = board_copy.is_game_over()
            p, value = self.policy_value_net.predict(board_copy)
            if not is_over:
                # Apply Dirichlet noise to the policy in self-play mode for exploration
                if self.is_self_play:
                    p = 0.75 * p + 0.25 * np.random.dirichlet(0.03 * np.ones(len(p)))
                node.expand(zip(board_copy.available_actions, p))
            else:
                # Set the value based on the winner of the game
                value = 1 if winner == board_copy.current_player else -1 if winner else 0

            # Backup the value along the visited path
            node.backup(-value)

        # Calculate the action probabilities π and select an action
        T = 1 if self.is_self_play and len(chess_board.state) <= 30 else 1e-3
        visits = np.array([child.N for child in self.root.children.values()])
        pi_ = self._calculate_pi(visits, T)

        # Select an action based on the action probabilities
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        # In self-play mode, return the action and action probability distribution π
        if self.is_self_play:
            pi = np.zeros(chess_board.board_len ** 2)
            pi[actions] = pi_
            # Update the root to the selected action's node for the next iteration
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            # Reset the root node for non-self-play mode
            self.reset_root()
            return action

    def _calculate_pi(self, visits: np.ndarray, T: float) -> np.ndarray:
        """
        Calculate the action probabilities π using visit counts and temperature.

        Parameters
        ----------
        visits : np.ndarray
            Array of visit counts for each child node.
        T : float
            Temperature parameter for controlling exploration.

        Returns
        -------
        pi : np.ndarray
            Normalized action probabilities.
        """
        # Use logarithmic scaling to prevent overflow in calculations
        log_visits = 1 / T * np.log(visits + 1e-11)
        exp_visits = np.exp(log_visits - log_visits.max())
        return exp_visits / exp_visits.sum()

    def reset_root(self) -> None:
        """
        Reset the root node for a new MCTS search.
        """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)

    def set_self_play(self, is_self_play: bool) -> None:
        """
        Set the self-play mode of the MCTS.

        Parameters
        ----------
        is_self_play : bool
            Whether to enable self-play mode.
        """
        self.is_self_play = is_self_play