# coding: utf-8
from math import sqrt
from typing import Tuple, Iterable, Dict


class Node:
    """
    Monte Carlo Tree Search (MCTS) Node.
    """

    def __init__(self, prior_prob: float, c_puct: float = 5, parent=None) -> None:
        """
        Initialize an MCTS node.

        Parameters
        ----------
        prior_prob : float
            Prior probability `P(s, a)` for the node.
        c_puct : float
            Exploration constant for controlling the balance between exploration and exploitation.
        parent : Node
            Parent node in the tree.
        """
        self.Q = 0.0  # Average reward Q(s, a)
        self.U = 0.0  # Upper confidence bound U(s, a)
        self.N = 0    # Visit count N(s, a)
        self.score = 0.0  # Combined score Q + U
        self.P = prior_prob  # Prior probability P(s, a)
        self.c_puct = c_puct
        self.parent = parent
        self.children = {}  # Dictionary to store child nodes, keyed by action

    def select(self) -> Tuple[int, "Node"]:
        """
        Select the child node with the highest score.

        Returns
        -------
        action : int
            The action corresponding to the selected child.
        child : Node
            The child node with the highest score.
        """
        return max(self.children.items(), key=lambda item: item[1].get_score())

    def expand(self, action_probs: Iterable[Tuple[int, float]]) -> None:
        """
        Expand the node by adding child nodes.

        Parameters
        ----------
        action_probs : Iterable[Tuple[int, float]]
            Iterable containing tuples of (action, prior_prob), where each tuple represents an action
            and its corresponding prior probability.
        """
        for action, prior_prob in action_probs:
            self.children[action] = Node(prior_prob, self.c_puct, self)

    def __update(self, value: float) -> None:
        """
        Update the node's visit count `N(s, a)` and average reward `Q(s, a)`.

        Parameters
        ----------
        value : float
            The value used for updating the node's statistics.
        """
        self.Q = (self.N * self.Q + value) / (self.N + 1)
        self.N += 1

    def backup(self, value: float) -> None:
        """
        Perform backpropagation to update the node and its ancestors.

        Parameters
        ----------
        value : float
            The value to propagate up the tree, with alternating signs for each level.
        """
        # Update the parent node with the negated value before updating the current node.
        if self.parent:
            self.parent.backup(-value)
        self.__update(value)

    def get_score(self) -> float:
        """
        Calculate and return the node's score, which is a combination of Q and U.

        Returns
        -------
        score : float
            The calculated score of the node.
        """
        # U is calculated using the formula c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        self.U = self.c_puct * self.P * sqrt(self.parent.N) / (1 + self.N)
        self.score = self.U + self.Q
        return self.score

    def is_leaf_node(self) -> bool:
        """
        Check if the node is a leaf node (has no children).

        Returns
        -------
        is_leaf : bool
            True if the node is a leaf node, False otherwise.
        """
        return len(self.children) == 0