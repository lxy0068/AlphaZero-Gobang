# coding: utf-8
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np


class ChessBoard:
    """
    ChessBoard class to manage the game state, actions, and feature planes.
    """

    # Constants representing board state
    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_len: int = 9, n_feature_planes: int = 7) -> None:
        """
        Initialize a chessboard with specified dimensions and feature planes.

        Parameters
        ----------
        board_len : int
            Side length of the square board.
        n_feature_planes : int
            Number of feature planes (must be odd).
        """
        self.board_len = board_len
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.available_actions = list(range(self.board_len ** 2))
        self.state = OrderedDict()  # Board state as {action: player}
        self.previous_action = None

    def copy(self) -> "ChessBoard":
        """
        Create a deep copy of the current chessboard state.

        Returns
        -------
        ChessBoard
            A deep copy of the current chessboard.
        """
        return deepcopy(self)

    def clear_board(self) -> None:
        """
        Reset the board to its initial state, clearing all moves.
        """
        self.state.clear()
        self.previous_action = None
        self.current_player = self.BLACK
        self.available_actions = list(range(self.board_len ** 2))

    def do_action(self, action: int) -> None:
        """
        Place a piece on the board and update the state.

        Parameters
        ----------
        action : int
            Position on the board where the piece is placed (range: `[0, board_len^2 -1]`).
        """
        self.previous_action = action
        self.available_actions.remove(action)
        self.state[action] = self.current_player
        # Switch player after each move
        self.current_player = self.WHITE + self.BLACK - self.current_player

    def do_action_(self, pos: Tuple[int, int]) -> bool:
        """
        Place a piece based on a tuple position, used primarily for app integration.

        Parameters
        ----------
        pos : Tuple[int, int]
            Position on the board as (row, column).

        Returns
        -------
        bool
            True if the move is valid and successful, False otherwise.
        """
        action = pos[0] * self.board_len + pos[1]
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def is_game_over(self) -> Tuple[bool, int]:
        """
        Check if the game has ended and determine the winner if any.

        Returns
        -------
        is_over : bool
            True if the game is over (win or draw), False otherwise.
        winner : int
            The winner of the game, either `ChessBoard.BLACK`, `ChessBoard.WHITE`, or `None` for a draw.
        """
        # Game cannot end if fewer than 9 moves have been made
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.previous_action
        player = self.state[act]
        row, col = divmod(act, n)

        # Directions to check for winning condition: horizontal, vertical, and two diagonals
        directions = [
            [(0, -1), (0, 1)],  # Horizontal
            [(-1, 0), (1, 0)],  # Vertical
            [(-1, -1), (1, 1)],  # Main diagonal
            [(1, -1), (-1, 1)]  # Anti-diagonal
        ]

        # Check each direction for 5 or more consecutive stones
        for direction in directions:
            count = 1  # Including the current stone
            for offset in direction:
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t += offset[0]
                    col_t += offset[1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state.get(row_t * n + col_t, self.EMPTY) == player:
                        count += 1
                    else:
                        flag = False
            if count >= 5:
                return True, player

        # Check for draw if no moves are available
        if not self.available_actions:
            return True, None

        return False, None

    def get_feature_planes(self) -> torch.Tensor:
        """
        Generate a tensor representing the board state for the current player.

        Returns
        -------
        feature_planes : torch.Tensor
            A tensor of shape `(n_feature_planes, board_len, board_len)`.
        """
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n ** 2))

        if self.state:
            actions = np.array(list(self.state.keys()))[::-1]
            players = np.array(list(self.state.values()))[::-1]
            Xt = actions[players == self.current_player]
            Yt = actions[players != self.current_player]

            # Populate feature planes with historical moves
            for i in range((self.n_feature_planes - 1) // 2):
                if i < len(Xt):
                    feature_planes[2 * i, Xt[i:]] = 1
                if i < len(Yt):
                    feature_planes[2 * i + 1, Yt[i:]] = 1

        return feature_planes.view(self.n_feature_planes, n, n)


class ColorError(ValueError):
    """
    Custom error for invalid color assignments in the ChessBoard.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)