# coding: utf-8
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .chess_board import ChessBoard


class ConvBlock(nn.Module):
    """
    Convolutional Block: Convolution layer followed by Batch Normalization and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channel: int, kernel_size: int, padding: int = 0) -> None:
        """
        Initialize the ConvBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channel : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        padding : int, optional
            Padding for the convolutional layer, default is 0.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying convolution, batch normalization, and ReLU.
        """
        return F.relu(self.batch_norm(self.conv(x)))


class ResidueBlock(nn.Module):
    """
    Residual Block: Implements a residual connection with two convolutional layers.
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 128) -> None:
        """
        Initialize the ResidueBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual connection.
        """
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)


class PolicyHead(nn.Module):
    """
    Policy Head: Outputs action probabilities for the given board state.
    """

    def __init__(self, in_channels: int = 128, board_len: int = 9) -> None:
        """
        Initialize the PolicyHead.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        board_len : int
            Size of the board.
        """
        super().__init__()
        self.conv = ConvBlock(in_channels, 2, 1)
        self.fc = nn.Linear(2 * board_len ** 2, board_len ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Log-softmax probabilities over actions.
        """
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    """
    Value Head: Outputs a scalar value representing the expected outcome of the game.
    """

    def __init__(self, in_channels: int = 128, board_len: int = 9) -> None:
        """
        Initialize the ValueHead.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        board_len : int
            Size of the board.
        """
        super().__init__()
        self.conv = ConvBlock(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(board_len ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar value representing the board evaluation.
        """
        x = self.conv(x)
        return self.fc(x.flatten(1))


class PolicyValueNet(nn.Module):
    """
    Policy-Value Network: Integrates a convolutional body with a policy head and a value head.
    """

    def __init__(self, board_len: int = 9, n_feature_planes: int = 6, is_use_gpu: bool = True) -> None:
        """
        Initialize the PolicyValueNet.

        Parameters
        ----------
        board_len : int
            Size of the board.
        n_feature_planes : int
            Number of input feature planes.
        is_use_gpu : bool
            Whether to use GPU for computation.
        """
        super().__init__()
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
        self.conv = ConvBlock(n_feature_planes, 128, 3, padding=1)
        self.residues = nn.Sequential(*[ResidueBlock(128, 128) for _ in range(4)])
        self.policy_head = PolicyHead(128, board_len)
        self.value_head = ValueHead(128, board_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the board state.

        Returns
        -------
        p_hat : torch.Tensor
            Log probabilities of actions.
        value : torch.Tensor
            Scalar value estimation.
        """
        x = self.conv(x)
        x = self.residues(x)
        p_hat = self.policy_head(x)
        value = self.value_head(x)
        return p_hat, value

    def predict(self, chess_board: ChessBoard) -> tuple[np.ndarray, float]:
        """
        Predict action probabilities and board value for a given chessboard state.

        Parameters
        ----------
        chess_board : ChessBoard
            The chessboard state.

        Returns
        -------
        probs : np.ndarray
            Action probabilities for all available actions.
        value : float
            Value estimation of the current board state.
        """
        feature_planes = chess_board.get_feature_planes().to(self.device)
        feature_planes.unsqueeze_(0)  # Add batch dimension
        p_hat, value = self(feature_planes)

        # Convert log probabilities to probabilities
        p = torch.exp(p_hat).flatten()

        # Filter probabilities for available actions only
        p = p[chess_board.available_actions].cpu().detach().numpy() if self.device.type == 'cuda' else p[chess_board.available_actions].detach().numpy()

        return p, value[0].item()

    def set_device(self, is_use_gpu: bool) -> None:
        """
        Set the device for the network.

        Parameters
        ----------
        is_use_gpu : bool
            Whether to use GPU for computation.
        """
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')