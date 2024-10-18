# coding: utf-8
from collections import deque, namedtuple
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Named tuple for storing self-play data
SelfPlayData = namedtuple('SelfPlayData', ['pi_list', 'z_list', 'feature_planes_list'])


class SelfPlayDataSet(Dataset):
    """
    Dataset class for self-play data, where each sample is a tuple `(feature_planes, pi, z)`.
    """

    def __init__(self, board_len: int = 9, maxlen: int = 10000) -> None:
        """
        Initialize the dataset with a fixed-length deque for storing samples.

        Parameters
        ----------
        board_len : int
            The side length of the board.
        maxlen : int
            Maximum number of samples stored in the dataset.
        """
        super().__init__()
        self.__data_deque = deque(maxlen=maxlen)
        self.board_len = board_len

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.__data_deque)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a sample by index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A sample in the form `(feature_planes, pi, z)`.
        """
        return self.__data_deque[index]

    def clear(self) -> None:
        """
        Clear all samples from the dataset.
        """
        self.__data_deque.clear()

    def append(self, self_play_data: SelfPlayData) -> None:
        """
        Append new self-play data to the dataset, including data augmentation.

        Parameters
        ----------
        self_play_data : SelfPlayData
            Contains `pi_list`, `z_list`, and `feature_planes_list` for data augmentation.
        """
        n = self.board_len
        z_list = Tensor(self_play_data.z_list)
        pi_list = self_play_data.pi_list
        feature_planes_list = self_play_data.feature_planes_list

        # Augment the dataset using rotations and horizontal flips
        for z, pi, feature_planes in zip(z_list, pi_list, feature_planes_list):
            for i in range(4):
                # Rotate feature planes and policy vector counterclockwise by i*90°
                rot_features = torch.rot90(Tensor(feature_planes), i, (1, 2))
                rot_pi = torch.rot90(Tensor(pi.reshape(n, n)), i)
                self.__data_deque.append((rot_features, rot_pi.flatten(), z))

                # Flip the rotated feature planes and policy vector horizontally
                flip_features = torch.flip(rot_features, [2])
                flip_pi = torch.fliplr(rot_pi)
                self.__data_deque.append((flip_features, flip_pi.flatten(), z))