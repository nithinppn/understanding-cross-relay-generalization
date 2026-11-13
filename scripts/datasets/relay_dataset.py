import numpy as np
import torch
from torch.utils.data import Dataset


class RelayDataset(Dataset):
    """
    Memory-efficient dataset.

    Holds only references to the original dataset and
    a list of indices.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        x = torch.from_numpy(
            self.X[sample_idx]
        ).float()

        y = torch.tensor(
            self.y[sample_idx],
            dtype=torch.float32,
        )

        return x, y
