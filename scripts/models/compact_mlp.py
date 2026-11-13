import torch
import torch.nn as nn


class CompactMLP_FD(nn.Module):

    def __init__(
        self,
        input_dim=480 * 6,
        h1: int = 256,
        h2: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),

            nn.Linear(h1, h2),
            nn.ReLU(),

            nn.Linear(h2, 1),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.net[:4](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_embedding(x)
        return self.net[-1](embedding).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))