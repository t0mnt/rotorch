import torch
from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import GeometricProduct, MVLinear


class ConvexHullCGMLP(nn.Module):
    """Regress the volume of a convex hull from the multivectors of its points."""

    def __init__(self, in_features=16, hidden_features=32, out_features=1, num_layers=4,
                 normalization_init=0):
        super().__init__()

        self.net = nn.Sequential(
            MVLinear(in_features, hidden_features, subspaces=False),
            *(GeometricProduct(hidden_features, normalization_init=normalization_init)
              for _ in range(num_layers)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, input: MultiVector) -> torch.Tensor:
        y = self.net(input)
        y = sum(v ** 2 for v in y.values()).sqrt()  # Euclidean blade norm, invariant under the group.
        return self.mlp(y).squeeze(-1)
