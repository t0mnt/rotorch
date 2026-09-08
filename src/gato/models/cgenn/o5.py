import torch
from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import GeometricProduct, MVLinear


class O5CGMLP(nn.Module):
    """Regress an O(5) invariant from two vectors, read off the scalars."""

    def __init__(self, in_features=2, hidden_features=8, mlp_features=580, out_features=1,
                 normalization_init=0):
        super().__init__()

        self.gp = nn.Sequential(
            MVLinear(in_features, hidden_features, subspaces=False),
            GeometricProduct(hidden_features, normalization_init=normalization_init),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, mlp_features),
            nn.ReLU(),
            nn.Linear(mlp_features, mlp_features),
            nn.ReLU(),
            nn.Linear(mlp_features, out_features),
        )

    def forward(self, input: MultiVector) -> torch.Tensor:
        return self.mlp(self.gp(input).e)
