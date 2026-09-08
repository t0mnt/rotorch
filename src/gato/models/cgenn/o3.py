import torch
from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import FullyConnectedGeometricProduct, MVSiLU


class O3CGMLP(nn.Module):
    """Regress an O(3) invariant from three vectors, read off the pseudoscalar."""

    def __init__(self, in_features=3, hidden_features=32, out_features=1, num_layers=6,
                 normalization_init=0):
        super().__init__()

        product = lambda i, o: FullyConnectedGeometricProduct(
            i, o, normalization_init=normalization_init)
        self.net = nn.Sequential(
            product(in_features, hidden_features),
            # As in cgenn, the nonlinearities are stacked without products in between.
            *(MVSiLU() for _ in range(num_layers - 1)),
            product(hidden_features, out_features),
        )

    def forward(self, input: MultiVector) -> torch.Tensor:
        return self.net(input).e123[..., 0]
