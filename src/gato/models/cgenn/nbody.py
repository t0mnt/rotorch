import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import torch
from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import GeometricProduct, MVLayerNorm, MVLinear, MVSiLU


def cat(mvs: list[MultiVector]) -> MultiVector:
    """Concatenate multivectors along their feature axis."""
    packed, _ = einops.pack([mv.asmvtype() for mv in mvs], "n *")
    return packed


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    """Clifford equivariant MLP: a stack of linear, nonlinear and product layers."""

    def __init__(self, in_features, hidden_features, out_features, n_layers=2,
                 normalization_init=0):
        super().__init__()

        features = [in_features] + [hidden_features] * (n_layers - 1) + [out_features]
        self.layers = nn.Sequential(*(
            nn.Sequential(
                MVLinear(i, o),
                MVSiLU(),
                GeometricProduct(o, normalization_init=normalization_init),
                MVLayerNorm(),
            )
            for i, o in zip(features, features[1:])
        ))

    def forward(self, input: MultiVector) -> MultiVector:
        return self.layers(input)


class EGCL(nn.Module):
    """Equivariant graph convolution: message, mean aggregation and update."""

    def __init__(self, in_features, hidden_features, out_features, edge_attr_features=0,
                 node_attr_features=0, residual=True, normalization_init=0):
        super().__init__()

        self.residual = residual
        self.edge_model = CEMLP(in_features + edge_attr_features, hidden_features,
                                out_features, normalization_init=normalization_init)
        self.node_model = CEMLP(in_features + out_features + node_attr_features, hidden_features,
                                out_features, normalization_init=normalization_init)

    def message(self, h_i, h_j, edge_attr=None):
        input = h_i - h_j if edge_attr is None else cat([h_i - h_j, edge_attr])
        return self.edge_model(input)

    def aggregate(self, h_msg, segment_ids, num_segments):
        return h_msg.map(lambda v: unsorted_segment_mean(v, segment_ids, num_segments))

    def update(self, h_agg, h, node_attr=None):
        input = [h, h_agg] if node_attr is None else [h, h_agg, node_attr]
        out_h = self.node_model(cat(input))
        return h + out_h if self.residual else out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        rows, cols = edge_index
        h_msg = self.message(h[rows], h[cols], edge_attr)
        h_agg = self.aggregate(h_msg, rows, num_segments=h.shape[0])
        return self.update(h_agg, h, node_attr)


class NBodyCGGNN(nn.Module):
    """Predict the displacement of charged particles from their positions and velocities."""

    def __init__(self, in_features=3, hidden_features=28, out_features=1, edge_features_in=1,
                 n_layers=3, normalization_init=0, residual=True):
        super().__init__()

        self.embedding = MVLinear(in_features, hidden_features, subspaces=False)
        self.layers = nn.ModuleList(
            EGCL(hidden_features, hidden_features, hidden_features, edge_features_in,
                 residual=residual, normalization_init=normalization_init)
            for _ in range(n_layers)
        )
        self.projection = MVLinear(hidden_features, out_features)

    def forward(self, h: MultiVector, edges, edge_attr=None) -> MultiVector:
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        return self.projection(h)
