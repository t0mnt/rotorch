import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .utils import EPS, free_constants, norm


class MVLayerNorm(LazyModuleMixin, nn.Module):
    """Divide by the norm of the input, averaged over the channels."""

    a: UninitializedParameter

    def __init__(self):
        super().__init__()

        self.a = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            self.a.materialize((input.shape[-1],))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.a)

    def forward(self, input: MultiVector) -> MultiVector:
        input = free_constants(input)
        norms = einops.reduce(norm(input), "... f -> ... 1", "mean") + EPS
        return input.map(lambda v: self.a * v / norms)
