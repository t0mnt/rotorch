import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .utils import EPS, free_constants, norm


class NormalizationLayer(LazyModuleMixin, nn.Module):
    """Interpolate grade-wise between the input and its normalized version."""

    a: UninitializedParameter

    def __init__(self, init: float = 0):
        super().__init__()

        self.init = init
        self.a = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            self.grade_index = {g: i for i, g in enumerate(input.grades)}
            self.a.materialize((len(self.grade_index), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.a, self.init)

    def forward(self, input: MultiVector) -> MultiVector:
        input = free_constants(input)
        s_a = torch.sigmoid(self.a)
        # Interpolate between 1 and the norm of each grade.
        norms = {g: s_a[i] * (norm(input.grade(g)) - 1) + 1 for g, i in self.grade_index.items()}
        return input.map(lambda k, v: v / (norms[k.bit_count()] + EPS))
