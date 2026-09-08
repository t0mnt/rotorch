import math

import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .utils import free_constants

def gradewise_linear(X: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """
    Apply a weight to every grade of X seperatelly.
    """
    tot = 0
    for g, w in zip(X.grades, weights):
        tot += w * X.grade(g)
    return tot

class MVLinear(LazyModuleMixin, nn.Module):
    """Grade-wise linear map: every grade gets its own mixing matrix."""

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            self.grade_index = {g: i for i, g in enumerate(input.grades)}
            self.weight.materialize((len(self.grade_index), self.out_features, self.in_features))
            if self.bias is not None:
                self.bias.materialize((self.out_features,))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: MultiVector) -> MultiVector:
        input = free_constants(input)
        result = input.map(lambda k, v: einops.einsum(
            v, self.weight[self.grade_index[k.bit_count()]], "... i, o i -> ... o"))
        if self.bias is not None:
            result = result + input.algebra.scalar(e=self.bias)
        return result
