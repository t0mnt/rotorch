import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .utils import free_constants, mag2, norm


class MVSiLU(LazyModuleMixin, nn.Module):
    """
    Gate every grade with a sigmoid of an invariant of that grade. Because the gate is
    invariant, the gated multivector transforms just like the input.
    """

    a: UninitializedParameter
    b: UninitializedParameter

    def __init__(self, invariant="mag2"):
        super().__init__()

        self.invariant = {"mag2": mag2, "norm": norm}[invariant]
        self.a = UninitializedParameter()
        self.b = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            self.grade_index = {g: i for i, g in enumerate(input.grades)}
            self.a.materialize((len(self.grade_index), input.shape[-1]))
            self.b.materialize((len(self.grade_index), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.a)
        nn.init.zeros_(self.b)

    def forward(self, input: MultiVector) -> MultiVector:
        input = free_constants(input)
        gates = {}
        for g, i in self.grade_index.items():
            invariant = input.e if g == 0 else self.invariant(input.grade(g))
            gates[g] = torch.sigmoid(self.a[i] * invariant + self.b[i])
        return input.map(lambda k, v: gates[k.bit_count()] * v)
