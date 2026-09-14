"""
The five experiments of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn): the
O(3) and O(5) invariant regressions, the volume of a convex hull, charged n-body, and
tagging the jets of top quarks.
"""

from .hulls import ConvexHullCGMLP
from .lorentz import CGLayer, GradeGate, LorentzCGGNN
from .nbody import CEMLP, EGCL, NBodyCGGNN
from .o3 import O3CGMLP
from .o5 import O5CGMLP
