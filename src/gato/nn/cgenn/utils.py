import torch
from kingdon import MultiVector

EPS = 1e-6


def materialize_constants(mv: MultiVector) -> MultiVector:
    """
    Turn the structural constants of a fixed layout, e.g. the scalar 1.0 of a
    Translation, into values, since only those are visible to `MultiVector.map`.
    """
    if any(v is not ... for v in mv.type_layout.values()):
        return mv.asmvtype()
    return mv


def grade_of_blades(mv: MultiVector) -> torch.Tensor:
    """
    For every blade of `mv`, the index of its grade among the grades present. Lets a layer hold
    one parameter per grade and still apply them to the coefficients in one go, rather than a
    blade at a time.
    """
    index = {g: i for i, g in enumerate(mv.grades)}
    return torch.tensor([index[k.bit_count()] for k in mv.keys()])


def register(algebra, expr):
    """
    Compile `expr` for `algebra`, or hand back the operator registered under its name
    before, since registering anew would drop the codegen cached on it. The name of
    `expr` therefore has to be unique within `algebra.registry`.
    """
    if expr.__name__ not in algebra.registry:
        algebra.add_operator(expr, symbolic=True)
    return algebra.registry[expr.__name__]


def scalar_normsq(X: MultiVector) -> MultiVector:
    """Scalar part of X times its reverse, unlike kingdon's normsq which keeps all grades."""
    return (~X * X).grade(0)


def mag2(X: MultiVector):
    """Squared magnitude of the single grade multivector X. Zero for null blades."""
    return sum(register(X.algebra, scalar_normsq)(X).values())


def norm(X: MultiVector):
    """Magnitude of the single grade multivector X, smoothed to stay differentiable at zero."""
    return (mag2(X) ** 2 + 1e-16) ** 0.25
