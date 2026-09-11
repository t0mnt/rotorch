from kingdon import Algebra, EvenMV
import torch
import pytest


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)

@pytest.fixture
def alg():
    return Algebra(3, 0, 1, backends=['torch'])

@pytest.fixture
def alg3():
    return Algebra(3, backends=['torch'])

@pytest.fixture
def alg5():
    return Algebra(5, backends=['torch'])

@pytest.fixture
def rotor():
    def rotor(alg):
        """Composition of four reflections, so a unit rotor."""
        v1, v2, v3, v4 = (alg.vector(torch.randn(alg.d)).normalized() for _ in range(4))
        return v1 * v2 * v3 * v4
    return rotor

@pytest.fixture
def assert_equivariant():
    def assert_equivariant(layer, rotor, a):
        """Assert that f(w >> x) == w >> f(x), to within a few hundred ulps."""
        out = layer(a)
        eps = torch.finfo(out.values()[0].dtype).eps
        tol = 256 * eps * max(v.abs().max() for v in out.values())
        diff = layer(rotor >> a) - (rotor >> out)
        assert all(v.abs().max() < tol for v in diff.values())
    return assert_equivariant