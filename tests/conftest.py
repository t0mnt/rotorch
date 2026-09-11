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
def sta():
    return Algebra(1, 3, backends=['torch'])

@pytest.fixture
def double():
    """Run a test in double precision, parameters and all, since the lazy layers follow this."""
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)

@pytest.fixture
def rotor():
    def rotor(alg):
        """Composition of four reflections, so a unit rotor."""
        def reflection():
            # Only a vector that squares to a positive number has a norm to divide by, which in a
            # mixed signature leaves the timelike ones. The bar is one rather than zero because a
            # vector that barely clears the light cone normalizes to a boost of absurd rapidity.
            # Turning away a length and not a direction, so the rotor is uniform either way.
            while ((v := alg.vector(torch.randn(alg.d))) ** 2).e <= 1:
                pass
            return v.normalized()
        v1, v2, v3, v4 = (reflection() for _ in range(4))
        return v1 * v2 * v3 * v4
    return rotor

@pytest.fixture
def assert_equivariant():
    def assert_equivariant(layer, rotor, a, ulps=256):
        """Assert that f(w >> x) == w >> f(x), to within a few hundred ulps."""
        out = layer(a)
        eps = torch.finfo(out.values()[0].dtype).eps
        tol = ulps * eps * max(v.abs().max() for v in out.values())
        diff = layer(rotor >> a) - (rotor >> out)
        assert all(v.abs().max() < tol for v in diff.values())
    return assert_equivariant