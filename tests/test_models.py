import torch
from gato.models.cgenn import ConvexHullCGMLP, NBodyCGGNN, O3CGMLP, O5CGMLP


def test_hulls(alg5, rotor, assert_equivariant):
    a = alg5.vector(torch.randn(5, 7, 16))
    model = ConvexHullCGMLP(16, 8, num_layers=2)
    b = model(a)
    assert b.shape == (7, 1) and b.keys() == (0,)
    assert_equivariant(model, rotor(alg5), a)

def test_o3(alg3, rotor, assert_equivariant):
    a = alg3.vector(torch.randn(3, 7, 3))
    model = O3CGMLP(hidden_features=8, num_layers=3)
    b = model(a)
    assert b.shape == (7, 1) and b.keys() == (7,)
    assert_equivariant(model, rotor(alg3), a)

def test_o5(alg5, rotor, assert_equivariant):
    a = alg5.vector(torch.randn(5, 7, 2))
    model = O5CGMLP(mlp_features=16)
    b = model(a)
    assert b.shape == (7, 1) and b.keys() == (0,)
    assert_equivariant(model, rotor(alg5), a)

def test_nbody(alg3, rotor, assert_equivariant):
    h = alg3.multivector(torch.randn(8, 5, 3))
    edges = (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 0]))
    edge_attr = alg3.scalar(e=torch.randn(5, 1))  # Invariant, so it does not rotate along.
    model = NBodyCGGNN(hidden_features=6, n_layers=2)
    b = model(h, edges, edge_attr)
    assert b.shape == (5, 1) and b.keys() == (1, 2, 4)
    assert_equivariant(lambda x: model(x, edges, edge_attr), rotor(alg3), h)
