import torch
from gato.models.cgenn import ConvexHullCGMLP, LorentzCGGNN, NBodyCGGNN, O3CGMLP, O5CGMLP
from gato.nn.cgenn.utils import cat


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

def test_lorentz(sta, rotor, assert_equivariant, double):
    jets, nodes = 2, 4
    p = sta.vector(torch.randn(4, jets * nodes, 1))
    h = sta.scalar(e=torch.randn(jets * nodes, 2))
    pairs = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
    rows, cols = torch.tensor([(i + nodes * n, j + nodes * n) for n in range(jets) for i, j in pairs]).T
    model = LorentzCGGNN(features_x=4, features_h=8, decoder_features=8, n_layers=2, dropout=0.0)

    def forward(x):
        """The jets, the scalars that go with them, and every edge labelled with its endpoints."""
        edge_attr_x = cat([x[rows] - x[cols], x[rows], x[cols]])
        return model(h, x, (rows, cols), h, x, edge_attr_x, nodes)

    b = forward(p)
    assert b.shape == (jets, 2) and b.keys() == (0,)
    # A boost stretches the momenta tens of times over before the products square them, which
    # leaves fewer digits than a rotation would, and none worth checking in single precision.
    assert_equivariant(forward, rotor(sta), p, ulps=2 ** 20)
