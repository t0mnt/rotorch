"""
Regress the volume of the convex hull of 16 points in 5D, the hulls example of cgenn.

    python examples/hulls.py
    python examples/hulls.py --compile model
    python examples/hulls.py --backend triton
    python examples/hulls.py --impl cgenn [--cgenn-path /path/to/cgenn]
"""
import einops
import numpy as np
import torch

import benchmark

N_POINTS, DIM = 16, 5


def generate(n_samples, seed):
    """Random point clouds in 5D, labelled with the volume of their convex hull."""
    from scipy.spatial import ConvexHull

    points = np.random.default_rng(seed).standard_normal((n_samples, N_POINTS, DIM))
    volumes = [ConvexHull(p).volume for p in points]
    return points.astype(np.float32), np.array(volumes, dtype=np.float32)


def embed(algebra, points, volumes):
    """The points are vectors, and the volume they enclose is invariant, so a scalar."""
    return (algebra.vector(einops.rearrange(points, "batch point coord -> coord batch point")),
            algebra.scalar(e=einops.rearrange(volumes, "batch -> batch 1")))


def rotorch(args):
    from kingdon import Algebra
    from rotorch.models.cgenn import ConvexHullCGMLP

    algebra = Algebra(DIM, **benchmark.codegen(args))
    model = ConvexHullCGMLP(N_POINTS, args.hidden_features, num_layers=args.num_layers).to(args.device)

    def loss_fn(points, volumes):
        input, target = embed(algebra, points, volumes)
        return benchmark.mse_loss(model(input), target)

    return model, loss_fn


def cgenn(args):
    from models.hulls_cgmlp import ConvexHullCGMLP

    model = ConvexHullCGMLP(N_POINTS, args.hidden_features, num_layers=args.num_layers).to(args.device)
    return model, lambda points, volumes: model((points, volumes), 0)[0]


task = benchmark.Task(name="hulls", generate=generate,
                      models=dict(rotorch=rotorch, cgenn=cgenn),
                      defaults=dict(hidden_features=32, num_layers=4))

if __name__ == "__main__":
    benchmark.run(task)
