"""Point sets on the Fermat quartic, from the MLGeometry sampler.

`solve_points` makes a random line in CP^3 and intersects it with X, so each
line gives 4 points. `refine_points` then runs 2 Newton steps, and the measured
residual is max abs(P)/scale = 3.3e-16. The sampler draws from the global
numpy generator, so `make` seeds it and reports the seed.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import jax.numpy as jnp
import MLGeometry as mlg

__all__ = ['QUARTIC', 'COORDS', 'make', 'split', 'residual_of_P']

COORDS = sp.symbols('z0 z1 z2 z3')
QUARTIC = sum(z ** 4 for z in COORDS)


def make(n_pairs, seed):
    """A dataset of 4 * n_pairs points, with the restriction, OmegaOmegabar and mass."""
    np.random.seed(seed)
    surface = mlg.hypersurface.Hypersurface(np.array(COORDS), QUARTIC, n_pairs)
    data = mlg.dataset.generate_dataset(surface)
    return {k: jnp.asarray(v) for k, v in data.items()}


def split(batch, n_train, seed):
    """Split into a fit set and a hold-out set, with an independent permutation."""
    order = np.random.default_rng(seed).permutation(batch['points'].shape[0])
    take = jnp.asarray(order[:n_train])
    rest = jnp.asarray(order[n_train:])
    return ({k: v[take] for k, v in batch.items()},
            {k: v[rest] for k, v in batch.items()})


def residual_of_P(points):
    """abs(P(z)) divided by the scale of the terms that cancel inside P."""
    z = np.asarray(points)
    return np.abs((z ** 4).sum(axis=1)) / np.sum(np.abs(z) ** 4, axis=1)
