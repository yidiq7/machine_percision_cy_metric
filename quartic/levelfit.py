"""Helper functions of the level fits: the HN parameter counts, the layout of each degree, the
inputs of the points, a summary of f, and the linear solve with a cutoff.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import samples as cf
from . import sections


ROOT = Path(__file__).resolve().parents[1]


def hn_counts(k):
    """n_theta(j) for j = 1 to k, from the character theory of the notes."""
    spec = importlib.util.spec_from_file_location(
        'counts', ROOT / 'notes' / 'symmetry_param_counts_s4.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {j: mod.n_theta(j)[0] for j in range(1, k + 1)}


class Layouts(dict):
    """The plain layout of each degree, built when the build first asks for it."""

    def __missing__(self, j):
        self[j] = sections.layout(j, normalize=False)
        return self[j]


def point_inputs(batch):
    return (batch['points_hi'], batch['restriction'],
            (batch['points_hi'], batch['points_lo'], batch['patch']))


def summary(f, batch):
    """mean abs(f), rms, max and the regions of min abs(z) over max abs(z) for one f."""
    mass = np.asarray(batch['mass'] / jnp.sum(batch['mass']))
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    loss = float(np.sum(mass * f * f))
    regions = []
    for low, high in zip(cf.EDGES[:-1], cf.EDGES[1:]):
        take = (ratio >= low) & (ratio < high) if high < 1.0 else (ratio >= low)
        wt = float(mass[take].sum())
        regions.append({'bin': [low, high], 'points': float(take.mean()), 'weight': wt,
                        'loss_share': float(np.sum(mass[take] * f[take] ** 2)) / loss,
                        'mape': float(np.sum(mass[take] * np.abs(f[take]))) / wt,
                        'rms': float(np.sqrt(np.sum(mass[take] * f[take] ** 2) / wt))})
    return {'mape': float(np.sum(mass * np.abs(f))), 'rms': math.sqrt(loss),
            'max': float(np.max(np.abs(f))), 'loss': loss, 'regions': regions}


def solve(jac, r0, cutoffs):
    """The spectrum of the column-scaled J, and -J^+ r0 for each cutoff of s / s_max."""
    norms = jnp.linalg.norm(jac, axis=0)
    q, r = jnp.linalg.qr(jac / norms)
    u, s, vt = jnp.linalg.svd(r)
    beta = u.T @ (q.T @ r0)
    loss0 = float(r0 @ r0)
    rel = np.asarray(s / s[0])
    b2 = np.asarray(beta ** 2)
    shares = {}
    for e in range(0, 24):
        take = (rel < 10.0 ** -e) & (rel >= 10.0 ** -(e + 1))
        if take.any():
            shares[f'1e-{e} to 1e-{e + 1}'] = [int(take.sum()), float(b2[take].sum() / loss0)]
    low = np.abs(np.asarray(vt[-1]))
    spec = {'cond_scaled': float(rel[0] / rel[-1]), 'smallest': [float(x) for x in rel[-4:]],
            'column_norms': [float(norms.min()), float(norms.max())],
            'loss0': loss0, 'outside': float((loss0 - b2.sum()) / loss0), 'shares': shares,
            'smallest_vector_top': [[int(i), float(low[i])] for i in np.argsort(low)[::-1][:6]],
            'smallest_step': float(beta[-1] / s[-1])}
    sols, predicted = [], []
    for tau in cutoffs:
        keep = jnp.asarray(rel >= tau)
        y = -vt.T @ jnp.where(keep, beta / s, 0.0)
        sols.append(np.asarray(y / norms))
        predicted.append(loss0 - float(np.sum(np.where(np.asarray(keep), b2, 0.0))))

    def step(res, tau):
        """-J^+ res with the cutoff tau, for a refinement from the residual res."""
        b = u.T @ (q.T @ res)
        return np.asarray(-(vt.T @ jnp.where(jnp.asarray(rel >= tau), b / s, 0.0)) / norms)
    return spec, np.stack(sols, axis=1), predicted, step
