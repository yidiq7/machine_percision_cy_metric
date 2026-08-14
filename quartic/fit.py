"""The LM fit of theta on the MSPE, with optimistix.

`optax` and `jax.scipy.optimize` have no LM. On a test problem with the shape of
v, LM reached the FP64 floor at every condition number from 1e2 to 1e11, and
L-BFGS stalled 9 orders above it as soon as cond(J) passed 1e5.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from . import metric, sections

__all__ = ['fit', 'condition_number']


def fit(lay, batch, theta0=None, max_steps=200, rtol=1e-15, atol=1e-32,
        verbose=False):
    """Minimize the MSPE over theta. Returns the solution and its statistics."""
    theta0 = sections.fubini_study_theta(lay) if theta0 is None else theta0

    def resid(theta, args):
        return metric.weighted_residual(lay, theta, batch)

    # optimistix 0.1.0 takes a bool or a callable here, and not a frozenset.
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=verbose)

    start = time.perf_counter()
    sol = optx.least_squares(resid, solver, theta0, max_steps=max_steps,
                             throw=False)
    steps = int(sol.stats['num_steps'])
    # RESULTS has no readable name, so compare it and report the step limit too.
    return sol.value, {
        'steps': steps,
        'converged': bool(sol.result == optx.RESULTS.successful),
        'hit_max_steps': steps >= max_steps,
        'message': str(optx.RESULTS[sol.result]) or 'successful',
        'seconds': time.perf_counter() - start,
    }


def condition_number(lay, theta, batch, drop_scale=True):
    """Singular values of J = d(weighted residual)/d(theta), at theta.

    `h -> lambda h` leaves the metric unchanged, so theta has one exact null
    direction. `drop_scale` removes the smallest singular value with it.
    """
    jac = np.asarray(jax.jacfwd(
        lambda t: metric.weighted_residual(lay, t, batch))(theta))
    sing = np.linalg.svd(jac, compute_uv=False)
    kept = sing[:-1] if drop_scale else sing
    return {
        'sigma_max': float(kept[0]),
        'sigma_min': float(kept[-1]),
        'cond': float(kept[0] / kept[-1]),
        'null_sigma': float(sing[-1]),
        'shape': jac.shape,
    }
