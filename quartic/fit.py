"""The LM fit of theta on the MSPE, with optimistix.

`optax` and `jax.scipy.optimize` have no LM. On a test problem with the shape of
v, LM reached the FP64 floor at every condition number from 1e2 to 1e11, and
L-BFGS stalled 9 orders above it as soon as cond(J) passed 1e5.

Two settings control the cost:

* `analytic` selects `analytic.weighted_residual_fast`, whose derivative rule is
  hand written. `lineax` materializes the LM operator with `jax.vmap` over the
  basis of theta, and `vmap` only batches an operation whose input carries a
  batch axis, so v and the block matrices W come out once and only the Frobenius
  product runs for each direction. On one A100 this made the Jacobian 14.7 times
  faster at k = 6 and 126 times faster at k = 20.
* `chunk` puts `lax.map` over the points, so the peak memory follows the block
  and not the point count.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from . import analytic, metric, sections

__all__ = ['residual_fn', 'fit', 'condition_number', 'jacobian_memory',
           'suggest_chunk']


def residual_fn(lay, batch, chunk, analytic_jac=True):
    """The weighted residual, either with the analytic rule or with autodiff."""
    if analytic_jac:
        return lambda theta: analytic.weighted_residual_fast(lay, theta, batch, chunk)
    return lambda theta: metric.weighted_residual(lay, theta, batch, chunk)


def device_memory_gb():
    """Memory that JAX can use on device 0. JAX takes 75 percent by default.

    Raise it with XLA_PYTHON_CLIENT_MEM_FRACTION, for example to 0.95.
    """
    try:
        return (jax.devices()[0].memory_stats() or {}).get('bytes_limit', 0) / 1e9
    except Exception:
        return 0.0


def _solver(rtol, atol, verbose):
    # optimistix 0.1.0 takes a bool or a callable here, and not a frozenset.
    return optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=verbose)


def solve_memory(lay, batch, chunk, theta=None, max_steps=200, rtol=1e-15,
                 atol=1e-32, analytic_jac=True):
    """Peak memory of the whole LM solve, in GB, from the compiled program.

    One Jacobian is not the peak. The same program also holds J, the stacked
    [J; sqrt(lambda) I], the QR of it, and the candidates of the trust region.
    Sizing against the Jacobian alone under-counts, and the run then fails.
    """
    theta = sections.fubini_study_theta(lay) if theta is None else theta
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    solver = _solver(rtol, atol, False)

    def solve(start):
        return optx.least_squares(lambda q, args: residual(q), solver, start,
                                  max_steps=max_steps, throw=False).value

    info = jax.jit(solve).lower(theta).compile().memory_analysis()
    return (info.temp_size_in_bytes + info.output_size_in_bytes) / 1e9


def fit(lay, batch, theta0=None, max_steps=200, rtol=1e-15, atol=1e-32,
        chunk=2000, analytic_jac=True, verbose=False):
    """Minimize the MSPE over theta. Returns the solution and its statistics."""
    theta0 = sections.fubini_study_theta(lay) if theta0 is None else theta0
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    solver = _solver(rtol, atol, verbose)

    start = time.perf_counter()
    sol = optx.least_squares(lambda theta, args: residual(theta), solver, theta0,
                             max_steps=max_steps, throw=False)
    steps = int(sol.stats['num_steps'])
    # RESULTS has no readable name, so compare it and report the step limit too.
    return sol.value, {
        'steps': steps,
        'converged': bool(sol.result == optx.RESULTS.successful),
        'hit_max_steps': steps >= max_steps,
        'message': str(optx.RESULTS[sol.result]) or 'successful',
        'chunk': chunk,
        'analytic_jac': analytic_jac,
        'seconds': time.perf_counter() - start,
    }


def jacobian_memory(lay, batch, chunk=2000, theta=None, analytic_jac=True):
    """Peak device memory of one Jacobian, in GB, without a run.

    XLA reports this from the compiled program, so the call costs a compile and
    no execution. `temp` follows the chunk, and `output` is J itself, which is
    N * n_theta * 8 bytes.
    """
    theta = sections.fubini_study_theta(lay) if theta is None else theta
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    info = jax.jit(jax.jacfwd(residual)).lower(theta).compile().memory_analysis()
    return {
        'temp_gb': info.temp_size_in_bytes / 1e9,
        'output_gb': info.output_size_in_bytes / 1e9,
        'total_gb': (info.temp_size_in_bytes + info.output_size_in_bytes) / 1e9,
        'gb_per_point_in_chunk': info.temp_size_in_bytes / 1e9 / chunk,
        'chunk': chunk,
        'analytic_jac': analytic_jac,
    }


def suggest_chunk(lay, batch, budget_gb=None, probe=256, analytic_jac=True,
                  tries=8, max_steps=200):
    """Largest chunk that keeps the whole LM solve under `budget_gb`.

    Two probes separate the fixed cost from the part that follows the chunk.
    The prediction is then checked by a compile at the chosen chunk, and the
    chunk halves until it fits. A compile costs seconds and an out-of-memory
    run costs the whole job, so the check is worth it.
    """
    n_points = int(batch['points'].shape[0])
    if budget_gb is None:
        budget_gb = 0.45 * device_memory_gb() or 8.0

    def peak(size):
        return solve_memory(lay, batch, min(size, n_points), max_steps=max_steps,
                            analytic_jac=analytic_jac)

    small, large = min(probe, n_points), min(2 * probe, n_points)
    peak_small, peak_large = peak(small), peak(large)
    per_point = max((peak_large - peak_small) / max(large - small, 1), 1e-12)
    fixed = max(peak_small - per_point * small, 0.0)

    chunk = max(1, min(n_points, int((budget_gb - fixed) / per_point)))
    for _ in range(tries):
        measured = peak(chunk)
        if measured <= budget_gb or chunk == 1:
            return chunk, {'gb_per_point_in_chunk': per_point,
                           'fixed_gb': fixed, 'measured_total_gb': measured,
                           'budget_gb': budget_gb}
        chunk = max(1, chunk // 2)
    return chunk, {'gb_per_point_in_chunk': per_point, 'fixed_gb': fixed,
                   'measured_total_gb': peak(chunk), 'budget_gb': budget_gb}


def condition_number(lay, theta, batch, chunk=2000, drop_scale=True,
                     analytic_jac=True):
    """Singular values of J = d(weighted residual)/d(theta), at theta.

    `h -> lambda h` leaves the metric unchanged, so theta has one exact null
    direction. `drop_scale` removes the smallest singular value with it.
    """
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    jac = np.asarray(jax.jacfwd(residual)(theta))
    sing = np.linalg.svd(jac, compute_uv=False)
    kept = sing[:-1] if drop_scale else sing
    return {
        'sigma_max': float(kept[0]),
        'sigma_min': float(kept[-1]),
        'cond': float(kept[0] / kept[-1]),
        'null_sigma': float(sing[-1]),
        'shape': jac.shape,
    }
