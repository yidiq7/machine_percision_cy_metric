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
import lineax as lx
import optimistix as optx

from . import analytic, metric, sections

__all__ = ['residual_fn', 'fit', 'condition_number', 'jacobian_memory',
           'plan_fit', 'suggest_chunk', 'solve_memory', 'device_memory_gb']


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


def _solver(rtol, atol, verbose, rcond=None, best_so_far=False):
    """LM, with a rank-revealing step when `rcond` is set.

    cond(J) grows by 5.65 for each unit of k, and it passes 1/eps near k = 23.
    Past that J is numerically rank deficient, and lineax QR returns a step
    built from noise: at k = 24 LM stopped at the start point, and at k = 28 it
    threw theta out of the region where p is positive. `lx.SVD` drops every
    direction below rcond * sigma_max instead, and those directions carry
    almost none of the residual.

    `best_so_far` keeps a copy of the theta with the smallest loss. Use it with
    rtol = atol = 0, which stops the convergence test from firing, so that the
    fit runs to `max_steps` and the answer does not depend on that test.
    """
    linear = lx.QR() if rcond is None else lx.SVD(rcond=rcond)
    # optimistix 0.1.0 takes a bool or a callable here, and not a frozenset.
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=verbose,
                                     linear_solver=linear)
    return optx.BestSoFarLeastSquares(solver) if best_so_far else solver


def solve_memory(lay, batch, chunk, theta=None, max_steps=200, rtol=1e-15,
                 atol=1e-32, analytic_jac=True, rcond=None, best_so_far=False):
    """Peak memory of the whole LM solve, in GB, from the compiled program.

    One Jacobian is not the peak. The same program also holds J, the stacked
    [J; sqrt(lambda) I], the QR of it, and the candidates of the trust region.
    Sizing against the Jacobian alone under-counts, and the run then fails.
    """
    theta = sections.fubini_study_theta(lay) if theta is None else theta
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    solver = _solver(rtol, atol, False, rcond, best_so_far)

    def solve(start):
        return optx.least_squares(lambda q, args: residual(q), solver, start,
                                  max_steps=max_steps, throw=False).value

    info = jax.jit(solve).lower(theta).compile().memory_analysis()
    return (info.temp_size_in_bytes + info.output_size_in_bytes) / 1e9


def fit(lay, batch, theta0=None, max_steps=200, rtol=1e-15, atol=1e-32,
        chunk=2000, analytic_jac=True, verbose=False, rcond=None,
        best_so_far=False):
    """Minimize the MSPE over theta. Returns the solution and its statistics.

    The convergence test of optimistix measures the step in theta, so it fires
    where the step is small and not where the residual stops falling. Two exact
    reparametrizations therefore stop at different residuals. `best_so_far` with
    rtol = atol = 0 removes that dependence, at the cost of every step.
    """
    theta0 = sections.fubini_study_theta(lay) if theta0 is None else theta0
    residual = residual_fn(lay, batch, chunk, analytic_jac)
    solver = _solver(rtol, atol, verbose, rcond, best_so_far)

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
        'rcond': rcond,
        'best_so_far': best_so_far,
        'rtol': rtol,
        'atol': atol,
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


def plan_fit(lay, batch, budget_gb=None, chunk_cap=4096, analytic_jac=True,
             max_steps=200, tries=8, min_chunk=64, target_risk=5e-3,
             best_so_far=False):
    """Fit-set size and block size that keep the whole LM solve in the budget.

    The peak follows `a * N_fit + b * chunk + c`. The `a` term is the surprise:
    the LM `while_loop` keeps one residual's per-point work alive for every
    point, and the `lax.map` block does not bound it. Measured at k = 12 it is
    4554 doubles for each point, which is sigma, dsigma, H sigma, H dsigma and W
    together, and it grows as `sum_r d_r^2`.

    So a block alone cannot fit a large k. The fit set must shrink as well, and
    the cost of that is the excess risk `n_theta / N_fit`.
    """
    n_points = int(batch['points'].shape[0])
    if budget_gb is None:
        budget_gb = 0.45 * device_memory_gb() or 8.0

    def peak(n_fit, chunk):
        small = {k: v[:n_fit] for k, v in batch.items()}
        return solve_memory(lay, small, min(chunk, n_fit), max_steps=max_steps,
                            analytic_jac=analytic_jac,
                            best_so_far=best_so_far)

    # Two point counts at one block, and one more block, give a, b and c.
    n_lo, n_hi = min(2000, n_points), min(8000, n_points)
    c_lo, c_hi = min(250, n_lo), min(1000, n_lo)
    p_lo, p_hi = peak(n_lo, c_lo), peak(n_hi, c_lo)
    p_chunk = peak(n_lo, c_hi)
    per_point = max((p_hi - p_lo) / max(n_hi - n_lo, 1), 1e-12)
    per_chunk = max((p_chunk - p_lo) / max(c_hi - c_lo, 1), 1e-12)
    fixed = max(p_lo - per_point * n_lo - per_chunk * c_lo, 0.0)

    # A fit set below this carries no information about n_theta parameters, so
    # a plan that needs one is a failure and not a small answer.
    floor = min(n_points, max(4 * lay.n_theta, 1000))

    # Enough fit points to bring the excess risk n_theta/N_fit down to
    # `target_risk`, and no more. Past that the points cost far more than they
    # buy: at k = 20, 141948 points against 100000 lowered the risk from 2.9e-3
    # to 2.0e-3, moved sigma by 0.04 percent, and made the run 2.5 times longer,
    # because the budget went to the fit set and left a block of 64.
    want = max(floor, int(lay.n_theta / max(target_risk, 1e-9)))
    small = min(min_chunk, n_points)
    afford = int((budget_gb - fixed - per_chunk * small) / per_point)
    n_fit = max(floor, min(n_points, want, afford))

    # Everything left goes to the block, which only buys speed.
    chunk = int((budget_gb - fixed - per_point * n_fit) / per_chunk)
    chunk = max(1, min(chunk_cap, n_fit, chunk))

    measured = peak(n_fit, chunk)
    for _ in range(tries):
        if measured <= budget_gb:
            break
        if chunk > 1:
            chunk = max(1, chunk // 4)          # time before information
        elif n_fit > floor:
            n_fit = max(floor, int(n_fit * min(0.8, budget_gb / measured)))
        else:
            break
        measured = peak(n_fit, chunk)

    if measured > budget_gb:
        need = fixed + per_chunk + per_point * floor
        raise MemoryError(
            f"k={lay.k} needs approximately {need:.1f} GB for the smallest "
            f"useful fit set of {floor} points with a block of 1, and the budget "
            f"is {budget_gb:.1f} GB on a {device_memory_gb():.1f} GB device. "
            f"Raise --budget-gb, set XLA_PYTHON_CLIENT_MEM_FRACTION=0.95, turn "
            f"--gram-dd off, or lower k.")

    return n_fit, chunk, {
        'gb_per_fit_point': per_point, 'gb_per_chunk_point': per_chunk,
        'fixed_gb': fixed, 'measured_total_gb': measured,
        'budget_gb': budget_gb, 'excess_risk': lay.n_theta / max(n_fit, 1),
        'target_risk': target_risk, 'n_points_available': n_points,
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
