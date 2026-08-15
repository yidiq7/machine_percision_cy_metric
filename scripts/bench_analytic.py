"""Compare the analytic path against nested autodiff, for speed and for accuracy.

Three configurations:

  baseline   phi from `complex_hessian`, and dv/dtheta from `jacfwd`.
  A          phi from the Gram sums of sigma and its 4 derivatives, dv/dtheta
             still from `jacfwd`.
  A+B        the same phi, and dv/dtheta from Jacobi's formula.

B does not stand alone. The analytic Jacobian contracts the same Gram sums that
A builds, so it needs A.

Usage:
    uv run python -u scripts/bench_analytic.py --k 8 12 16 --n-points 2000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import analytic, data, metric, sections        # noqa: E402


def timed(fn, reps=3):
    """Wall time of fn, after one call that pays the compile."""
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / reps


def peak_gb(fn, *args):
    info = jax.jit(fn).lower(*args).compile().memory_analysis()
    return (info.temp_size_in_bytes + info.output_size_in_bytes) / 1e9


def jac_from_dv(v, dv, mass):
    """The residual Jacobian from dv/dtheta, through the sampled mean."""
    weights = mass / jnp.sum(mass)
    mean = jnp.sum(weights * v)
    dmean = jnp.einsum('i,ia->a', weights, dv)
    grad = dv / mean - jnp.einsum('i,a->ia', v, dmean) / mean ** 2
    return jnp.sqrt(weights)[:, None] * grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[8, 12, 16])
    ap.add_argument('--n-points', type=int, default=2000)
    ap.add_argument('--chunk', type=int, default=250)
    args = ap.parse_args()

    batch = data.make(args.n_points // 4, 3)
    n = int(batch['points'].shape[0])
    print(f"{n} points, chunk {args.chunk}\n")

    head = (f"{'k':>3} {'n_theta':>7} | {'resid base':>10} {'resid A':>9} {'gain':>6} "
            f"| {'jac base':>9} {'jac A':>8} {'jac A+B':>9} {'gain':>7} "
            f"| {'mem base':>9} {'mem A+B':>8} | {'J agree':>9}")
    print(head)
    print('-' * len(head))

    for k in args.k:
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)

        r_base = jax.jit(lambda t: metric.weighted_residual(lay, t, batch, args.chunk))
        r_a = jax.jit(lambda t: analytic.weighted_residual_analytic(lay, t, batch, args.chunk))
        j_base = jax.jit(jax.jacfwd(lambda t: metric.weighted_residual(lay, t, batch, args.chunk)))
        j_a = jax.jit(jax.jacfwd(lambda t: analytic.weighted_residual_analytic(lay, t, batch, args.chunk)))
        j_ab = jax.jit(lambda t: analytic.residual_jacobian(lay, t, batch, args.chunk))

        t_rb = timed(lambda: r_base(theta))
        t_ra = timed(lambda: r_a(theta))
        t_jb = timed(lambda: j_base(theta), 1)
        t_ja = timed(lambda: j_a(theta), 1)
        t_jab = timed(lambda: j_ab(theta), 1)

        m_base = peak_gb(lambda t: jax.jacfwd(
            lambda q: metric.weighted_residual(lay, q, batch, args.chunk))(t), theta)
        m_ab = peak_gb(lambda t: analytic.residual_jacobian(lay, t, batch, args.chunk), theta)

        v, dv = j_ab(theta)
        agree = float(jnp.abs(jac_from_dv(v, dv, batch['mass']) - j_base(theta)).max()
                      / jnp.abs(j_base(theta)).max())

        print(f"{k:>3} {lay.n_theta:>7} | {t_rb:>9.4f}s {t_ra:>8.4f}s {t_rb/t_ra:>5.1f}x "
              f"| {t_jb:>8.3f}s {t_ja:>7.3f}s {t_jab:>8.3f}s {t_jb/t_jab:>6.1f}x "
              f"| {m_base:>8.2f}G {m_ab:>7.2f}G | {agree:>9.2e}")


def lm_analytic(lay, batch, chunk, max_steps=200, tol=1e-15):
    """A plain LM loop on the analytic Jacobian, to check the fit end to end.

    The damped step comes from a least-squares solve on [J; sqrt(lambda) I],
    which is the same system that lineax QR solves inside optimistix.
    """
    theta = sections.fubini_study_theta(lay)
    lam = 1e-3

    def parts(t):
        v, dv = analytic.residual_jacobian(lay, t, batch, chunk)
        return jac_from_dv(v, dv, batch['mass']), _resid(lay, t, batch, chunk)

    def _resid(lay_, t, b, c):
        return analytic.weighted_residual_analytic(lay_, t, b, c)

    jac, res = parts(theta)
    cost = float(jnp.sum(res ** 2))
    for step in range(max_steps):
        n_par = theta.shape[0]
        stacked = jnp.concatenate([jac, jnp.sqrt(lam) * jnp.eye(n_par)])
        rhs = jnp.concatenate([-res, jnp.zeros(n_par)])
        delta = jnp.linalg.lstsq(stacked, rhs, rcond=None)[0]
        trial = theta + delta
        res_new = _resid(lay, trial, batch, chunk)
        cost_new = float(jnp.sum(res_new ** 2))
        if cost_new < cost:
            improve = (cost - cost_new) / max(cost, 1e-300)
            theta, res, cost = trial, res_new, cost_new
            jac, _ = parts(theta)
            lam = max(lam * 0.3, 1e-14)
            if improve < tol:
                return theta, step + 1, 'converged'
        else:
            lam *= 10.0
            if lam > 1e12:
                return theta, step + 1, 'damping limit'
    return theta, max_steps, 'max steps'


def compare_fits(ks, batch, chunk):
    """Fit with optimistix on the baseline, and with LM on the analytic path."""
    from quartic import fit as qfit
    print(f"\n{'k':>3} {'n_theta':>7} | {'optimistix':>16} {'steps':>6} {'sec':>7} "
          f"| {'analytic':>16} {'steps':>6} {'sec':>7} | {'sigma rel diff':>14} {'theta rel':>10}")
    for k in ks:
        lay = sections.layout(k)
        start = time.perf_counter()
        th_o, stats = qfit.fit(lay, batch, chunk=chunk)
        t_o = time.perf_counter() - start
        start = time.perf_counter()
        th_a, steps, _why = lm_analytic(lay, batch, chunk)
        t_a = time.perf_counter() - start
        rep_o = metric.report(lay, th_o, batch)
        rep_a = metric.report(lay, th_a, batch)
        d_theta = float(jnp.linalg.norm(th_a - th_o) / jnp.linalg.norm(th_o))
        print(f"{k:>3} {lay.n_theta:>7} | {rep_o['mape']:>16.8e} {stats['steps']:>6} "
              f"{t_o:>6.1f}s | {rep_a['mape']:>16.8e} {steps:>6} {t_a:>6.1f}s "
              f"| {abs(rep_a['mape'] / rep_o['mape'] - 1):>14.2e} {d_theta:>10.2e}")


if __name__ == '__main__':
    main()
