"""Compare the analytic path against nested autodiff, on whatever device runs it.

Three configurations:

  baseline   phi from `complex_hessian`, and dv/dtheta from `jacfwd`.
  A          phi from the Gram sums of sigma and its 4 derivatives, dv/dtheta
             still from `jacfwd`.
  A+B        the same phi, and dv/dtheta from Jacobi's formula.

B does not stand alone. The analytic Jacobian contracts the same Gram sums that
A builds, so it needs A.

The laptop CPU gave a Jacobian gain of 10x at k = 6 and 110x at k = 16. A GPU
has a different balance of arithmetic against bandwidth, and the baseline builds
larger fused kernels, so the result can differ. This script measures it.

Usage on the cluster:

    uv run python -u scripts/bench_analytic.py --k 6 8 12 16 20 --out bench.json
    uv run python -u scripts/bench_analytic.py --k 8 --fit          # end to end

Every configuration is guarded, so an out-of-memory case prints OOM and the
table continues.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
import numpy as np                                          # noqa: E402
from MLGeometry import complex_math                         # noqa: E402
from quartic import analytic, data, metric, sections        # noqa: E402

OOM = (RuntimeError, MemoryError, jax.errors.JaxRuntimeError)


def describe_device():
    dev = jax.devices()[0]
    out = {'platform': dev.platform, 'kind': dev.device_kind,
           'count': len(jax.devices()), 'jax': jax.__version__,
           'x64': bool(jax.config.jax_enable_x64)}
    try:
        stats = dev.memory_stats() or {}
        out['device_gb'] = stats.get('bytes_limit', 0) / 1e9
    except Exception:
        out['device_gb'] = 0.0
    return out


def timed(fn, reps=3):
    """Wall time of fn, after one call that pays the compile."""
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / reps


def peak_gb(fn, *args):
    """Peak device memory of the compiled program, with no execution."""
    info = jax.jit(fn).lower(*args).compile().memory_analysis()
    return (info.temp_size_in_bytes + info.output_size_in_bytes) / 1e9


def jac_from_dv(v, dv, mass):
    """The residual Jacobian from dv/dtheta, through the sampled mean."""
    weights = mass / jnp.sum(mass)
    mean = jnp.sum(weights * v)
    dmean = jnp.einsum('i,ia->a', weights, dv)
    grad = dv / mean - jnp.einsum('i,a->ia', v, dmean) / mean ** 2
    return jnp.sqrt(weights)[:, None] * grad


def pick_chunk(lay, batch, budget_gb, probes=(100, 20, 4, 1)):
    """Largest chunk that keeps the baseline Jacobian inside the budget.

    Both paths then use the same chunk, so the comparison is of the algorithms
    and not of the block size. The temporary memory is proportional to the
    chunk, so one probe gives the constant. A high k needs a small probe,
    because the probe itself holds that memory.
    """
    n_points = int(batch['points'].shape[0])
    theta = sections.fubini_study_theta(lay)
    for probe in probes:
        try:
            total = peak_gb(lambda t: jax.jacfwd(
                lambda q: metric.weighted_residual(lay, q, batch, probe))(t), theta)
        except OOM:
            continue
        per_point = total / probe
        chunk = max(1, min(n_points, int(budget_gb / max(per_point, 1e-12))))
        if chunk < 8:
            print(f"    warning: k={lay.k} needs {per_point * 1e3:.0f} MB for each "
                  f"point on the baseline, so the block is only {chunk}. The "
                  f"baseline will be slow. Use --chunk to force a value.")
        return chunk, per_point
    raise MemoryError('the baseline Jacobian does not fit even for one point')


def check_agreement(lay, theta, batch, chunk):
    """phi, the residual and the Jacobian against the present path."""
    mats = sections.prepare(lay, theta)
    z = batch['points'][0]
    phi_a, _ = analytic.phi_analytic(lay, mats, z)
    phi_b = complex_math.complex_hessian(
        lambda w: sections.log_potential_at(lay, mats, w), z)
    d_phi = float(jnp.abs(phi_a - phi_b).max() / jnp.abs(phi_b).max())

    r_a = analytic.weighted_residual_analytic(lay, theta, batch, chunk)
    r_b = metric.weighted_residual(lay, theta, batch, chunk)
    d_res = float(jnp.abs(r_a - r_b).max() / jnp.abs(r_b).max())

    v, dv = analytic.residual_jacobian(lay, theta, batch, chunk)
    j_a = jac_from_dv(v, dv, batch['mass'])
    j_b = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, batch, chunk))(theta)
    d_jac = float(jnp.abs(j_a - j_b).max() / jnp.abs(j_b).max())
    return {'phi': d_phi, 'residual': d_res, 'jacobian': d_jac}


def bench_one(k, batch, budget_gb, forced_chunk):
    """Every configuration at one degree k. Returns a record."""
    lay = sections.layout(k)
    theta = sections.fubini_study_theta(lay)
    rec = {'k': k, 'n_theta': lay.n_theta, 'macs': lay.structure.macs}

    if forced_chunk:
        chunk, per_point = forced_chunk, float('nan')
    else:
        chunk, per_point = pick_chunk(lay, batch, budget_gb)
    rec['chunk'] = chunk
    rec['baseline_gb_per_point'] = per_point

    def guard(name, fn):
        try:
            rec[name] = fn()
        except OOM as exc:
            rec[name] = None
            rec.setdefault('errors', {})[name] = str(exc)[:120]

    r_base = jax.jit(lambda t: metric.weighted_residual(lay, t, batch, chunk))
    r_a = jax.jit(lambda t: analytic.weighted_residual_analytic(lay, t, batch, chunk))
    j_base = jax.jit(jax.jacfwd(lambda t: metric.weighted_residual(lay, t, batch, chunk)))
    j_a = jax.jit(jax.jacfwd(
        lambda t: analytic.weighted_residual_analytic(lay, t, batch, chunk)))
    j_ab = jax.jit(lambda t: analytic.residual_jacobian(lay, t, batch, chunk))

    guard('t_resid_base', lambda: timed(lambda: r_base(theta)))
    guard('t_resid_A', lambda: timed(lambda: r_a(theta)))
    guard('t_jac_base', lambda: timed(lambda: j_base(theta), 1))
    guard('t_jac_A', lambda: timed(lambda: j_a(theta), 1))
    guard('t_jac_AB', lambda: timed(lambda: j_ab(theta), 1))
    guard('mem_base_gb', lambda: peak_gb(lambda t: jax.jacfwd(
        lambda q: metric.weighted_residual(lay, q, batch, chunk))(t), theta))
    guard('mem_AB_gb', lambda: peak_gb(
        lambda t: analytic.residual_jacobian(lay, t, batch, chunk), theta))
    guard('agreement', lambda: check_agreement(lay, theta, batch, chunk))
    return rec


def show(rec):
    def fmt(key, unit='s', width=9, digits=3):
        val = rec.get(key)
        return f"{'OOM':>{width}}" if val is None else f"{val:>{width}.{digits}f}{unit}"

    gain_r = gain_j = '     -'
    if rec.get('t_resid_base') and rec.get('t_resid_A'):
        gain_r = f"{rec['t_resid_base'] / rec['t_resid_A']:>5.1f}x"
    if rec.get('t_jac_base') and rec.get('t_jac_AB'):
        gain_j = f"{rec['t_jac_base'] / rec['t_jac_AB']:>6.1f}x"
    agree = rec.get('agreement')
    worst = f"{max(agree.values()):.1e}" if agree else 'n/a'

    print(f"{rec['k']:>3} {rec['n_theta']:>7} {rec['chunk']:>6} | "
          f"{fmt('t_resid_base', 's', 8, 4)} {fmt('t_resid_A', 's', 8, 4)} {gain_r} | "
          f"{fmt('t_jac_base')} {fmt('t_jac_A')} {fmt('t_jac_AB')} {gain_j} | "
          f"{fmt('mem_base_gb', 'G', 7, 2)} {fmt('mem_AB_gb', 'G', 7, 2)} | {worst:>9}")


def lm_analytic(lay, batch, chunk, max_steps=200, tol=1e-15):
    """A plain LM loop on the analytic Jacobian, to check the fit end to end.

    The damped step comes from a least-squares solve on [J; sqrt(lambda) I],
    which is the same system that lineax QR solves inside optimistix.
    """
    theta = sections.fubini_study_theta(lay)
    lam = 1e-3

    def residual_of(t):
        return analytic.weighted_residual_analytic(lay, t, batch, chunk)

    def jacobian_of(t):
        v, dv = analytic.residual_jacobian(lay, t, batch, chunk)
        return jac_from_dv(v, dv, batch['mass'])

    res = residual_of(theta)
    jac = jacobian_of(theta)
    cost = float(jnp.sum(res ** 2))
    for step in range(max_steps):
        n_par = theta.shape[0]
        stacked = jnp.concatenate([jac, jnp.sqrt(lam) * jnp.eye(n_par)])
        rhs = jnp.concatenate([-res, jnp.zeros(n_par)])
        delta = jnp.linalg.lstsq(stacked, rhs, rcond=None)[0]
        trial = theta + delta
        res_new = residual_of(trial)
        cost_new = float(jnp.sum(res_new ** 2))
        if cost_new < cost:
            improve = (cost - cost_new) / max(cost, 1e-300)
            theta, res, cost = trial, res_new, cost_new
            jac = jacobian_of(theta)
            lam = max(lam * 0.3, 1e-14)
            if improve < tol:
                return theta, step + 1, 'converged'
        else:
            lam *= 10.0
            if lam > 1e12:
                return theta, step + 1, 'damping limit'
    return theta, max_steps, 'max steps'


def compare_fits(ks, batch, budget_gb, forced_chunk):
    """Fit with optimistix on the baseline, and with LM on the analytic path."""
    from quartic import fit as qfit
    print(f"\n{'k':>3} {'n_theta':>7} | {'optimistix':>16} {'steps':>6} {'sec':>8} "
          f"| {'analytic':>16} {'steps':>6} {'sec':>8} | {'sigma rel':>10} {'speedup':>8}")
    out = []
    for k in ks:
        lay = sections.layout(k)
        chunk = forced_chunk or pick_chunk(lay, batch, budget_gb)[0]
        try:
            start = time.perf_counter()
            th_o, stats = qfit.fit(lay, batch, chunk=chunk)
            t_o = time.perf_counter() - start
            sig_o = metric.report(lay, th_o, batch)['mape']
            steps_o = stats['steps']
        except OOM:
            t_o, sig_o, steps_o = float('nan'), float('nan'), -1

        start = time.perf_counter()
        th_a, steps_a, _why = lm_analytic(lay, batch, chunk)
        t_a = time.perf_counter() - start
        sig_a = metric.report(lay, th_a, batch)['mape']

        rel = abs(sig_a / sig_o - 1) if sig_o == sig_o else float('nan')
        print(f"{k:>3} {lay.n_theta:>7} | {sig_o:>16.8e} {steps_o:>6} {t_o:>7.1f}s "
              f"| {sig_a:>16.8e} {steps_a:>6} {t_a:>7.1f}s | {rel:>10.2e} "
              f"{t_o / t_a:>7.1f}x")
        out.append({'k': k, 'sigma_optimistix': sig_o, 'sigma_analytic': sig_a,
                    'seconds_optimistix': t_o, 'seconds_analytic': t_a,
                    'steps_optimistix': steps_o, 'steps_analytic': steps_a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[6, 8, 12, 16, 20])
    ap.add_argument('--n-points', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=None,
                    help='force the block size; the default fits --budget-gb')
    ap.add_argument('--budget-gb', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--fit', action='store_true',
                    help='also fit end to end with both paths, which is slower')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    info = describe_device()
    print(f"device: {info['platform']} {info['kind']} x{info['count']}, "
          f"{info['device_gb']:.0f} GB, jax {info['jax']}, x64 {info['x64']}")

    batch = data.make(args.n_points // 4, args.seed)
    n = int(batch['points'].shape[0])
    print(f"{n} points, budget {args.budget_gb:.0f} GB\n")

    head = (f"{'k':>3} {'n_theta':>7} {'chunk':>6} | {'resid base':>9} {'resid A':>9} "
            f"{'gain':>6} | {'jac base':>10} {'jac A':>10} {'jac A+B':>10} {'gain':>7} "
            f"| {'mem base':>8} {'mem A+B':>8} | {'agree':>9}")
    print(head)
    print('-' * len(head))

    records = []
    for k in args.k:
        try:
            rec = bench_one(k, batch, args.budget_gb, args.chunk)
        except OOM as exc:
            rec = {'k': k, 'n_theta': sections.layout(k).n_theta, 'chunk': 0,
                   'fatal': str(exc)[:200]}
            print(f"{k:>3} {rec['n_theta']:>7}      0 | out of memory before the timing")
        except Exception:
            traceback.print_exc()
            continue
        else:
            show(rec)
        records.append(rec)

    fits = compare_fits(args.k, batch, args.budget_gb, args.chunk) if args.fit else []

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'device': info, 'n_points': n, 'budget_gb': args.budget_gb,
             'bench': records, 'fits': fits}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
