"""Lift a fit from degree k_low to degree k_high through the potential.

If p_high = p_low^(k_high/k_low) on X, then (1/k_high) ddbar log p_high equals
(1/k_low) ddbar log p_low, so the lifted metric is the metric of the low fit.
p_high is linear in theta, so the best lift on a point sample is one weighted
linear least-squares problem:

    min over theta of sum_i w_i (p_high(theta; z_i) / t_i - 1)^2,
    t_i = p_low(z_i)^(k_high/k_low).

A row of it is scale <B_a, sum_b Re(tau_b tau_b^*)> / t_i with tau_b = rot_b^T
sigma_b, formed in dd. The script reports the lifted theta with the whole path
in dd on a hold-out set, so a least-squares solve in FP64 cannot give a false
number.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/studies/lift.py --k-low 28 --k-high 32 \\
        --theta runs_end/theta_k28_seed20260813.npy --points 200000
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd, dd_metric, metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def potential_dd(lay, parts, z, extra):
    """p at one point from the dd Gram stage, at the dd point, rounded once."""
    z_dd, _, _ = analytic._dd_point(extra)
    sig, dsig, _, _ = analytic._section_vectors(lay, z, z_dd=z_dd)
    return dd.to_fp(analytic._gram_parts(lay, parts, sig, dsig)[4][0])


def potential_row(lay, rots, z, extra):
    """dp/dtheta at one point: scale <B_a, sum over the orbit of Re(tau tau^*)>."""
    z_dd, _, _ = analytic._dd_point(extra)
    sig, _ = analytic.section_vectors_dd(lay, z, z_dd=z_dd)
    row = jnp.zeros((lay.n_theta,), jnp.float64)
    for g, pair, s in zip(lay.groups, rots, sig):
        col = tuple(tuple(s[c][h][..., None] for h in (0, 1)) for c in (0, 1))
        tau = dd.cto_fp(analytic._apply_rotation_dd(pair, col))[..., 0]   # (b, d)
        w = jnp.real(jnp.einsum('bi,bj->bij', tau, jnp.conj(tau)))
        for local, sl in enumerate(g.slices):
            take = jnp.asarray(g.orbit_of == local)
            summed = jnp.sum(jnp.where(take[:, None, None], w, 0.0), axis=0)
            basis = jnp.asarray(g.basis[local, :sl.stop - sl.start])
            row = row.at[sl].add(jnp.einsum('aij,ij->a', basis, summed))
    return row * jnp.asarray(lay.scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-low', type=int, required=True)
    ap.add_argument('--k-high', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True, help='the plain theta at k-low')
    ap.add_argument('--points', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--n-eval', type=int, default=8_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--out', type=str, default='runs_lift')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.PHI_DD = True, True, True
    analytic.END_DD, analytic.F_DD = True, True
    sections.PREPARE_DD = True

    low = sections.layout(args.k_low)
    high = sections.layout(args.k_high)
    theta_low = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != low.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis at k-low')

    batch = data.make(math.ceil(args.points / 4), args.seed)
    batch = {key: val[:args.points] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    xs = analytic.point_inputs(batch)
    weights = np.asarray(batch['mass']) / float(np.sum(np.asarray(batch['mass'])))
    say(f'k = {args.k_low} to {args.k_high}, {args.points} points of seed {args.seed}')

    parts = jax.jit(lambda t: sections.prepare_parts(low, t))(theta_low)
    p_low = np.asarray(jax.jit(lambda x: jax.lax.map(
        lambda a: potential_dd(low, parts, a[0], a[3:]), x, batch_size=args.block))(xs))
    target = np.exp((args.k_high / args.k_low) * np.log(p_low))
    rots = analytic.whitened_rotations(high)
    rows = np.asarray(jax.jit(lambda x: jax.lax.map(
        lambda a: potential_row(high, rots, a[0], a[3:]), x, batch_size=args.block))(xs))
    say(f'rows of the least-squares problem: {rows.shape}')

    root = np.sqrt(weights)
    mat = jnp.asarray(rows * (root / target)[:, None])
    rhs = jnp.asarray(root)
    theta_high, _, rank, sing = jnp.linalg.lstsq(mat, rhs, rcond=None)
    theta_high = np.asarray(theta_high)
    rel = (rows @ theta_high) / target - 1.0
    say(f'least squares: rank {int(rank)} of {high.n_theta}, cond {float(sing[0] / sing[-1]):.3e}, '
        f'relative residual of p: rms {np.sqrt(np.sum(weights * rel ** 2)):.3e}, '
        f'max {np.abs(rel).max():.3e}')

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stem = f'theta_k{args.k_high}_from_k{args.k_low}'
    np.save(out_dir / f'{stem}.npy', theta_high)
    (out_dir / f'{stem}.basis').write_text(high.fingerprint() + '\n')

    hold = data.make(math.ceil(args.n_eval / 4), args.eval_seed)
    hold = {key: val[:args.n_eval] for key, val in hold.items()}
    hold.update(dd_metric.point_extras(hold['points']))
    rep = metric.report(high, jnp.asarray(theta_high), hold)
    say(f"the lift at k = {args.k_high} on {args.n_eval} hold-out points: mean abs(f) "
        f"{rep['mape']:.4e}, rms {rep['rms']:.3e}, max {rep['max_abs']:.3e}, "
        f"min lambda {rep['min_eig']:.3e}, min p {rep['min_p']:.3e}")
    (out_dir / f'{stem}.json').write_text(json.dumps(
        {'args': vars(args), 'lsq_rms': float(np.sqrt(np.sum(weights * rel ** 2))),
         'lsq_rank': int(rank), 'report': rep}, indent=1))
    say(f'saved to {out_dir}/{stem}.npy')


if __name__ == '__main__':
    main()
