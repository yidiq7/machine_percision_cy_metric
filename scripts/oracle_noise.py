"""Noise of the computed residual as a function of theta, which is what LM sees.

The second difference s = r(theta + h u) + r(theta - h u) - 2 r(theta) along a
direction u cancels the smooth part of r to second order. For a small step the
curvature term h^2 u^T Hess(r) u is negligible, and s holds only the round-off
of the three evaluations, so norm(s) / sqrt(6) estimates the noise of one
evaluation. Three step sizes check that: the noise does not depend on h, and
the curvature term grows as h^2.

u is theta times random normal numbers, so h is the relative size of the step
of each component. h = 1e-12 moves each component by approximately 4500 ulp.

The script measures the noise in several arithmetic configurations, at one
theta and on the fit set of the run that saved it, and it shows how the noise
spreads over the points. A noise that a few points carry has a different cause
from a noise that every point carries.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/oracle_noise.py --k 28 --n-fit 59312 \\
        --chunk 2653 --theta runs/theta_k28_seed20260813.npy
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd_metric, fit, sections   # noqa: E402

# name: (Gram stage, stages 1 to 3 in dd, H in dd, low part of H, stage 5 in dd,
#        the whole path in dd from a dd point)
CONFIGS = {
    'fit':     ('on', True, False, False, False, False),  # the fits to 2026-09
    'prepare': ('on', True, True, False, False, False),   # the same with H in dd
    'gram':    ('on', False, True, False, False, False),  # H and stage 4 in dd
    'fp64':    ('off', False, True, False, False, False), # H in dd, per-point FP64
    'carry':   ('on', False, True, True, False, False),   # and the low part of H
    'all':     ('on', True, True, True, False, False),    # and stages 1 to 3 in dd
    'phi':     ('on', False, True, True, True, False),    # carry, and stage 5 in dd
    'end':     ('on', False, True, True, True, True),     # END_DD at dd points
}
T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def spread(s2, key):
    """Share of sum(s2) that the top 0.1, 1 and 10 percent of points carry,
    with the points put in order by `key`, largest first."""
    order = np.argsort(key)[::-1]
    total = s2.sum()
    out = {}
    for share in (0.001, 0.01, 0.1):
        n = max(1, int(round(share * s2.size)))
        out[share] = float(s2[order[:n]].sum() / total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--n-fit', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument('--steps', type=float, nargs='+', default=[1e-12, 1e-10, 1e-8])
    ap.add_argument('--directions', type=int, default=2)
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    dev = jax.devices()[0]
    if dev.platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')

    lay = sections.layout(args.k)
    theta = np.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis '
                         'of this build')
    sample = data.make(math.ceil(args.n_train / 4), args.seed)
    batch = {key: val[:args.n_fit] for key, val in sample.items()}
    if any(CONFIGS[name][5] for name in args.configs):
        batch.update(dd_metric.point_extras(batch['points']))
    mass = np.asarray(batch['mass'])
    root = np.sqrt(mass / mass.sum())
    say(f'k = {args.k}, {args.n_fit} of {args.n_train} points, seed {args.seed}, '
        f'at {args.theta}')

    rng = np.random.default_rng(11)
    dirs = [theta * rng.normal(size=theta.size) for _ in range(args.directions)]

    results, base = {}, {}
    for name in args.configs:
        gram, section, prep, low, phi, end = CONFIGS[name]
        analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[gram]
        analytic.SECTION_DD = section
        sections.PREPARE_DD = prep
        analytic.H_LOW = low
        analytic.PHI_DD = phi
        analytic.END_DD = end
        # A new function object, so jit traces again with these flags.
        res = jax.jit(lambda t, lay=lay: fit.residual_fn(lay, batch, args.chunk)(t))
        r0 = np.asarray(res(jnp.asarray(theta)))
        base[name] = r0
        f0 = r0 / root
        norm_r = float(np.linalg.norm(r0))
        row = {'gram_dd': gram, 'section_dd': section, 'prepare_dd': prep,
               'h_low': low, 'phi_dd': phi, 'end_dd': end,
               'sigma_fit': float(np.sum(root * np.abs(r0))), 'norm_r': norm_r,
               'noise': {}}
        say(f"{name}: Gram {gram}, stages 1-3 dd {section}, H dd {prep}, low {low}, "
            f"stage 5 dd {phi}, end to end dd {end}: "
            f"fit-set sigma {row['sigma_fit']:.4e}, norm(r) {norm_r:.4e}")
        s2_small = None
        for h in args.steps:
            vals = []
            for u in dirs:
                plus = np.asarray(res(jnp.asarray(theta + h * u)))
                minus = np.asarray(res(jnp.asarray(theta - h * u)))
                s = plus + minus - 2.0 * r0
                vals.append(float(np.linalg.norm(s) / math.sqrt(6.0)))
                if h == min(args.steps):
                    s2_small = s * s if s2_small is None else s2_small + s * s
            row['noise'][h] = vals
            print(f"    h = {h:.0e}: noise " + '  '.join(f'{x:.3e}' for x in vals)
                  + f"   share of norm(r): {np.mean(vals) / norm_r:.3e}", flush=True)
        row['spread_by_noise'] = spread(s2_small, s2_small)
        row['spread_by_abs_f'] = spread(s2_small, np.abs(f0))
        print(f"    share of the noise on the noisiest 0.1, 1, 10 percent of "
              f"points: " + '  '.join(f'{v:.3f}' for v in row['spread_by_noise'].values()))
        print(f"    share of the noise on the points of largest abs(f), same "
              f"shares: " + '  '.join(f'{v:.3f}' for v in row['spread_by_abs_f'].values()),
              flush=True)
        results[name] = row

    names = list(base)
    print('\n  norm of the difference of r at theta between two configurations:')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = float(np.linalg.norm(base[a] - base[b]))
            print(f"    {a:>8} - {b:<8} {diff:.3e}, "
                  f"{diff / results[b]['norm_r']:.3e} of norm(r)")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'noise_k{args.k}_seed{args.seed}.json'
    path.write_text(json.dumps({'args': vars(args), 'results': results},
                               indent=1, default=str))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
