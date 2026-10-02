"""The FP64 error of the joint V+A recurrence for several builds, by degree.

The base is an HN fit at k0 in FP64. For each build setting (the highest degree
of a multiplier and the size of the build sample) the script builds the joint
basis of a level at k1 and repeats the recurrence in mpmath at a few points, as
`vpa_oracle.py` does, to find a build whose error does not grow with the degree.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/vpa_stability.py --k0 12 --k1 24 \\
        --settings 8:10000 4:10000 8:40000
"""
import argparse
import math
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

import vpa_oracle as vo                                     # noqa: E402
from vpa_fit import hn_counts, Layouts                      # noqa: E402
from quartic import analytic, arnoldi, data, fit, sections  # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def errors(basis, z_all, pick):
    """The worst absolute error of q for each degree over the points `pick`."""
    lays = dict(basis.layouts)
    degrees = np.asarray(basis.degrees)
    worst = {}
    for i in pick:
        z = jnp.asarray(z_all[i])
        y_fp = arnoldi.multiplier_values(basis, z)
        seeds_fp = None
        if basis.seeds:
            seeds_fp = jnp.concatenate([arnoldi.seed_values(lays[j], z, [a])
                                        for j, a in basis.seeds], axis=-1)
        q_fp = np.asarray(arnoldi.run_recurrence(basis, y_fp, seeds_fp))
        zm = [mp.mpc(complex(c).real, complex(c).imag) for c in z_all[i]]
        y_mp = vo.mp_multipliers(basis.entries, zm)
        if basis.shift is not None:
            y_mp = [(y - mp.mpf(a)) / mp.mpf(b)
                    for y, a, b in zip(y_mp, basis.shift, basis.scale)]
        r2 = sum(abs(c) ** 2 for c in zm)
        seeds_mp = []
        for j, a in basis.seeds:
            mats = sections.prepare_fp(lays[j], jnp.zeros(lays[j].n_theta).at[a].set(1.0))
            seeds_mp.append(vo.mp_potential(lays[j], mats, zm) / r2 ** j)
        q_mp = np.array([float(x) for x in vo.mp_recurrence(basis, y_mp, seeds_mp)])
        err = np.abs(q_fp - q_mp)
        for j in np.unique(degrees):
            worst[int(j)] = max(worst.get(int(j), 0.0), float(err[degrees == j].max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, default=12)
    ap.add_argument('--k1', type=int, default=24)
    ap.add_argument('--settings', type=str, nargs='+', default=['8:10000'],
                    help='d_max:n_build for each build, then :flat for q orthonormal in mu '
                         'alone, and :center for centered and scaled multipliers')
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--points', type=int, default=2)
    ap.add_argument('--dps', type=int, default=40)
    args = ap.parse_args()
    mp.mp.dps = args.dps
    analytic.GRAM_DD, analytic.SECTION_DD, analytic.END_DD = False, False, False
    sections.PREPARE_DD = False

    def sample(n, seed):
        batch = data.make(math.ceil(n / 4), seed)
        return {key: val[:n] for key, val in batch.items()}
    fit_set = sample(20_000, args.seed)
    lay0 = sections.layout(args.k0)
    theta0, st0 = fit.fit(lay0, fit_set, max_steps=300, chunk=512)
    mats0 = sections.prepare(lay0, theta0)
    say(f'HN at k0 = {args.k0}: {st0["steps"]} steps')
    alpha = args.k1 / args.k0

    def weight_fn(z):
        return jnp.exp(-alpha * sections.log_potential_at(lay0, mats0, z))
    weights = jax.jit(jax.vmap(weight_fn))
    counts = hn_counts(args.k1)
    layouts = Layouts()
    test = np.asarray(data.make(1000, 20263001)['points'])
    size = np.abs(test)
    ratio = size.min(axis=1) / size.max(axis=1)
    pick = list(range(args.points)) + list(np.argsort(ratio)[:args.points])
    table = {}
    for setting in args.settings:
        parts = setting.split(':')
        d_max, n_build = int(parts[0]), int(parts[1])
        flat = 'flat' in parts[2:]
        center = 'center' in parts[2:]
        build = sample(n_build, args.seed + 2000)
        r2 = jnp.sum(jnp.abs(build['points']) ** 2, axis=1)
        scale = (jnp.ones(n_build) if flat
                 else weights(build['points']) * r2 ** args.k1)
        t1 = time.perf_counter()
        basis, trace = arnoldi.build_joint(
            args.k1, np.asarray(build['points']), scale,
            build['mass'] / jnp.sum(build['mass']), d_max=d_max, n_by_degree=counts,
            layouts=layouts, center=center)
        used = sorted({int(basis.entries[m][0]) for m in basis.mults if m >= 0})
        say(f'd_max {d_max}, {n_build} build points: dimension {basis.h.shape[0]} of '
            f'{counts[args.k1]} in {time.perf_counter() - t1:.0f}s, {len(basis.seeds)} '
            f'seeds, degrees of the multipliers used {used}')
        mults = np.asarray(basis.mults)
        step = np.array([basis.entries[m][0] if m >= 0 else 0 for m in mults])
        ratios = [r['worst_product_ratio'] for r in trace]
        print(f'    mean degree of a step {step[step > 0].mean():.2f}; worst pivot ratio '
              f'{min(ratios):.1e}', flush=True)
        table[setting] = errors(basis, test, pick)
    degrees = sorted(set().union(*[set(t) for t in table.values()]))
    print('worst absolute error of q for each degree:')
    print('    degree  ' + '  '.join(f'{s:>12}' for s in args.settings))
    for j in degrees:
        print(f'    {j:6d}  ' + '  '.join(f'{table[s].get(j, float("nan")):12.2e}'
                                       for s in args.settings))


if __name__ == '__main__':
    main()
