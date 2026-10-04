"""σ of a level fit on a hold-out that several `vpa_dd_levels.py --eval-only` runs share.

Each run saves v in dd for its slice of the hold-out, with the masses and min/max
abs(z_a). Here every slice joins one sample, so ⟨v⟩ = Σ w v / Σ w comes in dd over
all the points, and f, σ, rms, max and the regions follow for each column.

With --k0 it also takes f = v / mean_v - 1 against the exact mean of `quartic.volume`,
which is f with g and Omega both of volume 1, and it gives the offset of the sampled
mean from the exact one.

Usage:
    uv run python scripts/vpa_levels_combine.py runs_x/vpa_levels_v_k0_28_44_40_36_*.npz \\
        --k0 28
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, dd, volume                    # noqa: E402

EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', type=str, nargs='+')
    ap.add_argument('--k0', type=int, default=None,
                    help='also f against the exact mean of v of the class k0')
    args = ap.parse_args()
    parts = [np.load(p) for p in sorted(args.paths)]
    hi = np.concatenate([p['hi'] for p in parts])
    lo = np.concatenate([p['lo'] for p in parts])
    mass = np.concatenate([p['mass'] for p in parts])
    ratio = np.concatenate([p['ratio'] for p in parts])
    w = mass / mass.sum()
    v = (jnp.asarray(hi), jnp.asarray(lo))
    weights = jnp.asarray(w)[:, None]
    forms = [('sampled mean', np.asarray(analytic.residual_from_volumes(v, weights)))]
    if args.k0 is not None:
        exact = volume.mean_v_dd(args.k0)
        forms.append(('exact mean', np.asarray(
            analytic.residual_from_volumes(v, weights, mean=exact))))
        mean = analytic.sampled_mean(v, weights)
        off = dd.to_fp(dd.add(dd.mul(mean, dd.reciprocal(tuple(jnp.asarray(x) for x in exact))),
                              dd.to_dd(-jnp.ones_like(mean[0]))))
        print(f'exact mean of v for k0 = {args.k0}: {exact[0]!r}; the sampled mean over it '
              f'- 1, for each column: {["%.4e" % x for x in np.asarray(off)]}')
    print(f'{len(w)} hold-out points from {len(parts)} slices')
    for name, f in forms:
        for i in range(f.shape[1]):
            fi = f[:, i]
            loss = float(np.sum(w * fi * fi))
            print(f'column {i}, {name}: sigma {np.sum(w * np.abs(fi)):.4e}, rms '
                  f'{math.sqrt(loss):.3e}, max {np.abs(fi).max():.3e}')
            for a, b in zip(EDGES[:-1], EDGES[1:]):
                take = (ratio >= a) & (ratio < b) if b < 1.0 else (ratio >= a)
                wt = w[take].sum()
                print(f'    min/max abs(z) in {a:.2f} to {b:.2f}: weight {wt:.4f}, mean abs(f) '
                      f'{np.sum(w[take] * np.abs(fi[take])) / wt:.3e}')


if __name__ == '__main__':
    main()
