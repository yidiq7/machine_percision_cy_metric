"""σ of a level fit on a hold-out that several `vpa_dd_levels.py --eval-only` runs share.

Each run saves v in dd for its slice of the hold-out, with the masses and min/max
abs(z_a). Here every slice joins one sample, so ⟨v⟩ = Σ w v / Σ w comes in dd over
all the points, and f, σ, rms, max and the regions follow for each column.

Usage:
    uv run python scripts/vpa_levels_combine.py runs_x/vpa_levels_v_k0_28_44_40_36_*.npz
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

from quartic import analytic                                # noqa: E402

EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', type=str, nargs='+')
    args = ap.parse_args()
    parts = [np.load(p) for p in sorted(args.paths)]
    hi = np.concatenate([p['hi'] for p in parts])
    lo = np.concatenate([p['lo'] for p in parts])
    mass = np.concatenate([p['mass'] for p in parts])
    ratio = np.concatenate([p['ratio'] for p in parts])
    w = mass / mass.sum()
    f = np.asarray(analytic.residual_from_volumes((jnp.asarray(hi), jnp.asarray(lo)),
                                                  jnp.asarray(w)[:, None]))
    print(f'{len(w)} hold-out points from {len(parts)} slices')
    for i in range(f.shape[1]):
        fi = f[:, i]
        loss = float(np.sum(w * fi * fi))
        print(f'column {i}: sigma {np.sum(w * np.abs(fi)):.4e}, rms {math.sqrt(loss):.3e}, '
              f'max {np.abs(fi).max():.3e}')
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            take = (ratio >= a) & (ratio < b) if b < 1.0 else (ratio >= a)
            wt = w[take].sum()
            print(f'    min/max abs(z) in {a:.2f} to {b:.2f}: weight {wt:.4f}, mean abs(f) '
                  f'{np.sum(w[take] * np.abs(fi[take])) / wt:.3e}')


if __name__ == '__main__':
    main()
