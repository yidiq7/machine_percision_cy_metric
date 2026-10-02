"""The distribution of f of a saved level on its hold-out, and its largest values.

`vpa_dd_fit.py` saves f on the fit set and the hold-out for each column of the
coefficients. This script reports quantiles of f, the share of f > 0, and the
points with the largest abs(f), with their min/max abs(z_a) and weight, and the
part of the weighted mean of v that those points carry.

Usage:
    uv run python scripts/vpa_f_stats.py runs_vpa_dd36_600k/vpa_dd_f_hold_k0_28_k1_36.npy \\
        --n 2000000 --seed 20261813 --column 1
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

from quartic import data                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--column', type=int, default=1)
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    f = np.load(args.path)[:, args.column]
    batch = data.make(math.ceil(args.n / 4), args.seed)
    z = np.asarray(batch['points'])[:args.n]
    mass = np.asarray(batch['mass'])[:args.n]
    w = mass / mass.sum()
    size = np.abs(z)
    ratio = size.min(axis=1) / size.max(axis=1)
    print(f'{len(f)} points; weighted mean of f {np.sum(w * f):.3e}; mean abs(f) '
          f'{np.sum(w * np.abs(f)):.4e}; rms {math.sqrt(np.sum(w * f * f)):.3e}')
    print(f'share of the weight with f > 0: {w[f > 0].sum():.4f}')
    # f = v/<v> - 1 has weighted mean 0 exactly; an offset comes from weights whose
    # FP64 normalization does not sum to 1, and it moves every f by the same amount.
    g = f - np.sum(w * f) / np.sum(w)
    print(f'with the weighted mean removed: mean abs(f) {np.sum(w * np.abs(g)):.4e}, rms '
          f'{math.sqrt(np.sum(w * g * g)):.3e}, max abs(f) {np.abs(g).max():.3e}')
    size_bins = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01)
    for low, high in zip(size_bins[:-1], size_bins[1:]):
        take = (ratio >= low) & (ratio < high)
        print(f'    min/max abs(z) in {low:.2f} to {min(high, 1.0):.2f}: mean abs(f) '
              f'{np.sum(w[take] * np.abs(g[take])) / w[take].sum():.3e}')
    for q in (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
        print(f'    quantile {q}: f = {np.quantile(f, q):.3e}')
    order = np.argsort(np.abs(f))[::-1][:args.top]
    print('largest abs(f):')
    for i in order:
        print(f'    f {f[i]:.3e}, min/max abs(z) {ratio[i]:.2e}, weight x N {w[i] * len(f):.3e}')
    for k in (10, 100, 1000, 10000):
        top = np.argsort(np.abs(f))[::-1][:k]
        print(f'    the {k} largest abs(f) carry {np.sum(w[top] * f[top]):.3e} of the weighted '
              f'sum of f, and the rest {np.sum(w * f) - np.sum(w[top] * f[top]):.3e}')


if __name__ == '__main__':
    main()
