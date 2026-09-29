"""Where on X the loss of a fit sits, by the size of the smallest coordinate.

For bins of min_a abs(z_a) / max_a abs(z_a), the script reports the share of
the points, the share of the weight, and the share of the loss sum w f^2, with
the whole path in dd. If a small share of the points carries a large share of
the loss, a fit set of N points holds few points for the directions that
reduce it.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/residual_regions.py --k 32 \\
        --theta runs_end_remat/theta_k32_seed20260813.npy --points 400000
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd_metric, metric, sections   # noqa: E402

EDGES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--points', type=int, default=400_000)
    ap.add_argument('--seed', type=int, default=20261813)
    ap.add_argument('--out', type=str, default='runs_regions')
    args = ap.parse_args()

    analytic.GRAM_DD, analytic.H_LOW, analytic.PHI_DD, analytic.END_DD = True, True, True, True
    sections.PREPARE_DD = True
    lay = sections.layout(args.k)
    theta = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis')
    batch = data.make(math.ceil(args.points / 4), args.seed)
    batch = {key: val[:args.points] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))

    v = np.asarray(metric.volumes_fit(lay, theta, batch))
    mass = np.asarray(batch['mass'])
    w = mass / mass.sum()
    f = v / np.sum(w * v) - 1.0
    z = np.abs(np.asarray(batch['points']))
    ratio = z.min(axis=1) / z.max(axis=1)
    loss = w * f * f
    print(f"k = {args.k}, {f.size} points of seed {args.seed}: mean abs(f) "
          f"{np.sum(w * np.abs(f)):.4e}, rms {np.sqrt(loss.sum()):.4e}")
    print(f"  {'min/max coord':>15} {'points':>8} {'weight':>8} {'loss':>8} "
          f"{'mean abs f':>11} {'rms f':>10}")
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (ratio >= lo) & (ratio < hi)
        wt = w[sel].sum()
        row = {'lo': lo, 'hi': min(hi, 1.0), 'points': float(sel.mean()),
               'weight': float(wt), 'loss': float(loss[sel].sum() / loss.sum()),
               'mean_abs_f': float(np.sum(w[sel] * np.abs(f[sel])) / max(wt, 1e-300)),
               'rms_f': float(np.sqrt(np.sum(loss[sel]) / max(wt, 1e-300)))}
        rows.append(row)
        print(f"  {lo:>6.2f} to {min(hi, 1.0):>5.2f} {row['points']:>8.4f} "
              f"{row['weight']:>8.4f} {row['loss']:>8.4f} {row['mean_abs_f']:>11.3e} "
              f"{row['rms_f']:>10.3e}")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'regions_{Path(args.theta).stem}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    print(f"saved to {path}")


if __name__ == '__main__':
    main()
