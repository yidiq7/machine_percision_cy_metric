"""norm(f)_L2, norm(grad f)_L2 and their ratio R at a saved theta, in dd.

`quartic/gradient_dd.py` takes grad f by central differences along X, with the
whole path in double-double. Two step sizes bound the error: the truncation
error falls as eps^2 and the rounding error grows as 1/eps.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/gradient_dd_report.py --k 32 \\
        --theta runs_end/theta_k32_seed20260813.npy --points 20000
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

from quartic import analytic, data, dd_metric, gradient_dd, sections, whiten   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--points', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=20261813,
                    help='the sample; the default is the hold-out seed of the fits')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--eps', type=float, nargs='+', default=[2.0 ** -12, 2.0 ** -14],
                    help='step sizes along X; powers of 2 keep the step exact')
    ap.add_argument('--h-base', type=str, default=None,
                    help='the base theta of a fit with --h-base; --theta is then '
                         'the correction')
    ap.add_argument('--whiten', action='store_true',
                    help='the correction is in the whitened basis, as in run_fit.py')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    ap.add_argument('--out', type=str, default='runs_gradient')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.END_DD = True, True

    plain = sections.layout(args.k)
    lay = (whiten.whitened_layout(args.k, n_pairs=args.gram_pairs, seed=args.gram_seed)
           if args.whiten else plain)
    theta = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the basis of this build')
    if args.h_base is not None:
        side = Path(args.h_base).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != plain.fingerprint():
            raise SystemExit('the basis of --h-base is absent or not the plain basis')
        base = jnp.asarray(np.load(args.h_base))
        sections.H_BASE = jax.jit(lambda t: sections.base_matrices(plain, t))(base)
    batch = data.make(math.ceil(args.points / 4), args.seed)
    batch = {key: val[:args.points] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    say(f'k = {args.k}, {args.points} points of seed {args.seed}, at {args.theta}')

    reports = []
    for eps in args.eps:
        rep = gradient_dd.gradient_report_dd(lay, theta, batch, eps=eps, chunk=args.chunk)
        rep.pop('grad_sq')
        reports.append(rep)
        say(f"eps = 2^{math.log2(eps):.0f}: mean abs(f) {rep['mean_abs_f']:.4e}, "
            f"norm(f)_L2 {rep['l2_f']:.4e}, norm(grad f)_L2 {rep['l2_grad_f']:.4e}, "
            f"R = {rep['ratio']:.4f}")
    if len(reports) > 1:
        spread = max(abs(r['l2_grad_f'] / reports[-1]['l2_grad_f'] - 1) for r in reports)
        print(f"  norm(grad f)_L2 moves by {spread:.2e} across the step sizes")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'gradient_dd_k{args.k}.json'
    path.write_text(json.dumps({'args': vars(args), 'reports': reports}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
