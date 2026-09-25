"""sigma and the margins at a saved theta, with the whole path in dd.

The report of run_fit.py uses the arithmetic of the fit that made theta. This
script reports any saved theta in one arithmetic, END_DD, on the hold-out set
and, if asked, on a fit set, so that fits from different arithmetics compare.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/report_theta.py --k 32 \\
        --theta runs_full/theta_k32_seed20260813.npy --n-fit 63335
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

from quartic import analytic, data, dd_metric, metric, sections, whiten   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--h-base', type=str, default=None)
    ap.add_argument('--whiten', action='store_true')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    ap.add_argument('--n-eval', type=int, default=8_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--n-fit', type=int, default=0,
                    help='also report on the first n-fit points of the fit sample')
    ap.add_argument('--n-train', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, default=100_000)
    ap.add_argument('--out', type=str, default='runs_report')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.PHI_DD, analytic.END_DD = True, True, True, True
    sections.PREPARE_DD = True

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

    sets = [('hold-out', args.n_eval, args.eval_seed)]
    if args.n_fit:
        sets.append(('fit set', args.n_fit, args.seed))
    rows = {}
    for tag, n, seed in sets:
        size = n if tag == 'hold-out' else args.n_train
        batch = data.make(math.ceil(size / 4), seed)
        batch = {key: val[:n] for key, val in batch.items()}
        batch.update(dd_metric.point_extras(batch['points']))
        rep = metric.report(lay, theta, batch, chunk=args.chunk)
        rows[tag] = rep
        say(f"{tag}, {n} points of seed {seed}: mean abs(f) {rep['mape']:.4e}, "
            f"rms {rep['rms']:.3e}, max {rep['max_abs']:.3e}, FP64 check "
            f"{rep['mape_check']:.4e}, min lambda {rep['min_eig']:.3e}, "
            f"min p {rep['min_p']:.3e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'report_{Path(args.theta).stem}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
