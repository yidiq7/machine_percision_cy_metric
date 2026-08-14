"""Fit theta at one or more degrees k, and report on a hold-out set.

Usage:
    .venv/bin/python scripts/run_fit.py --k 2 4 6 --n-pairs 4000 --seed 20260813
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quartic                                              # noqa: E402,F401
import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

from quartic import data, fit, metric, sections             # noqa: E402

# sigma = mean abs(f) from the earlier fits, for comparison. CLAUDE.md holds them.
KNOWN_SIGMA = {2: 1.90e-2, 4: 1.68e-3, 6: 1.60e-4, 8: 1.69e-5,
               10: 1.85e-6, 12: 2.96e-7, 16: 8.5e-9, 20: 6.3e-10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', required=True)
    ap.add_argument('--n-pairs', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--train-frac', type=float, default=0.5)
    ap.add_argument('--n-train', type=int, default=None,
                    help='absolute fit-set size; it overrides --train-frac')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--cond', action='store_true', help='also measure cond(J)')
    ap.add_argument('--out', type=str, default='runs')
    args = ap.parse_args()

    t0 = time.perf_counter()
    batch = data.make(args.n_pairs, args.seed)
    n_total = int(batch['points'].shape[0])
    n_train = args.n_train or int(args.train_frac * n_total)
    train, hold = data.split(batch, n_train, args.seed + 1)
    res_p = data.residual_of_P(batch['points'])
    print(f"points: {n_total} total, {n_train} fit, {n_total - n_train} hold-out, "
          f"{time.perf_counter() - t0:.1f}s")
    print(f"max abs(P)/scale = {res_p.max():.3e}, median = {np.median(res_p):.3e}, "
          f"seed = {args.seed}\n")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    print(f"{'k':>3} {'n_theta':>8} {'steps':>6} {'seconds':>9} "
          f"{'mean abs(f)':>13} {'known':>10} {'ratio':>7} "
          f"{'rms(f)':>11} {'max abs(f)':>11} {'min lambda':>11} {'min p':>10}")

    for k in args.k:
        lay = sections.layout(k)
        theta, stats = fit.fit(lay, train, max_steps=args.max_steps)
        rep = metric.report(lay, theta, hold)
        known = KNOWN_SIGMA.get(k)
        ratio = f"{rep['mape'] / known:.2f}" if known else ''
        flag = '' if stats['converged'] else (
            ' HIT MAX STEPS' if stats['hit_max_steps'] else f" {stats['message']}")
        print(f"{k:>3} {lay.n_theta:>8} {stats['steps']:>6} {stats['seconds']:>8.1f}s "
              f"{rep['mape']:>13.4e} {known if known else '':>10} {ratio:>7} "
              f"{rep['rms']:>11.3e} {rep['max_abs']:>11.3e} "
              f"{rep['min_eig']:>11.3e} {rep['min_p']:>10.3e}{flag}")

        record = {'k': k, 'n_theta': lay.n_theta, 'seed': args.seed,
                  'n_pairs': args.n_pairs, 'n_train': n_train,
                  'normalization': 'sampled mean of v over the hold-out set',
                  'loss': 'MSPE, LM from optimistix', **stats, **rep}
        if args.cond:
            record['cond'] = fit.condition_number(lay, theta, train)
            print(f"    cond(J) = {record['cond']['cond']:.3e}, "
                  f"null sigma = {record['cond']['null_sigma']:.3e}, "
                  f"J shape = {record['cond']['shape']}")

        np.save(out_dir / f"theta_k{k}_seed{args.seed}.npy", np.asarray(theta))
        (out_dir / f"fit_k{k}_seed{args.seed}.json").write_text(
            json.dumps(record, indent=2))

    print(f"\nsaved to {out_dir}/")


if __name__ == '__main__':
    main()
