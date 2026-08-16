"""Fit theta at one or more degrees k, and report on a hold-out set.

The fit set and the hold-out set are two independent samples, because they have
different limits:

  * the fit set is bounded by memory. LM materializes [J; sqrt(lambda) I] of
    shape (N_fit + n_theta, n_theta) in FP64, and lineax needs a few copies.
  * the hold-out set is bounded by nothing much. It costs one forward pass in
    chunks, and the standard error of `mean abs(f)` falls as 1/sqrt(N).

Usage:
    uv run python -u scripts/run_fit.py --k 12 --n-train 100000 --n-eval 8000000
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

from quartic import data, fit, metric, sections             # noqa: E402

# sigma = mean abs(f) from the earlier fits, for comparison. CLAUDE.md holds them.
KNOWN_SIGMA = {2: 1.90e-2, 4: 1.68e-3, 6: 1.60e-4, 8: 1.69e-5,
               10: 1.85e-6, 12: 2.96e-7, 16: 8.5e-9, 20: 6.3e-10}


def sample(n_points, seed, tag):
    """One point set of at least n_points, with its residual of P."""
    t0 = time.perf_counter()
    batch = data.make(math.ceil(n_points / 4), seed)
    res = data.residual_of_P(batch['points'])
    print(f"{tag}: {batch['points'].shape[0]} points, seed {seed}, "
          f"{time.perf_counter() - t0:.1f}s, "
          f"max abs(P)/scale = {res.max():.3e}, median = {np.median(res):.3e}")
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', required=True)
    ap.add_argument('--n-train', type=int, default=100_000)
    ap.add_argument('--n-eval', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=100_000,
                    help='points per chunk when the hold-out set is evaluated')
    ap.add_argument('--fit-chunk', type=int, default=None,
                    help='points per chunk inside the LM Jacobian. The peak '
                         'memory follows this and not --n-train. The default '
                         'picks it from --budget-gb.')
    ap.add_argument('--budget-gb', type=float, default=40.0,
                    help='device memory for one Jacobian; sets --fit-chunk')
    ap.add_argument('--autodiff-jac', action='store_true',
                    help='use nested autodiff for the Jacobian; the default is '
                         'the analytic rule, which is much faster above k = 8')
    ap.add_argument('--cond', action='store_true', help='also measure cond(J)')
    ap.add_argument('--out', type=str, default='runs')
    args = ap.parse_args()

    train = sample(args.n_train, args.seed, 'fit set     ')
    hold = sample(args.n_eval, args.seed + 1000, 'hold-out set')
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    print(f"{'k':>3} {'n_theta':>8} {'steps':>6} {'seconds':>9} "
          f"{'mean abs(f)':>13} {'+/-':>7} {'known':>10} {'ratio':>7} "
          f"{'rms(f)':>11} {'max abs(f)':>11} {'min lambda':>11} {'min p':>10}")

    for k in args.k:
        lay = sections.layout(k)

        analytic_jac = not args.autodiff_jac
        fit_chunk = args.fit_chunk
        if fit_chunk is None:
            fit_chunk, info = fit.suggest_chunk(lay, train, args.budget_gb,
                                                analytic_jac=analytic_jac)
            print(f"    k={k}: fit chunk {fit_chunk}, predicted peak "
                  f"{info['predicted_total_gb']:.1f} GB of {args.budget_gb:.0f} GB "
                  f"({info['gb_per_point_in_chunk'] * 1e3:.2f} MB per point, "
                  f"J is {info['output_gb']:.2f} GB)")

        theta, stats = fit.fit(lay, train, max_steps=args.max_steps,
                               chunk=fit_chunk, analytic_jac=analytic_jac)
        rep = metric.report(lay, theta, hold, chunk=args.chunk)
        known = KNOWN_SIGMA.get(k)
        ratio = f"{rep['mape'] / known:.2f}" if known else ''
        flag = '' if stats['converged'] else (
            ' HIT MAX STEPS' if stats['hit_max_steps'] else f" {stats['message']}")
        print(f"{k:>3} {lay.n_theta:>8} {stats['steps']:>6} {stats['seconds']:>8.1f}s "
              f"{rep['mape']:>13.4e} {rep['mape_stderr'] / rep['mape']:>6.2%} "
              f"{known if known else '':>10} {ratio:>7} "
              f"{rep['rms']:>11.3e} {rep['max_abs']:>11.3e} "
              f"{rep['min_eig']:>11.3e} {rep['min_p']:>10.3e}{flag}")

        record = {'k': k, 'n_theta': lay.n_theta, 'seed': args.seed,
                  'n_train': int(train['points'].shape[0]),
                  'normalization': 'sampled mean of v over the hold-out set',
                  'loss': 'MSPE, LM from optimistix',
                  'max_abs_note': 'a lower bound on sup abs(f); a sample cannot '
                                  'bound a supremum',
                  **stats, **rep}
        if args.cond:
            record['cond'] = fit.condition_number(lay, theta, train,
                                                  chunk=fit_chunk,
                                                  analytic_jac=analytic_jac)
            print(f"    cond(J) = {record['cond']['cond']:.3e}, "
                  f"null sigma = {record['cond']['null_sigma']:.3e}, "
                  f"J shape = {record['cond']['shape']}")

        np.save(out_dir / f"theta_k{k}_seed{args.seed}.npy", np.asarray(theta))
        (out_dir / f"fit_k{k}_seed{args.seed}.json").write_text(
            json.dumps(record, indent=2))

    print(f"\nsaved to {out_dir}/")


if __name__ == '__main__':
    main()
