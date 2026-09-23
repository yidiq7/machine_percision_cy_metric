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
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

from quartic import analytic, data, fit, metric, sections, whiten  # noqa: E402

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
    ap.add_argument('--budget-gb', type=float, default=None,
                    help='device memory for the whole LM solve; sets '
                         '--fit-chunk. The default is 45 percent of the device.')
    ap.add_argument('--autodiff-jac', action='store_true',
                    help='use nested autodiff for the Jacobian; the default is '
                         'the analytic rule, which is much faster above k = 8')
    ap.add_argument('--target-risk', type=float, default=5e-3,
                    help='excess risk n_theta/N_fit to aim for. The planner '
                         'takes only as many fit points as this needs and gives '
                         'the rest of the budget to the block, which buys speed.')
    ap.add_argument('--gram-dd', choices=['off', 'inner', 'on'], default='off',
                    help='precision of the Gram stage. `on` puts the matrix-'
                         'vector products and the inner products in double-'
                         'double, which lowers the floor on abs(f) from 3.97e-13'
                         ' to 1.61e-14 at k = 28 and costs 41 times the time.')
    ap.add_argument('--section-dd', action='store_true',
                    help='stages 1 to 3 in double-double as well: the monomials '
                         'of u, the frame of the sections, and sigma with its 4 '
                         'derivatives. The Gram stage alone leaves a floor on '
                         'abs(f) that grows by 1.28 for each unit of k; with '
                         'these it is approximately 2.5e-15 and flat in k.')
    ap.add_argument('--prepare-dd', action='store_true',
                    help='form the block matrices H_r in double-double and '
                         'round each entry once. The FP64 form moves the '
                         'residual at the stalled theta of k = 28 by 38 percent '
                         'of its norm. H is built once for each LM step, so '
                         'the cost is small.')
    ap.add_argument('--rcond', type=float, default=None,
                    help='rank cutoff for the LM step. None uses QR, which '
                         'fails once cond(J) passes 1/eps near k = 23. Set it '
                         'near a tenth of the sigma you expect, because the '
                         'sigma reached is approximately rcond.')
    ap.add_argument('--theta-init', type=str, default=None,
                    help='start from a saved theta instead of Fubini-Study. '
                         'This is not a test of convergence: LM starts again at '
                         'lambda = 1, and at k = 24 and k = 28 it then stopped '
                         'with no gain where steps that remove most of the '
                         'loss existed. One --k only.')
    ap.add_argument('--whiten', action='store_true',
                    help='whiten the section basis in L2(X). The monomial frame '
                         'is orthonormal in the coefficients and not on X, so '
                         'its Gram condition number reaches 4.5e5 at k = 24 and '
                         'enters twice, because h is quadratic in the sections. '
                         'The change is an exact reparametrization.')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS,
                    help='lines for the Gram sample; each line gives 4 points')
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED,
                    help='seed of the Gram sample. It is part of the convention '
                         'for h, so the fingerprint covers it.')
    ap.add_argument('--require-gpu', action='store_true',
                    help='stop if JAX did not find a GPU. `uv run` without '
                         '--extra gpu removes jax[cuda12] and falls back to the '
                         'CPU without failing.')
    ap.add_argument('--run-to-limit', action='store_true',
                    help='set rtol = atol = 0, run every one of --max-steps, '
                         'and keep the theta with the smallest loss. The '
                         'convergence test of optimistix measures the step in '
                         'theta, so it fires where the step is small and not '
                         'where the residual stops falling. Two exact '
                         'reparametrizations of the same h then stop 6.6 times '
                         'apart in sigma at k = 28.')
    ap.add_argument('--dry-run', action='store_true',
                    help='size the block and report the memory, then stop')
    ap.add_argument('--cond', action='store_true', help='also measure cond(J)')
    ap.add_argument('--out', type=str, default='runs')
    args = ap.parse_args()

    import jax
    dev = jax.devices()[0]
    print(f"device: {dev.platform} {dev.device_kind} x{len(jax.devices())}, "
          f"{fit.device_memory_gb():.1f} GB")
    if args.require_gpu and dev.platform != 'gpu':
        raise SystemExit(
            f"--require-gpu, but JAX is on {dev.platform!r}. `uv run` syncs "
            f"before it runs, and without `--extra gpu` it removes "
            f"jax[cuda12]. Use `uv run --extra gpu ...`, or "
            f"`./scripts/setup.sh` and then `./scripts/run_cluster.sh`.")

    analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[args.gram_dd]
    analytic.SECTION_DD = args.section_dd
    if args.section_dd and not args.gram_dd == 'on':
        raise SystemExit('--section-dd needs --gram-dd on. Stages 1 to 3 in '
                         'extended precision buy nothing while stage 4, which '
                         'carries almost all of the error, stays in FP64.')
    sections.PREPARE_DD = args.prepare_dd
    print(f"Gram stage: {args.gram_dd}, stages 1 to 3 in dd: {args.section_dd}, "
          f"H in dd: {args.prepare_dd}")

    train = sample(args.n_train, args.seed, 'fit set     ')
    hold = sample(args.n_eval, args.seed + 1000, 'hold-out set')
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    print(f"{'k':>3} {'n_theta':>8} {'steps':>6} {'seconds':>9} "
          f"{'mean abs(f)':>13} {'+/-':>7} {'known':>10} {'ratio':>7} "
          f"{'rms(f)':>11} {'max abs(f)':>11} {'min lambda':>11} {'min p':>10}")

    if args.theta_init is not None and len(args.k) != 1:
        ap.error('--theta-init needs exactly one --k')

    for k in args.k:
        if args.whiten:
            t0 = time.perf_counter()
            lay = whiten.whitened_layout(k, n_pairs=args.gram_pairs,
                                         seed=args.gram_seed)
            print(f"    k={k}: whitened basis from {lay.gram_id}, "
                  f"{time.perf_counter() - t0:.1f}s")
        else:
            lay = sections.layout(k)

        # The whitening and the scale are part of the convention, so the
        # fingerprint must cover them and not the symmetry structure alone.
        basis_id = lay.fingerprint()

        theta0 = None
        if args.theta_init is not None:
            theta0 = jnp.asarray(np.load(args.theta_init))
            if theta0.shape != (lay.n_theta,):
                raise SystemExit(f'theta has {theta0.shape}, k={k} needs '
                                 f'({lay.n_theta},)')
            # theta means nothing without the basis it belongs to, so refuse a
            # theta from another basis instead of fitting from a wrong start.
            side = Path(args.theta_init).with_suffix('.basis')
            if side.exists():
                saved = side.read_text().strip()
                if saved != basis_id:
                    raise SystemExit(
                        f'theta was saved against basis {saved} and this build '
                        f'has {basis_id}. The two are different conventions, so '
                        f'the values do not carry over. Refit, or rebuild on the '
                        f'machine that made it.')
            else:
                print(f"    warning: {side.name} is absent, so the basis of "
                      f"{args.theta_init} cannot be checked")
            start = metric.report(lay, theta0, hold, chunk=args.chunk)
            print(f"    restart from {args.theta_init}: "
                  f"mean abs(f) = {start['mape']:.4e}")

        analytic_jac = not args.autodiff_jac
        fit_set, fit_chunk = train, args.fit_chunk
        if fit_chunk is None:
            n_fit, fit_chunk, info = fit.plan_fit(
                lay, train, args.budget_gb, analytic_jac=analytic_jac,
                max_steps=args.max_steps, target_risk=args.target_risk,
                best_so_far=args.run_to_limit)
            fit_set = {key: val[:n_fit] for key, val in train.items()}
            print(f"    k={k}: fit set {n_fit} of {int(train['points'].shape[0])}, "
                  f"block {fit_chunk}, measured peak "
                  f"{info['measured_total_gb']:.1f} GB of {info['budget_gb']:.1f} GB "
                  f"on a {fit.device_memory_gb():.1f} GB device. "
                  f"{info['gb_per_fit_point'] * 1e6:.2f} kB for each fit point, "
                  f"excess risk {info['excess_risk']:.1e}")

        if args.dry_run:
            continue

        tols = {'rtol': 0.0, 'atol': 0.0} if args.run_to_limit else {}
        theta, stats = fit.fit(lay, fit_set, theta0=theta0,
                               max_steps=args.max_steps,
                               chunk=fit_chunk, analytic_jac=analytic_jac,
                               rcond=args.rcond,
                               best_so_far=args.run_to_limit, **tols)
        rep = metric.report(lay, theta, hold, chunk=args.chunk)
        # sigma on the fit set as well. The two together separate a fit that
        # stalls from a fit that lands on a different point of a flat valley:
        # the second one keeps the fit-set value and moves the hold-out value.
        rep_fit = metric.report(lay, theta, fit_set, chunk=args.chunk)
        known = KNOWN_SIGMA.get(k)
        ratio = f"{rep['mape'] / known:.2f}" if known else ''
        # --run-to-limit means to use every step, so the step limit is the
        # intent there and not a warning.
        if args.run_to_limit:
            flag = ' best of the whole budget'
        elif stats['converged']:
            flag = ''
        else:
            flag = (' HIT MAX STEPS' if stats['hit_max_steps']
                    else f" {stats['message']}")
        print(f"{k:>3} {lay.n_theta:>8} {stats['steps']:>6} {stats['seconds']:>8.1f}s "
              f"{rep['mape']:>13.4e} {rep['mape_stderr'] / rep['mape']:>6.2%} "
              f"{known if known else '':>10} {ratio:>7} "
              f"{rep['rms']:>11.3e} {rep['max_abs']:>11.3e} "
              f"{rep['min_eig']:>11.3e} {rep['min_p']:>10.3e}{flag}")
        print(f"    fit-set mean abs(f) = {rep_fit['mape']:.4e}, "
              f"hold-out over fit set = {rep['mape'] / rep_fit['mape']:.3f}")
        print(f"    through complex_hessian in plain FP64: hold-out "
              f"{rep['mape_check']:.4e}, fit set {rep_fit['mape_check']:.4e}")

        record = {'k': k, 'n_theta': lay.n_theta, 'seed': args.seed,
                  'n_train': int(fit_set['points'].shape[0]),
                  'normalization': 'sampled mean of v over the hold-out set',
                  'loss': 'MSPE, LM from optimistix',
                  'gram_dd': args.gram_dd,
                  'section_dd': bool(args.section_dd),
                  'prepare_dd': bool(args.prepare_dd),
                  'sigma_path': 'analytic.volume_at, in the arithmetic of the '
                                'fit; mape_check uses complex_hessian in FP64',
                  'mape_fit_check': rep_fit['mape_check'],
                  'theta_init': args.theta_init,
                  'basis_fingerprint': basis_id,
                  'whiten': bool(args.whiten),
                  'gram_id': lay.gram_id,
                  'max_abs_note': 'a lower bound on sup abs(f); a sample cannot '
                                  'bound a supremum',
                  'mape_fit': rep_fit['mape'],
                  'rms_fit': rep_fit['rms'],
                  **stats, **rep}
        if args.cond:
            record['cond'] = fit.condition_number(lay, theta, fit_set,
                                                  chunk=fit_chunk,
                                                  analytic_jac=analytic_jac)
            print(f"    cond(J) = {record['cond']['cond']:.3e}, "
                  f"null sigma = {record['cond']['null_sigma']:.3e}, "
                  f"J shape = {record['cond']['shape']}")

        np.save(out_dir / f"theta_k{k}_seed{args.seed}.npy", np.asarray(theta))
        (out_dir / f"theta_k{k}_seed{args.seed}.basis").write_text(basis_id + "\n")
        (out_dir / f"fit_k{k}_seed{args.seed}.json").write_text(
            json.dumps(record, indent=2))

    print(f"\nsaved to {out_dir}/")


if __name__ == '__main__':
    main()
