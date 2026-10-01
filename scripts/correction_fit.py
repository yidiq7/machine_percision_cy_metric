"""Fit corrections of the potential on a saved HN base, in the mixed precision.

The base theta at k0 gives ddbar log p_0 in dd at every point, once. Each level
of --chain fits psi = sigma^* C sigma / p^(k/k0) of degree k with LM on the
fixed potential before it, and the next level takes the sum as its base. f comes
in dd on a hold-out set, and the base alone is reported first as the control.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/correction_fit.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --chain 32
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
import optimistix as optx                                   # noqa: E402

from quartic import analytic, correction, data, dd_metric, sections, whiten   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def load_points(n, seed):
    """n points of the sampler with `seed`, with the dd points of `point_extras`."""
    batch = data.make(math.ceil(n / 4), seed)
    batch = {key: val[:n] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    return batch


EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def report(lay0, parts0, levels, batch, chunk):
    """mean abs(f), rms(f) and max abs(f) for the base and `levels`, with f in dd.

    `regions` splits the loss by bins of min abs(z_a) over max abs(z_a).
    """
    xs = (batch['points_hi'], (batch['points_hi'], batch['points_lo'], batch['patch']))
    v = jax.jit(lambda x: jax.lax.map(
        lambda a: correction.chain_volume(lay0, parts0, levels, a[0], a[1]), x,
        batch_size=chunk))(xs)
    mass = batch['mass'] / jnp.sum(batch['mass'])
    f = analytic.residual_from_volumes(v, mass)
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    w, ff = np.asarray(mass), np.asarray(f)
    loss = float(np.sum(w * ff * ff))
    regions = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        take = (ratio >= low) & (ratio < high) if high < 1.0 else (ratio >= low)
        wt = float(w[take].sum())
        regions.append({'bin': [low, high], 'points': float(take.mean()), 'weight': wt,
                        'loss_share': float(np.sum(w[take] * ff[take] ** 2)) / loss,
                        'mape': float(np.sum(w[take] * np.abs(ff[take]))) / wt,
                        'rms': float(np.sqrt(np.sum(w[take] * ff[take] ** 2) / wt))})
    return {'mape': float(jnp.sum(mass * jnp.abs(f))),
            'rms': float(jnp.sqrt(jnp.sum(mass * f * f))),
            'max': float(jnp.max(jnp.abs(f))), 'regions': regions}


def regions_text(rep):
    lines = []
    for row in rep['regions']:
        lines.append(f"    min/max abs(z) in {row['bin'][0]:.2f} to {row['bin'][1]:.2f}: "
                     f"points {row['points']:.4f}, weight {row['weight']:.4f}, loss "
                     f"{row['loss_share']:.3f}, mean abs(f) {row['mape']:.3e}, rms "
                     f"{row['rms']:.3e}")
    return '\n'.join(lines)


def describe(name, rep):
    return (f"{name}: mean abs(f) {rep['mape']:.4e}, rms {rep['rms']:.3e}, "
            f"max {rep['max']:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True, help='the theta of the base at k0')
    ap.add_argument('--h-base', type=str, default=None,
                    help='a plain theta at k0 whose H is added to the H of --theta, in dd')
    ap.add_argument('--whiten-file', type=str, default=None,
                    help='W and W^-1 of a whitened --theta, as the fit saved them')
    ap.add_argument('--chain', type=int, nargs='+', required=True)
    ap.add_argument('--n-fit', type=int, default=300_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--n-eval', type=int, default=8_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--chunk', type=int, default=1024, help='points in one block of J')
    ap.add_argument('--report-chunk', type=int, default=8192)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--verbose', action='store_true', help='print every LM step')
    ap.add_argument('--fp64-b', action='store_true',
                    help='form b of the tangent in FP64 and not in dd')
    ap.add_argument('--psi-dd', action='store_true',
                    help='form psi with its Gram sums in dd, in the fit and the report')
    ap.add_argument('--j-rows', action='store_true',
                    help='form each row of J from tau and beta in dd')
    ap.add_argument('--init', type=str, default=None,
                    help='saved coefficients of the first level, as the start of LM')
    ap.add_argument('--whiten-level', action='store_true',
                    help='whiten the sections of each level in L2(X), as --whiten does for HN')
    ap.add_argument('--unit', type=float, default=1.0,
                    help='run LM in the scaled coordinates over this unit, so that its '
                         'start at lambda = 1 is lambda = 1/unit^2 for them')
    ap.add_argument('--load', type=str, nargs='+', default=None,
                    help='saved coefficients of the levels of --chain, in place of a fit')
    ap.add_argument('--out', type=str, default='runs_corr')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    sections.PREPARE_DD = True
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    # psi in dd holds the Gram stage of the level for each point, so a report of
    # a level must take the block of the fit: 8192 points at degree 34 need 181 GB.
    level_chunk = min(args.report_chunk, args.chunk) if args.psi_dd else args.report_chunk

    plain = sections.layout(args.k0)
    lay0 = plain if args.whiten_file is None else whiten.load_layout(args.k0,
                                                                     args.whiten_file)
    theta0 = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay0.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the basis of the base')
    if args.h_base is not None:
        side = Path(args.h_base).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != plain.fingerprint():
            raise SystemExit('the basis of --h-base is absent or not the plain basis')
        base = jnp.asarray(np.load(args.h_base))
        sections.H_BASE = jax.jit(lambda t: sections.base_matrices(plain, t))(base)
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    # The base is in parts0 now, and no other H of this run takes it.
    sections.H_BASE = None

    fit_set = load_points(args.n_fit, args.seed)
    hold = load_points(args.n_eval, args.eval_seed)
    say(f'base k0 = {args.k0} from {args.theta}; {args.n_fit} fit points of seed '
        f'{args.seed}, {args.n_eval} hold-out points of seed {args.eval_seed}')
    rows = {'base': report(lay0, parts0, [], hold, args.report_chunk)}
    say(describe('the base on the hold-out', rows['base']))
    print(regions_text(rows['base']), flush=True)
    if args.load is not None:
        if len(args.load) != len(args.chain):
            raise SystemExit('--load needs one file for each level of --chain')
        levels = []
        for level, (k1, path) in enumerate(zip(args.chain, args.load), start=1):
            lay_c = sections.layout(k1, normalize=False)
            side = Path(path).with_suffix('.basis')
            if not side.exists() or side.read_text().strip() != lay_c.fingerprint():
                raise SystemExit(f'the basis of {path} is absent or not the layout at {k1}')
            coeffs = jnp.asarray(np.load(path))
            cmats = sections.prepare_fp(lay_c, coeffs)
            parts_c = sections.prepare_parts(lay_c, coeffs) if args.psi_dd else None
            levels.append((lay_c, cmats, parts_c, k1 / args.k0))
            rep = report(lay0, parts0, levels, hold, level_chunk)
            say(describe(f'level {level} at k = {k1} from {path}, hold-out', rep))
            print(regions_text(rep), flush=True)
            rows[f'level_{level}_k{k1}'] = {'hold': rep, 'loaded': path}
        (out_dir / f'corr_k{args.k0}_loaded.json').write_text(
            json.dumps({'args': vars(args), 'rows': rows}, indent=1))
        say(f'saved to {out_dir}')
        return

    xs = (fit_set['points_hi'],
          (fit_set['points_hi'], fit_set['points_lo'], fit_set['patch']))
    state, hi = jax.jit(lambda x: jax.lax.map(
        lambda a: correction.base_state_dd(lay0, parts0, a[0], a[1]), x,
        batch_size=args.report_chunk))(xs)

    levels = []
    for level, k1 in enumerate(args.chain, start=1):
        lay_c = (whiten.whitened_layout(k1) if args.whiten_level
                 else sections.layout(k1, normalize=False))
        alpha = k1 / args.k0
        res = correction.residual_fn_mixed(lay_c, fit_set, state, hi, args.chunk,
                                           alpha, dd_b=not args.fp64_b,
                                           psi_dd=args.psi_dd, j_rows=args.j_rows)
        zero = jnp.zeros(lay_c.n_theta)
        t1 = time.perf_counter()
        norms = jax.jit(lambda c: jnp.linalg.norm(jax.jacfwd(res)(c), axis=0))(zero)
        scale = 1.0 / norms
        start = zero
        if args.init is not None and level == 1:
            side = Path(args.init).with_suffix('.basis')
            if not side.exists() or side.read_text().strip() != lay_c.fingerprint():
                raise SystemExit('the basis of --init is absent or not the layout of level 1')
            start = jnp.asarray(np.load(args.init)) / scale
        loss0 = float(jnp.sum(res(scale * start) ** 2))
        say(f'level {level} at k = {k1}: n = {lay_c.n_theta}, alpha = {alpha:.6f}, '
            f'column norms {float(norms.min()):.3e} to {float(norms.max()):.3e}, '
            f'loss at 0 {loss0:.4e}, {time.perf_counter() - t1:.0f}s')

        solver = optx.LevenbergMarquardt(rtol=1e-15, atol=1e-32, verbose=args.verbose)
        t1 = time.perf_counter()
        unit = args.unit
        sol = optx.least_squares(lambda q, a: res(scale * unit * q), solver,
                                 start / unit, max_steps=args.max_steps, throw=False)
        coeffs = scale * unit * sol.value
        steps = int(sol.stats['num_steps'])
        loss = float(jnp.sum(res(coeffs) ** 2))
        seconds = time.perf_counter() - t1
        say(f'level {level} at k = {k1}: {steps} steps, {seconds:.0f}s, loss '
            f'{loss0:.4e} to {loss:.4e}')

        stem = out_dir / f'corr_k{args.k0}_level{level}_k{k1}'
        np.save(stem.with_suffix('.npy'), np.asarray(coeffs))
        stem.with_suffix('.basis').write_text(lay_c.fingerprint() + '\n')
        if args.whiten_level:
            # W of a whitened level moves between processes, so it goes with the level.
            whiten.save_whitening(lay_c, str(stem) + '_whiten.npz')
        cmats = sections.prepare_fp(lay_c, coeffs)
        parts_c = sections.prepare_parts(lay_c, coeffs) if args.psi_dd else None
        levels.append((lay_c, cmats, parts_c, alpha))
        rep_fit = report(lay0, parts0, levels, fit_set, level_chunk)
        rep_hold = report(lay0, parts0, levels, hold, level_chunk)
        say(describe(f'level {level}, fit set', rep_fit))
        say(describe(f'level {level}, hold-out', rep_hold))
        print(regions_text(rep_hold), flush=True)
        rows[f'level_{level}_k{k1}'] = {'steps': steps, 'seconds': seconds,
                                        'loss0': loss0, 'loss': loss,
                                        'fit': rep_fit, 'hold': rep_hold}
        (out_dir / f'corr_k{args.k0}.json').write_text(
            json.dumps({'args': vars(args), 'rows': rows}, indent=1))
        state, hi = correction.advance_mixed(lay_c, cmats, fit_set, state, hi, alpha,
                                             level_chunk, parts_c=parts_c)
    say(f'saved to {out_dir}')


if __name__ == '__main__':
    main()
