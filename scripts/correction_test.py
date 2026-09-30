"""Test 2 of the potential correction: one level at small k, in plain FP64.

The base is an HN fit at k0. The correction psi runs over the invariant
functions of bidegree (k1, k1), and LM fits it on the fixed base. HN fits at k0
and at k1 give the reference. If the correction reaches the sigma of HN at k1
or lower in a few steps, one level works as a Newton step on the Monge-Ampere
residual.

Each column of J at psi = 0 is scaled to norm 1 before the fit, so the fit is
free of the trivial scale of the raw functions, which spans many orders.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/correction_test.py --k0 12 --k1 16
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

from quartic import analytic, correction, data, fit, metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def sigma_corrected(lay0, theta0, lay_c, coeffs, batch, chunk):
    """mean abs(f), rms(f) and max abs(f) of log p_0 + psi on a point set."""
    mats0 = sections.prepare(lay0, theta0)
    cmats = sections.prepare_fp(lay_c, coeffs)
    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    v = jax.jit(lambda x: jax.lax.map(
        lambda a: correction.volume_corrected(lay0, mats0, lay_c, cmats, *a),
        x, batch_size=chunk))(xs)
    w = batch['mass'] / jnp.sum(batch['mass'])
    f = v / jnp.sum(w * v) - 1.0
    return (float(jnp.sum(w * jnp.abs(f))), float(jnp.sqrt(jnp.sum(w * f * f))),
            float(jnp.max(jnp.abs(f))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, default=12)
    ap.add_argument('--k1', type=int, nargs='+', default=[16])
    ap.add_argument('--n-fit', type=int, default=20_000)
    ap.add_argument('--n-eval', type=int, default=400_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--out', type=str, default='runs_correction')
    args = ap.parse_args()

    analytic.GRAM_DD, analytic.SECTION_DD = False, False
    sections.PREPARE_DD = False
    fit_set = data.make(math.ceil(args.n_fit / 4), args.seed)
    fit_set = {key: val[:args.n_fit] for key, val in fit_set.items()}
    hold = data.make(math.ceil(args.n_eval / 4), args.seed + 1000)
    hold = {key: val[:args.n_eval] for key, val in hold.items()}

    lay0 = sections.layout(args.k0)
    theta0, st0 = fit.fit(lay0, fit_set, max_steps=args.max_steps, chunk=args.chunk)
    rep0 = metric.report(lay0, theta0, hold, chunk=100_000, block=args.chunk)
    say(f"HN at k0 = {args.k0}: {st0['steps']} steps, sigma {rep0['mape']:.4e}")
    rows = {'k0': args.k0, 'sigma_k0': rep0['mape']}

    for k1 in args.k1:
        lay1 = sections.layout(k1)
        theta1, st1 = fit.fit(lay1, fit_set, max_steps=args.max_steps, chunk=args.chunk)
        rep1 = metric.report(lay1, theta1, hold, chunk=100_000, block=args.chunk)
        say(f"HN at k1 = {k1}: {st1['steps']} steps, sigma {rep1['mape']:.4e}")

        lay_c = sections.layout(k1, normalize=False)
        res = correction.residual_fn(lay0, theta0, lay_c, fit_set, args.chunk)
        zero = jnp.zeros(lay_c.n_theta)
        jac0 = np.asarray(jax.jit(jax.jacfwd(res))(zero))
        norms = np.linalg.norm(jac0, axis=0)
        sing = np.linalg.svd(jac0 / norms, compute_uv=False)
        scale = jnp.asarray(1.0 / norms)
        say(f"correction at k1 = {k1}: n = {lay_c.n_theta}, column norms of J from "
            f"{norms.min():.3e} to {norms.max():.3e}, cond of the scaled J "
            f"{sing[0] / sing[-1]:.3e}")

        solver = optx.LevenbergMarquardt(rtol=1e-15, atol=1e-32)
        t1 = time.perf_counter()
        sol = optx.least_squares(lambda q, a: res(scale * q), solver, zero,
                                 max_steps=args.max_steps, throw=False)
        coeffs = scale * sol.value
        mape, rms, top = sigma_corrected(lay0, theta0, lay_c, coeffs, hold, args.chunk)
        fit_mape = sigma_corrected(lay0, theta0, lay_c, coeffs, fit_set, args.chunk)[0]
        steps = int(sol.stats['num_steps'])
        say(f"correction at k1 = {k1}: {steps} steps, {time.perf_counter() - t1:.0f}s, "
            f"sigma {mape:.4e} on the hold-out, {fit_mape:.4e} on the fit set, rms "
            f"{rms:.3e}, max {top:.3e}; HN at k1 gives {rep1['mape']:.4e}")
        rows[f'k1_{k1}'] = {'sigma_hn': rep1['mape'], 'sigma_corrected': mape,
                            'sigma_corrected_fit': fit_mape, 'rms': rms, 'max': top,
                            'steps': steps, 'cond_scaled_j': float(sing[0] / sing[-1])}

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'correction_k0_{args.k0}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
