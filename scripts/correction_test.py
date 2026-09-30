"""Test 2 of the potential correction: one level at small k, in plain FP64.

The base is an HN fit at k0. The correction psi = sigma^* C sigma / D runs over
the invariant functions of bidegree (k1, k1), and LM fits it on the fixed base.
Two weights D, as `quartic.correction` defines them:

  * fs: D = abs(z)^(2 k1).
  * base: D = p_0^(k1/k0), which contains the first-order step of HN at k1.

HN fits at k0 and at k1 give the reference. The script also reports the
singular values of J, each column scaled to norm 1. An exact null direction is
removed first: theta itself for HN, and C = H_FS for the fs weight, because
there psi is a constant. The base weight has no exact null direction.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/correction_test.py --k0 12 --k1 16 20
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


def spectrum(jac, null=None):
    """Column norms of J and the singular values of the scaled J, null removed."""
    norms = np.linalg.norm(jac, axis=0)
    out = {}

    def singular(mat, direction):
        if direction is None:
            return np.linalg.svd(mat, compute_uv=False)
        q = direction / np.linalg.norm(direction)
        out.setdefault('null_sigma', float(np.linalg.norm(mat @ q)))
        return np.linalg.svd(mat - np.outer(mat @ q, q), compute_uv=False)[:-1]

    sing = singular(jac / norms, None if null is None else np.asarray(null) * norms)
    raw = singular(jac, None if null is None else np.asarray(null))
    out.update(s_max=float(sing[0]), s_min=float(sing[-1]),
               cond=float(sing[0] / sing[-1]), cond_raw=float(raw[0] / raw[-1]),
               smallest=[float(s) for s in sing[-4:]])
    return norms, out


def describe(name, norms, spec):
    null = (f", null direction {spec['null_sigma']:.2e}" if 'null_sigma' in spec
            else '')
    say(f"{name}: column norms {norms.min():.3e} to {norms.max():.3e}{null}; "
        f"scaled J: s_max {spec['s_max']:.3e}, cond {spec['cond']:.3e} (unscaled "
        f"{spec['cond_raw']:.3e}), 4 smallest "
        + ' '.join(f'{s:.2e}' for s in spec['smallest']))


def sigma_of(v, batch):
    """mean abs(f), rms(f) and max abs(f) from v on a point set."""
    w = batch['mass'] / jnp.sum(batch['mass'])
    f = v / jnp.sum(w * v) - 1.0
    return (float(jnp.sum(w * jnp.abs(f))), float(jnp.sqrt(jnp.sum(w * f * f))),
            float(jnp.max(jnp.abs(f))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, default=12)
    ap.add_argument('--k1', type=int, nargs='+', default=[16])
    ap.add_argument('--weight', nargs='+', default=['fs', 'base'],
                    choices=['fs', 'base'])
    ap.add_argument('--n-fit', type=int, default=20_000)
    ap.add_argument('--n-eval', type=int, default=400_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=300)
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

    mats0 = sections.prepare(lay0, theta0)

    def base(z):
        return correction.base_state(lay0, mats0, z)
    state_fit = correction.states(base, fit_set['points'], args.chunk)
    state_hold = correction.states(base, hold['points'], args.chunk)

    for k1 in args.k1:
        lay1 = sections.layout(k1)
        theta1, st1 = fit.fit(lay1, fit_set, max_steps=args.max_steps, chunk=args.chunk)
        rep1 = metric.report(lay1, theta1, hold, chunk=100_000, block=args.chunk)
        say(f"HN at k1 = {k1}: {st1['steps']} steps, sigma {rep1['mape']:.4e}")
        jac1 = np.asarray(jax.jacfwd(fit.residual_fn(lay1, fit_set, args.chunk))(theta1))
        norms1, spec1 = spectrum(jac1, null=np.asarray(theta1))
        describe(f"HN at k1 = {k1}, J at its fit", norms1, spec1)
        row = {'sigma_hn': rep1['mape'], 'steps_hn': int(st1['steps']), 'spec_hn': spec1}

        lay_c = sections.layout(k1, normalize=False)
        for weight in args.weight:
            fs_weight = weight == 'fs'
            alpha = float(k1) if fs_weight else k1 / args.k0
            res = correction.residual_fn(lay_c, fit_set, state_fit, args.chunk,
                                         alpha, fs_weight)
            zero = jnp.zeros(lay_c.n_theta)
            jac0 = np.asarray(jax.jit(jax.jacfwd(res))(zero))
            null = np.asarray(sections.fubini_study_theta(lay_c)) if fs_weight else None
            norms, spec = spectrum(jac0, null=null)
            describe(f"correction at k1 = {k1}, weight {weight}, J at 0", norms, spec)

            scale = jnp.asarray(1.0 / norms)
            solver = optx.LevenbergMarquardt(rtol=1e-15, atol=1e-32)
            t1 = time.perf_counter()
            sol = optx.least_squares(lambda q, a: res(scale * q), solver, zero,
                                     max_steps=args.max_steps, throw=False)
            coeffs = scale * sol.value
            cmats = sections.prepare_fp(lay_c, coeffs)
            vols = jax.jit(lambda cm, b, st: correction.volumes(
                lay_c, cm, b, st, args.chunk, alpha, fs_weight))
            v_hold = vols(cmats, hold, state_hold)
            v_fit = vols(cmats, fit_set, state_fit)
            mape, rms, top = sigma_of(v_hold, hold)
            fit_mape = sigma_of(v_fit, fit_set)[0]
            steps = int(sol.stats['num_steps'])
            say(f"correction at k1 = {k1}, weight {weight}: {steps} steps, "
                f"{time.perf_counter() - t1:.0f}s, sigma {mape:.4e} on the hold-out, "
                f"{fit_mape:.4e} on the fit set, rms {rms:.3e}, max {top:.3e}; HN at "
                f"k1 gives {rep1['mape']:.4e}")
            row[weight] = {'sigma': mape, 'sigma_fit': fit_mape, 'rms': rms,
                           'max': top, 'steps': steps, 'spec': spec}
        rows[f'k1_{k1}'] = row

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'correction_k0_{args.k0}_weights.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
