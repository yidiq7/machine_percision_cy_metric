"""The V+A basis against the product basis for one correction level, at small k.

The base is an HN fit at k0 in FP64. For each degree k1 the script builds the
V+A basis of the level on a sample, stores R ddbar G R^dag of every basis
function at the fit and hold-out points, and runs LM on the stored values:
g = R ddbar(log p_0) R^dag + sum_b c_b P_b, v = det(g) / OmegaOmegabar. It
reports the checks of the build, the orthonormality on an independent sample,
cond(J) at c = 0, the size of c and sigma, and the same for the product basis
on the same points.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/arnoldi_test.py --k0 12 --k1 16 20
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

from quartic import analytic, arnoldi, correction, data, fit, metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def stored_hessians(basis, batch, weight_fn, chunk):
    """P_b = R ddbar G_b R^dag at every point of `batch`, as (N, n, 2, 2)."""
    def one(a):
        z, rest = a
        return arnoldi.restricted_hessians(basis, z, rest, weight_fn)
    return jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(
        (batch['points'], batch['restriction']))


def base_metric(lay0, mats0, batch, chunk):
    """g_0 = R ddbar(log p_0) R^dag at every point, as (N, 2, 2)."""
    def one(a):
        z, rest = a
        phi = correction.base_state(lay0, mats0, z)[2]
        return rest @ phi @ jnp.conj(rest).T
    return jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(
        (batch['points'], batch['restriction']))


def residual_from_stored(g0, p, batch):
    """sqrt(w) f as a function of c, from g = g_0 + sum_b c_b P_b."""
    mass = batch['mass'] / jnp.sum(batch['mass'])
    root = jnp.sqrt(mass)
    omega = batch['Omega_Omegabar']

    def res(c):
        g = g0 + jnp.einsum('nbkl,b->nkl', p, c)
        det = jnp.real(g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] * g[:, 1, 0])
        v = det / omega
        return root * (v / jnp.sum(mass * v) - 1.0)
    return res


def sigma_of(r, batch):
    mass = batch['mass'] / jnp.sum(batch['mass'])
    f = r / jnp.sqrt(mass)
    return (float(jnp.sum(mass * jnp.abs(f))), float(jnp.sqrt(jnp.sum(mass * f * f))),
            float(jnp.max(jnp.abs(f))))


def spectrum(jac):
    norms = np.linalg.norm(jac, axis=0)
    raw = np.linalg.svd(jac, compute_uv=False)
    scaled = np.linalg.svd(jac / norms, compute_uv=False)
    return {'cond_raw': float(raw[0] / raw[-1]), 'cond_scaled': float(scaled[0] / scaled[-1]),
            'smallest_scaled': [float(s) for s in scaled[-4:] / scaled[0]],
            'column_norms': [float(norms.min()), float(norms.max())]}


def lm(res, n, max_steps, scale=None):
    scale = jnp.ones(n) if scale is None else scale
    solver = optx.LevenbergMarquardt(rtol=1e-15, atol=1e-32)
    sol = optx.least_squares(lambda q, a: res(scale * q), solver, jnp.zeros(n),
                             max_steps=max_steps, throw=False)
    return scale * sol.value, int(sol.stats['num_steps'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, default=12)
    ap.add_argument('--k1', type=int, nargs='+', default=[16])
    ap.add_argument('--n-fit', type=int, default=20_000)
    ap.add_argument('--n-eval', type=int, default=400_000)
    ap.add_argument('--n-build', type=int, default=5_000)
    ap.add_argument('--n-check', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--out', type=str, default='runs_arnoldi')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.SECTION_DD = False, False
    sections.PREPARE_DD = False

    def sample(n, seed):
        batch = data.make(math.ceil(n / 4), seed)
        return {key: val[:n] for key, val in batch.items()}
    fit_set = sample(args.n_fit, args.seed)
    hold = sample(args.n_eval, args.seed + 1000)
    build = sample(args.n_build, args.seed + 2000)
    check = sample(args.n_check, args.seed + 3000)

    lay0 = sections.layout(args.k0)
    theta0, st0 = fit.fit(lay0, fit_set, max_steps=args.max_steps, chunk=args.chunk)
    rep0 = metric.report(lay0, theta0, hold, chunk=100_000, block=args.chunk)
    say(f"HN at k0 = {args.k0}: {st0['steps']} steps, sigma {rep0['mape']:.4e}")
    mats0 = sections.prepare(lay0, theta0)
    g0_fit = base_metric(lay0, mats0, fit_set, args.chunk)
    g0_hold = base_metric(lay0, mats0, hold, args.chunk)
    rows = {'k0': args.k0, 'sigma_k0': rep0['mape']}

    for k1 in args.k1:
        alpha = k1 / args.k0
        lay_c = sections.layout(k1, normalize=False)

        def weight_fn(z):
            return jnp.exp(-alpha * sections.log_potential_at(lay0, mats0, z))
        w_build = jax.vmap(weight_fn)(build['points'])
        t1 = time.perf_counter()
        basis, report = arnoldi.build_basis(lay_c.structure, np.asarray(build['points']),
                                            np.asarray(w_build), np.asarray(build['mass']),
                                            alpha)
        dims_ok = all(r['dim_q'] == r['dim_q_expected'] for r in report['orbits'])
        gap = max(r['s_first_dropped'] / r['s_last_kept'] for r in report['orbits'])
        say(f"k1 = {k1}: V+A basis of n = {lay_c.n_theta} in {time.perf_counter() - t1:.0f}s; "
            f"full q spaces {dims_ok}, worst rank gap {gap:.1e}, cond across orbits "
            f"{report['cond_across_orbits']:.3e}")

        vals = arnoldi.values(basis, check['points'], jax.vmap(weight_fn)(check['points']))
        mu = check['mass'] / jnp.sum(check['mass'])
        gram = np.asarray(vals.T @ (mu[:, None] * vals))
        eig = np.linalg.eigvalsh(gram)
        say(f"k1 = {k1}: Gram of the basis on {args.n_check} independent points: "
            f"eigenvalues {eig[0]:.4f} to {eig[-1]:.4f}")

        t1 = time.perf_counter()
        p_fit = stored_hessians(basis, fit_set, weight_fn, args.chunk)
        p_hold = stored_hessians(basis, hold, weight_fn, 4 * args.chunk)
        say(f"k1 = {k1}: stored P at {args.n_fit} + {args.n_eval} points in "
            f"{time.perf_counter() - t1:.0f}s")

        res = residual_from_stored(g0_fit, p_fit, fit_set)
        n = lay_c.n_theta
        jac = np.asarray(jax.jit(jax.jacfwd(res))(jnp.zeros(n)))
        spec = spectrum(jac)
        t1 = time.perf_counter()
        coeffs, steps = lm(jax.jit(res), n, args.max_steps)
        mape, rms, top = sigma_of(residual_from_stored(g0_hold, p_hold, hold)(coeffs), hold)
        fit_mape = sigma_of(res(coeffs), fit_set)[0]
        say(f"k1 = {k1}, V+A: cond(J) at 0 {spec['cond_raw']:.3e} (columns scaled "
            f"{spec['cond_scaled']:.3e}), {steps} steps, {time.perf_counter() - t1:.0f}s, "
            f"sigma {mape:.4e} on the hold-out, {fit_mape:.4e} on the fit set, rms "
            f"{rms:.3e}, max {top:.3e}, norm of c {float(jnp.linalg.norm(coeffs)):.3e}")
        row = {'report': report, 'gram_eig': [float(eig[0]), float(eig[-1])],
               'vpa': {'spec': spec, 'steps': steps, 'sigma': mape, 'sigma_fit': fit_mape,
                       'rms': rms, 'max': top, 'norm_c': float(jnp.linalg.norm(coeffs))}}

        # The product basis on the same points, with each column of J scaled to 1.
        state_fit = correction.states(lambda z: correction.base_state(lay0, mats0, z),
                                      fit_set['points'], args.chunk)
        state_hold = correction.states(lambda z: correction.base_state(lay0, mats0, z),
                                       hold['points'], args.chunk)
        res_p = correction.residual_fn(lay_c, fit_set, state_fit, args.chunk, alpha, False)
        jac_p = np.asarray(jax.jit(jax.jacfwd(res_p))(jnp.zeros(n)))
        spec_p = spectrum(jac_p)
        t1 = time.perf_counter()
        coeffs_p, steps_p = lm(res_p, n, args.max_steps,
                               scale=jnp.asarray(1.0 / np.linalg.norm(jac_p, axis=0)))
        cmats = sections.prepare_fp(lay_c, coeffs_p)
        vols = jax.jit(lambda cm, b, st: correction.volumes(lay_c, cm, b, st, args.chunk,
                                                            alpha, False))
        v_hold = vols(cmats, hold, state_hold)
        w_hold = hold['mass'] / jnp.sum(hold['mass'])
        f_hold = v_hold / jnp.sum(w_hold * v_hold) - 1.0
        mape_p = float(jnp.sum(w_hold * jnp.abs(f_hold)))
        say(f"k1 = {k1}, product: cond(J) at 0 {spec_p['cond_raw']:.3e} (columns scaled "
            f"{spec_p['cond_scaled']:.3e}), {steps_p} steps, {time.perf_counter() - t1:.0f}s, "
            f"sigma {mape_p:.4e} on the hold-out, norm of c "
            f"{float(jnp.linalg.norm(coeffs_p)):.3e}")
        row['product'] = {'spec': spec_p, 'steps': steps_p, 'sigma': mape_p,
                          'norm_c': float(jnp.linalg.norm(coeffs_p))}
        rows[f'k1_{k1}'] = row

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'arnoldi_k0_{args.k0}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
