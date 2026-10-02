"""One correction level on an HN base in the V+A basis built and evaluated in dd.

As `vpa_fit.py`, with `arnoldi_dd` in place of the FP64 recurrence: the build of
the basis, the Hessians of every function and the sum of the level run in dd, and
the solve of sqrt(mu) (f_0 + J c) runs in FP64 from the rows of J rounded once.
With --null-degree, psi_N, the function of the degree <= j part closest to 1,
takes the place of the function closest to it, so its coefficient carries the
almost constant direction alone.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/vpa_dd_fit.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 36 --null-degree 9
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

import correction_fit as cf                                 # noqa: E402
import vpa_fit as vf                                        # noqa: E402
from quartic import (analytic, arnoldi, arnoldi_dd, correction, data, dd,  # noqa: E402
                     dd_metric, fit, metric, sections)

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def point_state(lay0, parts0, basis, h_lo, weight_fn, alpha, a):
    """The base in dd at one point and the dd Hessians of every function there."""
    z, rest, extra = a
    state, hi = correction.base_state_dd(lay0, parts0, z, extra)
    z_dd, m, free = analytic._dd_point(extra)
    rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
    scale = weight_fn(z) * jnp.sum(jnp.abs(z) ** 2) ** basis.k
    p = arnoldi_dd.joint_hessians_dd(basis, h_lo, z_dd, rest_dd, hi, alpha, scale)
    return state, hi, (z_dd, m, free), p


def with_null(p, null):
    """The Hessians with row b0 replaced by psi_N = sum over b < n_low of gamma_b G_b."""
    if null is None:
        return p, None
    n_low, gamma, b0 = null
    g = jnp.asarray(gamma)[:, None, None]
    p_n = tuple(dd.dot_dd_fp(jax.tree_util.tree_map(lambda x: x[:n_low], comp), g, axis=0)
                for comp in p)
    return jax.tree_util.tree_map(lambda x, y: x.at[b0].set(y), p, p_n), p_n


def fit_rows(lay0, parts0, basis, h_lo, weight_fn, alpha, batch, chunk, null):
    """v_0 as a dd pair and the rows L of dv/dc rounded to FP64, as (N,) and (N, n)."""
    def one(a, h, hl):
        lev = dataclasses.replace(basis, h=h)
        state, hi, (z_dd, m, free), p = point_state(lay0, parts0, lev, hl, weight_fn,
                                                    alpha, a)
        p, _ = with_null(p, null)
        v0 = analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True)
        rest = a[1]
        g0 = rest @ state[2] @ jnp.conj(rest).T
        rows = jnp.real(jnp.einsum('kl,blk->b', jnp.linalg.inv(g0), dd.cto_fp(p)))
        return v0, dd.to_fp(v0) * rows
    return jax.jit(lambda x, h, hl: jax.lax.map(lambda a: one(a, h, hl), x,
                                                batch_size=chunk))(
        vf.point_inputs(batch), jnp.asarray(basis.h), jnp.asarray(h_lo))


def residuals(lay0, parts0, basis, h_lo, weight_fn, alpha, coeffs, batch, chunk, null):
    """f in dd, rounded once, at every point for each column of coeffs, as (N, n_c)."""
    def one(a, cc, h, hl):
        lev = dataclasses.replace(basis, h=h)
        _, hi, (z_dd, m, free), p = point_state(lay0, parts0, lev, hl, weight_fn,
                                                alpha, a)
        p, _ = with_null(p, null)

        def volume(c):
            delta = tuple(dd.dot_dd_fp(comp, c[:, None, None], axis=0) for comp in p)
            return analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True,
                                                delta=delta)
        return jax.vmap(volume)(cc.T)
    v = jax.jit(lambda x, cc, h, hl: jax.lax.map(lambda a: one(a, cc, h, hl), x,
                                                 batch_size=chunk))(
        vf.point_inputs(batch), jnp.asarray(coeffs), jnp.asarray(basis.h), jnp.asarray(h_lo))
    mass = batch['mass'] / jnp.sum(batch['mass'])
    return np.asarray(analytic.residual_from_volumes(v, mass[:, None]))


def dd_values(basis, h_lo, weight, z, chunk):
    """W q of every function at FP64 points z, from the dd jets, rounded once."""
    rest = dd.cto_dd(jnp.zeros((2, 4), jnp.complex128).at[0, 1].set(1.0).at[1, 2].set(1.0))

    def one(a, h, hl):
        p, w = a
        lev = dataclasses.replace(basis, h=h)
        jets = arnoldi_dd.joint_jets_dd(
            lev, hl, arnoldi_dd.vpa_dd.basis_multiplier_jets(lev, dd.cto_dd(p), rest))
        return w * jnp.sum(jnp.abs(p) ** 2) ** basis.k * dd.to_fp(jets[0])
    return jax.jit(lambda x, h, hl: jax.lax.map(lambda a: one(a, h, hl), x,
                                                batch_size=chunk))(
        (z, weight), jnp.asarray(basis.h), jnp.asarray(h_lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None)
    ap.add_argument('--k1', type=int, nargs='+', required=True)
    ap.add_argument('--n-fit', type=int, default=300_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--n-eval', type=int, default=8_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--n-build', type=int, default=20_000)
    ap.add_argument('--build-seed', type=int, default=20262001)
    ap.add_argument('--n-check', type=int, default=20_000)
    ap.add_argument('--d-max', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[1e-9, 1e-12, 0.0])
    ap.add_argument('--null-degree', type=int, default=0,
                    help='psi_N from the functions of degree <= this replaces the one '
                         'closest to it; 0 keeps the plain basis')
    ap.add_argument('--refine', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--out', type=str, default='runs_vpa_dd')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    def sample(n, seed):
        batch = data.make(math.ceil(n / 4), seed)
        return {key: val[:n] for key, val in batch.items()}
    fit_set = cf.load_points(args.n_fit, args.seed)
    hold = cf.load_points(args.n_eval, args.eval_seed)
    build = sample(args.n_build, args.build_seed)
    check = sample(args.n_check, args.build_seed + 1000)
    say(f'{args.n_fit} fit points of seed {args.seed}, {args.n_eval} hold-out points of '
        f'seed {args.eval_seed}, {args.n_build} build points of seed {args.build_seed}')

    lay0 = sections.layout(args.k0)
    if args.theta is not None:
        theta0 = jnp.asarray(np.load(args.theta))
        side = Path(args.theta).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != lay0.fingerprint():
            raise SystemExit('the basis of --theta is absent or not the plain basis at k0')
    else:
        analytic.GRAM_DD, analytic.SECTION_DD, analytic.END_DD = False, False, False
        sections.PREPARE_DD = False
        theta0, st0 = fit.fit(lay0, fit_set, max_steps=args.max_steps, chunk=args.chunk)
        rep0 = metric.report(lay0, theta0, hold, chunk=100_000, block=args.chunk)
        say(f"HN at k0 = {args.k0} in FP64: {st0['steps']} steps, sigma {rep0['mape']:.4e}")
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    counts = vf.hn_counts(max(args.k1))
    rows = {'k0': args.k0, 'theta': args.theta}

    for k1 in args.k1:
        alpha = k1 / args.k0

        def weight_fn(z, alpha=alpha):
            return jnp.exp(-alpha * sections.log_potential_at(lay0, mats0, z))
        weights = jax.jit(jax.vmap(weight_fn))

        t1 = time.perf_counter()
        r2 = jnp.sum(jnp.abs(build['points']) ** 2, axis=1)
        basis, h_lo, trace = arnoldi_dd.build_joint_dd(
            k1, np.asarray(build['points']), weights(build['points']) * r2 ** k1,
            build['mass'] / jnp.sum(build['mass']), d_max=args.d_max, n_by_degree=counts,
            verbose=True)
        n = int(basis.h.shape[0])
        say(f'k1 = {k1}: dd V+A basis in {time.perf_counter() - t1:.0f}s: dimension {n} of '
            f'{counts[k1]}, {len(basis.entries)} multipliers of degree <= {args.d_max}')
        arnoldi.save_joint(out_dir / f'vpa_dd_basis_k0_{args.k0}_k1_{k1}.npz', basis)
        np.save(out_dir / f'vpa_dd_hlo_k0_{args.k0}_k1_{k1}.npy', h_lo)
        row = {'n': n, 'trace': trace}

        t1 = time.perf_counter()
        vals = dd_values(basis, h_lo, weights(check['points']), check['points'], args.chunk)
        mu = check['mass'] / jnp.sum(check['mass'])
        eig = np.linalg.eigvalsh(np.asarray(vals.T @ (mu[:, None] * vals)))
        row['gram_eig'] = [float(eig[0]), float(eig[-1])]
        say(f'k1 = {k1}: Gram of the dd basis on {args.n_check} independent points in '
            f'{time.perf_counter() - t1:.0f}s: eigenvalues {eig[0]:.4f} to {eig[-1]:.4f}')

        null = None
        if args.null_degree > 0:
            n_low = counts[args.null_degree]
            low = vals[:, :n_low] * jnp.sqrt(mu)[:, None]
            gamma = np.asarray(jnp.linalg.lstsq(low, jnp.sqrt(mu))[0])
            b0 = int(np.argmax(np.abs(gamma)))
            gap = float(jnp.linalg.norm(low @ gamma - jnp.sqrt(mu)))
            null = (n_low, gamma, b0)
            row['null'] = {'n_low': n_low, 'b0': b0, 'rms_gap': gap}
            say(f'k1 = {k1}: psi_N from the first {n_low} functions replaces function {b0}; '
                f'rms of psi_N - 1 on the check points {gap:.3e}')
        del vals

        t1 = time.perf_counter()
        v0, lrows = fit_rows(lay0, parts0, basis, h_lo, weight_fn, alpha, fit_set,
                             args.chunk, null)
        mass = fit_set['mass'] / jnp.sum(fit_set['mass'])
        f0 = analytic.residual_from_volumes(v0, mass)
        m0 = float(dd.to_fp(dd.dot_dd_fp(v0, mass, axis=0)))
        root = jnp.sqrt(mass)
        jac = root[:, None] * (lrows - (1.0 + f0)[:, None] * (mass @ lrows)[None]) / m0
        r0 = root * f0
        say(f'k1 = {k1}: rows of J at {args.n_fit} points in {time.perf_counter() - t1:.0f}s')
        spec, sols, predicted, step = vf.solve(jac, r0, args.cutoffs)
        del jac, lrows
        row['spectrum'] = spec
        say(f"k1 = {k1}: columns scaled, cond {spec['cond_scaled']:.3e}, the 4 smallest "
            f"s / s_max {['%.2e' % x for x in spec['smallest']]}; loss at c = 0 "
            f"{spec['loss0']:.4e}, share outside the range of J {spec['outside']:.3e}")
        for key, (count, share) in spec['shares'].items():
            print(f"    s / s_max in {key}: {count} values, share of the loss {share:.3e}",
                  flush=True)

        coeffs = np.concatenate([np.zeros((n, 1)), sols], axis=1)
        labels = ['base'] + [f'cutoff {tau:.0e}' for tau in args.cutoffs]
        t1 = time.perf_counter()
        f_fit = residuals(lay0, parts0, basis, h_lo, weight_fn, alpha, coeffs, fit_set,
                          args.chunk, null)
        say(f'k1 = {k1}: the fit set in dd in {time.perf_counter() - t1:.0f}s')
        row['solutions'] = []
        for i, label in enumerate(labels):
            got = vf.summary(f_fit[:, i], fit_set)
            pred = spec['loss0'] if i == 0 else predicted[i - 1]
            print(f"    {label}: norm of c {np.linalg.norm(coeffs[:, i]):.3e}; fit set sigma "
                  f"{got['mape']:.4e}, loss {got['loss']:.4e} against the linear model "
                  f"{pred:.4e}", flush=True)
            row['solutions'].append({'label': label, 'fit': got, 'predicted_loss': pred,
                                     'norm_c': float(np.linalg.norm(coeffs[:, i]))})
        for it in range(1, args.refine + 1):
            res = root[:, None] * jnp.asarray(f_fit)
            for i, tau in enumerate(args.cutoffs):
                coeffs[:, 1 + i] += step(res[:, 1 + i], tau)
            f_fit = residuals(lay0, parts0, basis, h_lo, weight_fn, alpha, coeffs, fit_set,
                              args.chunk, null)
            for i, tau in enumerate(args.cutoffs):
                got = vf.summary(f_fit[:, 1 + i], fit_set)
                row['solutions'][1 + i][f'fit_refined_{it}'] = got
                print(f"    refinement {it}, cutoff {tau:.0e}: fit set sigma {got['mape']:.4e}, "
                      f"loss {got['loss']:.4e}", flush=True)
        del step
        np.save(out_dir / f'vpa_dd_coeffs_k0_{args.k0}_k1_{k1}.npy', coeffs)

        t1 = time.perf_counter()
        f_hold = residuals(lay0, parts0, basis, h_lo, weight_fn, alpha, coeffs, hold,
                           args.chunk, null)
        say(f'k1 = {k1}: the hold-out in dd in {time.perf_counter() - t1:.0f}s')
        for i, label in enumerate(labels):
            got = vf.summary(f_hold[:, i], hold)
            row['solutions'][i]['hold'] = got
            say(f"k1 = {k1}, {label}: hold-out sigma {got['mape']:.4e}, rms {got['rms']:.3e}, "
                f"max {got['max']:.3e}")
            print(cf.regions_text(got), flush=True)
        rows[f'k1_{k1}'] = row
        (out_dir / f'vpa_dd_k0_{args.k0}.json').write_text(
            json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {out_dir}')


if __name__ == '__main__':
    main()
