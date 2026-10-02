"""One correction level on an HN base in the V+A basis, by one linear solve.

The base theta at k0 gives ddbar log p_0 and v_0 in dd at every point. The V+A
basis of the level comes from `arnoldi.build_joint` on a build sample, with the
weight w = p_0^(-k1/k0) in FP64. To first order in the coefficients c,
v = v_0 + L c with L_b = v_0 tr(g_0^-1 P_b) and P_b = R ddbar G_b R^dag. The term
of second order is det(sum c_b P_b) / OmegaOmegabar, of the size (delta g / g)^2,
so one least-squares solve of sqrt(mu) (f_0 + J c) gives c. The columns of J are
scaled to 1, and the SVD of its R factor gives the spectrum, the share of the
loss in each decade, and the solution for each cutoff. Each solution is then
evaluated with the dd base and psi in FP64, on the fit set and the hold-out.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/vpa_fit.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 36
Without --theta, an HN fit at k0 in FP64 on the fit set is the base.
"""
import argparse
import dataclasses
import importlib.util
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
from quartic import (analytic, arnoldi, correction, data, dd, dd_metric, fit,  # noqa: E402
                     metric, sections, vpa_dd)

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def hn_counts(k):
    """n_theta(j) for j = 1 to k, from the character theory of the notes."""
    spec = importlib.util.spec_from_file_location(
        'counts', ROOT / 'notes' / 'symmetry_param_counts_s4.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {j: mod.n_theta(j)[0] for j in range(1, k + 1)}


class Layouts(dict):
    """The plain layout of each degree, built when the build first asks for it."""

    def __missing__(self, j):
        self[j] = sections.layout(j, normalize=False)
        return self[j]


def save_basis(path, basis):
    """The arrays of a joint basis, so that another process can use the same one."""
    np.savez(path, k=basis.k, h=basis.h, parents=basis.parents, mults=basis.mults,
             degrees=basis.degrees, seed_slot=basis.seed_slot,
             seeds=np.array(basis.seeds, dtype=np.int64).reshape(-1, 2),
             entry_d=np.array([d for d, _, _ in basis.entries]),
             entry_a=np.array([a for _, a, _ in basis.entries]).reshape(-1, 4),
             entry_b=np.array([b for _, _, b in basis.entries]).reshape(-1, 4))


def point_inputs(batch):
    return (batch['points_hi'], batch['restriction'],
            (batch['points_hi'], batch['points_lo'], batch['patch']))


def fit_rows(lay0, parts0, basis, weight_fn, batch, chunk):
    """v_0 as a dd pair and the rows L of dv/dc at every point, as (N,) and (N, n)."""
    def one(a, h):
        z, rest, extra = a
        lev = dataclasses.replace(basis, h=h)
        state, hi = correction.base_state_dd(lay0, parts0, z, extra)
        z_dd, m, free = analytic._dd_point(extra)
        v0 = analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True)
        g0 = rest @ state[2] @ jnp.conj(rest).T
        p = arnoldi.joint_hessians(lev, z, rest, weight_fn)                  # (n, 2, 2)
        rows = jnp.real(jnp.einsum('kl,blk->b', jnp.linalg.inv(g0), p))
        return v0, dd.to_fp(v0) * rows
    return jax.jit(lambda x, h: jax.lax.map(lambda a: one(a, h), x, batch_size=chunk))(
        point_inputs(batch), jnp.asarray(basis.h))


def residuals(lay0, parts0, basis, weight_fn, coeffs, batch, chunk, alpha, n_low=0):
    """f in dd, rounded once, at every point for each column of coeffs, as (N, n_c).

    The first `n_low` functions take their Hessians from `vpa_dd` in dd, and the
    others from FP64 autodiff.
    """
    def one(a, cc, h):
        z, rest, extra = a
        lev = dataclasses.replace(basis, h=h)
        _, hi = correction.base_state_dd(lay0, parts0, z, extra)
        z_dd, m, free = analytic._dd_point(extra)
        p = arnoldi.joint_hessians(lev, z, rest, weight_fn)                  # (n, 2, 2)
        if n_low:
            rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
            scale = weight_fn(z) * jnp.sum(jnp.abs(z) ** 2) ** basis.k
            low = vpa_dd.low_hessians(lev, n_low, z_dd, rest_dd, hi, alpha, scale)

        def volume(c):
            delta = jnp.einsum('bkl,b->kl', p[n_low:], c[n_low:])
            if n_low:
                part = tuple(dd.dot_dd_fp(comp, c[:n_low, None, None], axis=0)
                             for comp in low)
                delta = dd.cadd(part, dd.cto_dd(delta))
            return analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True,
                                                delta=delta)
        return jax.vmap(volume)(cc.T)
    v = jax.jit(lambda x, cc, h: jax.lax.map(lambda a: one(a, cc, h), x,
                                             batch_size=chunk))(
        point_inputs(batch), jnp.asarray(coeffs), jnp.asarray(basis.h))
    mass = batch['mass'] / jnp.sum(batch['mass'])
    return np.asarray(analytic.residual_from_volumes(v, mass[:, None]))


def product_values(lay_p, parts_p, weight_fn, batch, chunk):
    """psi = w n at every point, with n = sigma^* C sigma from its Gram sums in dd."""
    def one(a):
        z, extra = a
        z_dd, _, _ = analytic._dd_point(extra)
        sig, dsig = analytic.section_vectors_dd(lay_p, None, z_dd=z_dd)
        n_dd = analytic._gram_parts(lay_p, parts_p, sig, dsig)[4][0]
        return weight_fn(z) * dd.to_fp(n_dd)
    xs = (batch['points_hi'], (batch['points_hi'], batch['points_lo'], batch['patch']))
    return jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(xs)


def summary(f, batch):
    """mean abs(f), rms, max and the regions of `correction_fit.report` for one f."""
    mass = np.asarray(batch['mass'] / jnp.sum(batch['mass']))
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    loss = float(np.sum(mass * f * f))
    regions = []
    for low, high in zip(cf.EDGES[:-1], cf.EDGES[1:]):
        take = (ratio >= low) & (ratio < high) if high < 1.0 else (ratio >= low)
        wt = float(mass[take].sum())
        regions.append({'bin': [low, high], 'points': float(take.mean()), 'weight': wt,
                        'loss_share': float(np.sum(mass[take] * f[take] ** 2)) / loss,
                        'mape': float(np.sum(mass[take] * np.abs(f[take]))) / wt,
                        'rms': float(np.sqrt(np.sum(mass[take] * f[take] ** 2) / wt))})
    return {'mape': float(np.sum(mass * np.abs(f))), 'rms': math.sqrt(loss),
            'max': float(np.max(np.abs(f))), 'loss': loss, 'regions': regions}


def solve(jac, r0, cutoffs):
    """The spectrum of the column-scaled J, and -J^+ r0 for each cutoff of s / s_max."""
    norms = jnp.linalg.norm(jac, axis=0)
    q, r = jnp.linalg.qr(jac / norms)
    u, s, vt = jnp.linalg.svd(r)
    beta = u.T @ (q.T @ r0)
    loss0 = float(r0 @ r0)
    rel = np.asarray(s / s[0])
    b2 = np.asarray(beta ** 2)
    shares = {}
    for e in range(0, 24):
        take = (rel < 10.0 ** -e) & (rel >= 10.0 ** -(e + 1))
        if take.any():
            shares[f'1e-{e} to 1e-{e + 1}'] = [int(take.sum()), float(b2[take].sum() / loss0)]
    low = np.abs(np.asarray(vt[-1]))
    spec = {'cond_scaled': float(rel[0] / rel[-1]), 'smallest': [float(x) for x in rel[-4:]],
            'column_norms': [float(norms.min()), float(norms.max())],
            'loss0': loss0, 'outside': float((loss0 - b2.sum()) / loss0), 'shares': shares,
            'smallest_vector_top': [[int(i), float(low[i])] for i in np.argsort(low)[::-1][:6]],
            'smallest_step': float(beta[-1] / s[-1])}
    sols, predicted = [], []
    for tau in cutoffs:
        keep = jnp.asarray(rel >= tau)
        y = -vt.T @ jnp.where(keep, beta / s, 0.0)
        sols.append(np.asarray(y / norms))
        predicted.append(loss0 - float(np.sum(np.where(np.asarray(keep), b2, 0.0))))

    def step(res, tau):
        """-J^+ res with the cutoff tau, for a refinement from the residual res."""
        b = u.T @ (q.T @ res)
        return np.asarray(-(vt.T @ jnp.where(jnp.asarray(rel >= tau), b / s, 0.0)) / norms)
    return spec, np.stack(sols, axis=1), predicted, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None, help='the theta of the base at k0')
    ap.add_argument('--k1', type=int, nargs='+', required=True)
    ap.add_argument('--n-fit', type=int, default=300_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--n-eval', type=int, default=8_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--n-build', type=int, default=20_000)
    ap.add_argument('--build-seed', type=int, default=20262001)
    ap.add_argument('--n-check', type=int, default=20_000)
    ap.add_argument('--d-max', type=int, default=8, help='highest degree of a multiplier')
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[1e-6, 1e-9, 1e-12, 0.0],
                    help='the smallest s / s_max that each solution keeps')
    ap.add_argument('--max-steps', type=int, default=300,
                    help='LM steps of the HN fit at k0, without --theta')
    ap.add_argument('--map', type=str, nargs='+', default=None,
                    help='saved product-basis coefficients of each level of --k1, to put '
                         'into the V+A coordinates by least squares on psi')
    ap.add_argument('--dd-degree', type=int, default=0,
                    help='the functions of degree up to this, and before the first seed, '
                         'take their Hessians in dd in the evaluation')
    ap.add_argument('--refine', type=int, default=0,
                    help='steps of -J^+ r from the residual in dd after the solve')
    ap.add_argument('--out', type=str, default='runs_vpa')
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
    # The weight of the level needs FP64 block matrices; these are formed in dd and rounded.
    mats0 = sections.prepare(lay0, theta0)
    counts = hn_counts(max(args.k1))
    for j in range(1, min(10, max(args.k1)) + 1):
        assert counts[j] == sections.layout(j, normalize=False).n_theta, j
    layouts = Layouts()
    rows = {'k0': args.k0, 'theta': args.theta}

    for k1 in args.k1:
        alpha = k1 / args.k0

        def weight_fn(z, alpha=alpha):
            return jnp.exp(-alpha * sections.log_potential_at(lay0, mats0, z))
        weights = jax.jit(jax.vmap(weight_fn))

        t1 = time.perf_counter()
        r2 = jnp.sum(jnp.abs(build['points']) ** 2, axis=1)
        basis, trace = arnoldi.build_joint(
            k1, np.asarray(build['points']), weights(build['points']) * r2 ** k1,
            build['mass'] / jnp.sum(build['mass']), d_max=args.d_max,
            n_by_degree=counts, layouts=layouts, verbose=True)
        n = int(basis.h.shape[0])
        say(f"k1 = {k1}: joint V+A basis in {time.perf_counter() - t1:.0f}s: dimension {n} "
            f"of n = {counts[k1]}, {len(basis.entries)} multipliers of degree <= "
            f"{args.d_max}, {len(basis.seeds)} seeds")
        for row in trace:
            if row['seeded'] or row['worst_product_ratio'] < 1e-3 or row['dim'] < row['want']:
                print(f"    degree {row['degree']}: {row['dim']} of {row['want']}, "
                      f"{row['from_products']} from products (worst ratio "
                      f"{row['worst_product_ratio']:.1e}), {row['seeded']} seeds (worst "
                      f"ratio {row['worst_seed_ratio']:.1e})", flush=True)
        save_basis(out_dir / f'vpa_basis_k0_{args.k0}_k1_{k1}.npz', basis)
        row = {'n': n, 'n_theta': counts[k1], 'trace': trace}
        if n < counts[k1]:
            rows[f'k1_{k1}'] = row
            say(f'k1 = {k1}: the basis is short, so the level stops here')
            continue

        vals = jax.jit(lambda z, w: arnoldi.joint_values(basis, z, w))(
            check['points'], weights(check['points']))
        mu = check['mass'] / jnp.sum(check['mass'])
        eig = np.linalg.eigvalsh(np.asarray(vals.T @ (mu[:, None] * vals)))
        row['gram_eig'] = [float(eig[0]), float(eig[-1])]
        say(f'k1 = {k1}: Gram of the basis on {args.n_check} independent points: '
            f'eigenvalues {eig[0]:.4f} to {eig[-1]:.4f}')

        t1 = time.perf_counter()
        v0, lrows = fit_rows(lay0, parts0, basis, weight_fn, fit_set, args.chunk)
        mass = fit_set['mass'] / jnp.sum(fit_set['mass'])
        f0 = analytic.residual_from_volumes(v0, mass)
        m0 = float(dd.to_fp(dd.dot_dd_fp(v0, mass, axis=0)))
        root = jnp.sqrt(mass)
        jac = root[:, None] * (lrows - (1.0 + f0)[:, None] * (mass @ lrows)[None]) / m0
        r0 = root * f0
        say(f'k1 = {k1}: rows of J at {args.n_fit} points in {time.perf_counter() - t1:.0f}s')

        mapped = None
        if args.map is not None:
            path = args.map[args.k1.index(k1)]
            lay_p = sections.layout(k1, normalize=False)
            side = Path(path).with_suffix('.basis')
            if not side.exists() or side.read_text().strip() != lay_p.fingerprint():
                raise SystemExit(f'the basis of {path} is absent or not the layout at {k1}')
            parts_p = sections.prepare_parts(lay_p, jnp.asarray(np.load(path)))
            psi = product_values(lay_p, parts_p, weight_fn, fit_set, args.chunk)
            vals = jax.jit(lambda z, w: arnoldi.joint_values(basis, z, w))(
                fit_set['points_hi'], weights(fit_set['points_hi']))
            q, r = jnp.linalg.qr(root[:, None] * vals)
            mapped = np.asarray(jax.scipy.linalg.solve_triangular(r, q.T @ (root * psi)))
            miss = float(jnp.linalg.norm(root * (vals @ mapped - psi))
                         / jnp.linalg.norm(root * psi))
            pred = float(jnp.sum((r0 + jac @ mapped) ** 2))
            top = np.argsort(np.abs(mapped))[::-1][:6]
            row['mapped'] = {'path': path, 'miss': miss, 'norm_c': float(np.linalg.norm(mapped)),
                             'top': [[int(i), float(mapped[i])] for i in top],
                             'predicted_loss': pred}
            say(f"k1 = {k1}: the product level {path} in V+A coordinates: L2 norm of psi "
                f"{float(jnp.linalg.norm(root * psi)):.3e}, relative miss {miss:.1e}, norm "
                f"of c {np.linalg.norm(mapped):.3e}, largest components "
                f"{row['mapped']['top']}, linear-model loss {pred:.4e}")
            del vals, q, r

        t1 = time.perf_counter()
        spec, sols, predicted, step = solve(jac, r0, args.cutoffs)
        del jac, lrows
        row['spectrum'] = spec
        say(f"k1 = {k1}: solve in {time.perf_counter() - t1:.0f}s; columns scaled, cond "
            f"{spec['cond_scaled']:.3e}, the 4 smallest s / s_max "
            f"{['%.2e' % x for x in spec['smallest']]}, column norms "
            f"{spec['column_norms'][0]:.3e} to {spec['column_norms'][1]:.3e}")
        print(f"    loss at c = 0: {spec['loss0']:.4e}; share outside the range of J "
              f"{spec['outside']:.3e}", flush=True)
        for key, (count, share) in spec['shares'].items():
            print(f"    s / s_max in {key}: {count} values, share of the loss {share:.3e}",
                  flush=True)
        print(f"    the singular vector of the smallest s: largest components "
              f"{spec['smallest_vector_top']}, step along it {spec['smallest_step']:.3e}",
              flush=True)

        n_low = 0
        if args.dd_degree > 0:
            seeded = np.nonzero(np.asarray(basis.seed_slot) >= 0)[0]
            n_low = min(counts[args.dd_degree], int(seeded[0]) if len(seeded) else n)
            say(f'k1 = {k1}: the first {n_low} functions take their Hessians in dd')
        row['n_low'] = n_low
        coeffs = np.concatenate([np.zeros((n, 1)), sols], axis=1)
        labels = ['base'] + [f'cutoff {tau:.0e}' for tau in args.cutoffs]
        predicted = list(predicted)
        if mapped is not None:
            coeffs = np.concatenate([coeffs, mapped[:, None]], axis=1)
            labels.append('mapped product level')
            predicted.append(row['mapped']['predicted_loss'])
        t1 = time.perf_counter()
        f_fit = residuals(lay0, parts0, basis, weight_fn, coeffs, fit_set, args.chunk,
                          alpha, n_low)
        say(f'k1 = {k1}: the fit set in dd in {time.perf_counter() - t1:.0f}s')
        row['solutions'] = []
        for i, label in enumerate(labels):
            got = summary(f_fit[:, i], fit_set)
            pred = spec['loss0'] if i == 0 else predicted[i - 1]
            print(f"    {label}: norm of c {np.linalg.norm(coeffs[:, i]):.3e}; fit set "
                  f"sigma {got['mape']:.4e}, loss {got['loss']:.4e} against the linear "
                  f"model {pred:.4e}", flush=True)
            row['solutions'].append({'label': label, 'norm_c': float(np.linalg.norm(
                coeffs[:, i])), 'fit': got, 'predicted_loss': pred})
        # Refinement: one more step of -J^+ r from the residual r in dd, at each cutoff.
        for it in range(1, args.refine + 1):
            t1 = time.perf_counter()
            res = root[:, None] * jnp.asarray(f_fit)
            for i, tau in enumerate(args.cutoffs):
                coeffs[:, 1 + i] += step(res[:, 1 + i], tau)
            f_fit = residuals(lay0, parts0, basis, weight_fn, coeffs, fit_set, args.chunk,
                              alpha, n_low)
            say(f'k1 = {k1}: refinement {it} in {time.perf_counter() - t1:.0f}s')
            for i, tau in enumerate(args.cutoffs):
                got = summary(f_fit[:, 1 + i], fit_set)
                row['solutions'][1 + i][f'fit_refined_{it}'] = got
                print(f"    cutoff {tau:.0e}: norm of c {np.linalg.norm(coeffs[:, 1 + i]):.3e}; "
                      f"fit set sigma {got['mape']:.4e}, loss {got['loss']:.4e}", flush=True)
        del step
        np.save(out_dir / f'vpa_coeffs_k0_{args.k0}_k1_{k1}.npy', coeffs)

        t1 = time.perf_counter()
        f_hold = residuals(lay0, parts0, basis, weight_fn, coeffs, hold, args.chunk,
                           alpha, n_low)
        say(f'k1 = {k1}: the hold-out in dd in {time.perf_counter() - t1:.0f}s')
        for i, label in enumerate(labels):
            got = summary(f_hold[:, i], hold)
            row['solutions'][i]['hold'] = got
            say(f"k1 = {k1}, {label}: hold-out sigma {got['mape']:.4e}, rms "
                f"{got['rms']:.3e}, max {got['max']:.3e}")
            print(cf.regions_text(got), flush=True)
        rows[f'k1_{k1}'] = row
        (out_dir / f'vpa_k0_{args.k0}.json').write_text(
            json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {out_dir}')


if __name__ == '__main__':
    main()
