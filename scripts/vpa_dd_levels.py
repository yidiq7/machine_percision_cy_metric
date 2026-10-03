"""Several correction levels on one HN base, fitted together in one linear solve.

Each level is a V+A basis of degree k1 with the weight p0^(-k1/k0), built and
evaluated in dd as in `vpa_dd_fit.py`. v is linear in the coefficients of all the
levels to (dg/g)^2, so the rows of J of the levels side by side give the best
joint fit, and each subset of the levels in --solve gets its own solve from the
same rows. One program then evaluates f in dd for every solution.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/vpa_dd_levels.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 36 40 \\
        --load runs_vpa_dd36_final --solve 36 40 36,40 --null-degree 9
"""
import argparse
import dataclasses
import functools
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
import vpa_dd_fit as vdf                                    # noqa: E402
import vpa_fit as vf                                        # noqa: E402
from quartic import (analytic, arnoldi, arnoldi_dd, correction, data, dd,  # noqa: E402
                     dd_metric, fit, joint_dd, metric, sections, vpa_dd)

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


@dataclasses.dataclass
class Level:
    k0: int
    k1: int
    alpha: float
    basis: arnoldi.JointBasis
    h_lo: np.ndarray
    uinv: tuple
    null: tuple            # (n_low, gamma, b0), or None
    weight_fn: object      # w = p0^(-alpha) at an FP64 point, for the build and psi_N


def truncated(basis, h_lo, uinv, n):
    """The first n functions of a basis, which must end at the end of a degree."""
    degrees = np.asarray(basis.degrees)
    if n < len(degrees) and degrees[n] == degrees[n - 1]:
        raise ValueError(f'the first {n} functions end inside degree {degrees[n]}')
    blocks = len(set(degrees[:n].tolist()) - {0})
    lev = dataclasses.replace(basis, h=basis.h[:n, :n], parents=basis.parents[:n],
                              mults=basis.mults[:n], degrees=basis.degrees[:n],
                              seed_slot=basis.seed_slot[:n])
    return lev, h_lo[:n, :n], uinv[:blocks]


def point_hessians(lay0, parts0, levels, hs, a, joint=None):
    """The base in dd at one point and the dd Hessians of every column of the fit.

    The weight W of each level comes in dd, so that the sums of two levels keep
    their digits where they cancel. Without `joint` the columns are the functions
    of each level with its psi_N in place; with it they are the joint functions,
    with psi_N, the projection of 1 onto all of them, in place of joint function
    `joint[1]`.
    """
    z, rest, extra = a
    state, hi, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
    z_dd, m, free = analytic._dd_point(extra)
    rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
    r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
    ps = []
    for lev, (h, hl) in zip(levels, hs):
        basis = dataclasses.replace(lev.basis, h=h)
        scale = vpa_dd.weight_dd(p_dd, r2, lev.k0, lev.k1)
        ps.append(arnoldi_dd.joint_hessians_dd(basis, hl, lev.uinv, z_dd, rest_dd, hi,
                                               lev.alpha, scale))
    if joint is None:
        cols = [vdf.with_null(p, lev.null)[0] for p, lev in zip(ps, levels)]
        cols = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *cols)
    else:
        jm, b0 = joint
        cols = joint_dd.transform(jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs),
                                                         *ps), jm)
        g_hi, g_lo = (jnp.asarray(x)[:, None, None] for x in jm.gamma)
        p_n = tuple(dd.add(dd.dot_dd_fp(comp, g_hi, axis=0), dd.dot_dd_fp(comp, g_lo, axis=0))
                    for comp in cols)
        cols = jax.tree_util.tree_map(lambda x, y: x.at[b0].set(y), cols, p_n)
    return state, hi, (z_dd, m, free), cols


def level_values_dd(lay0, parts0, lev, batch, chunk):
    """W q of every function of a level in dd at the dd points of `batch`, W in dd, (N, n).

    The joint map needs the values to dd accuracy: an FP64 W puts noise of 1e-16 on
    each function, and that noise then reads as a new part of 1e-16.
    """
    def one(a, h, hl):
        z, _, extra = a
        _, _, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
        z_dd, m, free = analytic._dd_point(extra)
        rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
        r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
        basis = dataclasses.replace(lev.basis, h=h)
        jets = arnoldi_dd.joint_jets_dd(
            basis, hl, vpa_dd.basis_multiplier_jets(basis, z_dd, rest_dd), lev.uinv)
        return dd.mul(jets[0], vpa_dd.weight_dd(p_dd, r2, lev.k0, lev.k1))
    return jax.jit(lambda x, h, hl: jax.lax.map(lambda a: one(a, h, hl), x,
                                                batch_size=chunk))(
        vf.point_inputs(batch), jnp.asarray(lev.basis.h), jnp.asarray(lev.h_lo))


def level_arrays(levels, joint=None):
    """The large arrays of the levels and of the joint map, as arguments of a program."""
    hs = tuple((jnp.asarray(lev.basis.h), jnp.asarray(lev.h_lo)) for lev in levels)
    if joint is None:
        return hs, None
    jm = joint[0]
    return hs, (jnp.asarray(jm.h), jnp.asarray(jm.h_lo),
                tuple(tuple(jnp.asarray(x) for x in inv) for inv in jm.uinv),
                tuple(jnp.asarray(x) for x in jm.gamma))


def _with_arrays(joint, ja):
    if joint is None:
        return None
    return (dataclasses.replace(joint[0], h=ja[0], h_lo=ja[1], uinv=ja[2], gamma=ja[3]),
            joint[1])


def fit_rows(lay0, parts0, levels, batch, chunk, joint=None):
    """v_0 as a dd pair and the rows L of dv/dc of all columns, rounded to FP64, (N, n)."""
    def one(a, hs, ja):
        state, hi, (z_dd, m, free), p = point_hessians(lay0, parts0, levels, hs, a,
                                                       _with_arrays(joint, ja))
        v0 = analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True)
        rest = a[1]
        ginv = jnp.linalg.inv(rest @ state[2] @ jnp.conj(rest).T)
        rows = jnp.real(jnp.einsum('kl,blk->b', ginv, dd.cto_fp(p)))
        return v0, dd.to_fp(v0) * rows
    hs, ja = level_arrays(levels, joint)
    return jax.jit(lambda x, hs, ja: jax.lax.map(lambda a: one(a, hs, ja), x,
                                                 batch_size=chunk))(
        vf.point_inputs(batch), hs, ja)


def residual_program(lay0, parts0, levels, chunk, joint=None):
    """One compiled program: the volumes in dd at a block of points, for each column of c."""
    def one(a, cc, hs, ja):
        _, hi, (z_dd, m, free), p = point_hessians(lay0, parts0, levels, hs, a,
                                                   _with_arrays(joint, ja))

        def volume(c):
            delta = tuple(dd.dot_dd_fp(comp, c[:, None, None], axis=0) for comp in p)
            return analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True,
                                                delta=delta)
        return jax.vmap(volume)(cc.T)
    program = jax.jit(lambda x, cc, hs, ja: jax.lax.map(lambda a: one(a, cc, hs, ja), x,
                                                        batch_size=chunk))
    hs, ja = level_arrays(levels, joint)
    return lambda x, cc: program(x, cc, hs, ja)


@functools.partial(jax.jit, donate_argnums=0)
def scaled_jacobian(rows, f0, mass, m0):
    """The column-scaled J from the rows of dv/dc, in the buffer of the rows, and the norms."""
    root = jnp.sqrt(mass)
    jac = root[:, None] * (rows - (1.0 + f0)[:, None] * (mass @ rows)[None]) / m0
    norms = jnp.linalg.norm(jac, axis=0)
    return jac / norms, norms


def factor(rows, f0, mass, m0):
    """Q, R and the column norms of the scaled J; the rows are consumed."""
    jac, norms = scaled_jacobian(rows, f0, mass, m0)
    q, r = jnp.linalg.qr(jac)
    del jac
    return q, r, norms


def solve(q, r, norms, r0, cutoffs):
    """`vpa_fit.solve` from the QR of the column-scaled J."""
    u, s, vt = jnp.linalg.svd(r)
    # res @ q in place of q.T @ res: XLA can copy the transpose of q, which is as large as J.
    qr0 = r0 @ q
    beta = u.T @ qr0
    loss0 = float(r0 @ r0)
    # The part of r0 outside the range comes as a vector: loss0 - sum beta^2 cancels.
    out = r0 - q @ qr0
    loss_out = float(out @ out)
    rel = np.asarray(s / s[0])
    b2 = np.asarray(beta ** 2)
    shares = {}
    for e in range(0, 24):
        take = (rel < 10.0 ** -e) & (rel >= 10.0 ** -(e + 1))
        if take.any():
            shares[f'1e-{e} to 1e-{e + 1}'] = [int(take.sum()), float(b2[take].sum() / loss0)]
    spec = {'cond_scaled': float(rel[0] / rel[-1]), 'smallest': [float(x) for x in rel[-4:]],
            'column_norms': [float(norms.min()), float(norms.max())],
            'loss0': loss0, 'outside': loss_out / loss0, 'shares': shares}
    sols, predicted = [], []
    for tau in cutoffs:
        keep = jnp.asarray(rel >= tau)
        sols.append(np.asarray(-vt.T @ jnp.where(keep, beta / s, 0.0) / norms))
        predicted.append(loss_out + float(np.sum(np.where(np.asarray(keep), 0.0, b2))))

    def step(res, tau):
        """-J^+ res with the cutoff tau, for a refinement from the residual res."""
        b = u.T @ (res @ q)
        return np.asarray(-(vt.T @ jnp.where(jnp.asarray(rel >= tau), b / s, 0.0)) / norms)
    return spec, np.stack(sols, axis=1), predicted, step


def build_level(k0, k1, lay0, mats0, build, check, counts, args, out_dir):
    """A level of degree k1 in dd from the build points, with psi_N from the check points."""
    alpha = k1 / k0

    def weight_fn(z, alpha=alpha):
        return jnp.exp(-alpha * sections.log_potential_at(lay0, mats0, z))
    weights = jax.jit(jax.vmap(weight_fn))
    stem = f'k0_{k0}_k1_{k1}'
    if args.load is not None and (Path(args.load) / f'vpa_dd_basis_{stem}.npz').exists():
        src = Path(args.load)
        basis = arnoldi.load_joint(src / f'vpa_dd_basis_{stem}.npz')
        h_lo = np.load(src / f'vpa_dd_hlo_{stem}.npy')
        gamma = np.load(src / f'vpa_dd_gamma_{stem}.npy')
        uinv = arnoldi_dd.block_inverses(basis, h_lo)
        null = (len(gamma), gamma, int(np.argmax(np.abs(gamma))))
        say(f'k1 = {k1}: loaded from {src}: dimension {basis.h.shape[0]}, psi_N from the '
            f'first {null[0]} functions replaces function {null[2]}')
        return Level(k0, k1, alpha, basis, h_lo, uinv, null, weight_fn), {'loaded': str(src)}

    t1 = time.perf_counter()
    r2 = jnp.sum(jnp.abs(build['points']) ** 2, axis=1)
    scale_b = weights(build['points']) * r2 ** k1
    mu_b = build['mass'] / jnp.sum(build['mass'])
    if args.entries_from is not None:
        entries = arnoldi.load_joint(args.entries_from).entries
        say(f'k1 = {k1}: {len(entries)} multipliers from {args.entries_from}')
    else:
        # A build in FP64 finds the multipliers that the pivoting uses; the dd build
        # then takes those only, so that its candidates fit in memory.
        fp_basis, _ = arnoldi.build_joint(k1, np.asarray(build['points']), scale_b, mu_b,
                                          d_max=args.d_max, n_by_degree=counts,
                                          layouts=vf.Layouts(), center=True)
        if fp_basis.seeds:
            raise SystemExit('the FP64 build needs seeds; take a larger --d-max')
        entries = fp_basis.entries
        say(f'k1 = {k1}: FP64 build in {time.perf_counter() - t1:.0f}s, '
            f'{len(entries)} multipliers used')
    t1 = time.perf_counter()
    basis, h_lo, trace = arnoldi_dd.build_joint_dd(
        k1, np.asarray(build['points']), scale_b, mu_b, d_max=args.d_max,
        n_by_degree=counts, verbose=True, entries=entries)
    n = int(basis.h.shape[0])
    say(f'k1 = {k1}: dd V+A basis in {time.perf_counter() - t1:.0f}s: dimension {n} of '
        f'{counts[k1]}, {len(basis.entries)} multipliers')
    arnoldi.save_joint(out_dir / f'vpa_dd_basis_{stem}.npz', basis)
    np.save(out_dir / f'vpa_dd_hlo_{stem}.npy', h_lo)
    uinv = arnoldi_dd.block_inverses(basis, h_lo)
    row = {'n': n, 'trace': trace}

    mu = check['mass'] / jnp.sum(check['mass'])
    null = None
    if args.null_degree > 0:
        t1 = time.perf_counter()
        n_low = counts[args.null_degree]
        low_basis, low_hlo, low_uinv = truncated(basis, h_lo, uinv, n_low)
        vals = vdf.dd_values(low_basis, low_hlo, low_uinv, weights(check['points']),
                             check['points'], args.chunk)
        low = vals * jnp.sqrt(mu)[:, None]
        gamma = np.asarray(jnp.linalg.lstsq(low, jnp.sqrt(mu))[0])
        b0 = int(np.argmax(np.abs(gamma)))
        gap = float(jnp.linalg.norm(low @ gamma - jnp.sqrt(mu)))
        null = (n_low, gamma, b0)
        row['null'] = {'n_low': n_low, 'b0': b0, 'rms_gap': gap}
        # gamma enters psi_N exactly, so a reuse of the level needs these very numbers.
        np.save(out_dir / f'vpa_dd_gamma_{stem}.npy', gamma)
        say(f'k1 = {k1}: psi_N from the first {n_low} functions replaces function {b0} in '
            f'{time.perf_counter() - t1:.0f}s; rms of psi_N - 1 on the check points {gap:.3e}')
    if args.gram:
        t1 = time.perf_counter()
        vals = vdf.dd_values(basis, h_lo, uinv, weights(check['points']), check['points'],
                             args.chunk)
        eig = np.linalg.eigvalsh(np.asarray(vals.T @ (mu[:, None] * vals)))
        row['gram_eig'] = [float(eig[0]), float(eig[-1])]
        say(f'k1 = {k1}: Gram on {args.n_check} independent points in '
            f'{time.perf_counter() - t1:.0f}s: eigenvalues {eig[0]:.4f} to {eig[-1]:.4f}')
        del vals
    return Level(k0, k1, alpha, basis, h_lo, uinv, null, weight_fn), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None)
    ap.add_argument('--k1', type=int, nargs='+', required=True)
    ap.add_argument('--solve', type=str, nargs='+', default=None,
                    help='subsets of --k1 to fit, as comma lists such as 36,40; '
                         'the default is all levels together')
    ap.add_argument('--load', type=str, default=None,
                    help='a directory of `vpa_dd_fit.py` or of this script; a level whose '
                         'basis is there is loaded and not built')
    ap.add_argument('--entries-from', type=str, default=None,
                    help='a saved basis whose multipliers the dd build takes, in place of '
                         'those of a build in FP64')
    ap.add_argument('--n-fit', type=int, default=600_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--n-eval', type=int, default=2_000_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--n-build', type=int, default=20_000)
    ap.add_argument('--build-seed', type=int, default=20262001)
    ap.add_argument('--n-check', type=int, default=20_000)
    ap.add_argument('--d-max', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--block', type=int, default=65_536,
                    help='points in one call of the compiled residual program')
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[0.0],
                    help='cutoffs of s / s_max of the scaled J; 0 keeps every direction')
    ap.add_argument('--null-degree', type=int, default=9)
    ap.add_argument('--joint-basis', action='store_true',
                    help='fit in one orthonormal basis of the levels, built in dd in the '
                         'order of --k1; the solves are then the prefixes of that order')
    ap.add_argument('--joint-tol', type=float, default=1e-15,
                    help='a function whose new part is below this share of its norm '
                         'does not enter the joint basis')
    ap.add_argument('--joint-block', type=int, default=256)
    ap.add_argument('--gram', action='store_true',
                    help='also report the Gram eigenvalues of each built basis')
    ap.add_argument('--refine', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--out', type=str, default='runs_vpa_levels')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    subsets = [sorted(int(x) for x in s.split(',')) for s in
               (args.solve or [','.join(str(k) for k in args.k1)])]
    for sub in subsets:
        if not set(sub) <= set(args.k1):
            raise SystemExit(f'--solve {sub} names a degree outside --k1')

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
    record = {'args': vars(args), 'levels': {}}

    levels = []
    for k1 in args.k1:
        lev, row = build_level(args.k0, k1, lay0, mats0, build, check, counts, args, out_dir)
        levels.append(lev)
        record['levels'][str(k1)] = row
    joint = None
    if args.joint_basis:
        t1 = time.perf_counter()
        jbuild = cf.load_points(args.n_build, args.build_seed)
        values = [level_values_dd(lay0, parts0, lev, jbuild, args.chunk) for lev in levels]
        say(f'dd values of every level at the {args.n_build} build points on X in dd in '
            f'{time.perf_counter() - t1:.0f}s')
        t1 = time.perf_counter()
        jm = joint_dd.build(values, jbuild['mass'] / jnp.sum(jbuild['mass']),
                            tol=args.joint_tol, block=args.joint_block, verbose=True)
        del values
        # psi_N takes the place of the joint function that carries most of it.
        b0 = int(np.argmax(np.abs(jm.gamma[0])))
        joint = (jm, b0)
        counts_in = [int(np.sum(jm.level == i)) for i in range(len(levels))]
        record['joint'] = {'entered': counts_in, 'b0': b0, 'gamma_b0': float(jm.gamma[0][b0]),
                           'gap': jm.gap, 'ratio': np.sort(jm.ratio).tolist()}
        say(f'joint map in {time.perf_counter() - t1:.0f}s: {counts_in} functions of the '
            f'levels enter, smallest new part {jm.ratio.min():.3e}; psi_N - 1 has the norm '
            f'{jm.gap:.3e} on the build points, and psi_N takes joint function {b0}')
        prefix = np.cumsum(counts_in)
        plans = [('+'.join(str(lev.k1) for lev in levels[:i + 1]) + ' joint',
                  np.arange(prefix[i]), [lev.k1 for lev in levels[:i + 1]])
                 for i in range(len(levels))]
        total = int(prefix[-1])
    else:
        sizes = [int(lev.basis.h.shape[0]) for lev in levels]
        starts = np.concatenate([[0], np.cumsum(sizes)])
        columns = {lev.k1: np.arange(starts[i], starts[i + 1]) for i, lev in enumerate(levels)}
        plans = [('+'.join(str(k) for k in sub), np.concatenate([columns[k] for k in sub]),
                  sub) for sub in subsets]
        total = int(starts[-1])

    t1 = time.perf_counter()
    v0, lrows = fit_rows(lay0, parts0, levels, fit_set, args.chunk, joint)
    mass = fit_set['mass'] / jnp.sum(fit_set['mass'])
    f0 = analytic.residual_from_volumes(v0, mass)
    m0 = float(dd.to_fp(dd.dot_dd_fp(v0, mass, axis=0)))
    r0 = jnp.sqrt(mass) * f0
    say(f'rows of J of {total} functions at {args.n_fit} points in '
        f'{time.perf_counter() - t1:.0f}s')

    # The plan of all columns, if any, comes last and takes the buffer of the rows.
    order = sorted(range(len(plans)), key=lambda i: len(plans[i][1]) == total)
    steps, coeffs, labels, preds = {}, [np.zeros(total)], ['base'], [float(r0 @ r0)]
    record['solutions'] = []
    for i in order:
        name, cols, sub = plans[i]
        t1 = time.perf_counter()
        if len(cols) == total:
            rows_sub, lrows = lrows, None
        else:
            rows_sub = lrows[:, jnp.asarray(cols)]
        q, r, norms = factor(rows_sub, f0, mass, m0)
        del rows_sub
        spec, sols, predicted, step = solve(q, r, norms, r0, args.cutoffs)
        del q, r
        say(f"levels {name}: {len(cols)} functions, QR and SVD in "
            f"{time.perf_counter() - t1:.0f}s; columns scaled, cond {spec['cond_scaled']:.3e}, "
            f"the 4 smallest s / s_max {['%.2e' % x for x in spec['smallest']]}; share "
            f"outside the range of J {spec['outside']:.3e}")
        for key, (count, share) in spec['shares'].items():
            print(f'    s / s_max in {key}: {count} values, share of the loss {share:.3e}',
                  flush=True)
        for j, tau in enumerate(args.cutoffs):
            label = name if tau == 0.0 else f'{name}, cutoff {tau:.0e}'
            c = np.zeros(total)
            c[cols] = sols[:, j]
            print(f'    {label}: linear model loss {predicted[j]:.4e}', flush=True)
            for k in ([] if joint is not None else sub):
                lev = levels[args.k1.index(k)]
                part = c[columns[k]]
                psi_n = part[lev.null[2]] if lev.null is not None else float('nan')
                print(f'        level {k}: norm of c {np.linalg.norm(part):.3e}, coefficient '
                      f'of psi_N {psi_n:.6f}', flush=True)
            if joint is not None:
                print(f'        norm of c {np.linalg.norm(c):.3e}, coefficient of psi_N '
                      f'{c[joint[1]]:.6f}', flush=True)
            coeffs.append(c)
            labels.append(label)
            preds.append(predicted[j])
            steps[label] = (cols, step, tau)
            record['solutions'].append({'levels': sub, 'label': label, 'cutoff': tau,
                                        'spectrum': spec, 'predicted_loss': predicted[j]})
    del lrows
    coeffs = np.stack(coeffs, axis=1)

    program = residual_program(lay0, parts0, levels, args.chunk, joint)
    t1 = time.perf_counter()
    f_fit = np.array(vdf.residuals(program, coeffs, fit_set, args.block))
    say(f'the fit set in dd in {time.perf_counter() - t1:.0f}s')
    fits = {}
    for i, label in enumerate(labels):
        got = vf.summary(f_fit[:, i], fit_set)
        fits[label] = [got]
        print(f"    {label}: norm of c {np.linalg.norm(coeffs[:, i]):.3e}; fit set sigma "
              f"{got['mape']:.4e}, loss {got['loss']:.4e} against the linear model "
              f"{preds[i]:.4e}", flush=True)
    root = jnp.sqrt(mass)
    best = np.array([fits[label][0]['loss'] for label in labels])
    for it in range(1, args.refine + 1):
        t1 = time.perf_counter()
        trial = coeffs.copy()
        for i, label in enumerate(labels[1:], start=1):
            cols, step, tau = steps[label]
            trial[cols, i] += step(root * jnp.asarray(f_fit[:, i]), tau)
        f_new = vdf.residuals(program, trial, fit_set, args.block)
        say(f'refinement {it} in {time.perf_counter() - t1:.0f}s')
        for i, label in enumerate(labels[1:], start=1):
            got = vf.summary(f_new[:, i], fit_set)
            fits[label].append(got)
            # A step from an ill-conditioned J can raise the loss; such a column keeps
            # its earlier coefficients.
            kept = got['loss'] < best[i]
            if kept:
                coeffs[:, i], f_fit[:, i], best[i] = trial[:, i], f_new[:, i], got['loss']
            print(f"    {label}: fit set sigma {got['mape']:.4e}, loss {got['loss']:.4e}"
                  f"{'' if kept else ', not kept'}", flush=True)
    del steps
    tag = f"k0_{args.k0}_" + '_'.join(str(k) for k in args.k1)
    np.save(out_dir / f'vpa_levels_coeffs_{tag}.npy', coeffs)
    np.save(out_dir / f'vpa_levels_f_fit_{tag}.npy', f_fit)

    t1 = time.perf_counter()
    f_hold = vdf.residuals(program, coeffs, hold, args.block)
    say(f'the hold-out in dd in {time.perf_counter() - t1:.0f}s')
    np.save(out_dir / f'vpa_levels_f_hold_{tag}.npy', f_hold)
    record['labels'] = labels
    record['fit'] = fits
    record['hold'] = {}
    for i, label in enumerate(labels):
        got = vf.summary(f_hold[:, i], hold)
        record['hold'][label] = got
        say(f"{label}: hold-out sigma {got['mape']:.4e}, rms {got['rms']:.3e}, "
            f"max {got['max']:.3e}")
        print(cf.regions_text(got), flush=True)
    (out_dir / f'vpa_levels_{tag}.json').write_text(json.dumps(record, indent=1))
    say(f'saved to {out_dir}')


if __name__ == '__main__':
    main()
