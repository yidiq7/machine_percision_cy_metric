"""The seam test of a level fit: f at the same points of X in two charts.

v = det(g|X) / OmegaOmegabar does not depend on the chart, so the change of f
between two charts is the arithmetic noise of the whole path. The level
functions are polynomials in multipliers that satisfy relations on X, so their
values depend on how a point is written, and this test includes that. The second
chart puts the second largest coordinate to 1, in dd, and takes the largest
abs(dP/dz) among the other coordinates as m.

It also reports the margin of the metric: the smallest eigenvalue of g0, the
base metric on X, over its largest, and the largest abs eigenvalue of g0^-1 dg,
with dg the part of the levels. g = g0 + dg is positive where that is below 1.

Usage, on the GPU node, with the arguments of the `--eval-only` runs:
    uv run --extra gpu python -u scripts/seam_levels.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 44 40 36 \\
        --load runs_levels_load --load-joint MAP --coeffs COEFFS --exact-mul
"""
import argparse
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

from quartic import samples as cf  # noqa: E402
import vpa_dd_levels as vl                                  # noqa: E402
from quartic import levelfit as vf  # noqa: E402
from quartic import analytic, dd, dd_metric, joint_dd, sections  # noqa: E402


def other_chart(batch):
    """The same dd points, in the chart where the second largest coordinate is 1."""
    hi, lo = jnp.asarray(batch['points_hi']), jnp.asarray(batch['points_lo'])
    n = hi.shape[0]
    rows = np.arange(n)
    j = np.argsort(-np.abs(np.asarray(hi)), axis=1)[:, 1]
    z = ((jnp.real(hi), jnp.real(lo)), (jnp.imag(hi), jnp.imag(lo)))
    inv = dd.creciprocal(jax.tree_util.tree_map(lambda a: a[rows, j][:, None], z))
    w = dd.cmul(z, inv)
    new_hi = np.asarray(w[0][0]) + 1j * np.asarray(w[1][0])
    new_lo = np.asarray(w[0][1]) + 1j * np.asarray(w[1][1])
    new_hi[rows, j], new_lo[rows, j] = 1.0, 0.0
    grad = np.abs(4.0 * new_hi ** 3)
    grad[rows, j] = -1.0
    m = np.argmax(grad, axis=1)
    mask = np.ones((n, 4), dtype=bool)
    mask[rows, j] = False
    mask[rows, m] = False
    free = np.nonzero(mask)[1].reshape(n, 2)
    out = dict(batch)
    out['points'] = out['points_hi'] = jnp.asarray(new_hi)
    out['points_lo'] = jnp.asarray(new_lo)
    out['patch'] = jnp.asarray(np.stack([m, free[:, 0], free[:, 1]], axis=1).astype(np.int32))
    return out


def seam_program(lay0, parts0, levels, chunk, joint):
    """v in dd and the largest abs eigenvalue of g0^-1 dg for each column of c, and the
    smallest eigenvalue of g0 over its largest, at a block of points."""
    def one(a, cc, hs, ja):
        _, hi, (z_dd, m, free), p = vl.point_hessians(lay0, parts0, levels, hs, a,
                                                      vl._with_arrays(joint, ja))
        rest = dd.cto_fp(dd_metric._restriction_dd(z_dd, m, free)[0])
        g0 = rest @ dd.cto_fp(hi[0]) @ jnp.conj(rest).T
        root = jnp.linalg.inv(jnp.linalg.cholesky(g0))
        pf = dd.cto_fp(p)

        def column(c):
            delta = tuple(dd.dot_dd_fp(comp, c[:, None, None], axis=0) for comp in p)
            v = analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True, delta=delta)
            dg = jnp.einsum('b,bkl->kl', c, pf)
            return v, jnp.max(jnp.abs(jnp.linalg.eigvalsh(root @ dg @ jnp.conj(root).T)))
        eig0 = jnp.linalg.eigvalsh(g0)
        return jax.vmap(column)(cc.T), eig0[0] / eig0[1]
    program = jax.jit(lambda x, cc, hs, ja: jax.lax.map(lambda a: one(a, cc, hs, ja), x,
                                                        batch_size=chunk))
    hs, ja = vl.level_arrays(levels, joint)
    return lambda x, cc: program(x, cc, hs, ja)


def evaluate(program, coeffs, batch, block):
    """v as (hi, lo), the eigenvalue ratio of dg for each column and that of g0, per point."""
    inputs = vf.point_inputs(batch)
    total = batch['mass'].shape[0]
    outs = []
    for start in range(0, total, block):
        stop = min(start + block, total)
        take = jnp.minimum(jnp.arange(start, start + block), stop - 1)
        (v, ratio), cond = program(jax.tree_util.tree_map(lambda a: a[take], inputs),
                                   jnp.asarray(coeffs))
        outs.append([np.asarray(x[:stop - start]) for x in (v[0], v[1], ratio, cond)])
    return [np.concatenate(xs) for xs in zip(*outs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--k1', type=int, nargs='+', required=True)
    ap.add_argument('--load', type=str, required=True)
    ap.add_argument('--load-joint', type=str, required=True)
    ap.add_argument('--coeffs', type=str, required=True)
    ap.add_argument('--exact-mul', action='store_true')
    ap.add_argument('--invariant-base', action='store_true',
                    help='the base in the monomial basis, exactly invariant under S_4')
    ap.add_argument('--n', type=int, default=131_072)
    ap.add_argument('--seed', type=int, default=20263813)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--block', type=int, default=65_536)
    args = ap.parse_args()

    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    analytic.EXACT_MUL = args.exact_mul
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    # With --invariant-base the dd paths take the base in the monomial basis, exactly
    # invariant under S_4; the level bases keep the frame form of lay0.
    lay0d = lay0
    if args.invariant_base:
        from quartic import invariant
        lay0d, parts0 = invariant.monomial_base(lay0, parts0)
    counts = vf.hn_counts(max(args.k1))

    class LoadArgs:
        load = args.load
    levels = [vl.build_level(args.k0, k1, lay0, mats0, None, None, counts, LoadArgs,
                             Path('.'))[0] for k1 in args.k1]
    jm = joint_dd.load(args.load_joint)
    prefix = np.cumsum([int(np.sum(jm.level == i)) for i in range(len(levels))])
    joint = (jm, None, tuple(int(x) for x in prefix))
    coeffs = np.load(args.coeffs)
    program = seam_program(lay0d, parts0, levels, args.chunk, joint)

    batch = cf.load_points(args.n, args.seed)
    second = other_chart(batch)
    mass = batch['mass'] / jnp.sum(batch['mass'])
    w = np.asarray(mass)
    fs = []
    for name, points in (('first', batch), ('second', second)):
        t1 = time.perf_counter()
        hi, lo, ratio, cond = evaluate(program, coeffs, points, args.block)
        fs.append(np.asarray(analytic.residual_from_volumes(
            (jnp.asarray(hi), jnp.asarray(lo)), mass[:, None])))
        print(f'{name} chart: {args.n} points in dd in {time.perf_counter() - t1:.0f}s; '
              f'smallest eigenvalue of g0 over its largest {cond.min():.3e}; largest abs '
              f'eigenvalue of g0^-1 dg for each column '
              f'{["%.3e" % x for x in ratio.max(axis=0)]}', flush=True)
    same = np.mean(np.asarray(batch['patch'])[:, 0] == np.asarray(second['patch'])[:, 0])
    print(f'{args.n} points of seed {args.seed}; the two charts take the same m at '
          f'{same:.3f} of the points')
    for i in range(coeffs.shape[1]):
        fa, fb = fs[0][:, i], fs[1][:, i]
        d = fb - fa
        print(f'column {i}: sigma {np.sum(w * np.abs(fa)):.4e} in the first chart and '
              f'{np.sum(w * np.abs(fb)):.4e} in the second; the change of f has mean abs '
              f'{np.sum(w * np.abs(d)):.3e}, rms {math.sqrt(np.sum(w * d * d)):.3e}, max '
              f'{np.abs(d).max():.3e} and weighted mean {np.sum(w * d):.3e}', flush=True)


if __name__ == '__main__':
    main()
