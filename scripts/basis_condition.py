"""The condition number on X of two bases for a correction of the Kaehler potential.

A correction phi = phi_0 + psi, with psi in the invariant functions of bidegree
(k, k), turns each Newton step into a problem whose condition number is that of
the Laplacian times that of the basis of psi. This script measures the second
factor for two bases of the same space:

  * product: F_a = sum_b tau_b^* B_a tau_b / abs(z)^(2k), with tau_b = rot_b^T
    sigma_b. These are the functions that the HN matrix h spans.
  * harmonic: the same space, orthonormal in L2(CP^3) with the Fubini-Study
    measure. A complex Gaussian in C^4 samples that measure.

F_a and the two Gram matrices come in dd, and mpmath at 50 digits gives the
eigenvalues, so no FP64 floor hides the small ones.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/basis_condition.py --k 4 8 12 16 20
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd, dd_metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def row_dd(lay, rots, z_dd):
    """F_a at one point for every a, as a dd pair of shape (n_theta,)."""
    sig, _ = analytic.section_vectors_dd(lay, None, z_dd=z_dd)
    hi = jnp.zeros((lay.n_theta,))
    lo = jnp.zeros((lay.n_theta,))
    for g, pair, s in zip(lay.groups, rots, sig):
        col = tuple(tuple(s[c][h][..., None] for h in (0, 1)) for c in (0, 1))
        re, im = analytic._apply_rotation_dd(pair, col)                    # (b, d, 1)
        re = (re[0][..., 0], re[1][..., 0])
        im = (im[0][..., 0], im[1][..., 0])

        def outer(x):
            return dd.mul((x[0][:, :, None], x[1][:, :, None]),
                          (x[0][:, None, :], x[1][:, None, :]))
        w = dd.add(outer(re), outer(im))                                  # (b, d, d)
        for local, sl in enumerate(g.slices):
            take = jnp.asarray(g.orbit_of == local, dtype=jnp.float64)[:, None, None]
            summed = dd.sum_dd(w[0] * take, w[1] * take, axis=0)          # (d, d)
            basis = jnp.asarray(g.basis[local, :sl.stop - sl.start])
            flat = (jnp.reshape(summed[0], (1, -1)), jnp.reshape(summed[1], (1, -1)))
            part = dd.dot_dd_fp(flat, jnp.reshape(basis, (basis.shape[0], -1)), axis=-1)
            hi = hi.at[sl].set(part[0])
            lo = lo.at[sl].set(part[1])
    # Divide by abs(z)^(2k), so that each F_a is a function of degree 0.
    norm = dd.to_dd(jnp.zeros(()))
    for a in range(4):
        re_a = (z_dd[0][0][a], z_dd[0][1][a])
        im_a = (z_dd[1][0][a], z_dd[1][1][a])
        norm = dd.add(norm, dd.add(dd.mul(re_a, re_a), dd.mul(im_a, im_a)))
    power = dd.to_dd(jnp.ones(()))
    for _ in range(lay.k):
        power = dd.mul(power, norm)
    inv = dd.reciprocal(power)
    return dd.mul((hi, lo), (inv[0], inv[1]))


def gram_dd(rows, weights, chunk=2000):
    """sum_i w_i F_i F_i^T from dd rows, accurate to dd for the hi parts."""
    root = jnp.sqrt(jnp.asarray(weights))
    r_hi = rows[0] * root[:, None]
    r_lo = rows[1] * root[:, None]
    n = r_hi.shape[1]
    acc = dd.to_dd(jnp.zeros((n, n)))
    for start in range(0, r_hi.shape[0], chunk):
        a = r_hi[start:start + chunk]
        piece = dd.dot(a[:, :, None], a[:, None, :], axis=0)
        acc = dd.add(acc, piece)
    cross = r_hi.T @ r_lo
    return dd.add(acc, dd.to_dd(cross + cross.T))


def to_mp(pair):
    hi, lo = (np.asarray(x) for x in pair)
    n = hi.shape[0]
    return mp.matrix([[mpf(float(hi[i, j])) + mpf(float(lo[i, j])) for j in range(n)]
                      for i in range(n)])


def cond_of(mat):
    vals = mp.eigsy(mat, eigvals_only=True)
    vals = sorted(vals)
    return float(vals[-1] / vals[0]), float(vals[0]), float(vals[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[4, 8, 12, 16, 20])
    ap.add_argument('--points', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=20261813)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--out', type=str, default='runs_basis')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    mp.dps = 50
    rng = np.random.default_rng(args.seed)

    xbatch = data.make(math.ceil(args.points / 4), args.seed)
    xbatch = {key: val[:args.points] for key, val in xbatch.items()}
    xbatch.update(dd_metric.point_extras(xbatch['points']))
    x_pts = (xbatch['points_hi'], xbatch['points_lo'])
    x_w = np.asarray(xbatch['mass']) / float(np.sum(np.asarray(xbatch['mass'])))
    gauss = rng.normal(size=(args.points, 4)) + 1j * rng.normal(size=(args.points, 4))
    c_pts = (jnp.asarray(gauss), jnp.zeros_like(jnp.asarray(gauss)))
    c_w = np.full(args.points, 1.0 / args.points)

    results = []
    for k in args.k:
        lay = sections.layout(k)
        rots = analytic.whitened_rotations(lay)

        def rows_of(pts):
            def one(a):
                h, l = a
                z_dd = ((jnp.real(h), jnp.real(l)), (jnp.imag(h), jnp.imag(l)))
                return row_dd(lay, rots, z_dd)
            return jax.jit(lambda x: jax.lax.map(one, x, batch_size=args.block))(pts)

        g_x = to_mp(gram_dd(rows_of(x_pts), x_w))
        g_c = to_mp(gram_dd(rows_of(c_pts), c_w))
        say(f'k = {k}: n_theta = {lay.n_theta}, Gram matrices in dd')
        cx = cond_of(g_x)
        cc = cond_of(g_c)
        chol = mp.cholesky(g_c)
        inv = chol ** -1
        s = inv * g_x * inv.T
        ch = cond_of(s)
        row = {'k': k, 'n_theta': lay.n_theta, 'cond_x_product': cx[0],
               'cond_cp3_product': cc[0], 'cond_x_harmonic': ch[0],
               'eig_x_harmonic': [ch[1], ch[2]]}
        results.append(row)
        print(f"  k = {k:>2}, n_theta = {lay.n_theta:>4}: cond on X, product "
              f"{cx[0]:.3e}; cond on CP^3, product {cc[0]:.3e}; cond on X, "
              f"harmonic {ch[0]:.3e} (eigenvalues {ch[1]:.3e} to {ch[2]:.3e})", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / 'basis_condition.json'
    path.write_text(json.dumps({'args': vars(args), 'results': results}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
