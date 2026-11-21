"""The dd error of a saved dd V+A basis, against the same recurrence in mpmath.

`arnoldi_dd.joint_jets_dd` gives q in dd from the blocks of h and the inverses U^-1
of `block_inverses`. The reference runs q_t = (y q_parent - sum_s h_st q_s) / h_tt
at 50 digits, with h + h_lo, the shift and the scale of the multipliers and the
point as exact numbers. The script reports the largest error of q for each degree,
also relative to the largest q of the degree, on random points and on points with
a small coordinate.

Usage:
    uv run python -u scripts/vpa_dd_oracle.py runs_vpa_dd36_final/vpa_dd_basis_k0_28_k1_36.npz \\
        runs_vpa_dd36_final/vpa_dd_hlo_k0_28_k1_36.npy --points 3
"""
import argparse
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import mpref as vo  # noqa: E402
from quartic import arnoldi, arnoldi_dd, data, dd, vpa_dd   # noqa: E402


def mp_values(basis, h_lo, z):
    """Every q of the basis at one point, by the recurrence in mpmath."""
    zm = [mp.mpc(complex(c).real, complex(c).imag) for c in np.asarray(z)]
    y = vo.mp_multipliers(basis.entries, zm)
    if basis.shift is not None:
        y = [(v - mp.mpf(float(a))) / mp.mpf(float(b))
             for v, a, b in zip(y, basis.shift, basis.scale)]
    n = basis.h.shape[0]
    h = basis.h
    q = [mp.mpf(0)] * n
    q[0] = 1 / (mp.mpf(float(h[0, 0])) + mp.mpf(float(h_lo[0, 0])))
    for t in range(1, n):
        col = [mp.mpf(float(h[s, t])) + mp.mpf(float(h_lo[s, t])) for s in range(t)]
        acc = mp.fdot(col, q[:t])
        grown = y[int(basis.mults[t])] * q[int(basis.parents[t])]
        q[t] = (grown - acc) / (mp.mpf(float(h[t, t])) + mp.mpf(float(h_lo[t, t])))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('basis', type=str)
    ap.add_argument('h_lo', type=str)
    ap.add_argument('--points', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    basis = arnoldi.load_joint(args.basis)
    h_lo = np.load(args.h_lo)
    t1 = time.perf_counter()
    uinv = arnoldi_dd.block_inverses(basis, h_lo)
    print(f'{basis.h.shape[0]} functions to degree {basis.k}; the inverses in '
          f'{time.perf_counter() - t1:.0f}s', flush=True)
    mp.mp.dps = 50
    pts = np.asarray(data.make(400, args.seed)['points'])
    size = np.abs(pts)
    ratio = size.min(axis=1) / size.max(axis=1)
    order = np.argsort(ratio)
    picks = list(order[-args.points:]) + list(order[:args.points])
    rest = dd.cto_dd(jnp.zeros((2, 4), jnp.complex128).at[0, 1].set(1.0).at[1, 2].set(1.0))

    def values(z):
        jets = arnoldi_dd.joint_jets_dd(
            basis, h_lo, vpa_dd.basis_multiplier_jets(basis, dd.cto_dd(z), rest), uinv)
        return jets[0]
    run = jax.jit(values)
    degrees = np.asarray(basis.degrees)
    for i in picks:
        z = jnp.asarray(pts[i])
        t1 = time.perf_counter()
        hi, lo = (np.asarray(x) for x in run(z))
        t2 = time.perf_counter()
        ref = mp_values(basis, h_lo, z)
        t3 = time.perf_counter()
        err = np.array([float(abs(mp.mpf(float(a)) + mp.mpf(float(b)) - r))
                        for a, b, r in zip(hi, lo, ref)])
        mag = np.abs(hi)
        print(f'point with min/max abs(z) {ratio[i]:.3f}: dd in {t2 - t1:.0f}s, mpmath in '
              f'{t3 - t2:.0f}s', flush=True)
        for j in sorted(set(degrees.tolist())):
            if j % 4 and j != degrees.max():
                continue
            take = degrees == j
            print(f'    degree {j:2d}: max abs(q) {mag[take].max():.3e}, max error '
                  f'{err[take].max():.3e}, relative {err[take].max() / mag[take].max():.3e}',
                  flush=True)


if __name__ == '__main__':
    main()
