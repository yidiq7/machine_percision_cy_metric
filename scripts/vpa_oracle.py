"""The FP64 error of a saved V+A basis, against the same recurrence in mpmath.

The recurrence q_t = (y q_parent - sum_s h_st q_s) / h_tt runs with the FP64
coefficients h of the build, so the reference takes h, the points and the frame
as exact numbers and repeats every step at 40 digits. The script reports the
error of q for each degree, on random points and on points with a small
coordinate, where the functions of a level are largest.

Usage, on a node:
    uv run python -u scripts/vpa_oracle.py runs_vpa36dd/vpa_basis_k0_28_k1_36.npz
"""
import argparse
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax.numpy as jnp                                     # noqa: E402

from quartic import arnoldi, data, sections                 # noqa: E402


def mp_multipliers(entries, z):
    """dictionary_values at one point in mpmath."""
    r2 = sum(abs(c) ** 2 for c in z)
    out = []
    for d, a_exp, b_exp in entries:
        total = mp.mpc(0)
        for perm in arnoldi.PERMS:
            term = mp.mpc(1)
            for c in range(4):
                w = z[perm[c]]
                term *= w ** int(a_exp[c]) * mp.conj(w) ** int(b_exp[c])
            total += term
        out.append(mp.re(total) / r2 ** d)
    return out


def mp_potential(lay, mats, z):
    """sections.potential_at at one point in mpmath, with the FP64 matrices as exact."""
    u = [c ** 4 for c in z]
    abs2 = [abs(c) ** 2 for c in z]
    total = mp.mpf(0)
    for g, mat in zip(lay.groups, mats):
        mono = [mp.fprod(u[a] ** int(e[a]) for a in range(4)) for e in np.asarray(g.exps)]
        frame = np.asarray(g.frame)
        psi = [mp.fsum(mp.mpf(frame[i, j]) * mono[i] for i in range(len(mono)))
               for j in range(frame.shape[1])]
        mat = np.asarray(mat)
        for b, res in enumerate(np.asarray(g.residues)):
            weight = mp.fprod(abs2[a] ** int(res[a]) for a in range(4))
            form = mp.mpf(0)
            for i in range(len(psi)):
                row = mp.fsum(mp.mpf(mat[b, i, j]) * psi[j] for j in range(len(psi)))
                form += mp.re(mp.conj(psi[i]) * row)
            total += weight * form
    return total


def mp_recurrence(basis, y, seeds):
    n = basis.h.shape[0]
    h = basis.h
    q = [mp.mpf(0)] * n
    q[0] = 1 / mp.mpf(h[0, 0])
    for t in range(1, n):
        if basis.seed_slot[t] >= 0:
            grown = seeds[int(basis.seed_slot[t])]
        else:
            grown = y[int(basis.mults[t])] * q[int(basis.parents[t])]
        acc = mp.fsum(mp.mpf(h[s, t]) * q[s] for s in range(t) if h[s, t] != 0.0)
        q[t] = (grown - acc) / mp.mpf(h[t, t])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('basis', type=str)
    ap.add_argument('--points', type=int, default=3, help='random points, and as many '
                    'with the smallest coordinate ratio among 4000')
    ap.add_argument('--dps', type=int, default=40)
    args = ap.parse_args()
    mp.mp.dps = args.dps

    basis = arnoldi.load_joint(args.basis)
    lays = dict(basis.layouts)
    batch = data.make(1000, 20263001)
    z_all = np.asarray(batch['points'])
    size = np.abs(z_all)
    ratio = size.min(axis=1) / size.max(axis=1)
    pick = list(range(args.points)) + list(np.argsort(ratio)[:args.points])
    degrees = np.asarray(basis.degrees)
    worst = {}
    for i in pick:
        z = jnp.asarray(z_all[i])
        y_fp = arnoldi.multiplier_values(basis, z)
        seeds_fp = None
        if basis.seeds:
            seeds_fp = jnp.concatenate([arnoldi.seed_values(lays[j], z, [a])
                                        for j, a in basis.seeds], axis=-1)
        q_fp = np.asarray(arnoldi.run_recurrence(basis, y_fp, seeds_fp))
        zm = [mp.mpc(complex(c).real, complex(c).imag) for c in z_all[i]]
        y_mp = mp_multipliers(basis.entries, zm)
        if basis.shift is not None:
            y_mp = [(y - mp.mpf(a)) / mp.mpf(b)
                    for y, a, b in zip(y_mp, basis.shift, basis.scale)]
        r2 = sum(abs(c) ** 2 for c in zm)
        seeds_mp = []
        for j, a in basis.seeds:
            mats = sections.prepare_fp(lays[j], jnp.zeros(lays[j].n_theta).at[a].set(1.0))
            seeds_mp.append(mp_potential(lays[j], mats, zm) / r2 ** j)
        q_mp = np.array([float(x) for x in mp_recurrence(basis, y_mp, seeds_mp)])
        err = np.abs(q_fp - q_mp)
        print(f"point {i}: min/max abs(z) {ratio[i]:.3f}, largest abs(q) "
              f"{np.abs(q_mp).max():.3e}, largest error {err.max():.3e}", flush=True)
        for j in np.unique(degrees):
            take = degrees == j
            worst[j] = max(worst.get(j, 0.0), float(err[take].max()))
    print('worst absolute error of q for each degree, the functions have unit norm on X:')
    for j in sorted(worst):
        print(f'    degree {j}: {worst[j]:.2e}')


if __name__ == '__main__':
    main()
