"""Accuracy of the FP64 evaluation of phi, against mpmath at 50 digits.

The block matrices H_r come in as exact input, so the test isolates what the
per-point path does: the monomials of u, the frame of the sections, the Gram
sums, and the centered form of phi. That is the part whose error sets the floor
under `mean abs(f)`.

Usage:
    uv run python -u scripts/oracle_precision.py --k 12 20 24 28 --points 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpc, mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
from quartic import analytic, data, sections                # noqa: E402


def phi_exact(lay, mats, z):
    """phi at one point, in mpmath, from the same H_r that FP64 uses."""
    zz = [mpc(complex(c)) for c in z]
    u = [c ** 4 for c in zz]

    sig, dsig, hs, hd = [], [], [], []
    p = mpf(0)
    p_i = [mpc(0)] * 4
    for g, mat in zip(lay.groups, mats):
        exps = np.asarray(g.exps)
        frame = np.asarray(g.frame)
        h_all = np.asarray(mat)

        # Monomials of u, and their holomorphic derivatives.
        pf, dpf = [], []
        for row in exps:
            val = mpc(1)
            for a in range(4):
                val *= u[a] ** int(row[a])
            pf.append(val)
            dpf.append([4 * int(row[a]) * val / zz[a] for a in range(4)])

        d = frame.shape[1]
        psi = [sum((mpf(float(frame[c, al])) * pf[c] for c in range(len(pf))), mpc(0))
               for al in range(d)]
        dpsi = [[sum((mpf(float(frame[c, al])) * dpf[c][i] for c in range(len(pf))),
                     mpc(0)) for i in range(4)] for al in range(d)]

        for bi, r in enumerate(np.asarray(g.residues)):
            zr = mpc(1)
            for a in range(4):
                zr *= zz[a] ** int(r[a])
            dzr = [int(r[a]) * zr / zz[a] for a in range(4)]

            s = [zr * psi[al] for al in range(d)]
            ds = [[dzr[i] * psi[al] + zr * dpsi[al][i] for i in range(4)]
                  for al in range(d)]
            hmat = [[mpf(float(h_all[bi, a, c])) for c in range(d)] for a in range(d)]

            h_s = [sum((hmat[a][c] * s[c] for c in range(d)), mpc(0)) for a in range(d)]
            h_d = [[sum((hmat[a][c] * ds[c][i] for c in range(d)), mpc(0))
                    for i in range(4)] for a in range(d)]

            p += sum((s[a].conjugate() * h_s[a] for a in range(d)), mpc(0)).real
            for i in range(4):
                p_i[i] += sum((s[a].conjugate() * h_d[a][i] for a in range(d)), mpc(0))
            sig.append(s), dsig.append(ds), hs.append(h_s), hd.append(h_d)

    phi = [[mpc(0)] * 4 for _ in range(4)]
    for s, ds, h_s, h_d in zip(sig, dsig, hs, hd):
        d = len(s)
        b = [[ds[a][i] - (p_i[i] / p) * s[a] for i in range(4)] for a in range(d)]
        hb = [[h_d[a][i] - (p_i[i] / p) * h_s[a] for i in range(4)] for a in range(d)]
        for i in range(4):
            for j in range(4):
                phi[i][j] += sum((b[a][j].conjugate() * hb[a][i] for a in range(d)),
                                 mpc(0))
    return np.array([[complex(phi[i][j] / p) for j in range(4)] for i in range(4)]), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[12, 20, 24, 28])
    ap.add_argument('--points', type=int, default=3)
    ap.add_argument('--dps', type=int, default=50)
    args = ap.parse_args()
    mp.dps = args.dps

    batch = data.make(40, 3)
    restriction = np.asarray(batch['restriction'])
    omega = np.asarray(batch['Omega_Omegabar'])
    print(f"mpmath at {args.dps} digits, {args.points} points, H_r taken as exact\n")
    print(f"{'k':>3} {'rel err phi':>12} {'rel err v':>11} {'sigma':>12} {'sigma/err':>10}")
    known = {12: 2.2018e-07, 16: 3.5096e-09, 20: 6.3276e-11,
             24: 1.2566e-12, 28: 1.7077e-13}

    for k in args.k:
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        mats = sections.prepare(lay, theta)
        worst_phi = worst_v = 0.0
        for idx in range(args.points):
            z = batch['points'][idx]
            got, _ = analytic.phi_analytic(lay, mats, z)
            got = np.asarray(got)
            ref, _ = phi_exact(lay, mats, z)
            worst_phi = max(worst_phi, np.abs(got - ref).max() / np.abs(ref).max())
            r_mat = restriction[idx]
            g_got = r_mat @ got @ np.conj(r_mat).T
            g_ref = r_mat @ ref @ np.conj(r_mat).T
            v_got = np.real(np.linalg.det(g_got)) / omega[idx]
            v_ref = np.real(np.linalg.det(g_ref)) / omega[idx]
            worst_v = max(worst_v, abs(v_got - v_ref) / abs(v_ref))
        s = known.get(k)
        print(f"{k:>3} {worst_phi:>12.3e} {worst_v:>11.3e} "
              f"{(f'{s:.4e}' if s else '-'):>12} "
              f"{(f'{s / worst_v:.1f}' if s and worst_v else '-'):>10}")


if __name__ == '__main__':
    main()
