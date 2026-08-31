"""Accuracy of the FP64 Jacobian dv/dtheta, against mpmath at 60 digits.

LM descends on `J^T r`. If `J` carries a relative error of e, then `J^T r` is
noise once `r` falls to e times its start, and the fit stalls there whatever the
coordinates are. `--whiten` took `cond(J)*eps` from 133 to 0.036 at k = 28 and
sigma did not improve, so the rank of `J` is not what stops the fit. Its error is
the explanation that is left, and no fit measures that.

The reference is a central difference of v in mpmath, and not the analytic
formula in mpmath. So this tests Jacobi's formula and the contraction with K as
well as their arithmetic. Two step sizes bound the truncation error.

`H_r` and `dH_r/dtheta` come in as exact input, as in `oracle_precision.py`, so
the test isolates the per-point path.

Usage:
    uv run python -u scripts/oracle_jacobian.py --k 12 20 24
    uv run python -u scripts/oracle_jacobian.py --k 28 --gram-dd on --points 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpc, mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax.numpy as jnp                                     # noqa: E402
from quartic import analytic, data, sections, whiten        # noqa: E402


def _phi_mp(lay, mats_mp, z):
    """phi and p at one point, in mpmath, from H_r given as exact input."""
    zz = [mpc(complex(c)) for c in z]
    u = [c ** 4 for c in zz]

    sig, dsig, hs, hd = [], [], [], []
    p = mpf(0)
    p_i = [mpc(0)] * 4
    for g, h_all in zip(lay.groups, mats_mp):
        exps = np.asarray(g.exps)
        frame = np.asarray(g.frame)

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
            hmat = h_all[bi]

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
    return [[phi[i][j] / p for j in range(4)] for i in range(4)]


def _v_mp(lay, mats_mp, z, r_mat, omega):
    """v = det(R phi R^dag) / OmegaOmegabar at one point, all in mpmath.

    X is a graph over 2 local coordinates, so R is 2 by 4 and the determinant is
    2 by 2. Write it out, because mpmath has no batched linear algebra.
    """
    phi = _phi_mp(lay, mats_mp, z)
    rr = [[mpc(complex(r_mat[i, a])) for a in range(4)] for i in range(2)]

    g = [[mpc(0)] * 2 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            acc = mpc(0)
            for a in range(4):
                for b in range(4):
                    acc += rr[i][a] * phi[a][b] * rr[j][b].conjugate()
            g[i][j] = acc
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    return det.real / mpf(float(omega))


def _mats_mp(mats):
    """Promote the block matrices of every group to mpmath."""
    out = []
    for mat in mats:
        arr = np.asarray(mat)
        out.append([[[mpf(float(arr[b, i, j])) for j in range(arr.shape[2])]
                     for i in range(arr.shape[1])] for b in range(arr.shape[0])])
    return out


def _shift(mats_mp, dmats_mp, step):
    """H(theta + step * e_alpha), from H and dH/dtheta_alpha, in mpmath."""
    out = []
    for base, grad in zip(mats_mp, dmats_mp):
        out.append([[[base[b][i][j] + step * grad[b][i][j]
                      for j in range(len(base[b][i]))]
                     for i in range(len(base[b]))] for b in range(len(base))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[12, 20, 24])
    ap.add_argument('--points', type=int, default=2)
    ap.add_argument('--directions', type=int, default=3,
                    help='how many components of theta to differentiate')
    ap.add_argument('--dps', type=int, default=60)
    ap.add_argument('--theta', type=str, default=None,
                    help='a saved theta. The error of J depends on theta, so '
                         'measure it where the fit landed. One --k only.')
    ap.add_argument('--gram-dd', choices=['off', 'inner', 'on'], default='off')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--step', type=int, default=15,
                    help='the central difference uses 10**-step and 10**(1-step). '
                         'The truncation falls as the square of the step and it '
                         'grows with k, so it reached 1.4e-17 at k = 20 with '
                         'step 12. mpmath at 60 digits leaves ample room.')
    ap.add_argument('--whiten', action='store_true',
                    help='build the whitened layout. A theta from a whitened '
                         'fit means nothing in the plain basis, so this must '
                         'match the run that saved --theta.')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    args = ap.parse_args()
    analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[args.gram_dd]
    if args.theta is not None and len(args.k) != 1:
        ap.error('--theta needs exactly one --k')
    mp.dps = args.dps

    batch = data.make(40, args.seed)
    restriction = np.asarray(batch['restriction'])
    omega = np.asarray(batch['Omega_Omegabar'])
    where = 'the saved theta' if args.theta else 'theta_FS'
    print(f"mpmath at {args.dps} digits, {args.points} points, "
          f"{args.directions} directions, at {where}, Gram stage {args.gram_dd}, "
          f"basis {'whitened' if args.whiten else 'plain'}")
    print("reference: central difference of v, two steps. "
          "H and dH/dtheta are exact input.\n")
    print(f"{'k':>3} {'rel err v':>11} {'rel err dv/dth':>15} "
          f"{'truncation':>11} {'worst alpha':>12}")

    for k in args.k:
        if args.whiten:
            lay = whiten.whitened_layout(k, n_pairs=args.gram_pairs,
                                         seed=args.gram_seed)
        else:
            lay = sections.layout(k)
        if args.theta is None:
            theta = sections.fubini_study_theta(lay)
        else:
            theta = jnp.asarray(np.load(args.theta))
            if theta.shape != (lay.n_theta,):
                raise SystemExit(f'theta has {theta.shape}, k={k} needs '
                                 f'({lay.n_theta},)')
            side = Path(args.theta).with_suffix('.basis')
            if side.exists() and side.read_text().strip() != lay.fingerprint():
                raise SystemExit(
                    f'theta was saved against basis {side.read_text().strip()} '
                    f'and this build has {lay.fingerprint()}. Add or drop '
                    f'--whiten so that the two agree.')

        alphas = np.linspace(0, lay.n_theta - 1, args.directions).astype(int)
        alphas = sorted(set(int(a) for a in alphas))

        mats_mp = _mats_mp(sections.prepare(lay, theta))
        # dH/dtheta_alpha, exact because H is linear in theta and the scale of
        # `prepare` is inside it. A basis vector gives the column directly.
        grads = {}
        for al in alphas:
            e = jnp.zeros(lay.n_theta).at[al].set(1.0)
            grads[al] = _mats_mp(sections.prepare(lay, e))

        v_got, dv_got = analytic.residual_jacobian(lay, theta, batch)
        v_got = np.asarray(v_got)
        dv_got = np.asarray(dv_got)

        worst_v = worst_d = 0.0
        worst_trunc = 0.0
        worst_al = -1
        for idx in range(args.points):
            z = batch['points'][idx]
            r_mat = restriction[idx]
            v_ref = _v_mp(lay, mats_mp, z, r_mat, omega[idx])
            worst_v = max(worst_v, abs(float(v_got[idx]) / float(v_ref) - 1))

            # Scale the error by the largest derivative of this point, and not
            # by each component. A component near zero gives a relative error of
            # order 1 and says nothing about the step that LM takes.
            refs, truncs = {}, {}
            for al in alphas:
                ders = []
                for step in (mpf(10) ** -args.step, mpf(10) ** (1 - args.step)):
                    plus = _v_mp(lay, _shift(mats_mp, grads[al], step), z,
                                 r_mat, omega[idx])
                    minus = _v_mp(lay, _shift(mats_mp, grads[al], -step), z,
                                  r_mat, omega[idx])
                    ders.append((plus - minus) / (2 * step))
                refs[al] = float(ders[0])
                truncs[al] = float(abs(ders[0] - ders[1]))
            biggest = max(abs(r) for r in refs.values()) or 1.0
            worst_trunc = max(worst_trunc,
                              max(truncs.values()) / biggest)
            for al in alphas:
                rel = abs(float(dv_got[idx, al]) - refs[al]) / biggest
                if rel > worst_d:
                    worst_d, worst_al = rel, al

        print(f"{k:>3} {worst_v:>11.3e} {worst_d:>15.3e} "
              f"{worst_trunc:>11.3e} {worst_al:>12}")

    print("\n`truncation` is the difference between the two steps. It must sit "
          "far below\n`rel err dv/dth`, or the reference is the limit and not `J`.")


if __name__ == '__main__':
    main()
