"""Where the FP64 error of phi is born, stage by stage.

One scalar code path serves both precisions, because mpmath `mpc` and Python
`complex` support the same operations. `exact_upto` says how many stages run in
mpmath before the values round to FP64 and the rest continues there. The drop in
the error from one stage to the next attributes the error to that stage.

  1  u = z^4, and the monomials u^C of each group
  2  psi = Q^T psi_full, the frame of the sections
  3  sigma = z^r psi and its 4 holomorphic derivatives
  4  H sigma, H dsigma, and the Gram sums p and dp
  5  the centered form of phi

Usage:
    uv run python -u scripts/oracle_stages.py --k 20 24 28 32 --points 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpc, mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

from quartic import data, sections                          # noqa: E402

STAGES = {1: 'monomials u^C', 2: 'frame Q^T psi', 3: 'sigma and dsigma',
          4: 'H matvecs and Gram sums', 5: 'centered phi'}

# Which stages run in mpmath. The last two price the fix that is on the table.
CONFIGS = [set(), {4}, {5}, {1, 2, 3}, {4, 5}]
LABELS = ['all FP64', 'dd on 4', 'dd on 5', 'dd on 1-3', 'dd on 4+5']


def phi_staged(lay, mats, z, exact):
    """phi at one point, with every stage in `exact` run in mpmath.

    `num` converts at the point of use, so a value made in an exact stage rounds
    to FP64 as soon as an inexact stage consumes it. Any set of stages therefore
    works, and not only a prefix.
    """
    exact = set(exact)

    def num(stage, x):
        """A scalar in the precision that `stage` runs at."""
        return mpc(x) if stage in exact else complex(x)

    def real(stage, x):
        return mpf(float(x)) if stage in exact else float(x)

    def demote(stage, x):
        return x

    zz = [num(1, complex(c)) for c in z]
    u = [c ** 4 for c in zz]

    groups = []
    for g, mat in zip(lay.groups, mats):
        exps = np.asarray(g.exps)
        frame = np.asarray(g.frame)
        h_all = np.asarray(mat)
        d = frame.shape[1]

        # Stage 1: the monomials of u, and their holomorphic derivatives.
        pf, dpf = [], []
        for row in exps:
            val = num(1, 1.0)
            for a in range(4):
                val = val * u[a] ** int(row[a])
            pf.append(demote(1, val))
            dpf.append([demote(1, 4 * int(row[a]) * val / zz[a]) for a in range(4)])

        # Stage 2: the frame of the sections.
        psi = [demote(2, sum((real(2, frame[c, al]) * num(2, pf[c])
                             for c in range(len(pf))), num(2, 0.0)))
               for al in range(d)]
        dpsi = [[demote(2, sum((real(2, frame[c, al]) * num(2, dpf[c][i])
                               for c in range(len(pf))), num(2, 0.0)))
                 for i in range(4)] for al in range(d)]

        # Stage 3: the z^r factor of each block.
        blocks_out = []
        for bi, r in enumerate(np.asarray(g.residues)):
            zr = num(3, 1.0)
            for a in range(4):
                zr = zr * num(3, zz[a]) ** int(r[a])
            dzr = [int(r[a]) * zr / num(3, zz[a]) for a in range(4)]
            s = [demote(3, zr * num(3, psi[al])) for al in range(d)]
            ds = [[demote(3, dzr[i] * num(3, psi[al]) + zr * num(3, dpsi[al][i]))
                   for i in range(4)] for al in range(d)]
            blocks_out.append((bi, s, ds, h_all))
        groups.append((d, blocks_out))

    # Stage 4: the matrix-vector products and the Gram sums.
    p = real(4, 0.0)
    p_i = [num(4, 0.0)] * 4
    store = []
    for d, blocks_out in groups:
        for bi, s, ds, h_all in blocks_out:
            hm = [[real(4, h_all[bi, a, c]) for c in range(d)] for a in range(d)]
            h_s = [sum((hm[a][c] * num(4, s[c]) for c in range(d)), num(4, 0.0))
                   for a in range(d)]
            h_d = [[sum((hm[a][c] * num(4, ds[c][i]) for c in range(d)), num(4, 0.0))
                    for i in range(4)] for a in range(d)]
            p = p + sum((num(4, s[a]).conjugate() * h_s[a] for a in range(d)),
                        num(4, 0.0)).real
            for i in range(4):
                p_i[i] = p_i[i] + sum((num(4, s[a]).conjugate() * h_d[a][i]
                                       for a in range(d)), num(4, 0.0))
            store.append((d, s, ds, [demote(4, x) for x in h_s],
                          [[demote(4, y) for y in row] for row in h_d]))
    p = p.real if hasattr(p, 'real') else p
    p_i = [demote(4, x) for x in p_i]

    # Stage 5: the centered form.
    phi = [[num(5, 0.0)] * 4 for _ in range(4)]
    for d, s, ds, h_s, h_d in store:
        ratio = [num(5, p_i[i]) / num(5, p) for i in range(4)]
        b = [[num(5, ds[a][i]) - ratio[i] * num(5, s[a]) for i in range(4)]
             for a in range(d)]
        hb = [[num(5, h_d[a][i]) - ratio[i] * num(5, h_s[a]) for i in range(4)]
              for a in range(d)]
        for i in range(4):
            for j in range(4):
                phi[i][j] = phi[i][j] + sum(
                    (b[a][j].conjugate() * hb[a][i] for a in range(d)), num(5, 0.0))
    return np.array([[complex(phi[i][j] / num(5, p)) for j in range(4)]
                     for i in range(4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[20, 24, 28])
    ap.add_argument('--points', type=int, default=2)
    ap.add_argument('--dps', type=int, default=50)
    args = ap.parse_args()
    mp.dps = args.dps

    batch = data.make(40, 3)
    print(f"mpmath at {args.dps} digits. Relative error of phi when stages 1..j "
          f"run exactly.\n")
    head = f"{'k':>3} " + ''.join(f"{name:>12}" for name in LABELS)
    print(head)
    print('-' * len(head))

    for k in args.k:
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        mats = sections.prepare(lay, theta)
        errs = []
        for cfg in CONFIGS:
            worst = 0.0
            for idx in range(args.points):
                z = batch['points'][idx]
                ref = phi_staged(lay, mats, z, {1, 2, 3, 4, 5})
                got = phi_staged(lay, mats, z, cfg)
                worst = max(worst, np.abs(got - ref).max() / np.abs(ref).max())
            errs.append(worst)
        print(f"{k:>3} " + ''.join(f"{e:>12.3e}" for e in errs))


if __name__ == '__main__':
    main()
