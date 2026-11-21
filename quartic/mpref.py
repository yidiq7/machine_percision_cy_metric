"""References in mpmath at one point: the multipliers of the V+A recurrence, and phi of the base
from the same blocks H_r that FP64 uses.
"""
import mpmath as mp
import numpy as np
from mpmath import mpc, mpf

from . import arnoldi


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


def phi_exact(lay, mats, z, keep=False):
    """phi at one point, in mpmath, from the same H_r that FP64 uses.

    `keep` returns the mpmath values instead of FP64 ones. Use it to compare two
    precisions against each other, because the FP64 return rounds away
    everything that such a comparison measures.
    """
    # `z` may already hold mpmath values. `complex(c)` would round them to FP64,
    # and a reference at a point 1e-17 away is not a reference: the error
    # propagates into p with the degree k as the amplification.
    zz = [c if isinstance(c, mpc) else mpc(complex(c)) for c in z]
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
    if keep:
        return [[phi[i][j] / p for j in range(4)] for i in range(4)], p
    return np.array([[complex(phi[i][j] / p) for j in range(4)] for i in range(4)]), float(p)
