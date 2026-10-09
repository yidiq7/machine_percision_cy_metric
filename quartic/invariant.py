"""The HN base in the monomial basis of each block, exactly invariant under S_4.

The per-point path takes p0 = sum over r of abs(z^r)^2 psi^T H_r psi with
psi = F^T u^E and F an FP64 frame. F spans the quotient by the relation sum u = 0
only to round-off, and the rotations F^T Pi F between the blocks of an orbit are
FP64 matrices, so that p0 is invariant only to approximately 1e-16.

In the monomial basis the same block is abs(z^r)^2 (u^E)^T C_r u^E with
C_r = F H_r F^T, and S_4 acts by a permutation of the indices. `monomial_base` forms
C of one block of each orbit in mpmath, averages it over the stabilizer of the block,
rounds it to dd, and copies those dd numbers to the other blocks of the orbit at the
permuted positions. The base is then exactly invariant as a polynomial in z and
conj(z), and it differs from p0 by the part of p0 that is not invariant.

The output takes the place of (lay, parts) in every per-point dd path: each frame is
the identity on the monomials, and each block is C_r. It has no parameters, so the
paths of the HN Jacobian refuse it.
"""
import dataclasses
import itertools

import mpmath as mp
import numpy as np
import jax.numpy as jnp

from . import blocks

S4 = list(itertools.permutations(range(4)))


def _dd(x):
    """The dd pair nearest to the mpmath number x."""
    hi = float(x)
    return hi, float(x - mp.mpf(hi))


def _pair_orbits(n, perms):
    """The orbits of the index pairs (i, j) under the permutations and the transposition."""
    seen = np.zeros((n, n), dtype=bool)
    out = []
    for i in range(n):
        for j in range(n):
            if seen[i, j]:
                continue
            orbit = set()
            for p in perms:
                a, b = p[i], p[j]
                orbit.add((a, b))
                orbit.add((b, a))
            for a, b in orbit:
                seen[a, b] = True
            out.append(sorted(orbit))
    return out


def _block_c(frame, hi, lo):
    """C = F H F^T in mpmath, with H = hi + lo."""
    n, d = frame.shape
    f = mp.matrix([[mp.mpf(float(x)) for x in row] for row in frame])
    h = mp.matrix(d, d)
    for i in range(d):
        for j in range(d):
            h[i, j] = mp.mpf(float(hi[i, j])) + mp.mpf(float(lo[i, j]))
    return f * h * f.T


def monomial_base(lay, parts, dps=60):
    """(lay_m, parts_m): the base of `analytic.point_matrices` in the monomial basis.

    `parts` holds one (hi, lo) pair of shape (n_blocks, d, d) for each group, as
    `analytic.point_matrices` gives it with `H_LOW`.
    """
    groups, out = [], []
    with mp.workdps(dps):
        for g, (hi, lo) in zip(lay.groups, parts):
            hi, lo = np.asarray(hi), np.asarray(lo)
            exps = [tuple(int(x) for x in e) for e in np.asarray(g.exps)]
            index = {e: i for i, e in enumerate(exps)}
            n = len(exps)
            residues = [tuple(int(x) for x in r) for r in np.asarray(g.residues)]
            orbit_of = np.asarray(g.orbit_of)
            c_hi = np.zeros((len(residues), n, n))
            c_lo = np.zeros((len(residues), n, n))
            for o in sorted(set(orbit_of.tolist())):
                members = [int(b) for b in np.flatnonzero(orbit_of == o)]
                b0, r0 = members[0], residues[members[0]]
                c = _block_c(np.asarray(g.frame), hi[b0], lo[b0])
                stab = [[index[blocks.act(s, e)] for e in exps]
                        for s in S4 if blocks.act(s, r0) == r0]
                # One value for each orbit of index pairs, copied to every pair of it,
                # so that the block is exactly invariant under its stabilizer.
                for pairs in _pair_orbits(n, stab):
                    val = mp.fsum(c[a, b] for a, b in pairs) / len(pairs)
                    vh, vl = _dd(val)
                    for a, b in pairs:
                        c_hi[b0, a, b], c_lo[b0, a, b] = vh, vl
                # The other blocks of the orbit: the same numbers at permuted positions.
                for b in members[1:]:
                    sigma = next(s for s in S4 if blocks.act(s, r0) == residues[b])
                    pi = np.array([index[blocks.act(sigma, e)] for e in exps])
                    c_hi[b][np.ix_(pi, pi)] = c_hi[b0]
                    c_lo[b][np.ix_(pi, pi)] = c_lo[b0]
            eye = np.eye(n)
            groups.append(dataclasses.replace(
                g, frame=eye, rot=np.broadcast_to(eye, (len(residues), n, n)).copy(),
                basis=np.zeros((0, 0, n, n)), slices=(), whiten=np.zeros((0, n, n))))
            out.append((jnp.asarray(c_hi), jnp.asarray(c_lo)))
    lay_m = dataclasses.replace(lay, groups=tuple(groups), n_theta=0, scale=np.zeros(0),
                                whiten_inv=(), whiten_fwd=(), gram_id='monomial invariant')
    return lay_m, tuple(out)


def potential_mp(lay_m, parts_m, z):
    """p0 of the monomial base at one point, in mpmath, for the tests."""
    u = [c ** 4 for c in z]
    total = mp.mpf(0)
    for g, (hi, lo) in zip(lay_m.groups, parts_m):
        hi, lo = np.asarray(hi), np.asarray(lo)
        mono = []
        for e in np.asarray(g.exps):
            val = mp.mpc(1)
            for a in range(4):
                if e[a]:
                    val *= u[a] ** int(e[a])
            mono.append(val)
        for b, r in enumerate(np.asarray(g.residues)):
            w = mp.mpf(1)
            for a in range(4):
                if r[a]:
                    w *= abs(z[a]) ** (2 * int(r[a]))
            form = mp.mpf(0)
            for i in range(len(mono)):
                for j in range(len(mono)):
                    cij = mp.mpf(float(hi[b, i, j])) + mp.mpf(float(lo[b, i, j]))
                    if cij:
                        form += cij * (mp.conj(mono[i]) * mono[j]).real
            total += w * form
    return total
