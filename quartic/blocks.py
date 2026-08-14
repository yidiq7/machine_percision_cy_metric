"""Symmetry structure of the Headrick-Nassar ansatz on the Fermat quartic.

The phase group Z_4^3 makes h block diagonal in the residue r = A mod 4. Inside
one block A = r + 4C, so with u_a = z_a^4 the sections are the degree-m
monomials u^C. On X the one relation is sum_a u_a = 0.

The relation space K is the image of multiplication by (sum_a u_a), and the code
represents the quotient as the orthogonal complement of K. S_4 permutes the
monomials, so the standard inner product is invariant, K is invariant, and the
action on K-perp is an orthogonal matrix. That frame keeps every projector well
conditioned, and it fixes the gauge of h, because h is orthogonal to K.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

import numpy as np

__all__ = ['S4', 'act', 'monomial_exponents', 'relation_columns',
           'section_frame', 'rho', 'invariant_basis',
           'Orbit', 'Block', 'Structure', 'build']

S4 = list(itertools.permutations(range(4)))


def act(sigma, vec):
    """Move slot i of `vec` to slot sigma[i]. This is a left action on tuples."""
    out = [0] * 4
    for i, v in enumerate(vec):
        out[sigma[i]] = v
    return tuple(out)


def monomial_exponents(m):
    """Exponents of u that sum to m, in descending lexicographic order.

    This order is the reported convention for the monomial index of h.
    """
    return [(a, b, c, m - a - b - c)
            for a in range(m, -1, -1)
            for b in range(m - a, -1, -1)
            for c in range(m - a - b, -1, -1)]


def relation_columns(m):
    """The subspace that (sum_a u_a) * Sym^{m-1} spans, in the degree-m basis.

    Every element of this subspace is zero on X, so it carries no section.
    """
    basis = monomial_exponents(m)
    index = {e: i for i, e in enumerate(basis)}
    lower = monomial_exponents(m - 1) if m >= 1 else []
    out = np.zeros((len(basis), len(lower)))
    for j, e in enumerate(lower):
        for a in range(4):
            shifted = list(e)
            shifted[a] += 1
            out[index[tuple(shifted)], j] = 1.0
    return out


def section_frame(m):
    """Orthonormal frame (n_monomials, d) of the complement of the relations.

    d = C(m+2,2), which is the dimension of the block section space.
    """
    n_mono = comb(m + 3, 3)
    rel = relation_columns(m)
    if rel.shape[1] == 0:
        return np.eye(n_mono)
    left, _, _ = np.linalg.svd(rel, full_matrices=True)
    return left[:, rel.shape[1]:]


def rho(m, sigma, frame=None):
    """Matrix of sigma on the block section space, in the frame of section_frame.

    S_4 permutes the monomials, so the matrix is orthogonal to round-off.
    """
    frame = section_frame(m) if frame is None else frame
    basis = monomial_exponents(m)
    index = {e: i for i, e in enumerate(basis)}
    perm = np.zeros((len(basis), len(basis)))
    for j, e in enumerate(basis):
        perm[index[act(sigma, e)], j] = 1.0
    return frame.T @ perm @ frame


def invariant_basis(m, subgroup, frame=None, tol=1e-8):
    """Orthonormal basis of {H real symmetric : rho(s) H rho(s)^T = H}.

    Pi(H) = mean_s rho(s) H rho(s)^T is an orthogonal projector in this frame,
    because every rho(s) is orthogonal. So its singular values are 0 or 1.
    """
    frame = section_frame(m) if frame is None else frame
    d = frame.shape[1]
    mats = [rho(m, s, frame) for s in subgroup]

    pi = np.zeros((d * d, d * d))
    for r_mat in mats:
        pi += np.kron(r_mat, r_mat)
    pi /= len(mats)

    sym = []
    for i in range(d):
        for j in range(i, d):
            e = np.zeros((d, d))
            e[i, j] = e[j, i] = 1.0
            sym.append(e.ravel())

    left, sing, _ = np.linalg.svd(pi @ np.stack(sym, axis=1), full_matrices=False)
    rank = int(np.sum(sing > 0.5))
    assert sing.size == rank or sing[rank] < tol, (m, sing[rank - 1], sing[rank])
    return left[:, :rank].T.reshape(rank, d, d)


@dataclass(frozen=True)
class Orbit:
    """One S_4 orbit of residues. It carries one parameter set."""
    rep: tuple          # representative residue r
    m: int              # degree in u, so the block dimension is C(m+2,2)
    stabilizer: tuple   # the elements of S_4 that fix rep
    basis: np.ndarray   # (n_params, d, d), the invariant symmetric matrices
    size: int           # number of residues in the orbit


@dataclass(frozen=True)
class Block:
    """One residue class r, and the map from its orbit representative."""
    r: tuple
    m: int
    orbit: int
    rot: np.ndarray     # (d, d) orthogonal; H_r = rot @ H_rep @ rot.T


@dataclass(frozen=True)
class Structure:
    """The complete block decomposition of h at degree k."""
    k: int
    orbits: tuple
    blocks: tuple
    frames: dict        # m -> (n_monomials, d) orthonormal frame

    @property
    def n_theta(self):
        return sum(o.basis.shape[0] for o in self.orbits)

    @property
    def n_sections(self):
        return sum(self.frames[b.m].shape[1] for b in self.blocks)

    @property
    def macs(self):
        """MACs of one matrix-vector product over every block, for one point."""
        return sum(self.frames[b.m].shape[1] ** 2 for b in self.blocks)

    def param_slices(self):
        """Slice of the flat parameter vector that belongs to each orbit."""
        out, start = [], 0
        for orb in self.orbits:
            out.append(slice(start, start + orb.basis.shape[0]))
            start += orb.basis.shape[0]
        return out


def build(k):
    """Block decomposition of h at degree k, with the invariant bases."""
    residues = [(r, (k - sum(r)) // 4)
                for r in itertools.product(range(4), repeat=4)
                if (k - sum(r)) % 4 == 0 and sum(r) <= k]
    frames = {m: section_frame(m) for m in sorted({m for _, m in residues})}

    orbits, blocks, seen = [], [], set()
    for r, m in residues:
        if r in seen:
            continue
        reach = {}
        for s in S4:
            reach.setdefault(act(s, r), s)
        stab = tuple(s for s in S4 if act(s, r) == r)
        index = len(orbits)
        orbits.append(Orbit(rep=r, m=m, stabilizer=stab,
                            basis=invariant_basis(m, stab, frames[m]),
                            size=len(reach)))
        for r_other, sigma in reach.items():
            seen.add(r_other)
            blocks.append(Block(r=r_other, m=m, orbit=index,
                                rot=rho(m, sigma, frames[m])))

    blocks.sort(key=lambda b: b.r)
    return Structure(k=k, orbits=tuple(orbits), blocks=tuple(blocks),
                     frames=frames)
