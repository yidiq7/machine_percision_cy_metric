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

import hashlib
import itertools
from dataclasses import dataclass
from math import comb

import numpy as np

__all__ = ['S4', 'act', 'monomial_exponents', 'relation_columns',
           'section_frame', 'rho', 'invariant_basis', 'orthonormalize',
           'fingerprint', 'Orbit', 'Block', 'Structure', 'build']

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


def orthonormalize(columns, against=(), tol=1e-9):
    """Gram-Schmidt over `columns` in the order given, after `against`.

    **The order is the convention.** An SVD picks an arbitrary rotation inside a
    degenerate subspace, and both bases here come from one: the section frame is
    a null space, where every singular value is zero. LAPACK then answers with
    *an* orthonormal basis, which is reproducible on one machine and not across
    nodes. A theta saved on one node and read on another became meaningless: at
    k = 28 it reported `mean abs(f) = 1.34` in place of 2.06e-13.

    Gram-Schmidt in a fixed order has no such freedom. It runs twice over the
    accepted vectors, because one pass loses orthogonality when a column is
    nearly dependent.
    """
    cols = [np.asarray(c, dtype=float).ravel() for c in columns]
    prior = [np.asarray(v, dtype=float).ravel() for v in against]
    if not cols:
        return []

    # One matrix for the accepted vectors, so each projection is two BLAS calls
    # instead of a Python loop. The order of the columns still fixes the answer;
    # only the summation inside a dot product can differ between machines, and
    # that moves the basis by round-off and not by a rotation.
    length = cols[0].size
    mat = np.zeros((len(prior) + len(cols), length))
    for i, v in enumerate(prior):
        mat[i] = v
    used = len(prior)

    out = []
    for col in cols:
        v = col.copy()
        scale = np.linalg.norm(v)
        for _ in range(2):
            if used:
                block = mat[:used]
                v -= block.T @ (block @ v)
        norm = np.linalg.norm(v)
        if norm > tol * max(scale, 1.0):
            v /= norm
            mat[used] = v
            used += 1
            out.append(v)
    return out


def section_frame(m):
    """Orthonormal frame (n_monomials, d) of the complement of the relations.

    d = C(m+2,2), which is the dimension of the block section space. The frame
    comes from Gram-Schmidt in a fixed order, so it is the same on every machine.
    """
    n_mono = comb(m + 3, 3)
    rel = relation_columns(m)
    if rel.shape[1] == 0:
        return np.eye(n_mono)
    span = orthonormalize(rel.T)
    eye = np.eye(n_mono)
    kept = orthonormalize((eye[i] for i in range(n_mono)), against=span)
    frame = np.stack(kept, axis=1)
    assert frame.shape == (n_mono, n_mono - rel.shape[1]), frame.shape
    return frame


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

    # Gram-Schmidt in the fixed order of the symmetric basis, for the same
    # reason as in `section_frame`: an SVD is free to rotate inside the image.
    image = pi @ np.stack(sym, axis=1)
    basis = np.stack(orthonormalize(image.T, tol=tol), axis=0)

    # Gram-Schmidt combines nearly dependent columns, and the normalization then
    # amplifies their round-off: the invariance residual came out at 1.5e-9 at
    # k = 38. Pi is a projector, so one pass of project-then-orthonormalize
    # restores invariance, and a few passes reach round-off in both properties.
    for _ in range(3):
        basis = np.stack(orthonormalize(basis @ pi.T, tol=0.0), axis=0)
    return basis.reshape(basis.shape[0], d, d)


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

    def fingerprint(self):
        """A short digest of the basis, so a saved theta names its convention.

        theta means nothing without the basis it belongs to. Save this with the
        fit and compare it on a restart.
        """
        return fingerprint(self)

    def param_slices(self):
        """Slice of the flat parameter vector that belongs to each orbit."""
        out, start = [], 0
        for orb in self.orbits:
            out.append(slice(start, start + orb.basis.shape[0]))
            start += orb.basis.shape[0]
        return out


def fingerprint(struct):
    """A digest of every frame and invariant basis of a structure."""
    digest = hashlib.sha256()
    digest.update(str(struct.k).encode())
    for m in sorted(struct.frames):
        digest.update(np.ascontiguousarray(
            np.round(struct.frames[m], 10)).tobytes())
    for orb in struct.orbits:
        digest.update(str(orb.rep).encode())
        digest.update(np.ascontiguousarray(np.round(orb.basis, 10)).tobytes())
    return digest.hexdigest()[:16]


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
