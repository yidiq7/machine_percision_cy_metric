"""An orthonormal basis on X of the functions of one correction level, by
Vandermonde with Arnoldi (V+A), so that no step forms the product basis.

For an orbit with representative residue r and degree m, the functions are

    f_q(z) = w(z) |u|^(2m) mean over pi in S_4 of |z_pi^r|^2 q(u_pi),

where z_pi permutes the coordinates of z, u = z^4, w is the weight of the
level, and q runs over the real (m, m) forms of u on the hyperplane sum u = 0,
written as functions of degree 0. Products of at most m factors

    y_ab(u) = Re(v_a conj(v_b)) / |v|^2,    v = F_1^T u,

span exactly those forms, as the polynomials of degree at most m do in the
univariate method. The Arnoldi recurrence runs in the full space of q, with an
inner product that weights each permuted copy as its block does, so every q
keeps its own digits. The S_4 mean then maps that basis onto the invariant
functions of the orbit, and an SVD on the sample gives their orthonormal basis.
A QR over all orbits gives the basis of the level.

Each step of the recurrence is q_t = (y_ab q_parent - sum_l h_lt q_l) / h_tt,
with the coefficients that the build found, so an evaluation at any point
repeats the build exactly and forms no large coefficient.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import blocks

__all__ = ['PAIRS', 'OrbitRecurrence', 'Basis', 'multipliers', 'permuted',
           'block_factor', 'run_recurrence', 'build_orbit', 'build_basis',
           'orbit_images', 'values', 'restricted_hessian_of',
           'restricted_hessians', 'dictionary', 'dictionary_values', 'JointBasis',
           'build_joint', 'joint_values', 'joint_hessians']

# The six products v_a conj(v_b) with a <= b, in a fixed order.
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
PERMS = np.array(blocks.S4, dtype=np.int32)                        # (24, 4)
FRAME_1 = np.asarray(blocks.section_frame(1))                      # (4, 3)


@dataclass(frozen=True)
class OrbitRecurrence:
    rep: tuple            # the representative residue r
    m: int                # the degree in u
    parents: np.ndarray   # (D,) int, the parent of each q; -1 for q_0
    mults: np.ndarray     # (D,) int, the index of y in PAIRS; -1 for q_0
    h: np.ndarray         # (D, D), h[l, t] for l < t, and h[t, t] the norm
    mix: np.ndarray       # (D, n_orbit), the images f_q to the orthonormal basis


@dataclass(frozen=True)
class Basis:
    orbits: tuple         # OrbitRecurrence of each orbit, in the order of the level
    transform: np.ndarray  # (n, n), the QR over all orbits
    alpha: float          # the weight is exp(-alpha Phi)


def multipliers(u):
    """y_ab(u) for the six pairs, as an array (..., 6)."""
    v = jnp.einsum('...c,ca->...a', u, jnp.asarray(FRAME_1))
    norm = jnp.sum(jnp.real(v) ** 2 + jnp.imag(v) ** 2, axis=-1)
    outs = [jnp.real(v[..., a] * jnp.conj(v[..., b])) for a, b in PAIRS]
    return jnp.stack(outs, axis=-1) / norm[..., None]


def permuted(z):
    """z with its coordinates permuted by every element of S_4, as (..., 24, 4)."""
    return z[..., jnp.asarray(PERMS)]


def block_factor(zp, rep):
    """|z^r|^2 at permuted points, with integer powers, as (..., 24)."""
    abs2 = jnp.real(zp) ** 2 + jnp.imag(zp) ** 2
    out = jnp.ones(abs2.shape[:-1], abs2.dtype)
    for a, power in enumerate(rep):
        for _ in range(int(power)):
            out = out * abs2[..., a]
    return out


def run_recurrence(rec, y):
    """Every q_t at points with multipliers y (..., 6), as (..., D)."""
    d = rec.h.shape[0]
    h = jnp.asarray(rec.h)
    parents = jnp.asarray(np.maximum(rec.parents, 0))
    mults = jnp.asarray(np.maximum(rec.mults, 0))
    q = jnp.zeros(y.shape[:-1] + (d,), y.dtype)
    q = q.at[..., 0].set(1.0 / h[0, 0])

    def step(t, q):
        grown = y[..., mults[t]] * q[..., parents[t]]
        # q_l is still zero for l >= t, so the full column is the sum over l < t.
        return q.at[..., t].set((grown - q @ h[:, t]) / h[t, t])
    return jax.lax.fori_loop(1, d, step, q)


def orbit_images(rec, z):
    """mean over pi of |z_pi^r|^2 q_t(u_pi) at points z (..., 4), as (..., D).

    The weight w(z) |u|^(2m) is common to the orbit, and the caller adds it.
    """
    zp = permuted(z)
    u = zp ** 4
    q = run_recurrence(rec, multipliers(u))                        # (..., 24, D)
    return jnp.mean(block_factor(zp, rec.rep)[..., None] * q, axis=-2)


def _orthogonalize(cols, weights, cand):
    """Two passes of classical Gram-Schmidt of `cand` against `cols`, weighted."""
    if cols.shape[1] == 0:
        return cand, np.zeros(0)
    coef = cols.T @ (weights * cand)
    cand = cand - cols @ coef
    again = cols.T @ (weights * cand)
    cand = cand - cols @ again
    return cand, coef + again


def build_orbit(rep, m, z, scale, mu, tol=1e-9):
    """The recurrence of one orbit on a sample, and the orthonormal images.

    `scale` is w(z) |u|^(2m) at each sample point and `mu` its measure. The
    inner product of q weights each permuted copy by mu (scale |z_pi^r|^2)^2.
    Returns the recurrence and the dimension that the sample sees.
    """
    zp = np.asarray(permuted(jnp.asarray(z)))
    y = np.asarray(multipliers(jnp.asarray(zp ** 4)))              # (N, 24, 6)
    factor = np.asarray(block_factor(jnp.asarray(zp), rep))        # (N, 24)
    weight = (mu[:, None] * (scale[:, None] * factor) ** 2).ravel()
    y = y.reshape(-1, 6)

    ones = np.ones(y.shape[0])
    norm0 = np.sqrt(np.sum(weight * ones * ones))
    cols = (ones / norm0)[:, None]
    parents, mults, coefs, norms = [-1], [-1], [np.zeros(0)], [norm0]
    frontier = [0]
    for _ in range(m):
        new = []
        for parent in frontier:
            for ab in range(len(PAIRS)):
                cand = y[:, ab] * cols[:, parent]
                size = np.sqrt(np.sum(weight * cand * cand))
                cand, coef = _orthogonalize(cols, weight, cand)
                rest = np.sqrt(np.sum(weight * cand * cand))
                if rest <= tol * size:
                    continue
                cols = np.concatenate([cols, (cand / rest)[:, None]], axis=1)
                parents.append(parent)
                mults.append(ab)
                coefs.append(coef)
                norms.append(rest)
                new.append(cols.shape[1] - 1)
        frontier = new

    d = cols.shape[1]
    h = np.zeros((d, d))
    for t in range(d):
        h[:t, t] = coefs[t]
        h[t, t] = norms[t]
    return OrbitRecurrence(rep=tuple(int(a) for a in rep), m=int(m),
                           parents=np.array(parents), mults=np.array(mults), h=h,
                           mix=np.zeros((d, 0)))


def build_basis(struct, z, weight, mu, alpha, tol=1e-9):
    """The V+A basis of a level on a sample, with its checks.

    `weight` is w(z) at each sample point. Returns the basis and a report with,
    for each orbit, the dimension of q, the rank of the images and the gap of
    their singular values, and the condition number of the QR over all orbits.
    """
    mu = np.asarray(mu) / np.sum(mu)
    u_abs2 = np.sum(np.abs(np.asarray(z)) ** 8, axis=1)
    root = np.sqrt(mu)
    recs, images, report = [], [], []
    for orb in struct.orbits:
        scale = np.asarray(weight) * u_abs2 ** orb.m
        rec = build_orbit(orb.rep, orb.m, z, scale, mu, tol=tol)
        f = scale[:, None] * np.asarray(orbit_images(rec, jnp.asarray(z)))
        _, s, vt = np.linalg.svd(root[:, None] * f, full_matrices=False)
        n = orb.basis.shape[0]
        mix = vt[:n].T / s[:n]
        recs.append(OrbitRecurrence(rep=rec.rep, m=rec.m, parents=rec.parents,
                                    mults=rec.mults, h=rec.h, mix=mix))
        images.append(f @ mix)
        report.append({'rep': rec.rep, 'm': rec.m, 'dim_q': rec.h.shape[0],
                       'dim_q_expected': (lambda d: d * (d + 1) // 2)(
                           (orb.m + 1) * (orb.m + 2) // 2),
                       'n_params': n, 's_last_kept': float(s[n - 1] / s[0]),
                       's_first_dropped': float(s[n] / s[0]) if s.size > n else 0.0})
    stacked = np.concatenate(images, axis=1)
    _, r = np.linalg.qr(root[:, None] * stacked)
    sing = np.linalg.svd(r, compute_uv=False)
    transform = np.linalg.solve(r, np.eye(r.shape[0]))
    basis = Basis(orbits=tuple(recs), transform=transform, alpha=float(alpha))
    return basis, {'orbits': report, 'cond_across_orbits': float(sing[0] / sing[-1])}


def values(basis, z, weight):
    """Every basis function of the level at points z (..., 4), as (..., n)."""
    u_abs2 = jnp.sum((jnp.real(z) ** 2 + jnp.imag(z) ** 2) ** 4, axis=-1)
    parts = []
    for rec in basis.orbits:
        scale = weight * u_abs2 ** rec.m
        parts.append((scale[..., None] * orbit_images(rec, z)) @ jnp.asarray(rec.mix))
    return jnp.concatenate(parts, axis=-1) @ jnp.asarray(basis.transform)


def restricted_hessian_of(fn, z, restriction):
    """R ddbar f R^dag at one point for real functions fn(point) -> (n,), as (n, 2, 2).

    The real Hessian along T_k and i T_k, with T_k the rows of R, gives
    d_T dbar_S f = (D_S D_T + D_iS D_iT - i (D_S D_iT - D_iS D_T)) / 4.
    """
    dirs = jnp.stack([restriction[0], 1j * restriction[0],
                      restriction[1], 1j * restriction[1]])           # (4, 4)
    # Forward over forward, because reverse mode keeps every step of the recurrence.
    hess = jax.jacfwd(jax.jacfwd(lambda s: fn(z + s @ dirs)))(jnp.zeros(4))   # (n, 4, 4)
    out = []
    for k in range(2):
        row = []
        for l in range(2):
            t, it, s, is_ = 2 * k, 2 * k + 1, 2 * l, 2 * l + 1
            re = 0.25 * (hess[:, s, t] + hess[:, is_, it])
            im = -0.25 * (hess[:, s, it] - hess[:, is_, t])
            row.append(re + 1j * im)
        out.append(jnp.stack(row, axis=-1))
    return jnp.stack(out, axis=-2)


def restricted_hessians(basis, z, restriction, weight_fn):
    """R ddbar G R^dag for every basis function at one point, as (n, 2, 2).

    `weight_fn(z)` gives w at a point, so its derivatives enter too.
    """
    return restricted_hessian_of(lambda p: values(basis, p, weight_fn(p)), z,
                                 restriction)


# The joint recurrence over the whole invariant space of a level.
#
# The orbits of one level overlap on X: functions of different residues are all
# positive and smooth there, and the QR across orbits had a condition number of
# 1.1e5, 8.9e9 and 3.3e12 at k = 8, 16 and 20. So the recurrence must move
# between residues. Each multiplier is itself invariant: the S_4 sum of a
# balanced monomial prod_a z_a^A_a conj(z_a)^B_a with A_a = B_a mod 4 and
# |A| = |B| = d, real part, over |z|^(2d). Every function of the recurrence is
# then invariant, and the level is the span of the functions of degree <= k.


def dictionary(d_max):
    """The invariant multipliers of degree 1 to d_max, as (d, A, B) with A, B (4,).

    One entry for each orbit of (A, B) under S_4 and the swap of A and B.
    """
    keys, out = set(), []
    for d in range(1, d_max + 1):
        pairs = [(a, b) for a in range(d + 1) for b in range(d + 1) if (a - b) % 4 == 0]
        for combo in itertools.product(pairs, repeat=4):
            if sum(c[0] for c in combo) != d or sum(c[1] for c in combo) != d:
                continue
            key = tuple(sorted(combo, reverse=True))
            swap = tuple(sorted(((b, a) for a, b in combo), reverse=True))
            key = min(key, swap)
            if key in keys:
                continue
            keys.add(key)
            out.append((d, np.array([c[0] for c in key]), np.array([c[1] for c in key])))
    return out


def dictionary_values(entries, z):
    """Every multiplier of `entries` at points z (..., 4), as (..., n_entries)."""
    zp = permuted(z)                                                 # (..., 24, 4)
    top = max(max(int(a.max()), int(b.max())) for _, a, b in entries)
    pw = [jnp.ones_like(zp)]
    for _ in range(top):
        pw.append(pw[-1] * zp)
    pw = jnp.stack(pw, axis=-1)                                      # (..., 24, 4, top+1)
    cpw = jnp.conj(pw)
    r2 = jnp.sum(jnp.real(z) ** 2 + jnp.imag(z) ** 2, axis=-1)
    outs = []
    for d, a_exp, b_exp in entries:
        term = jnp.ones(zp.shape[:-1], zp.dtype)
        for c in range(4):
            term = term * pw[..., c, int(a_exp[c])] * cpw[..., c, int(b_exp[c])]
        outs.append(jnp.real(jnp.sum(term, axis=-1)) / r2 ** d)
    return jnp.stack(outs, axis=-1)


@dataclass(frozen=True)
class JointBasis:
    k: int
    entries: tuple        # the multipliers (d, A, B) that the recurrence uses
    parents: np.ndarray   # (n,) int, the parent of each q; -1 for q_0
    mults: np.ndarray     # (n,) int, the index of the multiplier; -1 for q_0
    degrees: np.ndarray   # (n,) int, the degree of each q
    h: np.ndarray         # (n, n), h[l, t] for l < t, and h[t, t] the norm
    alpha: float          # the weight is exp(-alpha Phi)


def build_joint(k, z, scale, mu, d_max=8, tol=1e-9, n_target=None):
    """The joint recurrence of a level of degree k on a sample.

    `scale` is w(z) |z|^(2k) at each sample point, so the functions of the level
    are scale times q, and q is orthonormal in the inner product of mu scale^2.
    Candidates are the products of a q of degree j with a multiplier of degree d,
    for j + d <= k, in the order of j + d. The build stops at `n_target`.
    Returns the recurrence and the dimension that it reached.
    """
    entries = dictionary(d_max)
    mvals = np.asarray(dictionary_values(entries, jnp.asarray(z)))   # (N, n_entries)
    degs = np.array([d for d, _, _ in entries])
    weight = np.asarray(mu) * np.asarray(scale) ** 2
    ones = np.ones(z.shape[0])
    norm0 = np.sqrt(np.sum(weight * ones * ones))
    cols = (ones / norm0)[:, None]
    parents, mults, degrees, coefs, norms = [-1], [-1], [0], [np.zeros(0)], [norm0]
    for target in range(1, k + 1):
        for e in np.nonzero(degs <= target)[0]:
            for parent in [t for t in range(len(degrees)) if degrees[t] == target - degs[e]]:
                cand = mvals[:, e] * cols[:, parent]
                size = np.sqrt(np.sum(weight * cand * cand))
                cand, coef = _orthogonalize(cols, weight, cand)
                rest = np.sqrt(np.sum(weight * cand * cand))
                if rest <= tol * size:
                    continue
                cols = np.concatenate([cols, (cand / rest)[:, None]], axis=1)
                parents.append(parent)
                mults.append(int(e))
                degrees.append(target)
                coefs.append(coef)
                norms.append(rest)
                if n_target is not None and cols.shape[1] >= n_target:
                    break
            if n_target is not None and cols.shape[1] >= n_target:
                break
        if n_target is not None and cols.shape[1] >= n_target:
            break
    n = cols.shape[1]
    h = np.zeros((n, n))
    for t in range(n):
        h[:t, t] = coefs[t]
        h[t, t] = norms[t]
    used = sorted(set(m for m in mults if m >= 0))
    remap = {m: i for i, m in enumerate(used)}
    return JointBasis(k=int(k), entries=tuple(entries[m] for m in used),
                      parents=np.array(parents),
                      mults=np.array([remap[m] if m >= 0 else -1 for m in mults]),
                      degrees=np.array(degrees), h=h, alpha=0.0)


def joint_values(basis, z, weight):
    """Every function of the level at points z (..., 4), as (..., n).

    `weight` is w(z), and the functions are w |z|^(2k) q.
    """
    y = dictionary_values(list(basis.entries), z)
    q = run_recurrence(basis, y)
    r2 = jnp.sum(jnp.real(z) ** 2 + jnp.imag(z) ** 2, axis=-1)
    return (weight * r2 ** basis.k)[..., None] * q


def joint_hessians(basis, z, restriction, weight_fn):
    """R ddbar G R^dag for every function of the level at one point, as (n, 2, 2)."""
    return restricted_hessian_of(lambda p: joint_values(basis, p, weight_fn(p)), z,
                                 restriction)
