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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import blocks

__all__ = ['PAIRS', 'OrbitRecurrence', 'Basis', 'multipliers', 'permuted',
           'block_factor', 'run_recurrence', 'build_orbit', 'build_basis',
           'orbit_images', 'values', 'restricted_hessian_of',
           'restricted_hessians']

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
