"""`grad f` on X, and the ratio `R = norm(grad f)_L2 / norm(f)_L2`.

This is task 2. The perturbation argument of the collaborators states its
requirement in the `H1` norm, and we report a mean, so `R` is the only quantity
that converts one to the other.

`f = 1 - <v>/v`, so `d_i f = <v> d_i v / v^2`. The whole difficulty is `d_i v`,
because `v = det(g|X) / OmegaOmegabar` and **both the restriction matrix and
OmegaOmegabar change along X**. The dataset holds them as constants for each
point, so they cannot be differentiated there. Two closed forms remove that,
and each was measured against the sampler on 800 points:

  * `OmegaOmegabar = 1 / abs(dP/dz_m)^2`, to 5.6e-16,
  * `R[i, m] = -(dP/dt_i)/(dP/dz_m)`, exactly,

where `j` is the coordinate that the affine patch fixes to 1, `m` is the
coordinate with the largest `abs(dP/dz)` among the others, and `t` holds the
remaining two. `m` is the choice that the sampler makes.

X is a graph over the 2 local complex coordinates `t`, so a tangent direction is
one row of the restriction matrix. `v` needs no extension off X for a tangential
derivative, so the closed forms above serve at the point itself.

**The arithmetic is plain FP64 here.** Through a dd path the tangent is FP64
arithmetic, because `dd.split` gives hi a zero tangent and lo = a - hi carries
all of it, so dd would not make grad f more accurate. FP64 is enough to k = 20,
where sigma is 6.3e-11 against a floor of 4.4e-15. `gradient_dd` serves above
that.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import analytic

__all__ = ['patch_indices', 'volume_at', 'volume_tangents', 'grad_norm_sq',
           'gradient_report']


def patch_indices(batch):
    """`(j, m, free)` for every point of a batch, as integer arrays.

    `j` fixes the affine patch, `m` is the coordinate that `P = 0` eliminates,
    and `free` holds the two that are left, in increasing order. They come from
    the restriction matrix of the batch, so they match the sampler exactly.
    """
    z = np.asarray(batch['points'])
    rest = np.asarray(batch['restriction'])
    n = z.shape[0]
    j = np.argmin(np.abs(z - 1.0), axis=1)
    unit = np.isclose(np.abs(rest), 1.0)
    j_out, m_out, free_out = np.empty(n, int), np.empty(n, int), np.empty((n, 2), int)
    for i in range(n):
        cols = np.where(unit[i].any(axis=0))[0]
        rest_cols = sorted(set(range(4)) - {int(j[i])} - set(cols.tolist()))
        assert len(cols) == 2 and len(rest_cols) == 1, (i, cols, rest_cols)
        j_out[i], m_out[i], free_out[i] = j[i], rest_cols[0], sorted(cols.tolist())
    return j_out, m_out, free_out


def _restriction(z, m, free):
    """The (2, 4) restriction matrix from z, in closed form.

    Row i is `dz/dt_i`. It carries a 1 in the free slot of `t_i` and
    `-(dP/dt_i)/(dP/dz_m)` in slot `m`, which is the implicit function theorem
    applied to `P = 0`.
    """
    grad = 4.0 * z ** 3
    inv_m = 1.0 / grad[m]
    rows = []
    for i in range(2):
        row = jnp.zeros(4, dtype=z.dtype)
        row = row.at[free[i]].set(1.0)
        row = row.at[m].set(-grad[free[i]] * inv_m)
        rows.append(row)
    return jnp.stack(rows)


def volume_at(lay, mats, z, m, free):
    """`v = det(g|X) / OmegaOmegabar` from z alone, with no stored constants."""
    rest = _restriction(z, m, free)
    phi, _ = analytic.phi_analytic(lay, mats, z)
    g = rest @ phi @ jnp.conj(rest).T
    omega = 1.0 / jnp.abs(4.0 * z[m] ** 3) ** 2
    return jnp.real(jnp.linalg.det(g)) / omega


def volume_tangents(lay, mats, z, m, free):
    """`v`, `dv/dt_i` for i = 0 and 1, and the restricted metric at one point.

    `v` is real and depends on z and its conjugate, so one complex derivative
    needs two real directional derivatives. A step of `eps*T` gives
    `2 Re(T.dv)` and a step of `i*eps*T` gives `-2 Im(T.dv)`, therefore
    `T.dv = (d1 - i*d2)/2`.
    """
    def real_form(parts):
        return volume_at(lay, mats, parts[0] + 1j * parts[1], m, free)

    base = (jnp.real(z), jnp.imag(z))
    rest = _restriction(z, m, free)
    phi, _ = analytic.phi_analytic(lay, mats, z)
    g = rest @ phi @ jnp.conj(rest).T

    out = []
    for i in range(2):
        tangent = rest[i]
        v, d1 = jax.jvp(real_form, (base,),
                        ((jnp.real(tangent), jnp.imag(tangent)),))
        _, d2 = jax.jvp(real_form, (base,),
                        ((-jnp.imag(tangent), jnp.real(tangent)),))
        out.append(0.5 * (d1 - 1j * d2))
    return v, jnp.stack(out), g


def grad_norm_sq(g, df):
    """`g^{i jbar} d_i f conj(d_j f)` for a stack of metrics g[i, j] = g_{i jbar}.

    g^{i jbar} is the transpose of inv(g), so the sum is conj(df)^T inv(g) df. For
    the Riemannian metric Re(g) the squared length of grad f is 4 times this.
    """
    return jnp.real(jnp.einsum('ni,nij,nj->n', jnp.conj(df), jnp.linalg.inv(g), df))


def gradient_report(lay, theta, batch, chunk=None, volume=2.0 * np.pi ** 2):
    """`norm(f)_L2`, `norm(grad f)_L2` and their ratio on a point set.

    `abs(grad f)^2 = g^{i jbar} d_i f conj(d_j f)`, which is the form that pairs
    with `Delta = g^{i jbar} d_i d_jbar` under integration by parts, and so it is
    the convention of the draft. A real Riemannian gradient gives 2 times this.

    Both norms carry the same factor of the volume, so **the ratio does not
    depend on the normalization**. That closes open question 2 for the ratio.
    """
    from . import sections

    if analytic.GRAM_DD or analytic.SECTION_DD:
        raise RuntimeError('gradient_report is the plain FP64 path. Set '
                           'analytic.GRAM_DD and analytic.SECTION_DD to False, '
                           'or use gradient_dd.gradient_report_dd.')

    mats = sections.prepare(lay, theta)
    j, m, free = patch_indices(batch)
    z = jnp.asarray(batch['points'])
    mass = jnp.asarray(batch['mass'])
    weight = mass / jnp.sum(mass)

    def one(args):
        zi, mi, fi = args
        return volume_tangents(lay, mats, zi, mi, fi)

    xs = (z, jnp.asarray(m), jnp.asarray(free))
    if chunk is None:
        v, dv, g = jax.vmap(one)(xs)
    else:
        v, dv, g = jax.lax.map(one, xs, batch_size=chunk)

    mean_v = jnp.sum(weight * v)
    f = 1.0 - mean_v / v
    # d_i f = <v> d_i v / v^2
    df = mean_v * dv / v[:, None] ** 2
    grad_sq = grad_norm_sq(g, df)

    l2_f = float(jnp.sqrt(volume * jnp.sum(weight * f ** 2)))
    l2_df = float(jnp.sqrt(volume * jnp.sum(weight * grad_sq)))
    return {
        'mean_abs_f': float(jnp.sum(weight * jnp.abs(f))),
        'rms_f': float(jnp.sqrt(jnp.sum(weight * f ** 2))),
        'l2_f': l2_f,
        'l2_grad_f': l2_df,
        'ratio': l2_df / l2_f,
        'volume': float(volume),
        'n_points': int(z.shape[0]),
    }
