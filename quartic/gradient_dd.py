"""`grad f` on X in double-double, by central differences along X.

`gradient.py` differentiates the FP64 arithmetic with autodiff. `dd.split` is a
bitcast and carries no derivative, so that path cannot use dd, and at k = 32 the
FP64 arithmetic has an error larger than f.

This module moves each dd point of `dd_metric.point_extras` along X, from t_i
to t_i + h with h = eps or i*eps, and puts it back on X with one Newton step in
dd. v there comes from the END_DD path without the final rounding. v is real,
so the central difference over h gives 2 Re(h dv/dt_i) / eps.

eps is a power of 2, so the step is exact in dd. The truncation error is
eps^2 abs(d^3 v) / 6, and the rounding error is the dd error of v over 2 eps.
Two values of eps bound both.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from . import analytic, dd, dd_metric, sections

__all__ = ['moved_point', 'volume_keep', 'volume_gradient', 'gradient_report_dd']


def _times_h(val, eps, rotate):
    """`val * h` for a complex dd value, with h = eps, or i*eps if `rotate`."""
    re = (val[0][0] * eps, val[0][1] * eps)
    im = (val[1][0] * eps, val[1][1] * eps)
    return (dd.neg(im), re) if rotate else (re, im)


def moved_point(z_dd, m, free, i, eps, rotate):
    """The point of X at t_i + h, from the tangent line and one Newton step in dd."""
    rest, _ = dd_metric._restriction_dd(z_dd, m, free)
    line = dd.cadd(z_dd, _times_h(analytic._cindex(rest, i), eps, rotate))
    return dd_metric.newton_step(line, m)


def volume_keep(lay, mats, z_dd, m, free):
    """v at a dd point by the END_DD path as a dd pair, and phi in FP64."""
    sig, dsig = analytic.section_vectors_dd(lay, None, z_dd=z_dd)
    extras = analytic._gram_parts(lay, mats, sig, dsig)[4]
    phi = analytic._phi_stage5_dd(extras, sig, dsig, keep=True)
    return (analytic._volume_from_phi_dd(phi, z_dd, m, free, keep=True),
            dd.cto_fp(phi))


def volume_gradient(lay, mats, z_dd, m, free, eps):
    """v as a dd pair, dv/dt_i for i = 0 and 1, and g|X in FP64, at one point.

    A step of h = eps gives 2 Re(dv/dt_i) and a step of h = i*eps gives
    -2 Im(dv/dt_i), therefore dv/dt_i = (d1 - i*d2)/2.
    """
    v0, phi = volume_keep(lay, mats, z_dd, m, free)
    moved = [moved_point(z_dd, m, free, i, sign * eps, rotate)
             for i in range(2) for rotate in (False, True) for sign in (1.0, -1.0)]
    stacked = jtu.tree_map(lambda *a: jnp.stack(a), *moved)
    hi, lo = jax.vmap(lambda zz: volume_keep(lay, mats, zz, m, free)[0])(stacked)
    diff = dd.add((hi[0::2], lo[0::2]), dd.neg((hi[1::2], lo[1::2])))
    d = dd.to_fp(diff) / (2.0 * eps)
    dv = 0.5 * (d[0::2] - 1j * d[1::2])
    rest = dd.cto_fp(dd_metric._restriction_dd(z_dd, m, free)[0])
    return v0, dv, rest @ phi @ jnp.conj(rest).T


def gradient_report_dd(lay, theta, batch, eps=2.0 ** -14, chunk=32,
                       volume=2.0 * np.pi ** 2):
    """`norm(f)_L2`, `norm(grad f)_L2` and their ratio, with the whole path in dd.

    The conventions are those of `gradient.gradient_report`:
    `abs(grad f)^2 = g^{i jbar} d_i f conj(d_j f)`, and the ratio does not depend
    on the volume normalization. `batch` must carry the dd points of
    `dd_metric.point_extras`. f comes from v in dd, so it keeps its digits at
    abs(f) near 1e-15, where the FP64 rounding of v is 10 percent of f.
    """
    if not (analytic.END_DD and analytic.GRAM_DD is True):
        raise RuntimeError('gradient_report_dd needs analytic.END_DD and '
                           'analytic.GRAM_DD = True')
    mats = sections.prepare_parts(lay, theta)

    def one(args):
        hi, lo, row = args
        z_dd = ((jnp.real(hi), jnp.real(lo)), (jnp.imag(hi), jnp.imag(lo)))
        return volume_gradient(lay, mats, z_dd, row[0], (row[1], row[2]), eps)

    xs = (batch['points_hi'], batch['points_lo'], batch['patch'])
    v, dv, g = jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(xs)

    mass = jnp.asarray(batch['mass'])
    weight = mass / jnp.sum(mass)
    mean_v = dd.dot_dd_fp(v, weight, axis=0)
    ratio = dd.mul(mean_v, dd.reciprocal(v))
    f = dd.to_fp(dd.add(dd.to_dd(jnp.ones_like(v[0])), dd.neg(ratio)))
    v_fp, mean_fp = dd.to_fp(v), dd.to_fp(mean_v)
    # d_i f = <v> d_i v / v^2
    df = mean_fp * dv / v_fp[:, None] ** 2
    grad_sq = jnp.real(jnp.einsum('nij,ni,nj->n', jnp.linalg.inv(g), df, jnp.conj(df)))

    l2_f = float(jnp.sqrt(volume * jnp.sum(weight * f ** 2)))
    l2_df = float(jnp.sqrt(volume * jnp.sum(weight * grad_sq)))
    return {
        'mean_abs_f': float(jnp.sum(weight * jnp.abs(f))),
        'rms_f': float(jnp.sqrt(jnp.sum(weight * f ** 2))),
        'l2_f': l2_f,
        'l2_grad_f': l2_df,
        'ratio': l2_df / l2_f,
        'volume': float(volume),
        'eps': float(eps),
        'n_points': int(f.shape[0]),
        'grad_sq': np.asarray(grad_sq),
    }
