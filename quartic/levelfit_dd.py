"""Helper functions of the level fits in dd: the row of psi_N, f for each column of coefficients,
and the values of the level functions.
"""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from . import analytic, arnoldi_dd, dd
from . import levelfit as vf


def with_null(p, null):
    """The Hessians with row b0 replaced by psi_N = sum over b < n_low of gamma_b G_b."""
    if null is None:
        return p, None
    n_low, gamma, b0 = null
    g = jnp.asarray(gamma)[:, None, None]
    p_n = tuple(dd.dot_dd_fp(jax.tree_util.tree_map(lambda x: x[:n_low], comp), g, axis=0)
                for comp in p)
    return jax.tree_util.tree_map(lambda x, y: x.at[b0].set(y), p, p_n), p_n


def residuals(program, coeffs, batch, block):
    """f in dd, rounded once, for each column of coeffs, as (N, n_c), in blocks of points.

    Every block has `block` points, the last one padded with its own last point, so
    the program compiles once for all point sets.
    """
    inputs = vf.point_inputs(batch)
    total = batch['mass'].shape[0]
    his, los = [], []
    for start in range(0, total, block):
        stop = min(start + block, total)
        take = jnp.minimum(jnp.arange(start, start + block), stop - 1)
        part = jax.tree_util.tree_map(lambda a: a[take], inputs)
        v = program(part, jnp.asarray(coeffs))
        his.append(v[0][:stop - start])
        los.append(v[1][:stop - start])
    v = (jnp.concatenate(his), jnp.concatenate(los))
    mass = batch['mass'] / jnp.sum(batch['mass'])
    return np.asarray(analytic.residual_from_volumes(v, mass[:, None]))


def dd_values(basis, h_lo, uinv, weight, z, chunk):
    """W q of every function at FP64 points z, from the dd jets, rounded once."""
    rest = dd.cto_dd(jnp.zeros((2, 4), jnp.complex128).at[0, 1].set(1.0).at[1, 2].set(1.0))

    def one(a, h, hl):
        p, w = a
        lev = dataclasses.replace(basis, h=h)
        jets = arnoldi_dd.joint_jets_dd(
            lev, hl, arnoldi_dd.vpa_dd.basis_multiplier_jets(lev, dd.cto_dd(p), rest), uinv)
        return w * jnp.sum(jnp.abs(p) ** 2) ** basis.k * dd.to_fp(jets[0])
    return jax.jit(lambda x, h, hl: jax.lax.map(lambda a: one(a, h, hl), x,
                                                batch_size=chunk))(
        (z, weight), jnp.asarray(basis.h), jnp.asarray(h_lo))
