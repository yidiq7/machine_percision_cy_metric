"""A correction of the Kaehler potential: log p_0 + psi, with psi of degree 0.

psi = sigma^* C sigma / abs(z)^(2k), where C runs over the invariant symmetric
block matrices of degree k, so psi runs over the invariant functions of
bidegree (k, k) on X. In the units of the code, where phi = ddbar log p, the
metric is ddbar log p_0 + ddbar psi. p_0 is a fixed HN fit, and the fit moves C
only.

This is the plain FP64 form, for tests at small k.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import analytic, sections

__all__ = ['psi_hessian', 'volume_corrected', 'residual_fn']


def psi_hessian(lay_c, cmats, z):
    """ddbar psi at z, a 4 by 4 matrix with [i, j] = d_i d_jbar psi."""
    sig, dsig = analytic.section_vectors(lay_c, z)
    n, n_i, n_ij, _, _ = analytic._gram_parts_fp(lay_c, cmats, sig, dsig)
    k = lay_c.k
    r2 = jnp.sum(jnp.real(z) ** 2 + jnp.imag(z) ** 2)
    d = r2 ** k
    d_i = k * r2 ** (k - 1) * jnp.conj(z)
    d_ij = (k * r2 ** (k - 1) * jnp.eye(4)
            + k * (k - 1) * r2 ** (k - 2) * jnp.outer(jnp.conj(z), z))
    n_jbar, d_jbar = jnp.conj(n_i), jnp.conj(d_i)
    return (n_ij / d
            - (jnp.outer(n_i, d_jbar) + jnp.outer(d_i, n_jbar)) / d ** 2
            - n * d_ij / d ** 2
            + 2.0 * n * jnp.outer(d_i, d_jbar) / d ** 3)


def volume_corrected(lay0, mats0, lay_c, cmats, z, restriction, omega):
    """v = det(g|X) / OmegaOmegabar for the potential log p_0 + psi."""
    phi0, _ = analytic.phi_analytic(lay0, mats0, z)
    total = phi0 + psi_hessian(lay_c, cmats, z)
    g = restriction @ total @ jnp.conj(restriction).T
    return jnp.real(jnp.linalg.det(g)) / omega


def residual_fn(lay0, theta0, lay_c, batch, chunk, basis=None):
    """sqrt(w) f as a function of the correction coefficients.

    `basis` is an optional matrix that maps the fit coefficients to the
    coefficients of the invariant matrices, for a basis other than theirs.
    """
    mats0 = sections.prepare(lay0, theta0)
    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    weights = batch['mass'] / jnp.sum(batch['mass'])

    def res(c):
        coeffs = c if basis is None else basis @ c
        cmats = sections.prepare_fp(lay_c, coeffs)
        v = jax.lax.map(lambda a: volume_corrected(lay0, mats0, lay_c, cmats, *a),
                        xs, batch_size=chunk)
        return jnp.sqrt(weights) * (v / jnp.sum(weights * v) - 1.0)
    return res
