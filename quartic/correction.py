"""A correction of the Kaehler potential: Phi + psi, with psi of degree 0.

Phi is the potential of the metric so far, for example log p_0 of an HN fit at
degree k_0, with Phi(lambda z) = Phi(z) + d log abs(lambda)^2. The correction is

    psi = n exp(-alpha Phi_D),    n = sigma^* C sigma,

where C runs over the invariant symmetric block matrices of degree k and
alpha = k / d_D, so psi has degree 0. Phi_D sets the weight D = exp(alpha Phi_D):

  * Phi_D = log abs(z)^2 gives D = abs(z)^(2k), the Fubini-Study weight.
  * Phi_D = Phi gives D = p_0^(k/k_0). To first order psi is then the step of
    HN at degree k from the lift of p_0, so the correction contains that step.

In the units of the code, where phi = ddbar log p, the metric is ddbar Phi +
ddbar psi. Phi is fixed, and the fit moves C only. A potential is carried as
the state (Phi, d Phi, ddbar Phi) at each point, and `add_level` adds a fitted
psi to it, so the next level can use the sum as its Phi.

This is the plain FP64 form, for tests at small k.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import analytic, sections

__all__ = ['fs_state', 'base_state', 'psi_parts', 'add_level', 'states',
           'volume_corrected', 'volumes', 'residual_fn']


def fs_state(z):
    """(Phi, d Phi, ddbar Phi) for Phi = log abs(z)^2, of degree 1."""
    r2 = jnp.sum(jnp.real(z) ** 2 + jnp.imag(z) ** 2)
    zbar = jnp.conj(z)
    return (jnp.log(r2), zbar / r2,
            jnp.eye(4) / r2 - jnp.outer(zbar, z) / r2 ** 2)


def base_state(lay0, mats0, z):
    """(Phi, d Phi, ddbar Phi) for Phi = log p_0, of degree k_0, in FP64.

    ddbar Phi uses the centered form of `analytic.phi_analytic`.
    """
    sig, dsig = analytic.section_vectors(lay0, z)
    p, p_i, _, store, _ = analytic._gram_parts_fp(lay0, mats0, sig, dsig)
    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig, dsig):
        b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
        h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
        phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
    return jnp.log(p), p_i / p, phi / p


def psi_parts(lay_c, cmats, z, alpha, pot, pot_i, pot_ij):
    """psi = n exp(-alpha Phi_D), d psi and ddbar psi at z.

    (pot, pot_i, pot_ij) is the state of Phi_D. In a 4 by 4 matrix, [i, j] is
    d_i d_jbar.
    """
    sig, dsig = analytic.section_vectors(lay_c, z)
    n, n_i, n_ij, _, _ = analytic._gram_parts_fp(lay_c, cmats, sig, dsig)
    w = jnp.exp(-alpha * pot)
    a = alpha * pot_i
    cross = jnp.outer(n_i, jnp.conj(a)) + jnp.outer(a, jnp.conj(n_i))
    psi_ij = w * (n_ij - cross + n * (jnp.outer(a, jnp.conj(a)) - alpha * pot_ij))
    return w * n, w * (n_i - n * a), psi_ij


def add_level(state, parts):
    """The state of Phi + psi, from the state of Phi and `psi_parts`."""
    return tuple(s + q for s, q in zip(state, parts))


def states(fn, points, chunk):
    """fn(z) at every point, computed once, because the fit does not move Phi."""
    return jax.jit(lambda x: jax.lax.map(fn, x, batch_size=chunk))(points)


def volume_corrected(lay_c, cmats, z, restriction, omega, state, alpha, fs_weight):
    """v = det(g|X) / OmegaOmegabar for the potential Phi + psi.

    `state` is the state of Phi. `fs_weight` takes Phi_D = log abs(z)^2, and
    otherwise Phi_D = Phi.
    """
    weight = fs_state(z) if fs_weight else state
    _, _, psi_ij = psi_parts(lay_c, cmats, z, alpha, *weight)
    g = restriction @ (state[2] + psi_ij) @ jnp.conj(restriction).T
    return jnp.real(jnp.linalg.det(g)) / omega


def volumes(lay_c, cmats, batch, state, chunk, alpha, fs_weight):
    """v at every point of `batch`, with `state` the state of Phi at each point."""
    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'], state)
    return jax.lax.map(
        lambda a: volume_corrected(lay_c, cmats, *a, alpha, fs_weight), xs,
        batch_size=chunk)


def residual_fn(lay_c, batch, state, chunk, alpha, fs_weight, basis=None):
    """sqrt(w) f as a function of the correction coefficients.

    `basis` is an optional matrix that maps the fit coefficients to the
    coefficients of the invariant matrices, for a basis other than theirs.
    """
    weights = batch['mass'] / jnp.sum(batch['mass'])

    def res(c):
        coeffs = c if basis is None else basis @ c
        cmats = sections.prepare_fp(lay_c, coeffs)
        v = volumes(lay_c, cmats, batch, state, chunk, alpha, fs_weight)
        return jnp.sqrt(weights) * (v / jnp.sum(weights * v) - 1.0)
    return res
