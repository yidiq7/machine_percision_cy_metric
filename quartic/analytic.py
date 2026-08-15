"""Analytic phi and analytic dv/dtheta, as an alternative to nested autodiff.

The measured cost of the present path is the reason for this module. One
residual costs approximately 50 times one evaluation of p, because
`complex_hessian` differentiates through an 8-dimensional real split. One
`jacfwd` over theta then costs approximately 1.5 to 1.9 times n_theta residuals,
because forward mode carries n_theta independent tangents through all of it.

Both factors come from structure that autodiff cannot see:

* p, dp and ddbar-p are Gram sums of one holomorphic section vector sigma(z) and
  its 4 derivatives. Only 4 tangents are necessary, and not an 8 by 8 Hessian.
* p is linear in theta, so dv/dtheta_alpha contracts to one Frobenius product
  against a single matrix for each block, and not a full pass for each alpha.

Nothing here changes the model. `phi_analytic` and `residual_analytic` must
agree with `metric` to round-off, and the tests check that.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import sections

__all__ = ['section_vectors', 'phi_analytic', 'volumes_analytic',
           'residual_analytic', 'weighted_residual_analytic',
           'volume_jacobian', 'residual_jacobian']


def section_vectors(lay, z):
    """The holomorphic section vector of each block, and its 4 derivatives.

    Block r holds sigma_r = z^r * psi_r, where psi_r is the section frame
    applied to the monomials of u = z^4. The derivatives come from 4 tangents
    on this cheap holomorphic map, and not from an 8 by 8 real Hessian.
    """
    def sigma(w):
        z2 = w * w
        u = z2 * z2
        out = []
        for g in lay.groups:
            psi = jnp.asarray(g.frame).T @ sections._monomials(
                u, jnp.asarray(g.exps), g.m)
            powers = sections._integer_powers(w, 3)          # r_a is 0 to 3
            weights = jnp.prod(powers[jnp.asarray(g.residues), jnp.arange(4)],
                               axis=1)
            out.append(weights[:, None] * psi[None, :])      # (n_blocks, d)
        return tuple(out)

    return sigma(z), jax.jacfwd(sigma, holomorphic=True)(z.astype(jnp.complex128))


def _gram_parts(lay, mats, sig, dsig):
    """p, dp and ddbar-p, from the Gram sums over every block.

    `H sigma` and the four `H dsigma_i` are the only matrix-vector products, so
    one point costs 5 of them for each block, in place of approximately 50.
    """
    p = jnp.zeros((), jnp.float64)
    p_i = jnp.zeros((4,), jnp.complex128)
    p_ij = jnp.zeros((4, 4), jnp.complex128)
    store = []

    for mat, s, ds in zip(mats, sig, dsig):
        h_s = jnp.einsum('bij,bj->bi', mat, s)               # (n_blocks, d)
        h_d = jnp.einsum('bij,bjk->bik', mat, ds)            # (n_blocks, d, 4)
        p = p + jnp.real(jnp.einsum('bi,bi->', jnp.conj(s), h_s))
        p_i = p_i + jnp.einsum('bi,bik->k', jnp.conj(s), h_d)
        # Entry [i, j] is d_i d_jbar p, so the conjugated index comes second.
        p_ij = p_ij + jnp.einsum('bil,bik->kl', jnp.conj(ds), h_d)
        store.append((h_s, h_d))
    return p, p_i, p_ij, store


def phi_analytic(lay, mats, z):
    """phi_{i jbar} = d_i d_jbar log p, in the centered form.

    The difference form (p_ij p - p_i p_jbar)/p^2 cancels by a factor of k. The
    centered vectors b_i = dsigma_i - (p_i/p) sigma make the result a Gram
    matrix, so the cancellation goes into a vector subtraction instead.
    """
    sig, dsig = section_vectors(lay, z)
    p, p_i, _, store = _gram_parts(lay, mats, sig, dsig)

    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig, dsig):
        # b_i = dsigma_i - (p_i/p) sigma, and H b_i reuses the products above.
        b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
        h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
        phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
    return phi / p, p


def volumes_analytic(lay, theta, batch, chunk=None):
    """v = det(g|_X) / OmegaOmegabar, with phi from `phi_analytic`."""
    mats = sections.prepare(lay, theta)

    def one(args):
        z, restriction, omega = args
        phi, _ = phi_analytic(lay, mats, z)
        g = restriction @ phi @ jnp.conj(restriction).T
        return jnp.real(jnp.linalg.det(g)) / omega

    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    if chunk is None:
        return jax.vmap(one)(xs)
    return jax.lax.map(one, xs, batch_size=chunk)


def residual_analytic(lay, theta, batch, chunk=None):
    """f = v/<v> - 1, with v from `volumes_analytic`."""
    v = volumes_analytic(lay, theta, batch, chunk)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return v / jnp.sum(weights * v) - 1.0


def weighted_residual_analytic(lay, theta, batch, chunk=None):
    """sqrt(w) f, so that the sum of squares is exactly the MSPE."""
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return jnp.sqrt(weights) * residual_analytic(lay, theta, batch, chunk)


def volume_jacobian(lay, theta, z, restriction, omega):
    """v and dv/dtheta at one point, with no tangent for each parameter.

    Jacobi's formula gives dv/dtheta_alpha = v * tr(g^-1 R dphi_alpha R^dag).
    p is linear in theta, so dphi_alpha is linear in the Gram sums of B_alpha.
    The contraction with K = R^dag g^-1 R collapses every index except the block
    one, so each alpha costs one Frobenius product against a single matrix W_r.
    """
    sig, dsig = section_vectors(lay, z)
    mats = sections.prepare(lay, theta)
    p, p_i, p_ij, store = _gram_parts(lay, mats, sig, dsig)

    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig, dsig):
        b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
        h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
        phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
    phi = phi / p

    g = restriction @ phi @ jnp.conj(restriction).T
    v = jnp.real(jnp.linalg.det(g)) / omega
    k_mat = jnp.conj(restriction).T @ jnp.linalg.inv(g) @ restriction   # (4, 4)

    # Scalars that multiply the outer products inside W_r.
    k_p = jnp.einsum('lk,kl->', k_mat, p_ij)
    k_phi = jnp.einsum('lk,kl->', k_mat, phi)
    coeff = k_p / p ** 2 - 2.0 * k_phi / p
    left = jnp.einsum('kl,k->l', k_mat, jnp.conj(p_i))     # (K^T conj(p))
    right = jnp.einsum('kl,l->k', k_mat, p_i)              # (K p)

    grads = []
    for g_layout, s, ds in zip(lay.groups, sig, dsig):
        # W_r assembles every index except the block index, so the parameter
        # derivative is one Frobenius product against it.
        w = coeff * jnp.einsum('bi,bj->bij', s, jnp.conj(s))
        w = w + jnp.einsum('bik,lk,bjl->bij', ds, k_mat, jnp.conj(ds)) / p
        w = w - jnp.einsum('bik,k,bj->bij', ds, left, jnp.conj(s)) / p ** 2
        w = w - jnp.einsum('bi,bjl,l->bij', s, jnp.conj(ds), right) / p ** 2
        rot = jnp.asarray(g_layout.rot)
        # H_r = rot H_rep rot^T, so tr(B_r W) = tr(B_rep rot^T W rot).
        w_orbit = jnp.einsum('bik,bij,bjl->bkl', rot, w, rot)
        grads.append((g_layout, w_orbit))

    out = jnp.zeros((lay.n_theta,), jnp.float64)
    for g_layout, w_orbit in grads:
        for local, sl in enumerate(g_layout.slices):
            take = g_layout.orbit_of == local
            summed = jnp.sum(jnp.where(take[:, None, None], w_orbit, 0.0), axis=0)
            basis = jnp.asarray(g_layout.basis[local, :sl.stop - sl.start])
            out = out.at[sl].add(jnp.real(jnp.einsum('aij,ij->a', basis, summed)))
    return v, v * out


def residual_jacobian(lay, theta, batch, chunk=None):
    """v and dv/dtheta at every point, with no tangent for each parameter."""
    def one(args):
        z, restriction, omega = args
        return volume_jacobian(lay, theta, z, restriction, omega)

    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    if chunk is None:
        return jax.vmap(one)(xs)
    return jax.lax.map(one, xs, batch_size=chunk)
