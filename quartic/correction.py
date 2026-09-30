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

The plain FP64 form is for tests at small k. The mixed form keeps ddbar Phi in
dd at each point, and psi in FP64, because the error of psi is relative to its
own size: `base_state_dd`, `volume_mixed`, `residual_fn_mixed`, `chain_volume`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import analytic, dd, sections

__all__ = ['fs_state', 'base_state', 'psi_parts', 'add_level', 'states',
           'volume_corrected', 'volumes', 'residual_fn', 'base_state_dd',
           'volume_mixed', 'weights', 'residual_fn_mixed', 'advance_mixed',
           'chain_volume']


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


def base_state_dd(lay0, mats0, z, extra):
    """The state of log p_0 at the dd point, and ddbar log p_0 as a complex dd value.

    The Gram stage and the centered form run in the dd path of `analytic`, so its
    flags must select END_DD and GRAM_DD, and `mats0` is `analytic.point_matrices`.
    """
    z_dd, _, _ = analytic._dd_point(extra)
    sig, dsig, _, _ = analytic._section_vectors(lay0, z, z_dd=z_dd)
    p, p_i, _, _, extras = analytic._gram_parts(lay0, mats0, sig, dsig)
    phi_dd = analytic._phi_stage5_dd(extras, sig, dsig, keep=True)
    return (jnp.log(p), p_i / p, dd.cto_fp(phi_dd)), phi_dd


def volume_mixed(lay_c, cmats, extra, state, phi_dd, alpha):
    """v as a dd pair for Phi + psi, and ddbar psi, with psi in FP64 at the dd point.

    `phi_dd` is ddbar Phi in dd. The weight of psi is exp(alpha Phi).
    """
    z_dd, m, free = analytic._dd_point(extra)
    _, _, psi_ij = psi_parts(lay_c, cmats, extra[0], alpha, *state)
    total = dd.cadd(phi_dd, dd.cto_dd(psi_ij))
    return analytic._volume_from_phi_dd(total, z_dd, m, free, keep=True), psi_ij


def weights(lay_c, z, restriction, state, psi_ij, v, alpha):
    """W_b of each block, so that dv = sum over blocks of <dC_b, W_b>, in FP64.

    With b = dsigma - alpha (d Phi) sigma, K = R^dag g^-1 R and t = tr(K ddbar Phi),
    the quotient rule and Jacobi's formula give
    W = v exp(-alpha Phi) Re(b K b^* - alpha t sigma sigma^*).
    """
    pot, pot_i, pot_ij = state
    sig, dsig = analytic.section_vectors(lay_c, z)
    g = restriction @ (pot_ij + psi_ij) @ jnp.conj(restriction).T
    k_mat = jnp.conj(restriction).T @ jnp.linalg.inv(g) @ restriction
    trace = jnp.einsum('lk,kl->', k_mat, pot_ij)
    scale = v * jnp.exp(-alpha * pot)
    out = []
    for s, ds in zip(sig, dsig):
        b = ds - alpha * jnp.einsum('bi,k->bik', s, pot_i)
        w = jnp.einsum('bik,lk,bjl->bij', b, k_mat, jnp.conj(b))
        w = w - alpha * trace * jnp.einsum('bi,bj->bij', s, jnp.conj(s))
        out.append(scale * jnp.real(w))
    return tuple(out)


def residual_fn_mixed(lay_c, batch, state, phi_dd, chunk, alpha):
    """sqrt(w) f with v and f in dd, and an FP64 tangent from `weights`.

    `batch` carries the dd points of `dd_metric.point_extras`. The tangent pass
    forms W again for each point, so the linearization keeps no W.
    """
    mass = batch['mass'] / jnp.sum(batch['mass'])
    root = jnp.sqrt(mass)
    extra = (batch['points_hi'], batch['points_lo'], batch['patch'])

    def primal(c):
        cmats = sections.prepare_fp(lay_c, c)
        return jax.lax.map(
            lambda a: volume_mixed(lay_c, cmats, a[0], a[1], a[2], alpha),
            (extra, state, phi_dd), batch_size=chunk)

    @jax.custom_jvp
    def res(c):
        v, _ = primal(c)
        return root * analytic.residual_from_volumes(v, mass)

    @res.defjvp
    def res_jvp(primals, tangents):
        (c,), (dc,) = primals, tangents
        v, psi = primal(c)
        f = analytic.residual_from_volumes(v, mass)
        v_fp = dd.to_fp(v)
        dmats = sections.prepare_fp(lay_c, dc)

        @jax.checkpoint
        def one(a):
            z, rest, st, ps, vv = a
            return analytic.volume_tangent(weights(lay_c, z, rest, st, ps, vv, alpha),
                                           dmats)
        dv = jax.lax.map(one, (batch['points_hi'], batch['restriction'], state, psi,
                               v_fp), batch_size=chunk)
        mean = jnp.sum(mass * v_fp)
        df = dv / mean - v_fp * jnp.sum(mass * dv) / mean ** 2
        return root * f, root * df
    return res


def advance_mixed(lay_c, cmats, batch, state, phi_dd, alpha, chunk):
    """The state of Phi + psi and ddbar (Phi + psi) in dd, at every point of `batch`."""
    def one(a):
        z, st, ph = a
        parts = psi_parts(lay_c, cmats, z, alpha, *st)
        return add_level(st, parts), dd.cadd(ph, dd.cto_dd(parts[2]))
    return jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(
        (batch['points_hi'], state, phi_dd))


def chain_volume(lay0, mats0, levels, z, extra):
    """v as a dd pair at one point for log p_0 plus every level of `levels`.

    Each entry of `levels` is (lay_c, cmats, alpha), in the order of the fit.
    """
    state, phi_dd = base_state_dd(lay0, mats0, z, extra)
    for lay_c, cmats, alpha in levels:
        parts = psi_parts(lay_c, cmats, extra[0], alpha, *state)
        state = add_level(state, parts)
        phi_dd = dd.cadd(phi_dd, dd.cto_dd(parts[2]))
    z_dd, m, free = analytic._dd_point(extra)
    return analytic._volume_from_phi_dd(phi_dd, z_dd, m, free, keep=True)
