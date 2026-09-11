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

from functools import partial

import jax
import jax.numpy as jnp

import jax.tree_util as jtu

from . import dd, sections

__all__ = ['section_vectors', 'section_vectors_closed', 'section_vectors_dd',
           'phi_analytic', 'EXACT_MUL',
           'volumes_analytic',
           'residual_analytic', 'weighted_residual_analytic',
           'volume_and_weights', 'volume_tangent', 'volume_jacobian',
           'residual_jacobian', 'weighted_residual_fast']

# Stage 4 of the evaluation, the Gram stage, in double-double. The oracle in
# scripts/oracle_stages.py measured it as the source of almost all of the FP64
# error: 4.4e-13 at k = 32, against 1.1e-14 with this stage exact.
#
#   False    everything in FP64
#   'inner'  the inner products in dd, and the matrix-vector products in FP64
#   True     both in dd
#
# 'inner' exists because dd gives up the matrix-vector product: FP64 uses one
# matmul kernel, and dd has to broadcast and reduce by hand. That made the whole
# Jacobian 91 times slower at k = 20. The inner products run over (blocks, d)
# and not (blocks, d, d), so they cost far less.
GRAM_DD = True

# Stages 1 to 3 in double-double: the monomials of u, the frame of the sections,
# and sigma with its 4 derivatives. Stage 4 alone leaves a floor that grows by
# 1.28 for each unit of k, and the oracle puts dd on stages 1 to 4 at a relative
# error of 5e-16 for phi that is flat in k. This flag turns on the rest.
#
# It needs `section_vectors_closed`, because `dd.split` uses a bitcast, which
# carries no derivative, so `jax.jacfwd` cannot pass through it.
SECTION_DD = False

# `dd.mul_fast` against the exact `dd.mul` inside stages 1 to 3. `mul_fast`
# keeps the exact high-part product and sums the rest in FP64, which gives
# 2^-79, or about 24 digits, for 2 barriers against 20.
#
# Measured end to end, the relative error of phi is 1.61e-24 with `mul_fast` and
# 3.89e-32 with `mul`. So `mul_fast` holds a floor on abs(f) of approximately
# 3e-24, which is enough for stage 1 at 1e-15 and **30 times too high for stage
# 2 at 1e-25**. The exact form costs compile time: XLA cannot fuse across a
# barrier, and the residual at k = 8 did not compile in 600 s with it.
EXACT_MUL = False


def _cmul(a, b):
    """Complex double-double product, exact or fast by `EXACT_MUL`."""
    return dd.cmul(a, b) if EXACT_MUL else dd.cmul_fast(a, b)


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


def section_vectors_closed(lay, z):
    """The same, with the derivatives in closed form instead of by autodiff.

    With A = r + 4C the section is z^A, so

        d_i (z^r u^C) = (z^r / z_i) (r_i + 4 C_i) u^C,

    and the sum over C of Q_{C alpha} C_i u^C is one more frame projection. So 5
    projections replace the 4 autodiff tangents.

    This form is what extended precision needs for stages 1 to 3 of the
    evaluation: `dd.split` uses a bitcast, which carries no derivative, so
    `jax.jacfwd` cannot pass through it.

    It is **not** the default. It agrees with the autodiff form to 6.3e-16, and
    it is approximately 30 percent slower, because XLA fuses the 4 tangents
    better than it fuses the extra projection and the (blocks, d, 4) tensor. So
    it waits until stages 1 to 3 need it.
    """
    z2 = z * z
    u = z2 * z2
    sig, dsig = [], []
    for g in lay.groups:
        exps = jnp.asarray(g.exps)
        frame = jnp.asarray(g.frame)
        mono = sections._monomials(u, exps, g.m)             # (n_monomials,)
        psi = frame.T @ mono                                 # (d,)
        # The C_i weighted projection, for each of the 4 directions.
        psi_i = frame.T @ (exps.astype(mono.dtype) * mono[:, None])   # (d, 4)

        powers = sections._integer_powers(z, 3)              # r_a is 0 to 3
        res = jnp.asarray(g.residues)
        weight = jnp.prod(powers[res, jnp.arange(4)], axis=1)         # (n_blocks,)

        sig.append(weight[:, None] * psi[None, :])
        # (r_i psi + 4 psi_i) * z^r / z_i
        inner = res[:, None, :].astype(psi.dtype) * psi[None, :, None] \
            + 4.0 * psi_i[None, :, :]
        dsig.append(inner * (weight[:, None, None] / z[None, None, :]))
    return tuple(sig), tuple(dsig)


def _cstack(vals, axis=0):
    """Stack a list of complex double-double values along a new axis."""
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *vals)


def _cindex(val, *idx):
    """Index every leaf of a complex double-double value."""
    return jtu.tree_map(lambda a: a[idx], val)


def _cscale(val, factor):
    """A complex double-double value times an FP64 real array."""
    return (dd.mul_dd_fp(val[0], factor), dd.mul_dd_fp(val[1], factor))


def _integer_powers_dd(x, top):
    """x**0 .. x**top in double-double, by repeated multiplication.

    Repeated multiplication keeps every exponent an exact integer. A float
    exponent would send the value through `exp` and `log`, which costs 15
    digits at k = 32.
    """
    one = (dd.to_dd(jnp.ones_like(x[0][0])), dd.to_dd(jnp.zeros_like(x[0][0])))
    out = [one]
    for _ in range(top):
        out.append(_cmul(out[-1], x))
    return _cstack(out, axis=0)


def _monomials_dd(u, exps, m):
    """Every monomial u^E for the rows E of `exps`, in double-double."""
    table = _integer_powers_dd(u, m)                     # (m+1, 4)
    cols = _cindex(table, exps, jnp.arange(4))           # (n_monomials, 4)
    out = _cindex(cols, slice(None), 0)
    for a in range(1, 4):
        out = _cmul(out, _cindex(cols, slice(None), a))
    return out                                            # (n_monomials,)


def _project_dd(weighted_frame, val, extra_axes=0):
    """sum over the monomials of `weighted_frame` against a complex dd `val`.

    `val` carries the monomial axis first, and `weighted_frame` carries it first
    as well, so the contraction is one reduction over axis 0. `extra_axes` says
    how many axes `weighted_frame` has after the monomial and output axes, so
    that `val` broadcasts against it.
    """
    shape = (slice(None),) + (None,) * (1 + extra_axes)
    def one(part):
        return dd.dot_dd_fp((part[0][shape], part[1][shape]),
                            weighted_frame, axis=0)
    return one(val[0]), one(val[1])


def section_vectors_dd(lay, z, z_dd=None):
    """`sig` and `dsig` in double-double: stages 1 to 3 of the evaluation.

    `z_dd` takes the point itself as a complex double-double value, which the
    end-to-end path needs: an FP64 z holds `abs(P)/scale` at 1e-16 at best, and
    that error enters `v` with an amplification of up to 6.6.

    This is `section_vectors_closed` with every operation in extended
    precision. The derivatives use the closed form

        d_i (z^r u^C) = (z^r / z_i) (r_i + 4 C_i) u^C,

    so one reciprocal of z serves every block, and no autodiff has to pass
    through the bitcast inside `dd.split`.

    **Every product is shared across the groups.** The power tables of u and of
    z do not depend on the group, and the monomials and the block weights are
    one gather and one product chain over the concatenated exponents. A version
    that repeated them for each group made the compiled program large enough
    that XLA did not finish k = 6, which is the same fault that the note in
    `_matvec_dd` records.
    """
    zc = dd.cto_dd(z) if z_dd is None else z_dd
    z2 = _cmul(zc, zc)
    u = _cmul(z2, z2)                                   # u_a = z_a^4, exact
    inv_z = dd.creciprocal(zc)

    # One product chain over every group's monomials at once.
    top = max(g.m for g in lay.groups)
    u_powers = _integer_powers_dd(u, top)
    all_exps = jnp.concatenate([jnp.asarray(g.exps) for g in lay.groups], axis=0)
    cols = _cindex(u_powers, all_exps, jnp.arange(4))
    mono_all = _cindex(cols, slice(None), 0)
    for a in range(1, 4):
        mono_all = _cmul(mono_all, _cindex(cols, slice(None), a))

    # And one over every block weight z^r. r_a is 0 to 3.
    z_powers = _integer_powers_dd(zc, 3)
    all_res = jnp.concatenate([jnp.asarray(g.residues) for g in lay.groups],
                              axis=0)
    cols = _cindex(z_powers, all_res, jnp.arange(4))
    weight_all = _cindex(cols, slice(None), 0)
    for a in range(1, 4):
        weight_all = _cmul(weight_all, _cindex(cols, slice(None), a))
    # z^r / z_i, for every block and direction.
    wz_all = _cmul(_cindex(weight_all, slice(None), None),
                     _cindex(inv_z, None, slice(None)))

    sig, dsig = [], []
    m_at = r_at = 0
    for g in lay.groups:
        exps = jnp.asarray(g.exps)
        frame = jnp.asarray(g.frame)
        res = jnp.asarray(g.residues)
        n_mono, n_blocks = exps.shape[0], res.shape[0]
        mono = _cindex(mono_all, slice(m_at, m_at + n_mono))
        weight = _cindex(weight_all, slice(r_at, r_at + n_blocks))
        wz = _cindex(wz_all, slice(r_at, r_at + n_blocks))
        m_at += n_mono
        r_at += n_blocks

        psi = _project_dd(frame, mono)                    # (d,)
        # The C_i weighted projection, for each of the 4 directions.
        psi_i = _project_dd(frame[:, :, None] * exps.astype(frame.dtype)[:, None, :],
                            mono, extra_axes=1)           # (d, 4)

        sig.append(_cmul(_cindex(weight, slice(None), None),
                           _cindex(psi, None, slice(None))))

        # (r_i psi + 4 psi_i), then times z^r / z_i.
        term_r = _cscale(_cindex(psi, None, slice(None), None),
                         res[:, None, :].astype(frame.dtype))
        term_c = _cscale(_cindex(psi_i, None, slice(None), slice(None)),
                         jnp.asarray(4.0))
        inner = dd.cadd(term_r, term_c)                   # (n_blocks, d, 4)
        dsig.append(_cmul(inner, _cindex(wz, slice(None), None, slice(None))))

    return tuple(sig), tuple(dsig)


def _matvec_dd(mat, vecs):
    """H @ vecs, in double-double, for every column of `vecs` at once.

    `mat` is (blocks, d, d) and real, `vecs` is (blocks, d, c) and real. The
    reduction runs over the middle axis of `vecs`.

    Every column shares one reduction tree. That matters for the size of the
    program and not for the arithmetic: a Python loop over the columns, over the
    4 derivative directions and over the real and imaginary parts made the graph
    48 times the FP64 one, and XLA then took 79 s to compile k = 6 and 339 s to
    compile k = 8, against 0.5 s and 0.7 s for FP64.
    """
    if isinstance(vecs, tuple):
        # A double-double vector from stages 1 to 3. The low part is 2^-53 of
        # the high part, so its product with `mat` needs FP64 only, and that
        # saves a second dd reduction over the largest array of the step.
        main = _matvec_dd(mat, vecs[0])
        low = jnp.einsum('bij,bjc->bic', mat, vecs[1])
        return dd.add(main, dd.to_dd(low))

    a_hi, a_lo = dd.split(mat[..., None])            # (blocks, d, d, 1)
    v_hi, v_lo = dd.split(vecs[:, None, :, :])       # (blocks, 1, d, c)
    acc = None
    for term in (a_hi * v_hi, a_hi * v_lo, a_lo * v_hi, a_lo * v_lo):
        piece = dd.sum_fp(term, axis=-2)
        acc = piece if acc is None else dd.add(acc, piece)
    return acc                                        # (blocks, d, c) as dd


def _inner_dd(u, v, axes):
    """sum of u * v over `axes`, in double-double, for every column at once.

    v is double-double. u is FP64, or a double-double pair when stages 1 to 3
    run in extended precision; the low part of u then enters in FP64, because it
    is 2^-53 of the high part. One tree serves every column, for the same reason
    as in `_matvec_dd`.
    """
    if isinstance(u, tuple):
        main = _inner_dd(u[0], v, axes)
        term = dd.to_fp(v) * u[1]
        flat = jnp.reshape(term, (-1,) + term.shape[len(axes):])
        return dd.add(main, dd.sum_fp(flat, axis=0))

    acc = None
    for term in dd._partials(v[0], u) + (v[1] * u,):
        flat = jnp.reshape(term, (-1,) + term.shape[len(axes):])
        piece = dd.sum_fp(flat, axis=0)
        acc = piece if acc is None else dd.add(acc, piece)
    return acc


def _section_vectors(lay, z):
    """`sig` and `dsig` for the Gram stage, and FP64 copies for the centered form.

    Every caller of `_gram_parts` must come through here. `_gram_parts` branches
    on `SECTION_DD`, so a caller that picks the section vectors itself can pair
    an FP64 array with the double-double branch. That happened: the Jacobian
    path called `section_vectors` directly and the fit died inside `plan_fit`
    with `Too many indices`, because `s[0]` took a row of an array where it
    meant the high part of a pair.
    """
    if SECTION_DD:
        sig, dsig = section_vectors_dd(lay, z)
        return (sig, dsig,
                tuple(dd.cto_fp(a) for a in sig),
                tuple(dd.cto_fp(a) for a in dsig))
    sig, dsig = section_vectors(lay, z)
    return sig, dsig, sig, dsig


def _gram_parts(lay, mats, sig, dsig):
    """p, dp and ddbar-p, from the Gram sums over every block.

    `H sigma` and the four `H dsigma_i` are the only matrix-vector products, so
    one point costs 5 of them for each block, in place of approximately 50.

    With `GRAM_DD` the products and the sums run in double-double. That is stage
    4 of `scripts/oracle_stages.py`, which carries almost all of the FP64 error.
    The results round back to FP64 on the way out, which is what the oracle
    measured: the centered form of phi does not need the extra digits.
    """
    if not GRAM_DD:
        return _gram_parts_fp(lay, mats, sig, dsig)
    matvec_dd = GRAM_DD != 'inner'

    p = dd.to_dd(jnp.zeros(()))
    p_i_re = dd.to_dd(jnp.zeros((4,)))
    p_i_im = dd.to_dd(jnp.zeros((4,)))
    p_ij = jnp.zeros((4, 4), jnp.complex128)
    store = []

    for mat, s, ds in zip(mats, sig, dsig):
        if SECTION_DD:
            # sig and dsig arrive as complex double-double from stages 1 to 3.
            # Lay the real and imaginary parts out as the same 10 columns, and
            # keep both halves of each pair.
            def cat(part, sel):
                return jnp.concatenate(
                    [part[0][sel][..., None], part[1][sel]], axis=-1)
            stack = tuple(
                jnp.concatenate([cat((s[0], ds[0]), h), cat((s[1], ds[1]), h)],
                                axis=-1) for h in (0, 1))
            s_re = (s[0][0][..., None], s[0][1][..., None])
            s_im = (s[1][0][..., None], s[1][1][..., None])
        else:
            # sigma and its 4 derivatives, real and imaginary, as 10 columns.
            cols = jnp.concatenate([s[..., None], ds], axis=-1)      # (b, d, 5)
            stack = jnp.concatenate([jnp.real(cols), jnp.imag(cols)], axis=-1)
            s_re = jnp.real(s)[..., None]
            s_im = jnp.imag(s)[..., None]

        if matvec_dd:
            out = _matvec_dd(mat, stack)                               # (b, d, 10)
        else:
            plain = jnp.einsum(
                'bij,bjc->bic', mat,
                stack[0] + stack[1] if SECTION_DD else stack)
            out = dd.to_dd(plain)
        h_re = (out[0][..., :5], out[1][..., :5])
        h_im = (out[0][..., 5:], out[1][..., 5:])

        # conj(sigma) against each column, so p is column 0 and dp is 1 to 4.
        rr = _inner_dd(s_re, h_re, (0, 1))
        ii = _inner_dd(s_im, h_im, (0, 1))
        ri = _inner_dd(s_re, h_im, (0, 1))
        ir = _inner_dd(s_im, h_re, (0, 1))
        real_part = dd.add(rr, ii)
        imag_part = dd.add(ri, (-ir[0], -ir[1]))

        p = dd.add(p, (real_part[0][0], real_part[1][0]))
        p_i_re = dd.add(p_i_re, (real_part[0][1:], real_part[1][1:]))
        p_i_im = dd.add(p_i_im, (imag_part[0][1:], imag_part[1][1:]))

        h_s = dd.to_fp(h_re)[..., 0] + 1j * dd.to_fp(h_im)[..., 0]
        h_d_fp = dd.to_fp(h_re)[..., 1:] + 1j * dd.to_fp(h_im)[..., 1:]
        # Entry [i, j] is d_i d_jbar p, so the conjugated index comes second.
        ds_fp = dd.cto_fp(ds) if SECTION_DD else ds
        p_ij = p_ij + jnp.einsum('bil,bik->kl', jnp.conj(ds_fp), h_d_fp)
        store.append((h_s, h_d_fp))

    return (dd.to_fp(p), dd.to_fp(p_i_re) + 1j * dd.to_fp(p_i_im), p_ij, store)


def _gram_parts_fp(lay, mats, sig, dsig):
    """The same in plain FP64, kept so the two can be held against each other."""
    p = jnp.zeros((), jnp.float64)
    p_i = jnp.zeros((4,), jnp.complex128)
    p_ij = jnp.zeros((4, 4), jnp.complex128)
    store = []

    for mat, s, ds in zip(mats, sig, dsig):
        h_s = jnp.einsum('bij,bj->bi', mat, s)               # (n_blocks, d)
        h_d = jnp.einsum('bij,bjk->bik', mat, ds)            # (n_blocks, d, 4)
        p = p + jnp.real(jnp.einsum('bi,bi->', jnp.conj(s), h_s))
        p_i = p_i + jnp.einsum('bi,bik->k', jnp.conj(s), h_d)
        p_ij = p_ij + jnp.einsum('bil,bik->kl', jnp.conj(ds), h_d)
        store.append((h_s, h_d))
    return p, p_i, p_ij, store


def phi_analytic(lay, mats, z):
    """phi_{i jbar} = d_i d_jbar log p, in the centered form.

    The difference form (p_ij p - p_i p_jbar)/p^2 cancels by a factor of k. The
    centered vectors b_i = dsigma_i - (p_i/p) sigma make the result a Gram
    matrix, so the cancellation goes into a vector subtraction instead.
    """
    sig, dsig, sig_fp, dsig_fp = _section_vectors(lay, z)
    p, p_i, _, store = _gram_parts(lay, mats, sig, dsig)

    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig_fp, dsig_fp):
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


def volume_and_weights(lay, mats, z, restriction, omega):
    """v at one point, and the matrix W_r of each block that carries dv.

    Jacobi's formula gives dv/dtheta_alpha = v * tr(g^-1 R dphi_alpha R^dag).
    p is linear in theta, so dphi_alpha is linear in the Gram sums of B_alpha.
    The contraction with K = R^dag g^-1 R collapses every index except the block
    one, and what is left is

        dv = v * sum_r tr(dH_r W_r),

    where dH_r is the block matrix of the tangent direction. So one direction
    costs one Frobenius product for each block, and nothing more.
    """
    sig, dsig, sig_fp, dsig_fp = _section_vectors(lay, z)
    p, p_i, p_ij, store = _gram_parts(lay, mats, sig, dsig)

    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig_fp, dsig_fp):
        b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
        h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
        phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
    phi = phi / p

    g = restriction @ phi @ jnp.conj(restriction).T
    v = jnp.real(jnp.linalg.det(g)) / omega
    k_mat = jnp.conj(restriction).T @ jnp.linalg.inv(g) @ restriction   # (4, 4)

    # Scalars and vectors that multiply the outer products inside W_r.
    k_p = jnp.einsum('lk,kl->', k_mat, p_ij)
    k_phi = jnp.einsum('lk,kl->', k_mat, phi)
    coeff = k_p / p ** 2 - 2.0 * k_phi / p
    left = jnp.einsum('kl,k->l', k_mat, jnp.conj(p_i))     # (K^T conj(p))
    right = jnp.einsum('kl,l->k', k_mat, p_i)              # (K p)

    weights = []
    for s, ds in zip(sig_fp, dsig_fp):
        w = coeff * jnp.einsum('bi,bj->bij', s, jnp.conj(s))
        w = w + jnp.einsum('bik,lk,bjl->bij', ds, k_mat, jnp.conj(ds)) / p
        w = w - jnp.einsum('bik,k,bj->bij', ds, left, jnp.conj(s)) / p ** 2
        w = w - jnp.einsum('bi,bjl,l->bij', s, jnp.conj(ds), right) / p ** 2
        weights.append(v * jnp.real(w))
    return v, tuple(weights)


def volume_tangent(weights, dmats):
    """dv along one direction, from the block matrices of that direction.

    dH_r is symmetric, so tr(dH_r W_r) is the plain Frobenius product.
    """
    return sum(jnp.einsum('bij,bij->', dmat, w)
               for dmat, w in zip(dmats, weights))


def volume_jacobian(lay, theta, z, restriction, omega):
    """v and the full dv/dtheta at one point. This is the reference for tests."""
    mats = sections.prepare(lay, theta)
    v, weights = volume_and_weights(lay, mats, z, restriction, omega)

    out = jnp.zeros((lay.n_theta,), jnp.float64)
    for g_layout, w in zip(lay.groups, weights):
        rot = jnp.asarray(g_layout.rot)
        # H_r = rot H_rep rot^T, so tr(B_r W) = tr(B_rep rot^T W rot).
        w_orbit = jnp.einsum('bik,bij,bjl->bkl', rot, w, rot)
        whitening = jnp.asarray(g_layout.whiten)
        for local, sl in enumerate(g_layout.slices):
            take = g_layout.orbit_of == local
            summed = jnp.sum(jnp.where(take[:, None, None], w_orbit, 0.0), axis=0)
            # H_rep = W H_white W, so dH_rep/dtheta_a is W B_a W. W is symmetric,
            # so tr(W B_a W S) = tr(B_a W S W). W is the identity with no Gram.
            w_mat = whitening[local]
            summed = w_mat @ summed @ w_mat
            basis = jnp.asarray(g_layout.basis[local, :sl.stop - sl.start])
            out = out.at[sl].add(jnp.einsum('aij,ij->a', basis, summed))
    # The basis is the raw one, so the derivative needs the scale of the layout.
    return v, out * jnp.asarray(lay.scale)


def residual_jacobian(lay, theta, batch, chunk=None):
    """v and dv/dtheta at every point, with no tangent for each parameter."""
    def one(args):
        z, restriction, omega = args
        return volume_jacobian(lay, theta, z, restriction, omega)

    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    if chunk is None:
        return jax.vmap(one)(xs)
    return jax.lax.map(one, xs, batch_size=chunk)


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
def weighted_residual_fast(lay, theta, batch, chunk):
    """sqrt(w) f, with a hand-written derivative rule.

    `lineax` materializes the LM operator with `jax.vmap` over the basis of
    theta. `vmap` only batches an operation whose input carries a batch axis, so
    the rule below computes v and W once, outside the batch, and only the final
    Frobenius product runs for each direction.
    """
    return weighted_residual_analytic(lay, theta, batch, chunk)


@weighted_residual_fast.defjvp
def _weighted_residual_fast_jvp(lay, batch, chunk, primals, tangents):
    (theta,), (dtheta,) = primals, tangents
    mats = sections.prepare(lay, theta)
    dmats = sections.prepare(lay, dtheta)

    def one(args):
        z, restriction, omega = args
        v, w = volume_and_weights(lay, mats, z, restriction, omega)
        return v, volume_tangent(w, dmats)

    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    v, dv = (jax.vmap(one)(xs) if chunk is None
             else jax.lax.map(one, xs, batch_size=chunk))

    weights = batch['mass'] / jnp.sum(batch['mass'])
    mean = jnp.sum(weights * v)
    d_mean = jnp.sum(weights * dv)
    root = jnp.sqrt(weights)
    return root * (v / mean - 1.0), root * (dv / mean - v * d_mean / mean ** 2)
