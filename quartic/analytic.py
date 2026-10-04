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
           'phi_analytic', 'EXACT_MUL', 'H_LOW', 'PHI_DD', 'END_DD', 'REMAT_W',
           'W_CENTERED', 'J_ROWS', 'F_DD',
           'point_matrices', 'point_inputs',
           'volume_at', 'volumes_analytic',
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
# It needs `section_vectors_closed`: an autodiff tangent through dd is FP64
# arithmetic, because `dd.split` gives hi a zero tangent, so dsigma must be a
# primal output.
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

# Carry the low part of H into the dd products of the Gram stage. It needs
# `GRAM_DD = True`. At the noisiest points of the stall at k = 28, the FP64
# rounding of H alone moves f by 4e-13, and every per-point stage in dd leaves
# that unchanged. One more FP64 matrix-vector product for each block removes it.
H_LOW = False

# Stage 5 in double-double: the centered form b = dsigma - (p_i/p) sigma, H b,
# and the Gram sum of phi, from the dd values of the Gram stage, with phi
# rounded once. It needs `GRAM_DD = True`. With H carried in dd, the FP64 form
# of this stage leaves 4.8e-14 of noise at the noisiest points of k = 28.
PHI_DD = False

# The whole path in double-double, from a point on X to v. Each point comes as a
# dd pair after one dd Newton step, and R, det and OmegaOmegabar are dd too,
# from the closed forms. It needs GRAM_DD = True, and it holds stage 5 in dd by
# itself. At the endpoint of k = 28, the FP64 point moves f by 9.0e-16 rms, and
# the FP64 parts of the fit arithmetic move it by 2.1e-15 rms.
END_DD = False

# Keep only v and (p, p_i, p_ij, phi) of each point between the two passes of the
# LM linearization, and form W again from FP64 sections in the tangent pass.
# Else the linearization holds W for every fit point, 3.6 sum_r d_r^2 doubles.
REMAT_W = False

# W from the centered vectors b = dsigma - (p_i/p) sigma, as (v/p) Re(b K b^T* -
# 2 sigma sigma^*). The expanded form cancels by about (k/abs(z_a))^2 at points
# with a small coordinate. It needs GRAM_DD = True, and dd sections for full effect.
W_CENTERED = False

# J from one row dv/dtheta for each point, formed from tau = W rot^T sigma and
# beta = W rot^T b in dd. No FP64 dH enters, whose whitened form W B W carries a
# dynamic range of cond(W)^2. It needs END_DD.
J_ROWS = False

# f = v/<v> - 1 in dd from v as a dd pair, rounded once. With v rounded first,
# f carries 1e-16 of absolute noise, 7e-4 of norm(r) at k = 32. It needs END_DD.
F_DD = False


def _dd_sections():
    """True when the section vectors reach the Gram stage in dd."""
    return SECTION_DD or END_DD


def point_matrices(lay, theta):
    """The block matrices that the per-point path takes: (hi, lo) with `H_LOW`."""
    if H_LOW:
        if GRAM_DD is not True:
            raise ValueError('H_LOW needs GRAM_DD = True')
        return sections.prepare_parts(lay, theta)
    return sections.prepare(lay, theta)


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
    evaluation. An autodiff tangent through dd is FP64 arithmetic, because
    `dd.split` gives hi a zero tangent and lo = a - hi carries all of it, so the
    derivatives would lose the extra digits.

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
        # The C_i weighted projection, for each of the 4 directions. frame * C_i can
        # need 56 bits, so the weights enter as an exact pair of FP64 arrays.
        w_hi, w_lo = dd.mul_dd_fp(dd.to_dd(frame[:, :, None]),
                                  exps.astype(frame.dtype)[:, None, :])
        psi_i = dd.cadd(_project_dd(w_hi, mono, extra_axes=1),
                        _project_dd(w_lo, mono, extra_axes=1))   # (d, 4)

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


def _section_vectors(lay, z, z_dd=None):
    """`sig` and `dsig` for the Gram stage, and FP64 copies for the centered form.

    Every caller of `_gram_parts` must come through here. `_gram_parts` branches
    on `SECTION_DD`, so a caller that picks the section vectors itself can pair
    an FP64 array with the double-double branch. That happened: the Jacobian
    path called `section_vectors` directly and the fit died inside `plan_fit`
    with `Too many indices`, because `s[0]` took a row of an array where it
    meant the high part of a pair.
    """
    if _dd_sections():
        sig, dsig = section_vectors_dd(lay, z, z_dd=z_dd)
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
    store, store_dd = [], []

    for mat, s, ds in zip(mats, sig, dsig):
        if _dd_sections():
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

        mat, low = mat if isinstance(mat, tuple) else (mat, None)
        if matvec_dd:
            out = _matvec_dd(mat, stack)                               # (b, d, 10)
            if low is not None:
                # low is 2^-53 of mat, so its product needs FP64 only.
                cols = stack[0] + stack[1] if _dd_sections() else stack
                out = dd.add(out, dd.to_dd(jnp.einsum('bij,bjc->bic', low, cols)))
        else:
            plain = jnp.einsum(
                'bij,bjc->bic', mat,
                stack[0] + stack[1] if _dd_sections() else stack)
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
        ds_fp = dd.cto_fp(ds) if _dd_sections() else ds
        p_ij = p_ij + jnp.einsum('bil,bik->kl', jnp.conj(ds_fp), h_d_fp)
        store.append((h_s, h_d_fp))
        # H sigma and H dsigma as complex dd, for stage 5 in dd.
        store_dd.append((((h_re[0][..., 0], h_re[1][..., 0]),
                          (h_im[0][..., 0], h_im[1][..., 0])),
                         ((h_re[0][..., 1:], h_re[1][..., 1:]),
                          (h_im[0][..., 1:], h_im[1][..., 1:]))))

    extras = (p, (p_i_re, p_i_im), store_dd)
    return (dd.to_fp(p), dd.to_fp(p_i_re) + 1j * dd.to_fp(p_i_im), p_ij, store,
            extras)


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
    return p, p_i, p_ij, store, None


def _mul(x, y):
    """Real double-double product, exact or fast by `EXACT_MUL`."""
    return dd.mul(x, y) if EXACT_MUL else dd.mul_fast(x, y)


def _creduce_lead(val, n_axes):
    """Sum a complex double-double value over its first `n_axes` axes."""
    def one(pair):
        hi, lo = pair
        shape = (-1,) + hi.shape[n_axes:]
        return dd.sum_dd(jnp.reshape(hi, shape), jnp.reshape(lo, shape), axis=0)
    return one(val[0]), one(val[1])


def _phi_stage5_dd(extras, sig, dsig, keep=False, with_b=False):
    """phi from the centered form and its Gram sum in dd, rounded once to FP64.

    `extras` holds p, p_i and the products H sigma, H dsigma of the Gram stage
    as dd. sigma and dsigma are dd with `SECTION_DD`, and FP64 otherwise.
    `with_b` also returns the centered vectors b of each group, rounded to FP64,
    or in dd if it is 'dd'.
    """
    p, p_i, store = extras
    inv_p = dd.reciprocal(p)
    ratio = _cindex((_mul(p_i[0], inv_p), _mul(p_i[1], inv_p)), None, None, slice(None))
    zero = dd.to_dd(jnp.zeros((4, 4)))
    phi = (zero, zero)
    bs = []
    for (h_s, h_d), s, ds in zip(store, sig, dsig):
        if not _dd_sections():
            s, ds = dd.cto_dd(s), dd.cto_dd(ds)
        col = (slice(None), slice(None), None)
        b = dd.csub(ds, _cmul(_cindex(s, *col), ratio))            # (blocks, d, 4)
        if with_b:
            bs.append(b if with_b == 'dd' else dd.cto_fp(b))
        h_b = dd.csub(h_d, _cmul(_cindex(h_s, *col), ratio))
        b_conj = (b[0], dd.neg(b[1]))
        # phi[k, l] = sum over blocks and d of conj(b_l) H b_k, as in FP64.
        left = _cindex(h_b, slice(None), slice(None), slice(None), None)
        right = _cindex(b_conj, slice(None), slice(None), None, slice(None))
        phi = dd.cadd(phi, _creduce_lead(_cmul(left, right), 2))
    phi = (_mul(phi[0], inv_p), _mul(phi[1], inv_p))
    phi = phi if keep else dd.cto_fp(phi)
    return (phi, tuple(bs)) if with_b else phi


def _centered_dd(p, p_i, sig, dsig):
    """The centered vectors b = dsigma - (p_i/p) sigma of each group, in dd, rounded."""
    inv_p = dd.reciprocal(p)
    ratio = _cindex((_mul(p_i[0], inv_p), _mul(p_i[1], inv_p)), None, None, slice(None))
    out = []
    for s, ds in zip(sig, dsig):
        if not _dd_sections():
            s, ds = dd.cto_dd(s), dd.cto_dd(ds)
        col = (slice(None), slice(None), None)
        out.append(dd.cto_fp(dd.csub(ds, _cmul(_cindex(s, *col), ratio))))
    return tuple(out)


def phi_analytic(lay, mats, z):
    """phi_{i jbar} = d_i d_jbar log p, in the centered form.

    The difference form (p_ij p - p_i p_jbar)/p^2 cancels by a factor of k. The
    centered vectors b_i = dsigma_i - (p_i/p) sigma make the result a Gram
    matrix, so the cancellation goes into a vector subtraction instead.
    """
    sig, dsig, sig_fp, dsig_fp = _section_vectors(lay, z)
    p, p_i, _, store, extras = _gram_parts(lay, mats, sig, dsig)
    if PHI_DD:
        if GRAM_DD is not True:
            raise ValueError('PHI_DD needs GRAM_DD = True')
        return _phi_stage5_dd(extras, sig, dsig), p

    phi = jnp.zeros((4, 4), jnp.complex128)
    for (h_s, h_d), s, ds in zip(store, sig_fp, dsig_fp):
        # b_i = dsigma_i - (p_i/p) sigma, and H b_i reuses the products above.
        b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
        h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
        phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
    return phi / p, p


def point_inputs(batch):
    """What each point of `batch` carries into the per-point path.

    With `END_DD` that is the dd point and its patch indices as well, which
    `dd_metric.point_extras` computes once for a sample.
    """
    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    if END_DD:
        xs = xs + (batch['points_hi'], batch['points_lo'], batch['patch'])
    return xs


def _dd_point(extra):
    """The dd point, m and the two free coordinates, from the extra inputs."""
    z_hi, z_lo, patch = extra
    z_dd = ((jnp.real(z_hi), jnp.real(z_lo)), (jnp.imag(z_hi), jnp.imag(z_lo)))
    return z_dd, patch[0], (patch[1], patch[2])


def _volume_from_phi_dd(phi, z_dd, m, free, keep=False, delta=None):
    """v = det(R phi R^dag) |dP/dz_m|^2 in dd, with R and dP from the dd point.

    OmegaOmegabar is 1/abs(dP/dz_m)^2, so this is det(g|X) / OmegaOmegabar.
    `keep` returns the dd pair and not its FP64 rounding. `delta` is an optional
    (2, 2) correction, FP64 or complex dd, that is added to g = R phi R^dag before
    the determinant.
    """
    from . import dd_metric
    rest, grad = dd_metric._restriction_dd(z_dd, m, free)
    rest_conj = (rest[0], dd.neg(rest[1]))
    left = _cindex(rest, slice(None), None, slice(None), None)
    mid = _cindex(phi, None, None, slice(None), slice(None))
    right = _cindex(rest_conj, None, slice(None), None, slice(None))
    g = dd_metric._creduce_last_two(_cmul(_cmul(left, mid), right))
    if delta is not None:
        g = dd.cadd(g, delta if isinstance(delta, tuple) else dd.cto_dd(delta))
    det = dd.csub(_cmul(dd_metric._at(g, 0, 0), dd_metric._at(g, 1, 1)),
                  _cmul(dd_metric._at(g, 0, 1), dd_metric._at(g, 1, 0)))
    v = _mul(det[0], dd.cabs2(_cindex(grad, m)))
    return v if keep else dd.to_fp(v)


def volume_at(lay, mats, z, restriction, omega, *extra):
    """v = det(g|_X) / OmegaOmegabar at one point, from given block matrices."""
    if END_DD:
        if GRAM_DD is not True:
            raise ValueError('END_DD needs GRAM_DD = True')
        z_dd, m, free = _dd_point(extra)
        sig, dsig, _, _ = _section_vectors(lay, z, z_dd=z_dd)
        extras = _gram_parts(lay, mats, sig, dsig)[4]
        return _volume_from_phi_dd(_phi_stage5_dd(extras, sig, dsig, keep=True),
                                   z_dd, m, free, keep=F_DD)
    phi, _ = phi_analytic(lay, mats, z)
    g = restriction @ phi @ jnp.conj(restriction).T
    return jnp.real(jnp.linalg.det(g)) / omega


def volumes_analytic(lay, theta, batch, chunk=None):
    """v = det(g|_X) / OmegaOmegabar, with phi from `phi_analytic`."""
    mats = point_matrices(lay, theta)

    def one(args):
        return volume_at(lay, mats, *args)

    xs = point_inputs(batch)
    if chunk is None:
        return jax.vmap(one)(xs)
    return jax.lax.map(one, xs, batch_size=chunk)


def _fp(v):
    """v in FP64, from an FP64 array or a dd pair."""
    return dd.to_fp(v) if isinstance(v, tuple) else v


def sampled_mean(v, weights):
    """<v> = sum w v / sum w in dd, for v as a dd pair.

    Weights divided by their FP64 sum add up to 1 only to about 1e-16, and that
    offset would move every f by the same amount.
    """
    return dd.mul(dd.dot_dd_fp(v, weights, axis=0),
                  dd.reciprocal(dd.sum_fp(weights, axis=0)))


def residual_from_volumes(v, weights, mean=None):
    """f = v/<v> - 1 from v in FP64, or from a dd pair in dd with f rounded once.

    <v> is the sampled mean, or `mean`, the exact dd pair of `volume.mean_v_dd`.
    """
    if not isinstance(v, tuple):
        if mean is not None:
            return v / (mean[0] + mean[1]) - 1.0
        return v * jnp.sum(weights, axis=0) / jnp.sum(weights * v, axis=0) - 1.0
    if mean is None:
        mean = sampled_mean(v, weights)
    else:
        mean = (jnp.asarray(mean[0]), jnp.asarray(mean[1]))
    ratio = dd.mul(v, dd.reciprocal(mean))
    return dd.to_fp(dd.add(ratio, dd.to_dd(-jnp.ones_like(v[0]))))


def residual_analytic(lay, theta, batch, chunk=None):
    """f = v/<v> - 1, with v from `volumes_analytic`."""
    v = volumes_analytic(lay, theta, batch, chunk)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return residual_from_volumes(v, weights)


def weighted_residual_analytic(lay, theta, batch, chunk=None):
    """sqrt(w) f, so that the sum of squares is exactly the MSPE."""
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return jnp.sqrt(weights) * residual_analytic(lay, theta, batch, chunk)


def volume_and_weights(lay, mats, z, restriction, omega, *extra):
    """v at one point, and the matrix W_r of each block that carries dv.

    Jacobi's formula gives dv/dtheta_alpha = v * tr(g^-1 R dphi_alpha R^dag).
    p is linear in theta, so dphi_alpha is linear in the Gram sums of B_alpha.
    The contraction with K = R^dag g^-1 R collapses every index except the block
    one, and what is left is

        dv = v * sum_r tr(dH_r W_r),

    where dH_r is the block matrix of the tangent direction. So one direction
    costs one Frobenius product for each block, and nothing more.
    """
    v, small, sig_fp, dsig_fp, b_fp = _point_state(lay, mats, z, restriction, omega,
                                                   *extra)
    return v, _block_weights(sig_fp, dsig_fp, small, restriction, _fp(v), b_fp)


def _point_state(lay, mats, z, restriction, omega, *extra):
    """v, the per-point values (p, p_i, p_ij, phi) that W needs, and FP64 sections."""
    point = _dd_point(extra) if END_DD else None
    sig, dsig, sig_fp, dsig_fp = _section_vectors(
        lay, z, z_dd=None if point is None else point[0])
    p, p_i, p_ij, store, extras = _gram_parts(lay, mats, sig, dsig)

    if END_DD:
        if GRAM_DD is not True:
            raise ValueError('END_DD needs GRAM_DD = True')
        phi_dd, b_fp = _phi_stage5_dd(extras, sig, dsig, keep=True, with_b=True)
        phi = dd.cto_fp(phi_dd)
        # LM takes v as the residual, so it comes from the dd chain. W only
        # enters J, so FP64 is enough there.
        v = _volume_from_phi_dd(phi_dd, *point, keep=F_DD)
    else:
        if PHI_DD:
            # v must be the accurate one, because LM takes it as the residual.
            if GRAM_DD is not True:
                raise ValueError('PHI_DD needs GRAM_DD = True')
            phi, b_fp = _phi_stage5_dd(extras, sig, dsig, with_b=True)
        else:
            phi = jnp.zeros((4, 4), jnp.complex128)
            b_fp = []
            for (h_s, h_d), s, ds in zip(store, sig_fp, dsig_fp):
                b = ds - jnp.einsum('bi,k->bik', s, p_i / p)
                h_b = h_d - jnp.einsum('bi,k->bik', h_s, p_i / p)
                phi = phi + jnp.einsum('bil,bik->kl', jnp.conj(b), h_b)
                b_fp.append(b)
            phi = phi / p
            b_fp = tuple(b_fp)
        g = restriction @ phi @ jnp.conj(restriction).T
        v = jnp.real(jnp.linalg.det(g)) / omega
    if extras is None:
        p_dd = dd.to_dd(p)
        p_i_dd = (dd.to_dd(jnp.real(p_i)), dd.to_dd(jnp.imag(p_i)))
    else:
        p_dd, p_i_dd = extras[0], extras[1]
    return v, (p, p_i, p_ij, phi, p_dd, p_i_dd), sig_fp, dsig_fp, b_fp


def _block_weights(sig_fp, dsig_fp, small, restriction, v, b_fp=None):
    """W_r of each block, from FP64 sections and the per-point values.

    With `W_CENTERED` it takes the centered vectors b and not dsigma.
    """
    p, p_i, p_ij, phi = small[:4]
    g = restriction @ phi @ jnp.conj(restriction).T
    k_mat = jnp.conj(restriction).T @ jnp.linalg.inv(g) @ restriction   # (4, 4)
    if W_CENTERED:
        # dv/v = (1/p) [sum K_lk b_l^* dH b_k - 2 sigma^* dH sigma], since tr(K phi) = 2.
        out = []
        for s, b in zip(sig_fp, b_fp):
            w = jnp.einsum('bik,lk,bjl->bij', b, k_mat, jnp.conj(b))
            w = w - 2.0 * jnp.einsum('bi,bj->bij', s, jnp.conj(s))
            out.append((v / p) * jnp.real(w))
        return tuple(out)

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
    return tuple(weights)


def whitened_rotations(lay):
    """A_b = W rot_b^T for every block, as a dd pair for each group, so tau = A sigma."""
    out = []
    for g in lay.groups:
        white = jnp.asarray(g.whiten)[jnp.asarray(g.orbit_of)]           # (b, j, k)
        rot = jnp.asarray(g.rot)                                         # (b, i, k)
        prod = dd.dot_dd_fp(dd.to_dd(white[:, :, None, :]), rot[:, None, :, :], axis=-1)
        out.append(dd.two_sum(*prod))                                    # (b, j, i)
    return tuple(out)


def _apply_rotation_dd(pair, vec):
    """sum over i of A[b, j, i] vec[b, i, c], for A a dd pair and vec complex dd."""
    a_hi, a_lo = pair

    def part(x):
        xs = (x[0][:, None], x[1][:, None])                              # (b, 1, i, c)
        main = dd.dot_dd_fp(xs, a_hi[:, :, :, None], axis=2)             # (b, j, c)
        # a_lo is 2^-53 of a_hi, so its product needs FP64 only.
        return dd.add(main, dd.to_dd(jnp.einsum('bji,bic->bjc', a_lo, x[0])))
    return part(vec[0]), part(vec[1])


def _volume_and_row(lay, mats, z, restriction, omega, *extra, rots=None):
    """v at one point and the whole row dv/dtheta, from tau and beta in dd.

    p_b = tau_b^* H_white tau_b with tau_b = W rot_b^T sigma_b, so the row of an
    orbit is scale <B_a, sum_b (v/p) Re(beta K beta^* - 2 tau tau^*)>. `rots` is
    `whitened_rotations(lay)`, formed once outside the loop over points.
    """
    if not (END_DD and GRAM_DD is True):
        raise ValueError('J_ROWS needs END_DD and GRAM_DD = True')
    rots = whitened_rotations(lay) if rots is None else rots
    z_dd, m, free = _dd_point(extra)
    sig, dsig, _, _ = _section_vectors(lay, z, z_dd=z_dd)
    p, _, _, _, extras = _gram_parts(lay, mats, sig, dsig)
    phi_dd, b_dd = _phi_stage5_dd(extras, sig, dsig, keep=True, with_b='dd')
    phi = dd.cto_fp(phi_dd)
    v_out = _volume_from_phi_dd(phi_dd, z_dd, m, free, keep=F_DD)
    v = _fp(v_out)
    g = restriction @ phi @ jnp.conj(restriction).T
    k_mat = jnp.conj(restriction).T @ jnp.linalg.inv(g) @ restriction   # (4, 4)

    row = jnp.zeros((lay.n_theta,), jnp.float64)
    for g_layout, pair, s, b in zip(lay.groups, rots, sig, b_dd):
        # sigma and the 4 columns of b as 5 columns, so one product serves both.
        cols = tuple(tuple(jnp.concatenate([s[c][h][..., None], b[c][h]], axis=-1)
                           for h in (0, 1)) for c in (0, 1))
        both = dd.cto_fp(_apply_rotation_dd(pair, cols))                  # (b, d, 5)
        tau, beta = both[..., 0], both[..., 1:]
        w = jnp.einsum('bik,lk,bjl->bij', beta, k_mat, jnp.conj(beta))
        w = w - 2.0 * jnp.einsum('bi,bj->bij', tau, jnp.conj(tau))
        w = (v / p) * jnp.real(w)
        for local, sl in enumerate(g_layout.slices):
            take = jnp.asarray(g_layout.orbit_of == local)
            summed = jnp.sum(jnp.where(take[:, None, None], w, 0.0), axis=0)
            basis = jnp.asarray(g_layout.basis[local, :sl.stop - sl.start])
            row = row.at[sl].add(jnp.einsum('aij,ij->a', basis, summed))
    return v_out, row * jnp.asarray(lay.scale)


def volume_tangent(weights, dmats):
    """dv along one direction, from the block matrices of that direction.

    dH_r is symmetric, so tr(dH_r W_r) is the plain Frobenius product.
    """
    return sum(jnp.einsum('bij,bij->', dmat, w)
               for dmat, w in zip(dmats, weights))


def volume_jacobian(lay, theta, z, restriction, omega, *extra):
    """v and the full dv/dtheta at one point. This is the reference for tests."""
    mats = point_matrices(lay, theta)
    v, weights = volume_and_weights(lay, mats, z, restriction, omega, *extra)

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
        return volume_jacobian(lay, theta, *args)

    xs = point_inputs(batch)
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
    mats = point_matrices(lay, theta)
    # lineax batches the tangent over every direction of theta, so it keeps the
    # FP64 map: the dd form would hold its intermediates once for each direction.
    dmats = sections.prepare_fp(lay, dtheta)

    def one(args):
        v, w = volume_and_weights(lay, mats, *args)
        return v, volume_tangent(w, dmats)

    def run(fn, inputs):
        return (jax.vmap(fn)(inputs) if chunk is None
                else jax.lax.map(fn, inputs, batch_size=chunk))

    xs = point_inputs(batch)
    if J_ROWS:
        rots = whitened_rotations(lay)

        def one_row(args):
            v, row = _volume_and_row(lay, mats, *args, rots=rots)
            return v, jnp.dot(row, dtheta)

        v, dv = run(one_row, xs)
    elif REMAT_W:
        def primal(args):
            v, small, _, _, _ = _point_state(lay, mats, *args)
            return v, small

        # W comes again from the sections, so the linearization keeps only
        # v and the per-point values, and not W for every point.
        @jax.checkpoint
        def tangent(args):
            (z, restriction, _omega, *extra), v, small = args
            if W_CENTERED:
                point = _dd_point(extra) if END_DD else None
                sig, dsig, sig_fp, _ = _section_vectors(
                    lay, z, z_dd=None if point is None else point[0])
                b_fp = _centered_dd(small[4], small[5], sig, dsig)
                weights = _block_weights(sig_fp, None, small, restriction, _fp(v), b_fp)
            else:
                sig, dsig = section_vectors(lay, z)
                weights = _block_weights(sig, dsig, small, restriction, _fp(v))
            return volume_tangent(weights, dmats)

        v, small = run(primal, xs)
        dv = run(tangent, (xs, v, small))
    else:
        v, dv = run(one, xs)

    weights = batch['mass'] / jnp.sum(batch['mass'])
    f = residual_from_volumes(v, weights)
    v = _fp(v)
    mean = jnp.sum(weights * v)
    d_mean = jnp.sum(weights * dv)
    root = jnp.sqrt(weights)
    return root * f, root * (dv / mean - v * d_mean / mean ** 2)
