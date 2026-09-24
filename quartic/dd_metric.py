"""`v` end to end in double-double, for stage 2.

The accumulators of `analytic` put the Gram sums and the section vectors in dd
and then round phi to FP64. That is enough while sigma stays above
approximately 1e-15, and it is not enough below it, because **the cancellation
comes after phi**:

    phi -> g|X -> det -> v -> v - <v> -> f

`f = 1 - <v>/v` cancels by the whole size of `v` when the fit is good, so `v`
must carry the extra digits into that subtraction. This module keeps the chain
in dd from the point itself to `v`.

`f` can come back in FP64. It is small, so its FP64 form is accurate in the
relative sense: at `f = 1e-20` the rounding is 1e-36. The error was never in
holding `f`; it was in computing it.

**The point must be dd as well.** For an FP64 `z` the nearest representable root
still leaves `abs(P)/scale` near 1e-16, and that error enters `v` with an
amplification of 1.5 to 6.6. `refine` takes one Newton step in dd, which
reaches 1e-31 because Newton is quadratic.

This path is a reference and not a fast path. It loops over the 64 blocks in
Python, so use it on a few points to measure the floor, and not inside a fit.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import analytic, dd, sections

__all__ = ['refine', 'newton_step', 'residual_of_P', 'phi_dd', 'volume_dd',
           'point_extras', 'residual_of_P_batch']


def _reduce(pair, n_axes):
    """Sum a double-double array over its first `n_axes` axes."""
    hi, lo = pair
    shape = (-1,) + hi.shape[n_axes:]
    return dd.sum_dd(jnp.reshape(hi, shape), jnp.reshape(lo, shape), axis=0)


def _creduce(val, n_axes):
    """The same, for a complex double-double value."""
    return _reduce(val[0], n_axes), _reduce(val[1], n_axes)


def _matvec(mat, vec, extra):
    """`mat @ vec` over the last axis of `mat`, with `vec` complex dd.

    `mat` is a real FP64 array of shape `(blocks, d, d)`, or a (hi, lo) pair of
    them as `sections.prepare_parts` gives. `vec` carries the block axis first
    and `extra` trailing axes after the `d` axis.
    """
    mat, low = mat if isinstance(mat, tuple) else (mat, None)
    idx = (slice(None), None) + (slice(None),) * (1 + extra)
    pad = (slice(None), slice(None), slice(None)) + (None,) * extra
    mat_b = jnp.asarray(mat)[pad]

    def part(component):
        out = dd.dot_dd_fp((component[0][idx], component[1][idx]), mat_b, axis=2)
        if low is not None:
            # lo is 2^-53 of mat, so its product needs FP64 only.
            extra_part = jnp.sum(component[0][idx] * jnp.asarray(low)[pad], axis=2)
            out = dd.add(out, dd.to_dd(extra_part))
        return out
    return part(vec[0]), part(vec[1])


def _outer(vec, ratio, extra):
    """`vec ⊗ ratio`, with `vec` of shape (blocks, d) and `ratio` of shape (4,)."""
    left = analytic._cindex(vec, slice(None), slice(None), None)
    right = analytic._cindex(ratio, None, None, slice(None))
    del extra
    return dd.cmul(left, right)


def phi_dd(lay, mats, z_dd):
    """`phi_{i jbar}` and `p` at one point, entirely in complex double-double.

    The centered form is the same as `analytic.phi_analytic`: with
    `b_i = dsigma_i - (p_i/p) sigma`, phi is a Gram matrix, so the cancellation
    of the difference form goes into a vector subtraction.
    """
    sig, dsig = analytic.section_vectors_dd(lay, None, z_dd=z_dd)

    p = dd.to_dd(jnp.zeros(()))
    p_i = (dd.to_dd(jnp.zeros(4)), dd.to_dd(jnp.zeros(4)))
    store = []
    for mat, s, ds in zip(mats, sig, dsig):
        h_s = _matvec(mat, s, 0)                      # (blocks, d)
        h_d = _matvec(mat, ds, 1)                     # (blocks, d, 4)
        store.append((h_s, h_d))

        s_conj = (s[0], dd.neg(s[1]))
        p = dd.add(p, _reduce(dd.cmul(s_conj, h_s)[0], 2))
        term = dd.cmul(analytic._cindex(s_conj, slice(None), slice(None), None),
                       h_d)
        p_i = dd.cadd(p_i, _creduce(term, 2))

    # p is real, so divide each part of p_i by it.
    inv_p = dd.reciprocal(p)
    ratio = (dd.mul(p_i[0], inv_p), dd.mul(p_i[1], inv_p))

    phi = (dd.to_dd(jnp.zeros((4, 4))), dd.to_dd(jnp.zeros((4, 4))))
    for (h_s, h_d), s, ds in zip(store, sig, dsig):
        b = dd.csub(ds, _outer(s, ratio, 0))
        h_b = dd.csub(h_d, _outer(h_s, ratio, 0))
        b_conj = (b[0], dd.neg(b[1]))
        # phi[i][j] = sum over blocks and d of conj(b_j) h_b_i
        left = analytic._cindex(h_b, slice(None), slice(None),
                                slice(None), None)
        right = analytic._cindex(b_conj, slice(None), slice(None),
                                 None, slice(None))
        phi = dd.cadd(phi, _creduce(dd.cmul(left, right), 2))
    return (dd.mul(phi[0], inv_p), dd.mul(phi[1], inv_p)), p


def _restriction_dd(z_dd, m, free):
    """The (2, 4) restriction matrix in complex double-double.

    Row i is `dz/dt_i`: a 1 in the free slot of `t_i` and
    `-(dP/dt_i)/(dP/dz_m)` in slot `m`.
    """
    z2 = dd.cmul(z_dd, z_dd)
    cube = dd.cmul(z2, z_dd)
    grad = (dd.mul_dd_fp(cube[0], jnp.asarray(4.0)),
            dd.mul_dd_fp(cube[1], jnp.asarray(4.0)))
    inv_m = dd.creciprocal(analytic._cindex(grad, m))

    rows_re_hi, rows_re_lo, rows_im_hi, rows_im_lo = [], [], [], []
    for i in range(2):
        one = jnp.zeros(4).at[free[i]].set(1.0)
        col = dd.cneg(dd.cmul(analytic._cindex(grad, free[i]), inv_m))
        mask = jnp.zeros(4).at[m].set(1.0)
        re_hi = one + col[0][0] * mask
        re_lo = col[0][1] * mask
        im_hi = col[1][0] * mask
        im_lo = col[1][1] * mask
        rows_re_hi.append(re_hi), rows_re_lo.append(re_lo)
        rows_im_hi.append(im_hi), rows_im_lo.append(im_lo)
    return ((jnp.stack(rows_re_hi), jnp.stack(rows_re_lo)),
            (jnp.stack(rows_im_hi), jnp.stack(rows_im_lo))), grad


def volume_dd(lay, mats, z_dd, m, free):
    """`v = det(g|X) / OmegaOmegabar` at one point, in double-double."""
    rest, grad = _restriction_dd(z_dd, m, free)
    phi, _ = phi_dd(lay, mats, z_dd)

    rest_conj = (rest[0], dd.neg(rest[1]))
    # g[i][j] = sum_{a b} R[i,a] phi[a,b] conj(R[j,b])
    left = analytic._cindex(rest, slice(None), None, slice(None), None)
    mid = analytic._cindex(phi, None, None, slice(None), slice(None))
    right = analytic._cindex(rest_conj, None, slice(None), None, slice(None))
    g = _creduce_last_two(dd.cmul(dd.cmul(left, mid), right))

    det = dd.csub(dd.cmul(_at(g, 0, 0), _at(g, 1, 1)),
                  dd.cmul(_at(g, 0, 1), _at(g, 1, 0)))
    # OmegaOmegabar = 1/abs(dP/dz_m)^2, so v = det * abs(dP/dz_m)^2.
    return dd.mul(det[0], dd.cabs2(analytic._cindex(grad, m)))


def _at(val, i, j):
    """One entry of a complex double-double matrix."""
    return analytic._cindex(val, i, j)


def _creduce_last_two(val):
    """Sum a complex dd array of shape (2, 2, 4, 4) over its last two axes."""
    def one(pair):
        hi, lo = pair
        shape = hi.shape[:2] + (-1,)
        return dd.sum_dd(jnp.reshape(hi, shape), jnp.reshape(lo, shape), axis=-1)
    return one(val[0]), one(val[1])


def newton_index(z):
    """The coordinate that Newton corrects, in numpy.

    It is the largest `abs(dP/dz)` outside the affine patch, which is the choice
    that the sampler makes. It must come from numpy, because `jax.jit` cannot
    take an index from a traced value.
    """
    z_np = np.asarray(z)
    patch = int(np.argmin(np.abs(z_np - 1.0)))
    grad = np.abs(4.0 * z_np ** 3)
    grad[patch] = -1.0
    return int(np.argmax(grad))


def refine(z, m, steps=2):
    """Newton steps onto `P = 0` in double-double, in coordinate `m`.

    Newton is quadratic, so 1e-16 becomes 1e-31 in one step.
    """
    zc = dd.cto_dd(jnp.asarray(z))
    for _ in range(steps):
        zc = newton_step(zc, m)
    return zc, m


def newton_step(zc, m):
    """One Newton step onto `P = 0` for a complex dd point, in coordinate `m`."""
    z2 = dd.cmul(zc, zc)
    u = dd.cmul(z2, z2)
    total = _creduce(u, 1)
    z2_m = analytic._cindex(z2, m)
    cube = dd.cmul(z2_m, analytic._cindex(zc, m))
    deriv = (dd.mul_dd_fp(cube[0], jnp.asarray(4.0)),
             dd.mul_dd_fp(cube[1], jnp.asarray(4.0)))
    step = dd.cdiv(total, deriv)
    new_m = dd.csub(analytic._cindex(zc, m), step)
    mask = jnp.zeros(4).at[m].set(1.0)
    keep = 1.0 - mask
    return ((zc[0][0] * keep + new_m[0][0] * mask,
             zc[0][1] * keep + new_m[0][1] * mask),
            (zc[1][0] * keep + new_m[1][0] * mask,
             zc[1][1] * keep + new_m[1][1] * mask))


def residual_of_P(z_dd):
    """`abs(P)/scale` for a double-double point, as an FP64 number."""
    z2 = dd.cmul(z_dd, z_dd)
    u = dd.cmul(z2, z2)
    total = _creduce(u, 1)
    scale = float(jnp.sum(jnp.abs(dd.cto_fp(u))))
    return float(jnp.abs(dd.cto_fp(total))) / scale



def point_extras(points, chunk=200_000):
    """The dd point after one Newton step, and its patch indices, for a sample.

    Returns `points_hi` and `points_lo`, the complex high and low parts, and
    `patch`, the int32 rows (m, free_0, free_1) that `analytic.point_inputs`
    takes with `END_DD`. m follows `newton_index`. The coordinate of the patch
    becomes exactly 1, because OmegaOmegabar = 1/abs(dP/dz_m)^2 holds in that
    chart only.
    """
    z = np.array(points)
    n = z.shape[0]
    rows = np.arange(n)
    patch = np.argmin(np.abs(z - 1.0), axis=1)
    z[rows, patch] = 1.0
    grad = np.abs(4.0 * z ** 3)
    grad[rows, patch] = -1.0
    m = np.argmax(grad, axis=1)
    mask = np.ones((n, 4), dtype=bool)
    mask[rows, patch] = False
    mask[rows, m] = False
    free = np.nonzero(mask)[1].reshape(n, 2)

    step = jax.jit(jax.vmap(lambda zz, mm: refine(zz, mm, 1)[0]))
    his, los = [], []
    for start in range(0, n, chunk):
        (re_hi, re_lo), (im_hi, im_lo) = step(jnp.asarray(z[start:start + chunk]),
                                              jnp.asarray(m[start:start + chunk]))
        his.append(np.asarray(re_hi) + 1j * np.asarray(im_hi))
        los.append(np.asarray(re_lo) + 1j * np.asarray(im_lo))
    patch_rows = np.stack([m, free[:, 0], free[:, 1]], axis=1).astype(np.int32)
    return {'points_hi': jnp.asarray(np.concatenate(his)),
            'points_lo': jnp.asarray(np.concatenate(los)),
            'patch': jnp.asarray(patch_rows)}


def residual_of_P_batch(points_hi, points_lo, chunk=200_000):
    """`abs(P)/scale` at each dd point of a sample, as FP64."""
    def one(hi, lo):
        z_dd = ((jnp.real(hi), jnp.real(lo)), (jnp.imag(hi), jnp.imag(lo)))
        z2 = dd.cmul(z_dd, z_dd)
        u = dd.cmul(z2, z2)
        return jnp.abs(dd.cto_fp(_creduce(u, 1))) / jnp.sum(jnp.abs(dd.cto_fp(u)))

    run = jax.jit(jax.vmap(one))
    n = points_hi.shape[0]
    return np.concatenate([np.asarray(run(points_hi[s:s + chunk], points_lo[s:s + chunk]))
                           for s in range(0, n, chunk)])
