"""The functions of low degree of a V+A level, and their Hessians on X, in dd.

The function of a level that is almost constant on X lies on the functions of
low degree, and the fit puts a coefficient of order 1 on it. Its Hessian comes
from terms of the size of k that cancel, so those functions need dd. A jet of a
real function F at a point is (F, D_k F, D_k Dbar_l F), with D_k = T_k . d for
the two rows T_k of the restriction matrix, all in complex dd. The multipliers
come from their monomials in closed form, and the recurrence of
`arnoldi.run_recurrence` then runs on jets. Every product is the exact `dd.mul`:
the recurrence amplifies the 2^-79 of `dd.mul_fast` by about 3 for each degree.
"""
from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from . import analytic, arnoldi, dd, dd_metric

__all__ = ['jet_mul', 'multiplier_jets', 'basis_multiplier_jets', 'weight_jets',
           'weight_dd', 'low_jets', 'low_hessians']


def _conj(a):
    return a[0], dd.neg(a[1])


def _creal(a, x):
    """A complex dd value times a real dd value."""
    return dd.mul(a[0], x), dd.mul(a[1], x)


def _csum(a, axis):
    return dd.sum_dd(*a[0], axis=axis), dd.sum_dd(*a[1], axis=axis)


def _ex(x, *axes):
    """Insert new axes at the end of every leaf, for broadcasting."""
    return jtu.tree_map(lambda a: a[(Ellipsis,) + (None,) * len(axes)], x)


def _outer(a, b):
    """(a_k conj(b_l)) for complex dd vectors a, b with the last axis 2, as (..., 2, 2)."""
    left = analytic._cindex(a, Ellipsis, slice(None), None)
    right = _conj(analytic._cindex(b, Ellipsis, None, slice(None)))
    return dd.cmul(left, right)


def jet_mul(f, g):
    """The jet of F G from the jets of two real functions F and G."""
    fv, fd, fh = f
    gv, gd, gh = g
    v = dd.mul(fv, gv)
    d = dd.cadd(_creal(fd, _ex(gv, 0)), _creal(gd, _ex(fv, 0)))
    h = dd.cadd(dd.cadd(_creal(fh, _ex(gv, 0, 0)), _creal(gh, _ex(fv, 0, 0))),
                dd.cadd(_outer(fd, gd), _outer(gd, fd)))
    return v, d, h


def _radius_jets(z_dd, rest):
    """|z|^2 as a real dd value, D_k |z|^2 and D_k Dbar_l |z|^2, at one point."""
    r2 = dd.add(dd.sum_dd(*dd.mul(z_dd[0], z_dd[0]), axis=0),
                dd.sum_dd(*dd.mul(z_dd[1], z_dd[1]), axis=0))
    zbar = _conj(analytic._cindex(z_dd, None, slice(None)))
    d = _csum(dd.cmul(rest, zbar), axis=-1)                          # (2,)
    left = analytic._cindex(rest, slice(None), None, slice(None))
    right = _conj(analytic._cindex(rest, None, slice(None), slice(None)))
    h = _csum(dd.cmul(left, right), axis=-1)                         # (2, 2)
    return r2, d, h


def _inverse_power_jets(r2, d_r2, h_r2, degs):
    """The jets of |z|^(-2d) for every d in `degs`, from the jets of |z|^2."""
    inv = dd.reciprocal(r2)
    top = int(np.max(degs))
    pows = [dd.to_dd(jnp.ones(()))]
    for _ in range(top + 2):
        pows.append(dd.mul(pows[-1], inv))
    stack = (jnp.stack([p[0] for p in pows]), jnp.stack([p[1] for p in pows]))
    idx = jnp.asarray(degs)
    rho = jtu.tree_map(lambda a: a[idx], stack)                             # r2^-d
    rho1 = jtu.tree_map(lambda a: a[idx + 1], stack)                        # r2^-(d+1)
    rho2 = jtu.tree_map(lambda a: a[idx + 2], stack)                        # r2^-(d+2)
    fd = jnp.asarray(degs, jnp.float64)
    d = analytic._cscale(_creal(jtu.tree_map(lambda a: a[None], d_r2),
                                _ex(rho1, 0)), -fd[:, None])
    outer = _outer(d_r2, d_r2)                                              # (2, 2)
    first = analytic._cscale(_creal(jtu.tree_map(lambda a: a[None], outer),
                                    _ex(rho2, 0, 0)), (fd * (fd + 1.0))[:, None, None])
    second = analytic._cscale(_creal(jtu.tree_map(lambda a: a[None], h_r2),
                                     _ex(rho1, 0, 0)), -fd[:, None, None])
    return rho, d, dd.cadd(first, second)


def multiplier_jets(entries, z_dd, rest):
    """The jets of every multiplier Re(sum over pi of z_pi^A conj(z_pi)^B) / |z|^(2d).

    `entries` are the (d, A, B) of the basis, `z_dd` the dd point and `rest` the dd
    restriction matrix (2, 4). Each part of the jet has the leading axis E.
    """
    a_exp = np.stack([np.asarray(a) for _, a, _ in entries])               # (E, 4)
    b_exp = np.stack([np.asarray(b) for _, _, b in entries])
    degs = np.array([d for d, _, _ in entries])
    n_e, n_p = len(entries), arnoldi.PERMS.shape[0]
    ea = np.zeros((n_e, n_p, 4), np.int64)
    eb = np.zeros((n_e, n_p, 4), np.int64)
    for p, perm in enumerate(arnoldi.PERMS):
        ea[:, p, perm] = a_exp
        eb[:, p, perm] = b_exp
    top = int(max(ea.max(), eb.max()))
    pw = [dd.cto_dd(jnp.ones(4, jnp.complex128))]
    for _ in range(top):
        pw.append(dd.cmul(pw[-1], z_dd))
    pw = jtu.tree_map(lambda *xs: jnp.stack(xs), *pw)                       # (top+1, 4)
    cols = jnp.arange(4)

    def monomial(exps):
        part = jtu.tree_map(lambda a: a[jnp.asarray(exps), cols], pw)      # (E, P, 4)
        out = analytic._cindex(part, Ellipsis, 0)
        for c in range(1, 4):
            out = dd.cmul(out, analytic._cindex(part, Ellipsis, c))
        return out
    m = dd.cmul(monomial(ea), _conj(monomial(eb)))                  # (E, P)
    tz = dd.cmul(rest, analytic._cindex(dd.creciprocal(z_dd), None, slice(None)))
    tz_b = analytic._cindex(tz, None, None, slice(None), slice(None))      # (1, 1, 2, 4)
    alpha = _csum(analytic._cscale(tz_b, jnp.asarray(ea, jnp.float64)[:, :, None, :]), -1)
    beta = _csum(analytic._cscale(_conj(tz_b),
                                  jnp.asarray(eb, jnp.float64)[:, :, None, :]), -1)
    m_k = analytic._cindex(m, Ellipsis, None)
    s = _csum(m, axis=1)                                                    # (E,)
    ds = _csum(dd.cmul(m_k, alpha), axis=1)                          # (E, 2)
    dbs = _csum(dd.cmul(m_k, beta), axis=1)                          # (E, 2)
    al = analytic._cindex(alpha, Ellipsis, slice(None), None)
    be = analytic._cindex(beta, Ellipsis, None, slice(None))
    hs = _csum(dd.cmul(analytic._cindex(m, Ellipsis, None, None),
                              dd.cmul(al, be)), axis=1)              # (E, 2, 2)
    half = jnp.asarray(0.5)
    f_v = s[0]
    f_d = analytic._cscale(dd.cadd(ds, _conj(dbs)), half)
    hs_t = jtu.tree_map(lambda a: jnp.swapaxes(a, -1, -2), hs)
    f_h = analytic._cscale(dd.cadd(hs, _conj(hs_t)), half)
    r2, d_r2, h_r2 = _radius_jets(z_dd, rest)
    return jet_mul((f_v, f_d, f_h), _inverse_power_jets(r2, d_r2, h_r2, degs))


def basis_multiplier_jets(basis, z_dd, rest):
    """The jets of the multipliers that the recurrence of `basis` takes, (y - shift) / scale."""
    v, d, h = multiplier_jets(basis.entries, z_dd, rest)
    if basis.shift is None:
        return v, d, h
    inv = dd.reciprocal(dd.to_dd(jnp.asarray(basis.scale)))
    v = dd.mul(dd.add(v, dd.to_dd(-jnp.asarray(basis.shift))), inv)
    return v, _creal(d, _ex(inv, 0)), _creal(h, _ex(inv, 0, 0))


def alpha_dd(alpha):
    """alpha in dd: k1/k0 to 2^-106 for a Fraction, the FP64 value for a float.

    W itself takes the exact exponent, so its jets must take the exact one too.
    """
    if isinstance(alpha, Fraction):
        inv = dd.reciprocal(dd.to_dd(jnp.asarray(float(alpha.denominator))))
        return dd.mul_dd_fp(inv, jnp.asarray(float(alpha.numerator)))
    return dd.to_dd(jnp.asarray(alpha, jnp.float64))


def weight_jets(hi, z_dd, rest, alpha, k1):
    """D_k L and D_k Dbar_l L for L = k1 log |z|^2 - alpha log p_0, in dd.

    `hi` is the dd state (ddbar log p_0, d log p_0) of `correction.base_state_dd`.
    """
    phi_dd, a_dd = hi
    r2, d_r2, h_r2 = _radius_jets(z_dd, rest)
    inv = dd.reciprocal(r2)
    a = (a_dd[0], a_dd[1])
    ta = _csum(dd.cmul(rest, analytic._cindex(a, None, slice(None))), axis=-1)
    left = analytic._cindex(rest, slice(None), None, slice(None), None)
    mid = analytic._cindex(phi_dd, None, None, slice(None), slice(None))
    right = _conj(analytic._cindex(rest, None, slice(None), None, slice(None)))
    tpt = dd_metric._creduce_last_two(dd.cmul(dd.cmul(left, mid), right))
    scale = alpha_dd(alpha)
    d_l = dd.csub(analytic._cscale(_creal(d_r2, _ex(inv, 0)), jnp.asarray(float(k1))),
                  _creal(ta, _ex(scale, 0)))
    fs = dd.csub(_creal(h_r2, _ex(inv, 0, 0)),
                 _creal(_outer(d_r2, d_r2), _ex(dd.mul(inv, inv), 0, 0)))
    h_l = dd.csub(analytic._cscale(fs, jnp.asarray(float(k1))),
                  _creal(tpt, _ex(scale, 0, 0)))
    return d_l, h_l


def _power(x, n):
    """x^n for a real dd value x and an integer n >= 1, by repeated squares."""
    out, base = None, x
    while True:
        if n & 1:
            out = base if out is None else dd.mul(out, base)
        n >>= 1
        if not n:
            return out
        base = dd.mul(base, base)


def weight_dd(p_dd, r2, k0, k1):
    """W = |z|^(2 k1) p_0^(-k1/k0) in dd, from p_0 and |z|^2 in dd.

    t = p_0^(-1/k0) comes from one Newton step t (1 + (1 - p_0 t^k0) / k0) on the
    FP64 root, which squares its relative error, and then W = (|z|^2 t)^k1.
    """
    t = dd.to_dd(p_dd[0] ** (-1.0 / k0))
    gap = dd.add(dd.to_dd(jnp.ones_like(p_dd[0])), dd.neg(dd.mul(p_dd, _power(t, k0))))
    t = dd.add(t, dd.mul(t, dd.mul_dd_fp(gap, jnp.asarray(1.0 / k0))))
    return _power(dd.mul(r2, t), k1)


def low_jets(basis, n_low, y_jets):
    """The jets of q_t for t < n_low, from the jets of the multipliers of the basis."""
    if np.any(np.asarray(basis.seed_slot)[:n_low] >= 0):
        raise ValueError('a seed lies in the low part; take a smaller n_low')
    h = jnp.asarray(basis.h[:n_low, :n_low])
    inv_h = dd.reciprocal(dd.to_dd(jnp.diag(h)))
    parents = jnp.asarray(np.maximum(basis.parents[:n_low], 0))
    mults = jnp.asarray(np.maximum(basis.mults[:n_low], 0))
    yv, yd, yh = y_jets
    zero_v = dd.to_dd(jnp.zeros(n_low))
    zero_d = dd.cto_dd(jnp.zeros((n_low, 2), jnp.complex128))
    zero_h = dd.cto_dd(jnp.zeros((n_low, 2, 2), jnp.complex128))
    first = (jtu.tree_map(lambda a: a[0], inv_h), dd.cto_dd(jnp.zeros(2, jnp.complex128)),
             dd.cto_dd(jnp.zeros((2, 2), jnp.complex128)))
    jets = jtu.tree_map(lambda arr, x: arr.at[0].set(x), (zero_v, zero_d, zero_h), first)

    def take(tree, i):
        return jtu.tree_map(lambda a: a[i], tree)

    def step(t, jets):
        grown = jet_mul(take((yv, yd, yh), mults[t]), take(jets, parents[t]))
        col = h[:, t]
        acc_v = dd.dot_dd_fp(jets[0], col, axis=0)
        acc_d = tuple(dd.dot_dd_fp(part, col[:, None], axis=0) for part in jets[1])
        acc_h = tuple(dd.dot_dd_fp(part, col[:, None, None], axis=0) for part in jets[2])
        r = take(inv_h, t)
        new = (dd.mul(dd.add(grown[0], dd.neg(acc_v)), r),
               _creal(dd.csub(grown[1], acc_d), _ex(r, 0)),
               _creal(dd.csub(grown[2], acc_h), _ex(r, 0, 0)))
        return jtu.tree_map(lambda arr, x: arr.at[t].set(x), jets, new)
    return jax.lax.fori_loop(1, n_low, step, jets)


def low_hessians(basis, n_low, z_dd, rest, hi, alpha, weight):
    """R ddbar(W q_t) R^dag for t < n_low in complex dd, as (n_low, 2, 2).

    W = w |z|^(2k) and `weight` is its FP64 value at the point, a factor common
    to every function, so its rounding moves no cancellation.
    """
    q_v, q_d, q_h = low_jets(basis, n_low, basis_multiplier_jets(basis, z_dd, rest))
    d_l, h_l = weight_jets(hi, z_dd, rest, alpha, basis.k)
    d_lb = analytic._cindex(d_l, None)
    h_lb = analytic._cindex(h_l, None)
    lead = dd.cadd(dd.cadd(q_h, _outer(q_d, d_lb)), _outer(d_lb, q_d))
    rest_term = _creal(dd.cadd(h_lb, analytic._cindex(_outer(d_l, d_l), None)), _ex(q_v, 0, 0))
    return analytic._cscale(dd.cadd(lead, rest_term), weight)
