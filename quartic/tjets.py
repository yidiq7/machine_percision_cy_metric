"""Taylor jets of bidegree (2, 2) in the chart coordinates of X, in complex dd.

In a chart z_j = 1, the free coordinates are t = (t_1, t_2), and z_m solves P = 0.
A function F near the point t0 is F(t0 + s) = sum over a, b of F[a, b] s^M[a]
conj(s)^M[b], up to terms of degree 3 in s or in conj(s), where M holds the 6
multi-indices of degree at most 2 in two variables. These 36 Taylor coefficients
hold every derivative that the gradient of f, its Laplacian and the Ricci form
need, up to D D Dbar Dbar of the potential.

A holomorphic jet is a complex dd array with the last axis 6, and a jet of
bidegree (2, 2) has the last two axes (6, 6). For a real function F the jet is
Hermitian, F[b, a] = conj(F[a, b]), and `flat` keeps 36 real rows of it. Every
product is the exact `dd.mul`, so the jets keep the digits of dd.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from . import analytic, dd, dd_metric

__all__ = ['M', 'hol_mul', 'mul', 'outer', 'hermitize', 'series', 'log1p_nil', 'exp_nil',
           'pow_nil', 'nil', 'chart_jets', 'power_table', 'monomial_jets', 'radius_jet',
           'base_jet', 'multiplier_jets', 'weight_jet', 'T36', 'metric_parts']

M = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
_POS = {m: i for i, m in enumerate(M)}


def _pairs():
    """For each a, the pairs (a1, a2) with M[a1] + M[a2] = M[a], padded with 6."""
    p1, p2 = np.full((6, 4), 6), np.full((6, 4), 6)
    for a, (i, j) in enumerate(M):
        k = 0
        for a1, (i1, j1) in enumerate(M):
            a2 = _POS.get((i - i1, j - j1))
            if a2 is not None:
                p1[a, k], p2[a, k] = a1, a2
                k += 1
    return p1, p2


P1, P2 = _pairs()
_TRIU = [(a, b) for a in range(6) for b in range(a + 1, 6)]


def _tmap(f, x):
    return jtu.tree_map(f, x)


def _pad_last(x, n_axes):
    """One zero entry appended to each of the last `n_axes` axes of every leaf."""
    def one(a):
        for k in range(1, n_axes + 1):
            shape = list(a.shape)
            shape[-k] = 1
            a = jnp.concatenate([a, jnp.zeros(shape, a.dtype)], axis=-k)
        return a
    return _tmap(one, x)


def _csum(x, axis):
    return dd.sum_dd(*x[0], axis=axis), dd.sum_dd(*x[1], axis=axis)


def _conj(x):
    return x[0], dd.neg(x[1])


def hol_mul(f, g):
    """The product of two holomorphic jets, (..., 6)."""
    fa = _tmap(lambda a: a[..., P1], _pad_last(f, 1))
    ga = _tmap(lambda a: a[..., P2], _pad_last(g, 1))
    return _csum(dd.cmul(fa, ga), -1)


def mul(f, g):
    """The product of two jets of bidegree (2, 2), (..., 6, 6)."""
    ia, ib = P1[:, :, None, None], P1[None, None, :, :]
    ja, jb = P2[:, :, None, None], P2[None, None, :, :]
    fa = _tmap(lambda a: a[..., ia, ib], _pad_last(f, 2))          # (..., 6, 4, 6, 4)
    ga = _tmap(lambda a: a[..., ja, jb], _pad_last(g, 2))
    prod = dd.cmul(fa, ga)
    prod = _tmap(lambda a: jnp.moveaxis(a, -3, -1).reshape(a.shape[:-4] + (6, 6, 16)), prod)
    return _csum(prod, -1)


def outer(f, g):
    """The jet of F conj(G) for holomorphic jets f and g: f[a] conj(g[b])."""
    left = analytic._cindex(f, Ellipsis, slice(None), None)
    right = _conj(analytic._cindex(g, Ellipsis, None, slice(None)))
    return dd.cmul(left, right)


def hermitize(f):
    """The jet of Re F from the jet of F: (F[a, b] + conj(F[b, a])) / 2."""
    ft = _conj(_tmap(lambda a: jnp.swapaxes(a, -1, -2), f))
    s = dd.cadd(f, ft)
    return _tmap(lambda a: 0.5 * a, s)


def nil(f):
    """The jet without its value, the part that vanishes at the point."""
    mask = np.ones((6, 6))
    mask[0, 0] = 0.0
    return _tmap(lambda a: a * mask, f)


def value(f):
    """The value of a real jet, a real dd number."""
    return _tmap(lambda a: a[..., 0, 0], f[0])


def _add_value(f, c):
    """f plus the real dd number c at the value entry."""
    re = f[0]
    upd = dd.add((re[0][..., 0, 0], re[1][..., 0, 0]), c)
    re = (re[0].at[..., 0, 0].set(upd[0]), re[1].at[..., 0, 0].set(upd[1]))
    return re, f[1]


def _scale(f, c):
    """A jet times a real dd number c, which broadcasts against the batch axes."""
    c = _tmap(lambda a: a[..., None, None], c)
    return dd.mul(f[0], c), dd.mul(f[1], c)


def _const_dd(x, like):
    return dd.to_dd(jnp.full(like.shape[:-2], x, like.dtype))


def series(nj, coeffs):
    """sum over n of coeffs[n] nj^n for a jet nj with no value, n <= 4, by Horner.

    The parts of degree 5 and higher vanish, so 5 coefficients are exact. Each
    coefficient is a real dd number or an FP64 number.
    """
    like = nj[0][0]
    def as_dd(c):
        return _const_dd(c, like) if not isinstance(c, tuple) else c
    zero = _tmap(lambda a: jnp.zeros_like(a), nj)
    out = _add_value(zero, as_dd(coeffs[-1]))
    for c in reversed(coeffs[:-1]):
        out = _add_value(mul(nj, out), as_dd(c))
    return out


def _third(like):
    return dd.reciprocal(_const_dd(3.0, like))


def log1p_nil(x):
    """The jet of log(1 + X) for a jet x with no value."""
    like = x[0][0]
    return series(x, [0.0, 1.0, -0.5, _third(like), -0.25])


def exp_nil(x):
    """The jet of exp(X) for a jet x with no value."""
    like = x[0][0]
    sixth = dd.reciprocal(_const_dd(6.0, like))
    inv24 = dd.reciprocal(_const_dd(24.0, like))
    return series(x, [1.0, 1.0, 0.5, sixth, inv24])


def pow_nil(x, p):
    """The jet of (1 + X)^p for a jet x with no value and an integer p."""
    c = [1.0]
    for n in range(1, 5):
        c.append(c[-1] * (p - n + 1) / n)
    # For an integer p every coefficient is an integer, so FP64 holds it exactly.
    return series(x, [float(round(v)) for v in c])


@dataclass(frozen=True)
class Chart:
    z: tuple          # complex dd point (4,)
    m: object
    free: tuple
    rest: tuple       # complex dd (2, 4): dz/dt
    hol: tuple        # complex dd (4, 6): the holomorphic jets of z_a(t)


def chart_jets(z_dd, m, free):
    """The holomorphic jets of the 4 coordinates along X in the chart of the point.

    z_j = 1 is constant, z_free_k = t_k, and z_m solves z_m^4 = -(1 + t_1^4 + t_2^4).
    With R_k = dz_m/dt_k = -t_k^3/z_m^3 from the restriction matrix, the second
    derivatives are d_i d_k z_m = -3 (t_k^2 delta_ik + z_m^2 R_i R_k) / z_m^3.
    """
    rest, _ = dd_metric._restriction_dd(z_dd, m, free)
    zm = analytic._cindex(z_dd, m)
    inv_zm3 = dd.creciprocal(dd.cmul(dd.cmul(zm, zm), zm))
    zm2 = dd.cmul(zm, zm)
    r = [analytic._cindex(rest, k, m) for k in range(2)]
    t = [analytic._cindex(z_dd, free[k]) for k in range(2)]

    def second(i, k):
        term = dd.cmul(zm2, dd.cmul(r[i], r[k]))
        if i == k:
            term = dd.cadd(term, dd.cmul(t[k], t[k]))
        return analytic._cscale(dd.cmul(term, inv_zm3), jnp.asarray(-3.0))
    half = jnp.asarray(0.5)
    z20 = analytic._cscale(second(0, 0), half)
    z11 = second(0, 1)
    z02 = analytic._cscale(second(1, 1), half)
    idx = jnp.arange(4)
    is_m = (idx == m).astype(jnp.float64)
    is_f0 = (idx == free[0]).astype(jnp.float64)
    is_f1 = (idx == free[1]).astype(jnp.float64)

    def comp(val_m, base=None):
        """A component of the 4 coordinate jets: val_m in slot m and `base` elsewhere."""
        re = (val_m[0][0] * is_m, val_m[0][1] * is_m)
        im = (val_m[1][0] * is_m, val_m[1][1] * is_m)
        if base is not None:
            re = (re[0] + base, re[1])
        return re, im
    c0 = z_dd
    c1 = comp(r[0], is_f0)
    c2 = comp(r[1], is_f1)
    c3, c4, c5 = comp(z20), comp(z11), comp(z02)
    hol = _tmap(lambda *xs: jnp.stack(xs, axis=-1), c0, c1, c2, c3, c4, c5)   # (4, 6)
    return Chart(z_dd, m, free, rest, hol)


def power_table(hol, top):
    """The holomorphic jets of z_a^n for n = 0 .. top, (top + 1, 4, 6)."""
    one = np.zeros((4, 6))
    one[:, 0] = 1.0
    out = [(dd.to_dd(jnp.asarray(one)), dd.to_dd(jnp.zeros((4, 6))))]
    for _ in range(top):
        out.append(hol_mul(out[-1], hol))
    return _tmap(lambda *xs: jnp.stack(xs), *out)


def monomial_jets(table, exps):
    """The holomorphic jets of z^A for the rows A of `exps`, (..., 6)."""
    exps = np.asarray(exps)
    cols = _tmap(lambda a: a[exps, np.arange(4)], table)               # (..., 4, 6)
    out = analytic._cindex(cols, Ellipsis, 0, slice(None))
    for c in range(1, 4):
        out = hol_mul(out, analytic._cindex(cols, Ellipsis, c, slice(None)))
    return out


def radius_jet(chart):
    """The jet of |z|^2 along X."""
    return _csum(outer(chart.hol, chart.hol), 0)


def base_jet(lay, mats, chart, table):
    """The jet of p_0 = sum over blocks of sigma^T H conj(sigma), from its sections.

    Block r holds sigma_r = z^r psi_r(u) with psi the section frame applied to the
    monomials u^C of u = z^4, so z^(r + 4C) is a monomial of z. `mats` are the dd
    pairs of `sections.prepare_parts`.
    """
    total = None
    for g, mat in zip(lay.groups, mats):
        exps = np.asarray(g.exps)
        frame = np.asarray(g.frame)
        res = np.asarray(g.residues)
        mono = monomial_jets(table, 4 * exps)                         # (n_mono, 6)
        shape = (slice(None), None)

        def project(part):
            return dd.dot_dd_fp((part[0][shape], part[1][shape]),
                                jnp.asarray(frame)[:, :, None], axis=0)
        psi = (project(mono[0]), project(mono[1]))                    # (d, 6)
        zr = monomial_jets(table, res)                                # (blocks, 6)
        sig = hol_mul(analytic._cindex(zr, slice(None), None, slice(None)),
                      analytic._cindex(psi, None, slice(None), slice(None)))   # (b, d, 6)
        hs = dd_metric._matvec(mat, _conj(sig), 1)                    # (b, d, 6)
        left = analytic._cindex(sig, Ellipsis, slice(None), None)
        right = analytic._cindex(hs, Ellipsis, None, slice(None))
        part = dd_metric._creduce(dd.cmul(left, right), 2)            # (6, 6)
        total = part if total is None else dd.cadd(total, part)
    return total


def multiplier_jets(entries, table, r2):
    """The jets of Re(sum over S4 of z^(pi A) conj(z)^(pi B)) / |z|^(2d), (E, 6, 6)."""
    from . import arnoldi
    a_exp = np.stack([np.asarray(a) for _, a, _ in entries])
    b_exp = np.stack([np.asarray(b) for _, _, b in entries])
    degs = [int(d) for d, _, _ in entries]
    n_e, n_p = len(entries), arnoldi.PERMS.shape[0]
    ea = np.zeros((n_e, n_p, 4), np.int64)
    eb = np.zeros((n_e, n_p, 4), np.int64)
    for p, perm in enumerate(arnoldi.PERMS):
        ea[:, p, perm] = a_exp
        eb[:, p, perm] = b_exp
    num = hermitize(_csum(outer(monomial_jets(table, ea), monomial_jets(table, eb)), 1))
    r0 = value(r2)
    inv_r0 = dd.reciprocal(r0)
    x = _scale(nil(r2), inv_r0)
    out = []
    for e, d in enumerate(degs):
        # |z|^(-2d) = r0^(-d) (1 + x)^(-d).
        lead = inv_r0
        for _ in range(d - 1):
            lead = dd.mul(lead, inv_r0)
        den = _scale(pow_nil(x, -d), lead)
        out.append(mul(analytic._cindex(num, e), den))
    return _tmap(lambda *xs: jnp.stack(xs), *out)


def centered(y, basis):
    """(y - shift) / scale of the multipliers of `basis`."""
    if basis.shift is None:
        return y
    inv = dd.reciprocal(dd.to_dd(jnp.asarray(basis.scale)))
    y = _add_value(y, dd.to_dd(-jnp.asarray(basis.shift)))
    return _scale(y, inv)


def weight_jet(w0, log_nil):
    """The jet of W = W0 exp(L), from the dd value W0 and the jet of L without value."""
    return _scale(exp_nil(log_nil), w0)


def _flat36(jets):
    """A real jet (m, 6, 6) as 36 real dd rows (36, m)."""
    rows_hi, rows_lo = [], []
    for a in range(6):
        rows_hi.append(jets[0][0][:, a, a])
        rows_lo.append(jets[0][1][:, a, a])
    for a, b in _TRIU:
        for part in (jets[0], jets[1]):
            rows_hi.append(part[0][:, a, b])
            rows_lo.append(part[1][:, a, b])
    return jnp.stack(rows_hi), jnp.stack(rows_lo)


def _unflat36(rows):
    """The inverse of `_flat36`."""
    hi, lo = rows
    m = hi.shape[1]
    re_hi, re_lo, im_hi, im_lo = (jnp.zeros((6, 6, m)) for _ in range(4))
    for a in range(6):
        re_hi = re_hi.at[a, a].set(hi[a])
        re_lo = re_lo.at[a, a].set(lo[a])
    for k, (a, b) in enumerate(_TRIU):
        r, i = 6 + 2 * k, 7 + 2 * k
        re_hi = re_hi.at[a, b].set(hi[r]).at[b, a].set(hi[r])
        re_lo = re_lo.at[a, b].set(lo[r]).at[b, a].set(lo[r])
        im_hi = im_hi.at[a, b].set(hi[i]).at[b, a].set(-hi[i])
        im_lo = im_lo.at[a, b].set(lo[i]).at[b, a].set(-lo[i])
    move = lambda x: jnp.moveaxis(x, -1, 0)
    return (move(re_hi), move(re_lo)), (move(im_hi), move(im_lo))


def _first36(inv_h):
    """The jet of the constant function inv_h, (1, 6, 6)."""
    hi = jnp.zeros((1, 6, 6)).at[0, 0, 0].set(inv_h[0][0])
    lo = jnp.zeros((1, 6, 6)).at[0, 0, 0].set(inv_h[1][0])
    return (hi, lo), dd.to_dd(jnp.zeros((1, 6, 6)))


@dataclass(frozen=True)
class Algebra:
    rows: int
    flat: object
    unflat: object
    mul: object
    first: object


T36 = Algebra(rows=36, flat=_flat36, unflat=_unflat36, mul=mul, first=_first36)


def metric_parts(phi_nil, chart, mean):
    """f, D f, the Levi form of f, the Ricci form and g at the point, from the jet of Phi.

    g[i, j] = d_i dbar_j Phi has the jet g[i, j][c, e] = (c_i + 1)(e_j + 1)
    Phi[c + e_i, e + e_j] for c, e of degree at most 1. Then eta = det(g)
    |dP/dz_m|^2 / mean, f = eta - 1, and Ric = -ddbar log eta.
    """
    low = [0, 1, 2]
    unit = [(1, 0), (0, 1)]
    g = []
    for i in range(2):
        row = []
        for j in range(2):
            src_a, src_b, fac = np.zeros((6, 6), int), np.zeros((6, 6), int), np.zeros((6, 6))
            for c in low:
                for e in low:
                    mc, me = M[c], M[e]
                    a = _POS[(mc[0] + unit[i][0], mc[1] + unit[i][1])]
                    b = _POS[(me[0] + unit[j][0], me[1] + unit[j][1])]
                    src_a[c, e], src_b[c, e] = a, b
                    fac[c, e] = (mc[i] + 1) * (me[j] + 1)
            row.append(_tmap(lambda x, sa=src_a, sb=src_b, f=fac: x[..., sa, sb] * f, phi_nil))
        g.append(row)
    det = dd.csub(mul(g[0][0], g[1][1]), mul(g[0][1], g[1][0]))
    zm3 = analytic._cindex(power_table(chart.hol, 3), 3, chart.m)            # z_m^3 jet
    dp = analytic._cscale(zm3, jnp.asarray(4.0))
    vol = mul(det, outer(dp, dp))
    eta = _scale(vol, dd.reciprocal(mean))
    e0 = value(eta)
    f0 = dd.add(e0, dd.to_dd(-jnp.ones_like(e0[0])))
    log_eta = log1p_nil(_scale(nil(eta), dd.reciprocal(e0)))
    df = analytic._cindex(eta, Ellipsis, slice(1, 3), 0)                    # (2,)
    lev = analytic._cindex(eta, Ellipsis, slice(1, 3), slice(1, 3))         # (2, 2)
    ric = dd.cneg(analytic._cindex(log_eta, Ellipsis, slice(1, 3), slice(1, 3)))
    g0 = _tmap(lambda *xs: jnp.stack([jnp.stack(xs[:2], -1), jnp.stack(xs[2:], -1)], -2),
               *[analytic._cindex(g[i][j], Ellipsis, 0, 0) for i in range(2) for j in range(2)])
    return f0, df, lev, ric, g0
