"""Taylor models in complex double-double on the GPU, with strict bounds.

A Taylor model on the polydisc abs(x_v) <= delta of nvars complex variables is a
polynomial P of total degree <= n with complex dd coefficients, and a number R with
abs(f - P) <= R on the polydisc. A batch holds one model for each box, each with its
own delta. The polynomial part is the exact data; every rounding goes into R.

**Rounding.** Each dd operation (`dd.add`, `dd.mul`) has a relative error of at most
8 u^2, u = 2^-53. A coefficient that is a sum of at most w products a_i b_j then has
an error of at most G_w sum abs(a_i) abs(b_j), with G_w = (w + 20) 8 u^2 and abs taken
as abs(re) + abs(im). On the polydisc the sum of those errors times delta^(degree) is
at most G_w B(A) B(B), where B(A) = sum over the coefficients of abs(a) delta^(deg a).
So each product adds G_w B(A) B(B) to the remainder, and each sum adds 4 u^2 times
the bounds of its terms.

**Bounds in FP64.** Every bound is a sum or a product of non-negative FP64 numbers,
and each one is multiplied by SAFE = 1 + 1e-9. That is strict for fewer than 1e6
operations in each bound, because each operation has a relative error of at most u.
A difference 1 - eta takes eta rounded up.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import jax
import jax.numpy as jnp

from . import dd

U2 = 2.0 ** -106            # u^2 with u = 2^-53
SAFE = 1.0 + 1e-9


def compositions(total, k):
    """All k-tuples of non-negative integers with this sum, the first entry largest first."""
    if k == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest


class Space:
    """The monomials of total degree <= n in nvars variables, and the product tables."""

    def __init__(self, nvars, n):
        self.nvars, self.n = nvars, n
        self.mons = [e for d in range(n + 1) for e in compositions(d, nvars)]
        self.index = {e: i for i, e in enumerate(self.mons)}
        self.deg = np.array([sum(e) for e in self.mons], dtype=np.int64)
        self.size = len(self.mons)
        self.deg_onehot = jnp.asarray(
            (self.deg[:, None] == np.arange(n + 1)[None, :]).astype(np.float64))
        self._buckets = None

    def buckets(self):
        """For the product: (K, I, J, width) for each bucket of outputs.

        Output K[q] is the sum over t of A[I[q, t]] B[J[q, t]]. Index `size` points to
        an appended zero, so that the rows of a bucket have one width.
        """
        if self._buckets is None:
            base = self.n + 1
            codes = np.array([sum(c * base ** v for v, c in enumerate(e)) for e in self.mons])
            lookup = np.full(base ** self.nvars, -1, dtype=np.int64)
            lookup[codes] = np.arange(self.size)
            count = np.searchsorted(self.deg, np.arange(self.n + 1), side='right')
            ii, jj, kk = [], [], []
            for i in range(self.size):
                js = np.arange(count[self.n - self.deg[i]])
                ii.append(np.full(js.size, i))
                jj.append(js)
                kk.append(lookup[codes[i] + codes[js]])
            ii, jj, kk = np.concatenate(ii), np.concatenate(jj), np.concatenate(kk)
            order = np.argsort(kk, kind='stable')
            ii, jj, kk = ii[order], jj[order], kk[order]
            per = np.bincount(kk, minlength=self.size)
            start = np.concatenate([[0], np.cumsum(per)[:-1]])
            width = 2 ** np.ceil(np.log2(np.maximum(per, 1))).astype(np.int64)
            out = []
            for w in np.unique(width):
                ks = np.flatnonzero(width == w)
                pos = start[ks][:, None] + np.arange(w)[None, :]
                valid = np.arange(w)[None, :] < per[ks][:, None]
                pos = np.where(valid, pos, 0)
                big_i = np.where(valid, ii[pos], self.size)
                big_j = np.where(valid, jj[pos], self.size)
                # NumPy constants, so that a table made inside one trace serves the next.
                out.append((ks, big_i.astype(np.int32), big_j.astype(np.int32), int(w)))
            self._buckets = out
            self.max_width = int(width.max())
        return self._buckets


# A complex dd array is ((re_hi, re_lo), (im_hi, im_lo)), each of shape (B, size).

def czeros(shape):
    z = jnp.zeros(shape)
    return ((z, z), (z, z))


def cabs_ub(c):
    """An upper bound of abs(re) + abs(im) of a complex dd array, in FP64."""
    (rh, rl), (ih, il) = c
    return (jnp.abs(rh) + jnp.abs(rl) + jnp.abs(ih) + jnp.abs(il)) * (1.0 + 8e-16)


def _pad(c):
    """Append one zero coefficient, for the padded rows of the product buckets."""
    return jax.tree_util.tree_map(
        lambda a: jnp.concatenate([a, jnp.zeros(a.shape[:-1] + (1,))], axis=-1), c)


def _take(c, idx):
    return jax.tree_util.tree_map(lambda a: a[:, idx], c)


@dataclass
class TM:
    """A batch of Taylor models: c complex dd (B, size), rem (B,), delta (B,)."""
    c: tuple
    rem: jnp.ndarray
    delta: jnp.ndarray

    def tree_flatten(self):
        return (self.c, self.rem, self.delta), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


jax.tree_util.register_pytree_node(TM, TM.tree_flatten, TM.tree_unflatten)


def graded(sp, x):
    """g[b, d] = an upper bound of the sum of abs(coefficient) over the monomials of degree d."""
    a = cabs_ub(x.c)
    return (a @ sp.deg_onehot) * SAFE


def poly_bound_g(g, delta):
    """sum over d of g[:, d] delta^d, rounded up."""
    powers = delta[:, None] ** jnp.arange(g.shape[1])[None, :]
    return jnp.sum(g * powers, axis=1) * SAFE


def poly_bound(sp, x, delta=None):
    return poly_bound_g(graded(sp, x), x.delta if delta is None else delta)


def bound(sp, x):
    """An upper bound of abs(f) on the polydisc: B(P) + R."""
    return poly_bound(sp, x) + x.rem


def tail_g(ga, gb, delta, n):
    """sum over k + l > n of ga[k] gb[l] delta^(k+l), rounded up."""
    m = ga.shape[1]
    k = np.arange(m)[:, None] + np.arange(gb.shape[1])[None, :]
    mask = jnp.asarray((k > n).astype(np.float64))
    powers = delta[:, None, None] ** jnp.asarray(k)[None, :, :]
    return jnp.sum(ga[:, :, None] * gb[:, None, :] * mask[None] * powers, axis=(1, 2)) * SAFE


def from_coeffs(c, delta, rem=None):
    b = c[0][0].shape[0]
    return TM(c, jnp.zeros(b) if rem is None else rem, delta)


def const(sp, value, delta):
    """A constant model: value is a complex dd array of shape (B,)."""
    b = delta.shape[0]
    c = czeros((b, sp.size))
    c = jax.tree_util.tree_map(lambda a, v: a.at[:, 0].set(v), c, value)
    return TM(c, jnp.zeros(b), delta)


def add(sp, x, y):
    c = dd.cadd(x.c, y.c)
    err = 4 * U2 * (poly_bound(sp, x) + poly_bound(sp, y))
    return TM(c, (x.rem + y.rem + err) * SAFE, x.delta)


def sub(sp, x, y):
    c = dd.csub(x.c, y.c)
    err = 4 * U2 * (poly_bound(sp, x) + poly_bound(sp, y))
    return TM(c, (x.rem + y.rem + err) * SAFE, x.delta)


def scale(sp, x, k):
    """x times a complex dd number for each box, k of shape (B,)."""
    kk = jax.tree_util.tree_map(lambda a: a[:, None], k)
    c = dd.cmul(x.c, kk)
    kb = cabs_ub(k)
    b = poly_bound(sp, x)
    return TM(c, (x.rem * kb + 20 * U2 * b * kb) * SAFE, x.delta)


def add_const(sp, x, k):
    """x plus a complex dd number for each box."""
    c0 = jax.tree_util.tree_map(lambda a: a[:, 0], x.c)
    s = dd.cadd(c0, k)
    c = jax.tree_util.tree_map(lambda a, v: a.at[:, 0].set(v), x.c, s)
    err = 4 * U2 * (cabs_ub(c0) + cabs_ub(k))
    return TM(c, (x.rem + err) * SAFE, x.delta)


def mul(sp, x, y):
    """The product, truncated at degree n, with the tail, the cross terms and the rounding."""
    xp, yp = _pad(x.c), _pad(y.c)
    b = x.delta.shape[0]
    out = czeros((b, sp.size))
    for ks, big_i, big_j, w in sp.buckets():
        prods = dd.cmul(_take(xp, big_i), _take(yp, big_j))         # (B, n_k, w)
        (rh, rl), (ih, il) = prods
        s_re = dd.sum_dd(rh, rl, axis=-1)
        s_im = dd.sum_dd(ih, il, axis=-1)
        out = jax.tree_util.tree_map(lambda a, v: a.at[:, ks].set(v), out, (s_re, s_im))
    gx, gy = graded(sp, x), graded(sp, y)
    bx, by = poly_bound_g(gx, x.delta), poly_bound_g(gy, x.delta)
    gamma = (sp.max_width + 20) * 8 * U2
    rem = (tail_g(gx, gy, x.delta, sp.n) + bx * y.rem + x.rem * by + x.rem * y.rem
           + gamma * bx * by) * SAFE
    return TM(out, rem, x.delta)


def powers(sp, y, k):
    """[1, y, y^2, ..., y^k] as models; y^j costs one product each."""
    one = const(sp, dd.cto_dd(jnp.ones(y.delta.shape[0], dtype=jnp.complex128)), y.delta)
    out = [one, y]
    for _ in range(2, k + 1):
        out.append(mul(sp, out[-1], y))
    return out


def series(sp, pw, coeffs, eta, tail_coeff, ratio):
    """sum over j of coeffs[j] y^j from the power table pw, and a bound of the rest.

    coeffs are Fractions. The caller gives `ratio` with
    abs(coefficient j+1) <= ratio * abs(coefficient j) for every j > k, so the rest is
    at most abs(tail_coeff) eta^(k+1) / (1 - ratio eta) on the polydisc, for
    ratio eta < 1, with eta a bound of abs(y) there.
    """
    k = len(coeffs) - 1
    b = eta.shape[0]
    total = None
    for j, q in enumerate(coeffs):
        if q == 0:
            continue
        val = _fraction_dd(q, b)
        term = scale(sp, pw[j], val)
        total = term if total is None else add(sp, total, term)
    eta_up = eta * SAFE
    q_eta = float(ratio) * eta_up * SAFE
    one_minus = 1.0 - q_eta
    rest = jnp.where(q_eta < 1.0,
                     float(abs(tail_coeff)) * eta_up ** (k + 1)
                     / jnp.where(one_minus > 0, one_minus, 1.0),
                     jnp.inf) * SAFE
    return TM(total.c, (total.rem + rest) * SAFE, total.delta)


def _fraction_dd(q, b):
    """A Fraction as a complex dd constant for each of b boxes, with its rounding."""
    q = Fraction(q)
    hi = float(q)
    lo = float(q - Fraction(hi))
    re = (jnp.full((b,), hi), jnp.full((b,), lo))
    zero = (jnp.zeros(b), jnp.zeros(b))
    return re, zero


def binomial_coeffs(a, k):
    """C(a, j) for j = 0..k+1 as Fractions, for a rational a."""
    a = Fraction(a)
    out = [Fraction(1)]
    for j in range(k + 1):
        out.append(out[-1] * (a - j) / (j + 1))
    return out


def log1p_coeffs(k):
    """The coefficients of log(1 + y): 0, 1, -1/2, 1/3, ... up to j = k + 1."""
    return [Fraction(0)] + [Fraction((-1) ** (j + 1), j) for j in range(1, k + 2)]


def power_series(sp, pw, a, eta):
    """(1 + y)^a from the power table of y, a rational, with the bound of the rest.

    abs(C(a, j+1) / C(a, j)) = abs(a - j) / (j + 1) <= (abs(a) + k + 1) / (k + 2) for
    j >= k + 1, which is the ratio of the geometric bound of the rest.
    """
    k = len(pw) - 1
    co = binomial_coeffs(a, k)
    ratio = (abs(Fraction(a)) + k + 1) / Fraction(k + 2)
    return series(sp, pw, co[:k + 1], eta, co[k + 1], ratio)


def log1p_series(sp, pw, eta):
    """log(1 + y) from the power table; abs(coefficient j+1) / abs(coefficient j) <= 1."""
    k = len(pw) - 1
    co = log1p_coeffs(k)
    return series(sp, pw, co[:k + 1], eta, co[k + 1], 1)


def deriv(sp, x, axis):
    """d/dx_axis of the polynomial part, of degree <= n - 1; the remainder is not set."""
    src = []
    mult = []
    for e in sp.mons:
        f = list(e)
        f[axis] += 1
        f = tuple(f)
        if sum(f) <= sp.n:
            src.append(sp.index[f])
            mult.append(f[axis])
        else:
            src.append(sp.size)
            mult.append(0)
    src = jnp.asarray(src, dtype=jnp.int32)
    mult = jnp.asarray(mult, dtype=jnp.float64)[None, :]
    zero = jnp.zeros_like(mult)
    return dd.cmul(_take(_pad(x.c), src), ((mult, zero), (zero, zero)))


def cauchy_mixed(sp, x, a, b, delta_small):
    """d^2 / dx_a dx_b of x on the polydisc delta_small, with Cauchy's estimate.

    a and b are different variables. The polynomial part is differentiated exactly
    up to rounding; the rest of x, at most x.rem on the polydisc x.delta, has a
    mixed derivative of at most x.rem / (x.delta - delta_small)^2 on delta_small.
    """
    if a == b:
        raise ValueError('cauchy_mixed takes two different variables')
    first = TM(deriv(sp, x, a), jnp.zeros_like(x.rem), x.delta)
    second = deriv(sp, first, b)
    gap = (x.delta - delta_small) * (1.0 - 1e-15)
    cauchy = x.rem / (gap * gap)
    # Each coefficient times alpha_a beta_b <= n^2 rounds by 16 u^2 at most.
    err = 40 * U2 * poly_bound(sp, x) / (delta_small * delta_small) * (sp.n * sp.n)
    return TM(second, (cauchy + err) * SAFE, delta_small)


def deriv1(sp, x, a, delta_small):
    """d/dx_a of x on the polydisc delta_small, with Cauchy's estimate for the rest."""
    first = deriv(sp, x, a)
    gap = (x.delta - delta_small) * (1.0 - 1e-15)
    err = 20 * U2 * poly_bound(sp, x) / delta_small * sp.n
    return TM(first, (x.rem / gap + err) * SAFE, delta_small)


def split_index(sp4, sp2):
    """For each monomial (a0, a1, b0, b1) of sp4: its (a0, a1) and (b0, b1) in sp2."""
    rows, cols = [], []
    for a0, a1, b0, b1 in sp4.mons:
        rows.append(sp2.index.get((a0, a1), sp2.size))
        cols.append(sp2.index.get((b0, b1), sp2.size))
    return np.asarray(rows), np.asarray(cols)


def from_bidegree(sp4, sp2, q, delta, err):
    """The model of a polynomial with coefficients q[b, alpha, beta] of s^alpha r^beta.

    q is complex dd of shape (B, sp2.size, sp2.size); err (B,) bounds the effect of
    the errors of q on the polydisc. The terms of total degree > n go to the remainder.
    """
    rows, cols = split_index(sp4, sp2)
    pad = jax.tree_util.tree_map(
        lambda a: jnp.pad(a, ((0, 0), (0, 1), (0, 1))), q)
    c = jax.tree_util.tree_map(lambda a: a[:, rows, cols], pad)
    absq = cabs_ub(q)                                             # (B, n2, n2)
    oh = sp2.deg_onehot
    bideg = jnp.einsum('bij,id,je->bde', absq, oh, oh) * SAFE     # (B, D+1, D+1)
    d = np.arange(oh.shape[1])[:, None] + np.arange(oh.shape[1])[None, :]
    mask = jnp.asarray((d > sp4.n).astype(np.float64))
    powers = delta[:, None, None] ** jnp.asarray(d)[None]
    tail = jnp.sum(bideg * mask[None] * powers, axis=(1, 2)) * SAFE
    return TM(c, (err + tail) * SAFE, delta)


def tensor(sp4, sp2, z, w):
    """z(s) w(r) as a model in the four variables, from two models of degree n in sp2."""
    rows, cols = split_index(sp4, sp2)
    zc = _take(_pad(z.c), jnp.asarray(rows))
    wc = _take(_pad(w.c), jnp.asarray(cols))
    c = dd.cmul(zc, wc)
    gz, gw = graded(sp2, z), graded(sp2, w)
    bz, bw = poly_bound_g(gz, z.delta), poly_bound_g(gw, z.delta)
    rem = (tail_g(gz, gw, z.delta, sp4.n) + bz * w.rem + z.rem * bw + z.rem * w.rem
           + 20 * U2 * bz * bw) * SAFE
    return TM(c, rem, z.delta)


def restrict(x, delta_small):
    """The same model on a smaller polydisc; its remainder stays valid there."""
    return TM(x.c, x.rem, delta_small)
