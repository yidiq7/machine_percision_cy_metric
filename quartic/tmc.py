"""Taylor models in complex dd on the GPU with CUDA kernels (CuPy), with strict bounds.

A Taylor model on the polydisc abs(x_v) <= delta of nvars complex variables is a
polynomial P of total degree <= n with complex dd coefficients, and a number R with
abs(f - P) <= R on the polydisc. A batch holds one model for each box, each with its
own delta. The polynomial part is the exact data; every rounding goes into R.

**Layout.** c is a CuPy array (4, S, B): re_hi, re_lo, im_hi, im_lo of the S
coefficients for B boxes. rem and delta are (B,).

**Arithmetic.** Every dd operation is a CUDA device function with __dadd_rn,
__dsub_rn, __dmul_rn and __fma_rn, compiled with --fmad=false, so no compiler can
merge or reorder an error term. two_sum is exact; the product uses the exact FMA
error of the high parts. The relative error of dd_mul is at most 6 u^2 and of
dd_add at most 3.1 u^2 of the sum, u = 2^-53.

**Rounding bounds.** A coefficient that is a sum of c complex products has an error
of at most (6.2 c + 6) u^2 sum abs(a_i) abs(b_j), with abs = abs(re) + abs(im). So a
product adds G_w B(A) B(B) to the remainder, G_w = (2 w + 20) 8 u^2 with w the
largest c, and B(A) = sum of abs(a) delta^(deg a). A sum adds 4 u^2 (B(A) + B(B)).

**Bounds in FP64.** Every bound is a sum or a product of non-negative FP64 numbers,
times SAFE = 1 + 1e-9 at each step, which is strict for fewer than 1e6 operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import cupy as cp

U2 = 2.0 ** -106
SAFE = 1.0 + 1e-9

_SRC = r'''
__device__ __forceinline__ void two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    double bb = __dsub_rn(s, a);
    e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
}
__device__ __forceinline__ void quick_two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    e = __dsub_rn(b, __dsub_rn(s, a));
}
__device__ __forceinline__ void dd_add(double ah, double al, double bh, double bl,
                                       double &sh, double &sl) {
    double s1, s2, t1, t2;
    two_sum(ah, bh, s1, s2);
    two_sum(al, bl, t1, t2);
    two_sum(s1, __dadd_rn(s2, t1), s1, s2);
    two_sum(s1, __dadd_rn(s2, t2), sh, sl);
}
__device__ __forceinline__ void dd_mul(double ah, double al, double bh, double bl,
                                       double &ph, double &pl) {
    double p = __dmul_rn(ah, bh);
    double e = __fma_rn(ah, bh, -p);
    double t = __fma_rn(al, bl, __dmul_rn(ah, bl));
    t = __fma_rn(al, bh, t);
    quick_two_sum(p, __dadd_rn(e, t), ph, pl);
}
// Z[k] = sum over t of X[I[t]] Y[J[t]], t in [start[k], start[k] + cnt[k]); arrays (4, S, B).
extern "C" __global__ void tm_mul(const double* __restrict__ X, const double* __restrict__ Y,
                                  const int* __restrict__ start, const int* __restrict__ cnt,
                                  const int* __restrict__ I, const int* __restrict__ J,
                                  double* __restrict__ Z, int S, int B) {
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const size_t SB = (size_t)S * B;
    int s0 = start[k], c = cnt[k];
    double rh = 0.0, rl = 0.0, ih = 0.0, il = 0.0, ph, pl;
    for (int t = 0; t < c; ++t) {
        size_t xi = (size_t)I[s0 + t] * B + b, yj = (size_t)J[s0 + t] * B + b;
        double xrh = X[xi], xrl = X[SB + xi], xih = X[2 * SB + xi], xil = X[3 * SB + xi];
        double yrh = Y[yj], yrl = Y[SB + yj], yih = Y[2 * SB + yj], yil = Y[3 * SB + yj];
        dd_mul(xrh, xrl, yrh, yrl, ph, pl); dd_add(rh, rl, ph, pl, rh, rl);
        dd_mul(xih, xil, yih, yil, ph, pl); dd_add(rh, rl, -ph, -pl, rh, rl);
        dd_mul(xrh, xrl, yih, yil, ph, pl); dd_add(ih, il, ph, pl, ih, il);
        dd_mul(xih, xil, yrh, yrl, ph, pl); dd_add(ih, il, ph, pl, ih, il);
    }
    size_t zk = (size_t)k * B + b;
    Z[zk] = rh; Z[SB + zk] = rl; Z[2 * SB + zk] = ih; Z[3 * SB + zk] = il;
}
// Z = X + s Y elementwise for complex dd arrays (4, n), s = +1 or -1.
extern "C" __global__ void c_axpy(const double* __restrict__ X, const double* __restrict__ Y,
                                  double* __restrict__ Z, long n, double s) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double h, l;
    dd_add(X[i], X[n + i], s * Y[i], s * Y[n + i], h, l);
    Z[i] = h; Z[n + i] = l;
    dd_add(X[2 * n + i], X[3 * n + i], s * Y[2 * n + i], s * Y[3 * n + i], h, l);
    Z[2 * n + i] = h; Z[3 * n + i] = l;
}
// Z[:, s, b] = X[:, s, b] * K[:, b] for complex dd arrays X (4, S, B) and K (4, B).
extern "C" __global__ void c_scale(const double* __restrict__ X, const double* __restrict__ K,
                                   double* __restrict__ Z, int S, int B) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)S * B;
    if (i >= n) return;
    int b = (int)(i % B);
    double xrh = X[i], xrl = X[n + i], xih = X[2 * n + i], xil = X[3 * n + i];
    double krh = K[b], krl = K[B + b], kih = K[2 * B + b], kil = K[3 * B + b];
    double ah, al, bh, bl, h, l;
    dd_mul(xrh, xrl, krh, krl, ah, al); dd_mul(xih, xil, kih, kil, bh, bl);
    dd_add(ah, al, -bh, -bl, h, l); Z[i] = h; Z[n + i] = l;
    dd_mul(xrh, xrl, kih, kil, ah, al); dd_mul(xih, xil, krh, krl, bh, bl);
    dd_add(ah, al, bh, bl, h, l); Z[2 * n + i] = h; Z[3 * n + i] = l;
}
// Z[:, s, b] = X[:, src[s], b] * m[s] for an integer m[s] (0 where src[s] < 0).
extern "C" __global__ void c_mulint(const double* __restrict__ X, const int* __restrict__ src,
                                    const double* __restrict__ m, double* __restrict__ Z,
                                    int S, int B) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)S * B;
    if (i >= n) return;
    int s = (int)(i / B), b = (int)(i % B);
    if (src[s] < 0) { Z[i] = 0.0; Z[n + i] = 0.0; Z[2 * n + i] = 0.0; Z[3 * n + i] = 0.0; return; }
    long j = (long)src[s] * B + b;
    double h, l;
    dd_mul(X[j], X[n + j], m[s], 0.0, h, l); Z[i] = h; Z[n + i] = l;
    dd_mul(X[2 * n + j], X[3 * n + j], m[s], 0.0, h, l); Z[2 * n + i] = h; Z[3 * n + i] = l;
}
// Z = X + Y for complex dd arrays (4, B) of constants, in place in the first coefficient.
extern "C" __global__ void c_add0(double* __restrict__ X, const double* __restrict__ K,
                                  int S, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    long n = (long)S * B;
    double h, l;
    dd_add(X[b], X[n + b], K[b], K[B + b], h, l); X[b] = h; X[n + b] = l;
    dd_add(X[2 * n + b], X[3 * n + b], K[2 * B + b], K[3 * B + b], h, l);
    X[2 * n + b] = h; X[3 * n + b] = l;
}
'''
_MOD = None


def _kernels():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD


def compositions(total, k):
    """All k-tuples of non-negative integers with this sum, the first entry largest first."""
    if k == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest


class Space:
    """The monomials of total degree <= n in nvars variables, and the product table."""

    def __init__(self, nvars, n):
        self.nvars, self.n = nvars, n
        self.mons = [e for d in range(n + 1) for e in compositions(d, nvars)]
        self.index = {e: i for i, e in enumerate(self.mons)}
        self.deg = np.array([sum(e) for e in self.mons], dtype=np.int64)
        self.size = len(self.mons)
        onehot = (self.deg[:, None] == np.arange(n + 1)[None, :]).astype(np.float64)
        self.onehot = cp.asarray(onehot)                     # (S, n + 1)
        self._table = None

    def table(self):
        """start, cnt, I, J (CuPy int32) of the product, sorted by output, and max cnt."""
        if self._table is None:
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
            cnt = np.bincount(kk, minlength=self.size)
            start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            self._table = (cp.asarray(start, cp.int32), cp.asarray(cnt, cp.int32),
                           cp.asarray(ii, cp.int32), cp.asarray(jj, cp.int32), int(cnt.max()))
        return self._table


@dataclass
class TM:
    """A batch of Taylor models: c (4, S, B), rem (B,), delta (B,)."""
    c: cp.ndarray
    rem: cp.ndarray
    delta: cp.ndarray


def zeros(sp, b):
    return cp.zeros((4, sp.size, b))


def abs_ub(c):
    """An upper bound of abs(re) + abs(im) of each coefficient, (S, B)."""
    return (cp.abs(c[0]) + cp.abs(c[1]) + cp.abs(c[2]) + cp.abs(c[3])) * (1.0 + 8e-16)


def graded(sp, x):
    """g[d, b] = an upper bound of the sum of abs over the monomials of degree d."""
    return (sp.onehot.T @ abs_ub(x.c)) * SAFE


def poly_bound_g(g, delta):
    """sum over d of g[d] delta^d, rounded up; g (D + 1, B)."""
    powers = delta[None, :] ** cp.arange(g.shape[0])[:, None]
    return cp.sum(g * powers, axis=0) * SAFE


def poly_bound(sp, x, delta=None):
    return poly_bound_g(graded(sp, x), x.delta if delta is None else delta)


def bound(sp, x):
    """An upper bound of abs(f) on the polydisc: B(P) + R."""
    return (poly_bound(sp, x) + x.rem) * SAFE


def tail_g(ga, gb, delta, n):
    """sum over k + l > n of ga[k] gb[l] delta^(k+l), rounded up."""
    ka, kb = ga.shape[0], gb.shape[0]
    k = np.arange(ka)[:, None] + np.arange(kb)[None, :]
    mask = cp.asarray((k > n).astype(np.float64))
    powers = delta[None, None, :] ** cp.asarray(k)[:, :, None]
    return cp.sum(ga[:, None, :] * gb[None, :, :] * mask[:, :, None] * powers,
                  axis=(0, 1)) * SAFE


def const(sp, value, delta):
    """A constant model; value is a complex dd array (4, B)."""
    c = zeros(sp, delta.shape[0])
    c[:, 0, :] = value
    return TM(c, cp.zeros_like(delta), delta)


def cconst(z, b):
    """A complex number (Python complex, or a pair of Fractions) as dd (4, b), exactly rounded."""
    if isinstance(z, tuple):
        re, im = (Fraction(x) for x in z)
    else:
        re, im = Fraction(z.real), Fraction(z.imag)
    rh, ih = float(re), float(im)
    rl, il = float(re - Fraction(rh)), float(im - Fraction(ih))
    return cp.asarray(np.array([[rh] * b, [rl] * b, [ih] * b, [il] * b]))


def _launch_elem(fn, n, args):
    threads = 256
    fn(((n + threads - 1) // threads,), (threads,), args)


def _axpy(x, y, s):
    n = x[0].size
    z = cp.empty_like(x)
    _launch_elem(_kernels().get_function('c_axpy'), n,
                 (cp.ascontiguousarray(x), cp.ascontiguousarray(y), z, np.int64(n), np.float64(s)))
    return z


def add(sp, x, y):
    c = _axpy(x.c, y.c, 1.0)
    err = 4 * U2 * (poly_bound(sp, x) + poly_bound(sp, y))
    return TM(c, (x.rem + y.rem + err) * SAFE, x.delta)


def sub(sp, x, y):
    c = _axpy(x.c, y.c, -1.0)
    err = 4 * U2 * (poly_bound(sp, x) + poly_bound(sp, y))
    return TM(c, (x.rem + y.rem + err) * SAFE, x.delta)


def scale(sp, x, k):
    """x times a complex dd number for each box, k of shape (4, B)."""
    _, s, b = x.c.shape
    z = cp.empty_like(x.c)
    _launch_elem(_kernels().get_function('c_scale'), s * b,
                 (cp.ascontiguousarray(x.c), cp.ascontiguousarray(k), z, np.int32(s), np.int32(b)))
    kb = abs_ub(k[:, None, :])[0]
    return TM(z, (x.rem * kb + 20 * U2 * poly_bound(sp, x) * kb) * SAFE, x.delta)


def add_const(sp, x, k):
    """x plus a complex dd number for each box, k (4, B)."""
    _, s, b = x.c.shape
    c = cp.array(x.c, copy=True)
    c0 = abs_ub(x.c[:, :1, :])[0]
    fn = _kernels().get_function('c_add0')
    fn(((b + 255) // 256,), (256,), (c, cp.ascontiguousarray(k), np.int32(s), np.int32(b)))
    err = 4 * U2 * (c0 + abs_ub(k[:, None, :])[0])
    return TM(c, (x.rem + err) * SAFE, x.delta)


def mul(sp, x, y, threads=64):
    """The product, truncated at degree n, with the tail, the cross terms and the rounding."""
    start, cnt, ii, jj, wmax = sp.table()
    _, s, b = x.c.shape
    z = cp.empty_like(x.c)
    _kernels().get_function('tm_mul')(
        (s, (b + threads - 1) // threads), (min(threads, b),),
        (cp.ascontiguousarray(x.c), cp.ascontiguousarray(y.c), start, cnt, ii, jj, z,
         np.int32(s), np.int32(b)))
    gx, gy = graded(sp, x), graded(sp, y)
    bx, by = poly_bound_g(gx, x.delta), poly_bound_g(gy, x.delta)
    gamma = (2 * wmax + 20) * 8 * U2
    rem = (tail_g(gx, gy, x.delta, sp.n) + bx * y.rem + x.rem * by + x.rem * y.rem
           + gamma * bx * by) * SAFE
    return TM(z, rem, x.delta)


def powers(sp, y, k):
    """[1, y, y^2, ..., y^k]; y^j costs one product each."""
    b = y.delta.shape[0]
    out = [const(sp, cconst(1, b), y.delta), y]
    for _ in range(2, k + 1):
        out.append(mul(sp, out[-1], y))
    return out


def series(sp, pw, coeffs, rest):
    """sum over j of coeffs[j] y^j from the power table pw (Fractions), plus `rest` (B,).

    Each coefficient is rounded to dd with an error of at most 2 u^2 of its value, which
    adds 2 u^2 abs(q) times the bound of y^j.
    """
    b = pw[0].delta.shape[0]
    total = None
    extra = cp.zeros(b)
    for j, q in enumerate(coeffs):
        if q == 0:
            continue
        term = scale(sp, pw[j], cconst((q, 0), b))
        extra = extra + 2 * U2 * abs(float(q)) * bound(sp, pw[j])
        total = term if total is None else add(sp, total, term)
    return TM(total.c, (total.rem + rest + extra) * SAFE, total.delta)


def binomial_coeffs(a, k):
    """C(a, j) for j = 0..k+1 as Fractions, for a rational a."""
    a = Fraction(a)
    out = [Fraction(1)]
    for j in range(k + 1):
        out.append(out[-1] * (a - j) / (j + 1))
    return out


def power_series(sp, pw, a, eta):
    """(1 + y)^a for a rational a, with the Lagrange bound of the rest.

    For real a and abs(y) <= eta < 1 the rest after the term of degree k is at most
    abs(C(a, k+1)) eta^(k+1) max over abs(t) <= eta of abs((1 + t)^(a - k - 1)), and
    the last factor is at most (1 - eta)^(a - k - 1) for a < k + 1 and
    (1 + eta)^(a - k - 1) otherwise.
    """
    k = len(pw) - 1
    co = binomial_coeffs(a, k)
    e = eta * SAFE
    ok = e < 1.0
    p = float(Fraction(a) - k - 1)
    if p < 0:
        factor = cp.where(ok, (1.0 - e) * (1.0 - 1e-15), 1.0) ** p
    else:
        factor = (1.0 + e) ** p
    lag = cp.where(ok, abs(float(co[k + 1])) * e ** (k + 1) * factor, cp.inf) * SAFE
    return series(sp, pw, co[:k + 1], lag)


def log1p_series(sp, pw, eta):
    """log(1 + y) with the bound eta^(k+1) / ((k + 1) (1 - eta)) of the rest."""
    k = len(pw) - 1
    co = [Fraction(0)] + [Fraction((-1) ** (j + 1), j) for j in range(1, k + 1)]
    e = eta * SAFE
    ok = e < 1.0
    rest = cp.where(ok, e ** (k + 1) / ((k + 1) * cp.where(ok, (1.0 - e) * (1.0 - 1e-15), 1.0)),
                    cp.inf) * SAFE
    return series(sp, pw, co, rest)


_DERIV = {}


def _deriv_index(sp, axis):
    key = (sp.nvars, sp.n, axis)
    if key not in _DERIV:
        src, mult = [], []
        for e in sp.mons:
            f = list(e)
            f[axis] += 1
            f = tuple(f)
            if sum(f) <= sp.n:
                src.append(sp.index[f])
                mult.append(float(f[axis]))
            else:
                src.append(-1)
                mult.append(0.0)
        _DERIV[key] = (cp.asarray(src, cp.int32), cp.asarray(mult))
    return _DERIV[key]


def deriv_coeffs(sp, c, axis):
    """The coefficients of d/dx_axis of the polynomial part, each a dd product by an integer."""
    src, mult = _deriv_index(sp, axis)
    _, s, b = c.shape
    z = cp.empty_like(c)
    _launch_elem(_kernels().get_function('c_mulint'), s * b,
                 (cp.ascontiguousarray(c), src, mult, z, np.int32(s), np.int32(b)))
    return z


def cauchy_mixed(sp, x, a, b, delta_small):
    """d^2 / dx_a dx_b of x on delta_small: exact derivative and Cauchy's estimate.

    The rest of x is at most x.rem on x.delta, so its mixed derivative is at most
    x.rem / (x.delta - delta_small)^2 on delta_small. The two dd products by integers
    alpha_a, alpha_b <= n round by 6 u^2 each, which adds 20 u^2 n^2 B / delta_small^2.
    """
    if a == b:
        raise ValueError('cauchy_mixed takes two different variables')
    c2 = deriv_coeffs(sp, deriv_coeffs(sp, x.c, a), b)
    gap = (x.delta - delta_small) * (1.0 - 1e-15)
    err = 20 * U2 * poly_bound(sp, x) * sp.n ** 2 / delta_small ** 2
    return TM(c2, (x.rem / (gap * gap) + err) * SAFE, delta_small)


def deriv1(sp, x, a, delta_small):
    """d/dx_a of x on delta_small, with Cauchy's estimate for the rest."""
    c1 = deriv_coeffs(sp, x.c, a)
    gap = (x.delta - delta_small) * (1.0 - 1e-15)
    err = 10 * U2 * poly_bound(sp, x) * sp.n / delta_small
    return TM(c1, (x.rem / gap + err) * SAFE, delta_small)


def split_index(sp4, sp2):
    """For each monomial (a0, a1, b0, b1) of sp4: its (a0, a1) and (b0, b1) in sp2."""
    rows = np.array([sp2.index[(a0, a1)] for a0, a1, _, _ in sp4.mons])
    cols = np.array([sp2.index[(b0, b1)] for _, _, b0, b1 in sp4.mons])
    return rows, cols


def tensor(sp4, sp2, z, w):
    """z(s) w(r) as a model in four variables, from two models in sp2 (sp2.n >= sp4.n)."""
    rows, cols = split_index(sp4, sp2)
    zc = z.c[:, cp.asarray(rows), :]
    wc = w.c[:, cp.asarray(cols), :]
    b = z.delta.shape[0]
    out = TM(cp.empty_like(zc), cp.zeros(b), z.delta)
    # Elementwise complex dd products through the scale kernel, one column at a time is
    # slow; use the product kernel with a one-pair table instead.
    out.c = _pairwise(zc, wc)
    gz, gw = graded(sp2, z), graded(sp2, w)
    bz, bw = poly_bound_g(gz, z.delta), poly_bound_g(gw, z.delta)
    out.rem = (tail_g(gz, gw, z.delta, sp4.n) + bz * w.rem + z.rem * bw + z.rem * w.rem
               + 20 * 8 * U2 * bz * bw) * SAFE
    return out


_PAIR = {}


def _pairwise(x, y):
    """Elementwise complex dd products of two arrays (4, S, B)."""
    s = x.shape[1]
    if s not in _PAIR:
        idx = cp.arange(s, dtype=cp.int32)
        _PAIR[s] = (idx, cp.ones(s, cp.int32), idx, idx)
    start, cnt, ii, jj = _PAIR[s]
    b = x.shape[2]
    z = cp.empty_like(x)
    _kernels().get_function('tm_mul')(
        (s, (b + 63) // 64), (min(64, b),),
        (cp.ascontiguousarray(x), cp.ascontiguousarray(y), start, cnt, ii, jj, z,
         np.int32(s), np.int32(b)))
    return z


def restrict(x, delta_small):
    """The same model on a smaller polydisc; its remainder stays valid there."""
    return TM(x.c, x.rem, delta_small)
