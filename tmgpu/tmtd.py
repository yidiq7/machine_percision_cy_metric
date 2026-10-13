"""Taylor models in complex triple-double (td) on the GPU, with running error bounds.

A td number is three FP64 limbs x0 + x1 + x2. A batch of models has coefficients c of
shape (6, S, B): re0, re1, re2, im0, im1, im2. The polynomial part is the exact data and
every rounding goes into the remainder, as in `tmgpu.tm`.

**Arithmetic.** A product accumulates into three levels A0 + A1 + A2 (`acc_tt`): the
products of the high limbs split exactly by FMA, the sums into A0 and A1 are two_sum, and
only the sums into A2, the products a0 b2, a1 b1, a2 b0 and the dropped a1 b2 + a2 b1 +
a2 b2 err. The levels go back to three limbs by exact two_sums (`levels_td`).

**Running error bounds.** Each kernel also returns, for each output, an FP64 bound of its
error: u times the magnitude of every rounded result, plus the magnitude of every dropped
term, summed as it runs and multiplied by ERR_SAFE at the end. This holds for limbs in any
order, so no normalization of the inputs is assumed. The FP64 sum of N error terms rounds
by at most N u of itself; ERR_SAFE = 1 + 1e-9 covers N up to 9e6, and the largest N here
is about 2e4 (a GEMM of inner dimension 325, with 52 terms for each inner index).
Underflow adds TINY through `tm.poly_bound_g`, as in `tmgpu.tm`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cupy as cp

from . import tm
from .cuda import DD_PRIMITIVES, module
from .tm import SAFE

U = 2.0 ** -53
U2 = tm.U2
ERR_SAFE = 1.0 + 1e-9

TD_SRC = DD_PRIMITIVES + r'''
#define UNIT 1.1102230246251565e-16
// A0 + A1 + A2 += (a0 + a1 + a2)(b0 + b1 + b2), and E += a bound of the error of this step.
__device__ __forceinline__ void acc_tt(double &A0, double &A1, double &A2, double &E,
                                       double a0, double a1, double a2,
                                       double b0, double b1, double b2) {
    double p = __dmul_rn(a0, b0), e = __fma_rn(a0, b0, -p);
    double m1 = __dmul_rn(a0, b1), f1 = __fma_rn(a0, b1, -m1);
    double m2 = __dmul_rn(a1, b0), f2 = __fma_rn(a1, b0, -m2);
    double s, t;
    two_sum(A0, p, A0, s);
    two_sum(A1, s, A1, t); A2 = __dadd_rn(A2, t); E += fabs(A2);
    two_sum(A1, e, A1, t); A2 = __dadd_rn(A2, t); E += fabs(A2);
    two_sum(A1, m1, A1, t); A2 = __dadd_rn(A2, t); E += fabs(A2);
    two_sum(A1, m2, A1, t); A2 = __dadd_rn(A2, t); E += fabs(A2);
    double q11 = __dmul_rn(a1, b1), q02 = __dmul_rn(a0, b2), q20 = __dmul_rn(a2, b0);
    double r = __dadd_rn(f1, f2);
    double r2 = __dadd_rn(r, q11);
    double r3 = __dadd_rn(q02, q20);
    double r4 = __dadd_rn(r2, r3);
    A2 = __dadd_rn(A2, r4);
    // u times each rounded result in level 2, in units of UNIT (applied by the caller),
    // and the dropped products, which are not rounding errors (kept apart in E below).
    E += fabs(q11) + fabs(q02) + fabs(q20) + fabs(r) + fabs(r2) + fabs(r3) + fabs(r4) + fabs(A2);
    E += (fabs(a1) * fabs(b2) + fabs(a2) * fabs(b1) + fabs(a2) * fabs(b2)) / UNIT;
}
// c0 + c1 + c2 = A0 + A1 + A2 exactly.
__device__ __forceinline__ void levels_td(double A0, double A1, double A2,
                                          double &c0, double &c1, double &c2) {
    double x, y, z;
    two_sum(A1, A2, x, y);
    two_sum(A0, x, c0, z);
    two_sum(z, y, c1, c2);
}
// h + l = A0 + A1 + A2 up to UNIT abs(e0 + e1); E gets that term.
__device__ __forceinline__ void levels_dd(double A0, double A1, double A2,
                                          double &h, double &l, double &E) {
    double s1, e1, s0, e0;
    two_sum(A1, A2, s1, e1);
    two_sum(A0, s1, s0, e0);
    double t = __dadd_rn(e0, e1);
    E += fabs(t);
    two_sum(s0, t, h, l);
}
// Z[k] = sum over t of X[I[t]] Y[J[t]] for td arrays (6, S, B); Err[k, b] its error bound.
extern "C" __global__ void tmt_mul(const double* __restrict__ X, const double* __restrict__ Y,
                                   const int* __restrict__ start, const int* __restrict__ cnt,
                                   const int* __restrict__ I, const int* __restrict__ J,
                                   double* __restrict__ Z, double* __restrict__ Err,
                                   int S, int B) {
    int k = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const size_t SB = (size_t)S * B;
    int s0 = start[k], c = cnt[k];
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, i0 = 0.0, i1 = 0.0, i2 = 0.0, E = 0.0;
    for (int t = 0; t < c; ++t) {
        size_t xi = (size_t)I[s0 + t] * B + b, yj = (size_t)J[s0 + t] * B + b;
        double x[6], y[6];
        for (int q = 0; q < 6; ++q) { x[q] = X[q * SB + xi]; y[q] = Y[q * SB + yj]; }
        acc_tt(r0, r1, r2, E, x[0], x[1], x[2], y[0], y[1], y[2]);
        acc_tt(r0, r1, r2, E, -x[3], -x[4], -x[5], y[3], y[4], y[5]);
        acc_tt(i0, i1, i2, E, x[0], x[1], x[2], y[3], y[4], y[5]);
        acc_tt(i0, i1, i2, E, x[3], x[4], x[5], y[0], y[1], y[2]);
    }
    size_t zk = (size_t)k * B + b;
    double c0, c1, c2;
    levels_td(r0, r1, r2, c0, c1, c2);
    Z[zk] = c0; Z[SB + zk] = c1; Z[2 * SB + zk] = c2;
    levels_td(i0, i1, i2, c0, c1, c2);
    Z[3 * SB + zk] = c0; Z[4 * SB + zk] = c1; Z[5 * SB + zk] = c2;
    Err[zk] = E * UNIT;
}
// Z = X + s Y elementwise for td arrays (6, n), s = +1 or -1; Err (n,) its error bound.
extern "C" __global__ void tmt_axpy(const double* __restrict__ X, const double* __restrict__ Y,
                                    double* __restrict__ Z, double* __restrict__ Err,
                                    long n, double s) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double E = 0.0;
    for (int part = 0; part < 2; ++part) {
        long o = (long)part * 3 * n;
        double a0 = X[o + i], a1 = X[o + n + i], a2 = X[o + 2 * n + i];
        double b0 = s * Y[o + i], b1 = s * Y[o + n + i], b2 = s * Y[o + 2 * n + i];
        double s0, e0, s1, e1, t1, e2;
        two_sum(a0, b0, s0, e0);
        two_sum(a1, b1, s1, e1);
        two_sum(s1, e0, t1, e2);
        double u1 = __dadd_rn(a2, b2), u2 = __dadd_rn(e1, e2), s2 = __dadd_rn(u1, u2);
        E += fabs(u1) + fabs(u2) + fabs(s2);
        double c0, c1, c2;
        levels_td(s0, t1, s2, c0, c1, c2);
        Z[o + i] = c0; Z[o + n + i] = c1; Z[o + 2 * n + i] = c2;
    }
    Err[i] = E * UNIT;
}
// Z[:, s, b] = X[:, s, b] * K[:, b] for td arrays X (6, S, B) and a td number K (6, B).
extern "C" __global__ void tmt_scale(const double* __restrict__ X, const double* __restrict__ K,
                                     double* __restrict__ Z, double* __restrict__ Err,
                                     int S, int B) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)S * B;
    if (i >= n) return;
    int b = (int)(i % B);
    double x[6], k[6];
    for (int q = 0; q < 6; ++q) { x[q] = X[q * n + i]; k[q] = K[q * B + b]; }
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, i0 = 0.0, i1 = 0.0, i2 = 0.0, E = 0.0;
    acc_tt(r0, r1, r2, E, x[0], x[1], x[2], k[0], k[1], k[2]);
    acc_tt(r0, r1, r2, E, -x[3], -x[4], -x[5], k[3], k[4], k[5]);
    acc_tt(i0, i1, i2, E, x[0], x[1], x[2], k[3], k[4], k[5]);
    acc_tt(i0, i1, i2, E, x[3], x[4], x[5], k[0], k[1], k[2]);
    double c0, c1, c2;
    levels_td(r0, r1, r2, c0, c1, c2);
    Z[i] = c0; Z[n + i] = c1; Z[2 * n + i] = c2;
    levels_td(i0, i1, i2, c0, c1, c2);
    Z[3 * n + i] = c0; Z[4 * n + i] = c1; Z[5 * n + i] = c2;
    Err[i] = E * UNIT;
}
'''
_MOD = None


def _kernels():
    global _MOD
    if _MOD is None:
        _MOD = module(TD_SRC)
    return _MOD


@dataclass
class TMT:
    """A batch of td Taylor models: c (6, S, B), rem (B,), delta (B,)."""
    c: cp.ndarray
    rem: cp.ndarray
    delta: cp.ndarray

    def __post_init__(self):
        # The kernels read float64 arrays (6, S, B) with no check of their own.
        if self.c.dtype != cp.float64 or self.c.ndim != 3 or self.c.shape[0] != 6:
            raise ValueError(f'a TMT needs float64 coefficients (6, S, B), not '
                             f'{self.c.dtype} {self.c.shape}')


def abs_ub(c):
    """An upper bound of abs(re) + abs(im) of each td coefficient, (S, B)."""
    return cp.sum(cp.abs(c), axis=0) * (1.0 + 1e-15)


def graded(sp, x):
    return (sp.onehot.T @ abs_ub(x.c)) * SAFE


def poly_bound(sp, x, delta=None):
    return tm.poly_bound_g(graded(sp, x), x.delta if delta is None else delta)


def bound(sp, x):
    return (poly_bound(sp, x) + x.rem) * SAFE


def err_bound(sp, err, delta):
    """sum over the coefficients of err * delta^deg, for err (S, B)."""
    g = (sp.onehot.T @ err) * SAFE
    return tm.poly_bound_g(g, delta)


def from_dd(x):
    """A dd model of `tmgpu.tm` as a td model: exact, with a zero third limb."""
    _, s, b = x.c.shape
    c = cp.zeros((6, s, b))
    c[0], c[1], c[3], c[4] = x.c[0], x.c[1], x.c[2], x.c[3]
    return TMT(c, x.rem.copy(), x.delta)


def to_dd(sp, x):
    """A dd model of a td model: lo = fl(x1 + x2) rounds by at most u abs(lo), and
    (x0, lo) becomes a dd by an exact two_sum; that rounding goes to rem."""
    hi = x.c[[0, 3]]
    lo = x.c[[1, 4]] + x.c[[2, 5]]
    # Each CuPy operation below is one IEEE operation (no multiply, so nothing fuses).
    s = hi + lo
    bb = s - hi
    err = (hi - (s - bb)) + (lo - bb)
    c = cp.empty((4,) + x.c.shape[1:])
    c[0], c[1], c[2], c[3] = s[0], err[0], s[1], err[1]
    e = cp.sum(cp.abs(lo), axis=0) * U * (1 + 1e-12)
    return tm.TM(cp.ascontiguousarray(c), (x.rem + err_bound(sp, e, x.delta)) * SAFE, x.delta)


def tconst(z, b):
    """A td constant (6, b) from a dd constant (4, b) of tmgpu.tm: exact."""
    out = cp.zeros((6, b))
    out[0], out[1], out[3], out[4] = z[0], z[1], z[2], z[3]
    return out


def const(sp, value6, delta):
    c = cp.zeros((6, sp.size, delta.shape[0]))
    c[:, 0, :] = value6
    return TMT(c, cp.zeros_like(delta), delta)


def _launch(fn, n, args):
    threads = 256
    fn(((n + threads - 1) // threads,), (threads,), args)


def _axpy(sp, x, y, s):
    tm.check_pair(x, y)
    n = x.c[0].size
    z = cp.empty(x.c.shape)
    err = cp.empty(x.c.shape[1:])
    _launch(_kernels().get_function('tmt_axpy'), n,
            (cp.ascontiguousarray(x.c), cp.ascontiguousarray(y.c), z, err, np.int64(n),
             np.float64(s)))
    return TMT(z, (x.rem + y.rem + err_bound(sp, err * ERR_SAFE, x.delta)) * SAFE, x.delta)


def add(sp, x, y):
    return _axpy(sp, x, y, 1.0)


def sub(sp, x, y):
    return _axpy(sp, x, y, -1.0)


def neg(x):
    return TMT(-x.c, x.rem.copy(), x.delta)


def scale(sp, x, k6):
    """x times a td number for each box, k6 (6, B)."""
    _, s, b = x.c.shape
    z = cp.empty(x.c.shape)
    err = cp.empty((s, b))
    _launch(_kernels().get_function('tmt_scale'), s * b,
            (cp.ascontiguousarray(x.c), cp.ascontiguousarray(k6), z, err, np.int32(s), np.int32(b)))
    kb = abs_ub(k6[:, None, :])[0]
    return TMT(z, (x.rem * kb + err_bound(sp, err * ERR_SAFE, x.delta)) * SAFE, x.delta)


def add_const(sp, x, k6):
    """x plus a td number for each box, k6 (6, B)."""
    _, s, b = x.c.shape
    y = TMT(cp.zeros_like(x.c), cp.zeros_like(x.rem), x.delta)
    y.c[:, 0, :] = k6
    return add(sp, x, y)


def mul(sp, x, y, threads=64, return_err=False):
    """The product truncated at degree n, with the tail, the cross terms and the rounding.
    return_err also returns the error bound of each coefficient (S, B), for tests."""
    tm.check_pair(x, y)
    if x.c.shape[1] != sp.size:
        raise ValueError(f'the models have {x.c.shape[1]} coefficients, the space {sp.size}')
    start, cnt, ii, jj, _ = sp.table()
    _, s, b = x.c.shape
    z = cp.empty(x.c.shape)
    err = cp.empty((s, b))
    _kernels().get_function('tmt_mul')(
        (s, (b + threads - 1) // threads), (min(threads, b),),
        (cp.ascontiguousarray(x.c), cp.ascontiguousarray(y.c), start, cnt, ii, jj, z, err,
         np.int32(s), np.int32(b)))
    gx, gy = graded(sp, x), graded(sp, y)
    bx, by = tm.poly_bound_g(gx, x.delta), tm.poly_bound_g(gy, x.delta)
    rem = (tm.tail_g(gx, gy, x.delta, sp.n) + bx * y.rem + x.rem * by + x.rem * y.rem
           + err_bound(sp, err * ERR_SAFE, x.delta)) * SAFE
    if return_err:
        return TMT(z, rem, x.delta), err * ERR_SAFE
    return TMT(z, rem, x.delta)
