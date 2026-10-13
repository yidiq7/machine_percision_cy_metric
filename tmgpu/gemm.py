"""Batched products of complex dd matrices on the GPU (CuPy), with an a priori error bound.

C[b] = A[b] B[b] (mode 0) or C[b] = A[b] B[b]^H (mode 1). Each array is (nb, 4, rows,
cols) with the parts re_hi, re_lo, im_hi, im_lo; nb = 1 for B means one B for all.
Each thread computes one entry of C in a tile of 16 by 16, from tiles of A and B in
shared memory, with the dd operations of `tmgpu.tm` (__dadd_rn, __fma_rn,
--fmad=false).

**The bound.** An entry is a sum of K complex products, and each part (re or im) is a
sum of 2K dd products. With 7 u^2 for each product (`tmgpu.tm`), 3.1 u^2 of the
partial sum for each sum and an exact first sum, abs(error) <= (3.1 (2K - 1) + 7) u^2
(abs(A) abs(B))_ij = (6.2 K + 3.9) u^2 (abs(A) abs(B))_ij, with abs = abs(re) + abs(im).
`gamma(K)` returns (6.2 K + 6) u^2, which also holds the terms of order K^2 u^3. The bounds
of dd_mul and dd_add need normalized dd inputs, abs(lo) <= ulp(hi) / 2.

`zgemm_tt` multiplies td matrices with a running bound of its own round-off (`tmgpu.tmtd`).
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from .cuda import DD_PRIMITIVES, module
from .tmtd import ERR_SAFE, TD_SRC

U2 = 2.0 ** -106
TILE = 16

_SRC = DD_PRIMITIVES + f'\n#define TILE {TILE}\n' + r'''// C[b] = A[b] op(B[b]); A (nb, 4, M, K); B (nbB, 4, K, N) for mode 0 or (nbB, 4, N, K) for
// mode 1 (conjugate transpose); C (nb, 4, M, N).
extern "C" __global__ void zgemm_dd(const double* __restrict__ A, const double* __restrict__ Bm,
                                    double* __restrict__ C, int M, int N, int K, int bstride,
                                    int mode) {
    __shared__ double sa[4][TILE][TILE + 1];
    __shared__ double sb[4][TILE][TILE + 1];
    int b = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    const double* a = A + (size_t)b * 4 * M * K;
    const double* bb = Bm + (size_t)b * bstride;
    size_t partA = (size_t)M * K, partB = (size_t)K * N;
    double rh = 0.0, rl = 0.0, ih = 0.0, il = 0.0, ph, pl;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ka = k0 + threadIdx.x;
        for (int q = 0; q < 4; ++q)
            sa[q][threadIdx.y][threadIdx.x] = (row < M && ka < K) ? a[q * partA + (size_t)row * K + ka] : 0.0;
        int kb = k0 + threadIdx.y;
        for (int q = 0; q < 4; ++q) {
            double v = 0.0;
            if (col < N && kb < K) {
                if (mode == 0) v = bb[q * partB + (size_t)kb * N + col];
                else v = bb[q * partB + (size_t)col * K + kb];
            }
            // The conjugate of mode 1 flips the sign of the imaginary parts.
            sb[q][threadIdx.y][threadIdx.x] = (mode == 1 && q >= 2) ? -v : v;
        }
        __syncthreads();
        int kmax = min(TILE, K - k0);
        for (int t = 0; t < kmax; ++t) {
            double xrh = sa[0][threadIdx.y][t], xrl = sa[1][threadIdx.y][t];
            double xih = sa[2][threadIdx.y][t], xil = sa[3][threadIdx.y][t];
            double yrh = sb[0][t][threadIdx.x], yrl = sb[1][t][threadIdx.x];
            double yih = sb[2][t][threadIdx.x], yil = sb[3][t][threadIdx.x];
            dd_mul(xrh, xrl, yrh, yrl, ph, pl); dd_add(rh, rl, ph, pl, rh, rl);
            dd_mul(xih, xil, yih, yil, ph, pl); dd_add(rh, rl, -ph, -pl, rh, rl);
            dd_mul(xrh, xrl, yih, yil, ph, pl); dd_add(ih, il, ph, pl, ih, il);
            dd_mul(xih, xil, yrh, yrl, ph, pl); dd_add(ih, il, ph, pl, ih, il);
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        double* c = C + (size_t)b * 4 * M * N;
        size_t partC = (size_t)M * N, at = (size_t)row * N + col;
        c[at] = rh; c[partC + at] = rl; c[2 * partC + at] = ih; c[3 * partC + at] = il;
    }
}
'''
# The grid of zgemm_tt in Python and the tile of its kernel take the same TILE.
_TT_SRC = f'\n#define TT {TILE}\n' + r'''// C[b] = A[b] op(B[b]) for td arrays: A (nb, 6, M, K), B (nbB, 6, K, N) or (nbB, 6, N, K) for
// mode 1 (conjugate transpose). out_td = 1: C (nb, 6, M, N); else C (nb, 4, M, N) as dd.
// Err (nb, M, N): the running error bound of acc_tt / levels_dd.
extern "C" __global__ void zgemm_tt(const double* __restrict__ A, const double* __restrict__ Bm,
                                    double* __restrict__ C, double* __restrict__ Err,
                                    int M, int N, int K, int bstride, int mode, int out_td) {
    __shared__ double sa[6][TT][TT + 1];
    __shared__ double sb[6][TT][TT + 1];
    int b = blockIdx.z;
    int row = blockIdx.y * TT + threadIdx.y;
    int col = blockIdx.x * TT + threadIdx.x;
    const double* a = A + (size_t)b * 6 * M * K;
    const double* bb = Bm + (size_t)b * bstride;
    size_t partA = (size_t)M * K, partB = (size_t)K * N;
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, i0 = 0.0, i1 = 0.0, i2 = 0.0, E = 0.0;
    for (int k0 = 0; k0 < K; k0 += TT) {
        int ka = k0 + threadIdx.x;
        for (int q = 0; q < 6; ++q)
            sa[q][threadIdx.y][threadIdx.x] = (row < M && ka < K) ? a[q * partA + (size_t)row * K + ka] : 0.0;
        int kb = k0 + threadIdx.y;
        for (int q = 0; q < 6; ++q) {
            double v = 0.0;
            if (col < N && kb < K) {
                if (mode == 0) v = bb[q * partB + (size_t)kb * N + col];
                else v = bb[q * partB + (size_t)col * K + kb];
            }
            sb[q][threadIdx.y][threadIdx.x] = (mode == 1 && q >= 3) ? -v : v;
        }
        __syncthreads();
        int kmax = min(TT, K - k0);
        for (int t = 0; t < kmax; ++t) {
            double x0 = sa[0][threadIdx.y][t], x1 = sa[1][threadIdx.y][t], x2 = sa[2][threadIdx.y][t];
            double x3 = sa[3][threadIdx.y][t], x4 = sa[4][threadIdx.y][t], x5 = sa[5][threadIdx.y][t];
            double y0 = sb[0][t][threadIdx.x], y1 = sb[1][t][threadIdx.x], y2 = sb[2][t][threadIdx.x];
            double y3 = sb[3][t][threadIdx.x], y4 = sb[4][t][threadIdx.x], y5 = sb[5][t][threadIdx.x];
            acc_tt(r0, r1, r2, E, x0, x1, x2, y0, y1, y2);
            acc_tt(r0, r1, r2, E, -x3, -x4, -x5, y3, y4, y5);
            acc_tt(i0, i1, i2, E, x0, x1, x2, y3, y4, y5);
            acc_tt(i0, i1, i2, E, x3, x4, x5, y0, y1, y2);
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        size_t partC = (size_t)M * N, at = (size_t)row * N + col;
        if (out_td) {
            double* c = C + (size_t)b * 6 * M * N;
            double c0, c1, c2;
            levels_td(r0, r1, r2, c0, c1, c2);
            c[at] = c0; c[partC + at] = c1; c[2 * partC + at] = c2;
            levels_td(i0, i1, i2, c0, c1, c2);
            c[3 * partC + at] = c0; c[4 * partC + at] = c1; c[5 * partC + at] = c2;
        } else {
            double* c = C + (size_t)b * 4 * M * N;
            double h, l;
            levels_dd(r0, r1, r2, h, l, E);
            c[at] = h; c[partC + at] = l;
            levels_dd(i0, i1, i2, h, l, E);
            c[2 * partC + at] = h; c[3 * partC + at] = l;
        }
        Err[(size_t)b * M * N + at] = E * UNIT;
    }
}
'''
_MOD = None
_TT_MOD = None


def zgemm_tt(a, b, mode=0, out_td=True):
    """C = A op(B) for batches of complex td matrices (nb, 6, rows, cols); B may have a batch
    of 1. Returns C (td, or dd with out_td=False) and Err (nb, rows, cols), a bound of the
    error of each entry of C (abs re + abs im), from the running bound of the kernel."""
    global _TT_MOD
    if _TT_MOD is None:
        _TT_MOD = module(TD_SRC + _TT_SRC)
    nb, _, m, k = a.shape
    if mode == 0:
        nbb, _, kb, n = b.shape
    else:
        nbb, _, n, kb = b.shape
    if kb != k or nbb not in (1, nb):
        raise ValueError('shapes do not match')
    c = cp.empty((nb, 6 if out_td else 4, m, n))
    err = cp.empty((nb, m, n))
    stride = 0 if nbb == 1 else 6 * k * n
    grid = ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE, nb)
    _TT_MOD.get_function('zgemm_tt')(
        grid, (TILE, TILE), (cp.ascontiguousarray(a), cp.ascontiguousarray(b), c, err,
                             np.int32(m), np.int32(n), np.int32(k), np.int32(stride),
                             np.int32(mode), np.int32(1 if out_td else 0)))
    return c, err * ERR_SAFE


def _kernel():
    global _MOD
    if _MOD is None:
        _MOD = module(_SRC)
    return _MOD.get_function('zgemm_dd')


def gamma(k):
    """The factor of the a priori bound for an inner dimension k."""
    return (6.2 * k + 6) * U2


def zgemm(a, b, mode=0):
    """C = A op(B) for batches of complex dd matrices; B may have a batch of 1. Each entry
    of C errs by at most gamma(k) (abs(A) abs(B))_ij for the inner dimension k."""
    nb, _, m, k = a.shape
    if mode == 0:
        nbb, _, kb, n = b.shape
    else:
        nbb, _, n, kb = b.shape
    if kb != k or nbb not in (1, nb):
        raise ValueError('shapes do not match')
    c = cp.empty((nb, 4, m, n))
    stride = 0 if nbb == 1 else 4 * k * n
    grid = ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE, nb)
    _kernel()(grid, (TILE, TILE), (cp.ascontiguousarray(a), cp.ascontiguousarray(b), c,
                                 np.int32(m), np.int32(n), np.int32(k), np.int32(stride),
                                 np.int32(mode)))
    return c

