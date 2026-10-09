"""Batched products of complex dd matrices on the GPU (CuPy), with an a priori error bound.

C[b] = A[b] B[b] (mode 0) or C[b] = A[b] B[b]^H (mode 1). Each array is (nb, 4, rows,
cols) with the parts re_hi, re_lo, im_hi, im_lo; nb = 1 for B means one B for all.
Each thread computes one entry of C in a tile of 16 by 16, from tiles of A and B in
shared memory, with the dd operations of `quartic.tmc` (__dadd_rn, __fma_rn,
--fmad=false).

**The bound.** An entry is a sum of K complex products, and each part (re or im) is a
sum of 2K dd products. With 6 u^2 for each product and 3.1 u^2 of the partial sum for
each sum, abs(error) <= (6.2 K + 6) u^2 (abs(A) abs(B))_ij, with abs = abs(re) +
abs(im). `gamma(K)` returns that factor.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from .tmc import _SRC as _DD_SRC

U2 = 2.0 ** -106
TILE = 16

_SRC = _DD_SRC.split('// Z[k] = sum over t')[0] + r'''
#define TILE 16
// C[b] = A[b] op(B[b]); A (nb, 4, M, K); B (nbB, 4, K, N) for mode 0 or (nbB, 4, N, K) for
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
_MOD = None


def _kernel():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD.get_function('zgemm_dd')


def gamma(k):
    """The factor of the a priori bound for an inner dimension k."""
    return (6.2 * k + 6) * U2


def zgemm(a, b, mode=0):
    """C = A op(B) for batches of complex dd matrices; B may have a batch of 1."""
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


def cabs(x):
    """An upper bound of abs(re) + abs(im) of each entry, for arrays (..., 4, r, c)."""
    return (cp.abs(x[..., 0, :, :]) + cp.abs(x[..., 1, :, :]) + cp.abs(x[..., 2, :, :])
            + cp.abs(x[..., 3, :, :])) * (1.0 + 8e-16)
