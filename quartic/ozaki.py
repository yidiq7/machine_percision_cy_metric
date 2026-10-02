"""Matrix products in dd from FP64 matrix products of slices (the Ozaki splitting).

Each row of A and each column of B is cut into `SLICES` slices relative to its
largest entry: SLICES - 1 on power-of-2 grids of `BITS` bits each, and the rest. A
product of two grid slices is a sum of integers times one unit with at most
2 BITS + log2(K) bits, so an FP64 matrix product of two grid slices is exact in
any order of summation. A product with a rest slice rounds, but the rest is below
2^-((SLICES - 1) BITS) of the largest entry, so that error is below
2^-((SLICES - 1) BITS + 53) of it. The SLICES^2 slice products are added in dd. The
grids come from `trunc`, and no step is an identity that the simplifier of XLA
can remove. An inner dimension above 2^(53 - 2 BITS) is cut into parts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import dd

__all__ = ['BITS', 'matmul_fp', 'matmul']

# 2 BITS + log2(K) <= 53 must hold for the inner dimension K: 19 bits allow K <= 2^15.
# With 4 slices the rest is below 2^-57 of the largest entry of its row, so a product
# with it rounds at 2^-110; 3 slices leave 2^-91, which a V+A recurrence amplifies.
BITS = 19
SLICES = 4


def _slices(x, axis):
    """x as the sum of SLICES parts, on power-of-2 grids of each row (axis=1) or column."""
    top = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    top = jnp.where(top > 0, top, 1.0)
    # The power of 2 above top, from the exponent bits: exp2 on the device is not exact.
    bits = jax.lax.bitcast_convert_type(top, jnp.int64)
    biased = (bits >> 52) & 0x7FF
    unit = jax.lax.bitcast_convert_type((biased + 1) << 52, jnp.float64)
    out, rest = [], x
    for p in range(1, SLICES):
        grid = unit * 2.0 ** (-BITS * p)
        # Truncation keeps the sign of the rest, so rest - part takes bits of rest only
        # and is exact; floor of a small negative rest would need 60 bits.
        part = jnp.trunc(rest / grid) * grid
        out.append(part)
        rest = rest - part
    out.append(rest)
    return out


def matmul_fp(a, b):
    """a @ b for FP64 matrices (..., M, K) and (K, N), as a dd pair."""
    limit = 2 ** (53 - 2 * BITS)
    if a.shape[-1] > limit:
        acc = None
        for start in range(0, a.shape[-1], limit):
            piece = matmul_fp(a[..., start:start + limit], b[start:start + limit])
            acc = piece if acc is None else dd.add(acc, piece)
        return acc
    sa, sb = _slices(a, -1), _slices(b, 0)
    acc = None
    for xa in sa:
        for xb in sb:
            piece = dd.to_dd(xa @ xb)
            acc = piece if acc is None else dd.add(acc, piece)
    return acc


def matmul(a, b):
    """a @ b in dd, where each of a and b is an FP64 array or a dd pair (hi, lo)."""
    a_hi, a_lo = a if isinstance(a, tuple) else (a, None)
    b_hi, b_lo = b if isinstance(b, tuple) else (b, None)
    out = matmul_fp(a_hi, b_hi)
    # The low parts are 2^-53 of the high parts, so their products need FP64 only.
    small = None
    if a_lo is not None:
        small = a_lo @ b_hi
    if b_lo is not None:
        term = a_hi @ b_lo
        small = term if small is None else small + term
    return out if small is None else dd.add(out, dd.to_dd(small))
