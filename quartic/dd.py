"""Double-double arithmetic: a pair of FP64 values that carries about 32 digits.

This is the one place where the arithmetic of the model changes, which is rule 2
of "Output". The oracle measured the relative error of phi at k = 32 as 4.4e-13
in plain FP64, and 1.1e-14 with the Gram stage in extended precision.

A value is a pair `(hi, lo)`, the number is `hi + lo`, and arrays work
elementwise.

**Two warnings about XLA, both measured.**

1. The algebraic simplifier can rewrite `(a + b) - a` as `b`, which destroys an
   error-free transformation. `jax.lax.optimization_barrier` is **not** enough on
   its own. A product built from Veltkamp splitting gave 1.1e-32 outside
   `jax.jit` and 9.2e-17 inside it, on a dot product that cancels by 7.9e16, with
   barriers on every intermediate and on every returned pair. The same happened
   with the barriers placed at each level of the reduction.
2. The cure is to remove the identity, not to fence it. `split` masks the low 27
   mantissa bits through a bitcast, so `hi` is exact with no subtraction to
   simplify, and the partial products of two split values are exact in FP64. That
   version gives 1.1e-32 inside `jax.jit` and outside it.

`two_sum` does survive `jax.jit`, so the reduction still uses it. Any change here
must keep `tests/test_dd.py` passing, and that test runs everything through
`jax.jit`, which is the only way to see this class of fault.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ['split', 'two_sum', 'quick_two_sum', 'add', 'sum_dd', 'dot',
           'dot_dd_fp', 'mul_dd_fp', 'to_dd', 'to_fp']

# Keeps the top 26 mantissa bits, so a product of two high parts is exact.
_LOW27 = jnp.int64(~((1 << 27) - 1))


def _hold(*args):
    """Stop XLA from moving arithmetic across this point."""
    out = jax.lax.optimization_barrier(args)
    return out if len(args) > 1 else out[0]


def split(a):
    """a = hi + lo exactly, with hi carrying 26 mantissa bits.

    A bit mask has no algebraic identity, so the simplifier cannot undo it. The
    subtraction `a - hi` is exact, because hi is a truncation of a.
    """
    bits = jax.lax.bitcast_convert_type(a, jnp.int64)
    hi = jax.lax.bitcast_convert_type(bits & _LOW27, jnp.float64)
    return hi, a - hi


def _partials(a, b, axis):
    """The partial products of a * b, stacked along `axis`.

    Three of the four are exact in FP64. The fourth, lo times lo, can round, and
    it is smaller than the result by 2^-54, so its error sits below the precision
    that a pair of FP64 values can hold anyway.
    """
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    return jnp.concatenate([a_hi * b_hi, a_hi * b_lo, a_lo * b_hi, a_lo * b_lo],
                           axis=axis)


def two_sum(a, b):
    """s = a + b rounded, and err with s + err = a + b exactly."""
    s = _hold(a + b)
    bb = _hold(s - a)
    err = (a - (s - bb)) + (b - bb)
    return _hold(s, err)


def quick_two_sum(a, b):
    """The same, when abs(a) >= abs(b) is already known."""
    s = _hold(a + b)
    err = b - (s - a)
    return _hold(s, err)


def add(x, y):
    """Sum of two double-double values, with no assumption on the magnitudes.

    The shorter form, which folds the low parts with one FP64 addition and ends
    in `quick_two_sum`, needs `abs(hi) >= abs(lo)`. A sum that cancels breaks
    that, because the high parts annihilate and the low parts survive.
    """
    s1, s2 = two_sum(x[0], y[0])
    t1, t2 = two_sum(x[1], y[1])
    s1, s2 = two_sum(s1, s2 + t1)
    s1, s2 = two_sum(s1, s2 + t2)
    return _hold(s1, s2)


def sum_dd(hi, lo, axis=-1):
    """Sum a stack of double-double values along `axis`, by a pairwise tree.

    A tree needs log2(n) dependent steps instead of n, so the cost stays low
    while the accuracy of the whole sum stays at the double-double level.
    """
    hi = jnp.moveaxis(hi, axis, -1)
    lo = jnp.moveaxis(lo, axis, -1)
    n = hi.shape[-1]
    while n > 1:
        half = (n + 1) // 2
        if n % 2:
            hi = jnp.concatenate([hi, jnp.zeros_like(hi[..., :1])], axis=-1)
            lo = jnp.concatenate([lo, jnp.zeros_like(lo[..., :1])], axis=-1)
        hi, lo = add((hi[..., :half], lo[..., :half]),
                     (hi[..., half:], lo[..., half:]))
        n = half
    return hi[..., 0], lo[..., 0]


def dot(a, b, axis=-1):
    """Sum of a * b over `axis`, for FP64 arrays, in double-double."""
    parts = _partials(a, b, axis)
    return sum_dd(parts, jnp.zeros_like(parts), axis=axis)


def mul_dd_fp(x, b):
    """A double-double value times an FP64 value.

    The last term, lo times b, is the only one that rounds, and it is smaller
    than the result by 2^-53.
    """
    hi_hi, hi_lo = split(x[0])
    b_hi, b_lo = split(b)
    terms = jnp.stack([hi_hi * b_hi, hi_hi * b_lo, hi_lo * b_hi, hi_lo * b_lo,
                       x[1] * b], axis=-1)
    return sum_dd(terms, jnp.zeros_like(terms), axis=-1)


def dot_dd_fp(x, b, axis=-1):
    """Sum of x * b over `axis`, where x is double-double and b is FP64."""
    hi_hi, hi_lo = split(x[0])
    b_hi, b_lo = split(b)
    terms = jnp.concatenate([hi_hi * b_hi, hi_hi * b_lo, hi_lo * b_hi,
                             hi_lo * b_lo, x[1] * b], axis=axis)
    return sum_dd(terms, jnp.zeros_like(terms), axis=axis)


def to_dd(x):
    """An FP64 array as a double-double value."""
    return x, jnp.zeros_like(x)


def to_fp(x):
    """Round a double-double value back to FP64."""
    return x[0] + x[1]
