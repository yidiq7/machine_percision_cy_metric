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

__all__ = ['split', 'two_sum', 'quick_two_sum', 'add', 'neg', 'mul',
           'mul_fast', 'cmul_fast',
           'reciprocal', 'cadd', 'cneg', 'cmul', 'creciprocal',
           'csub', 'cabs2', 'cdiv', 'cto_dd',
           'cto_fp', 'sum_dd', 'sum_fp',
           'dot', 'dot_dd_fp', 'mul_dd_fp', 'to_dd', 'to_fp']

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


def _partials(a, b):
    """The partial products of a * b, as 4 separate arrays.

    Three of the four are exact in FP64. The fourth, lo times lo, can round, and
    it is smaller than the result by 2^-54, so its error sits below the precision
    that a pair of FP64 values can hold anyway.

    They stay separate on purpose. Stacking them along the reduction axis would
    make a tensor 4 times the size of the FP64 one, and the matrix-vector
    products of the model are already the largest arrays in the step.
    """
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    return (a_hi * b_hi, a_hi * b_lo, a_lo * b_hi, a_lo * b_lo)


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


def sum_fp(x, axis=-1):
    """Sum FP64 values along `axis`, in double-double.

    The first level pairs the input with `two_sum`, so no array of zeros is ever
    made for the low parts.
    """
    x = jnp.moveaxis(x, axis, -1)
    n = x.shape[-1]
    if n == 1:
        return x[..., 0], jnp.zeros_like(x[..., 0])
    half = (n + 1) // 2
    if n % 2:
        x = jnp.concatenate([x, jnp.zeros_like(x[..., :1])], axis=-1)
    hi, lo = two_sum(x[..., :half], x[..., half:])
    n = half
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
    """Sum of a * b over `axis`, for FP64 arrays, in double-double.

    Each partial product is reduced on its own and the 4 results are added, so
    the peak memory stays at the size of one FP64 term.
    """
    acc = None
    for term in _partials(a, b):
        piece = sum_fp(term, axis=axis)
        acc = piece if acc is None else add(acc, piece)
    return acc


def mul_dd_fp(x, b):
    """A double-double value times an FP64 value.

    The last term, lo times b, is the only one that rounds, and it is smaller
    than the result by 2^-53.
    """
    acc = to_dd(x[1] * b)
    for term in _partials(x[0], b):
        acc = add(acc, to_dd(term))
    return acc


def dot_dd_fp(x, b, axis=-1):
    """Sum of x * b over `axis`, where x is double-double and b is FP64."""
    acc = None
    for term in _partials(x[0], b) + (x[1] * b,):
        piece = sum_fp(term, axis=axis)
        acc = piece if acc is None else add(acc, piece)
    return acc


def neg(x):
    """Negate a double-double value."""
    return -x[0], -x[1]


def mul(x, y):
    """Product of two double-double values.

    The 7 partial products are summed largest first, with `two_sum` carrying the
    error of each step. The terms run from the size of the result down to
    2^-106 of it, so this reaches the precision of the pair.

    **Do not replace this with the classic `two_prod` form.** That one builds
    the error of the high product as `a_hi*b_hi - a*b`, and the XLA simplifier
    rewrites it: measured, `z**4` from two nested products came out at 2.2e-17
    inside one `jax.jit` and exact outside it. `two_sum` does survive `jax.jit`,
    and no expression here is an algebraic identity.
    """
    terms = _partials(x[0], y[0]) + (x[0] * y[1], x[1] * y[0], x[1] * y[1])
    hi = terms[0]
    lo = jnp.zeros_like(hi)
    for term in terms[1:]:
        hi, err = two_sum(hi, term)
        lo = lo + err
    return quick_two_sum(hi, lo)


def mul_fast(x, y):
    """Product of two double-double values, accurate to about 2^-79.

    The high-part product `a_hi * b_hi` is exact, and every remaining term is
    2^-26 of it or smaller, so they are summed in FP64. The result therefore
    carries approximately 24 digits and not 32.

    **This is enough for stages 1 to 3 of the evaluation**, where the target is
    a floor of 2.5e-15 on abs(f), which needs about 20 digits. It costs 2
    barriers against the 20 of `mul`, and the number of barriers is what sets
    the compile time: XLA cannot fuse across one, and the exact `mul` in the
    loop over the monomials did not finish the compile of k = 8.
    """
    a_hi, a_lo = split(x[0])
    b_hi, b_lo = split(y[0])
    hi = a_hi * b_hi
    lo = ((a_hi * b_lo + a_lo * b_hi)
          + (a_lo * b_lo + x[0] * y[1] + x[1] * y[0]))
    return quick_two_sum(hi, lo)


def cmul_fast(a, b):
    """`cmul` with `mul_fast` on the four products.

    The two sums keep the assumption-free `add`, because the real part
    `a_re b_re - a_im b_im` can cancel, and `quick_two_sum` would then be
    invalid.
    """
    return (add(mul_fast(a[0], b[0]), neg(mul_fast(a[1], b[1]))),
            add(mul_fast(a[0], b[1]), mul_fast(a[1], b[0])))


def reciprocal(x):
    """1/x for a double-double value, by one Newton step on an FP64 seed.

    The seed is correct to 2^-53, and a Newton step for the reciprocal squares
    the error, so one step reaches 2^-106. That is the precision of the pair.
    """
    r = to_dd(1.0 / x[0])
    one = to_dd(jnp.ones_like(x[0]))
    return add(r, mul(r, add(one, neg(mul(x, r)))))


def cadd(a, b):
    """Sum of two complex double-double values, each a pair of real pairs."""
    return add(a[0], b[0]), add(a[1], b[1])


def cneg(a):
    """Negate a complex double-double value."""
    return neg(a[0]), neg(a[1])


def cmul(a, b):
    """Product of two complex double-double values.

    The plain 4-product form, and not Karatsuba: 3 multiplications would save
    one product and add two sums that cancel, which is what extended precision
    is here to avoid.
    """
    return (add(mul(a[0], b[0]), neg(mul(a[1], b[1]))),
            add(mul(a[0], b[1]), mul(a[1], b[0])))


def creciprocal(a):
    """1/a for a complex double-double value, through the conjugate."""
    den = reciprocal(add(mul(a[0], a[0]), mul(a[1], a[1])))
    return mul(a[0], den), neg(mul(a[1], den))


def csub(a, b):
    """Difference of two complex double-double values."""
    return cadd(a, cneg(b))


def cabs2(a):
    """abs(a)^2 for a complex double-double value, as a real pair."""
    return add(mul(a[0], a[0]), mul(a[1], a[1]))


def cdiv(a, b):
    """a / b for complex double-double values."""
    return cmul(a, creciprocal(b))


def cto_dd(z):
    """An FP64 complex array as a complex double-double value."""
    return to_dd(jnp.real(z)), to_dd(jnp.imag(z))


def cto_fp(a):
    """Round a complex double-double value back to FP64 complex."""
    return to_fp(a[0]) + 1j * to_fp(a[1])


def to_dd(x):
    """An FP64 array as a double-double value."""
    return x, jnp.zeros_like(x)


def to_fp(x):
    """Round a double-double value back to FP64."""
    return x[0] + x[1]
