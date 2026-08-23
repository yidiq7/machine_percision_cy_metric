"""The error-free transformations must survive jax.jit and the XLA simplifier.

Run: uv run python -m tests.test_dd
"""
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import dd                                      # noqa: E402

mp.dps = 60


def gen(seed):
    """One generator for each test, so the order of the tests does not matter."""
    return np.random.default_rng(seed)


def exact(x):
    return mpf(float(x))


def test_two_sum_is_exact_under_jit():
    """s + err must equal a + b with no error at all."""
    rng = gen(1)
    a = jnp.asarray(rng.normal(size=400) * 10.0 ** rng.integers(-30, 30, 400))
    b = jnp.asarray(rng.normal(size=400) * 10.0 ** rng.integers(-30, 30, 400))
    s, err = jax.jit(dd.two_sum)(a, b)
    worst = 0.0
    for i in range(len(a)):
        ref = exact(a[i]) + exact(b[i])
        got = exact(s[i]) + exact(err[i])
        scale = abs(ref) if ref != 0 else mpf(1)
        worst = max(worst, float(abs(got - ref) / scale))
    assert worst == 0.0, worst
    print(f"  two_sum is exact for 400 pairs, worst {worst:.1e}")


def test_split_is_exact_and_masked():
    """a = hi + lo exactly, and hi must carry no more than 26 mantissa bits."""
    rng = gen(2)
    a = jnp.asarray(rng.normal(size=500) * 10.0 ** rng.integers(-200, 200, 500))
    hi, lo = jax.jit(dd.split)(a)
    assert np.array_equal(np.asarray(hi) + np.asarray(lo), np.asarray(a))
    assert np.all((np.asarray(hi).view(np.int64) & ((1 << 27) - 1)) == 0)
    print("  split is exact for 500 values, and hi uses 26 bits")


def test_dot_beats_fp64_on_a_cancelling_sum():
    """A dot product that cancels hard must keep its digits."""
    rng = gen(3)
    n = 64
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    # Force the sum to cancel: shift b so that a . b is tiny.
    ref_full = sum((exact(a[i]) * exact(b[i]) for i in range(n)), mpf(0))
    b = b - float(ref_full) / float(np.sum(a * a)) * a
    ref = sum((exact(a[i]) * exact(b[i]) for i in range(n)), mpf(0))
    scale = sum((abs(exact(a[i]) * exact(b[i])) for i in range(n)), mpf(0))
    cancel = float(scale / abs(ref))

    plain = float(np.dot(a, b))
    # Through jax.jit on purpose: that is the only place the fault appears.
    hi, lo = jax.jit(dd.dot)(jnp.asarray(a), jnp.asarray(b))
    fine = exact(hi) + exact(lo)

    e_plain = float(abs(exact(plain) - ref) / abs(ref))
    e_dd = float(abs(fine - ref) / abs(ref))
    print(f"  cancellation {cancel:.1e}: FP64 rel err {e_plain:.2e}, "
          f"dd rel err {e_dd:.2e}, gain {e_plain / max(e_dd, 1e-300):.1e}")
    assert e_dd < e_plain / 1e6, (e_plain, e_dd)


def test_sum_dd_matches_mpmath():
    """A pairwise tree in dd must reproduce the exact sum of many terms."""
    rng = gen(4)
    n = 300
    x = rng.normal(size=n) * 10.0 ** rng.integers(-12, 12, n)
    hi, lo = jax.jit(lambda v: dd.sum_dd(v, jnp.zeros_like(v)))(jnp.asarray(x))
    ref = sum((exact(v) for v in x), mpf(0))
    got = exact(hi) + exact(lo)
    err = float(abs(got - ref) / abs(ref))
    plain = float(abs(exact(float(np.sum(x))) - ref) / abs(ref))
    assert err < 1e-28, err
    print(f"  sum of {n} terms: FP64 {plain:.2e}, dd {err:.2e}")


if __name__ == "__main__":
    for fn in (test_two_sum_is_exact_under_jit, test_split_is_exact_and_masked,
               test_dot_beats_fp64_on_a_cancelling_sum, test_sum_dd_matches_mpmath):
        print(fn.__name__)
        fn()
    print("\nall dd tests pass")
