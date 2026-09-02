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


def test_mul_and_reciprocal_under_jit():
    """dd times dd, and 1/dd, must reach the precision of the pair.

    Stages 1 to 3 of the evaluation need these, because the monomials of u and
    the powers of z multiply values that are already double-double.
    """
    rng = gen(5)
    worst_mul = worst_rec = 0.0
    mul = jax.jit(lambda a, b: dd.mul(dd.to_dd(a), dd.to_dd(b)))
    rec = jax.jit(lambda a: dd.reciprocal(dd.to_dd(a)))
    for _ in range(12):
        a, b = rng.normal() * 10.0 ** rng.integers(-6, 6), rng.normal()
        pair = mul(jnp.float64(a), jnp.float64(b))
        ref = exact(a) * exact(b)
        got = exact(pair[0]) + exact(pair[1])
        worst_mul = max(worst_mul, float(abs((got - ref) / ref)))

        pair = rec(jnp.float64(a))
        ref = 1 / exact(a)
        got = exact(pair[0]) + exact(pair[1])
        worst_rec = max(worst_rec, float(abs((got - ref) / ref)))
    assert worst_mul < 1e-30, worst_mul
    assert worst_rec < 1e-30, worst_rec
    print(f"  mul {worst_mul:.2e}, reciprocal {worst_rec:.2e}")


def test_mul_fast_is_enough_for_the_sections():
    """`mul_fast` must reach about 2^-79, which stages 1 to 3 need.

    The exact `mul` costs 20 barriers against 2, and the number of barriers sets
    the compile time. The target for stages 1 to 3 is a floor of 2.5e-15 on
    abs(f), so about 20 digits, and this form gives 24.
    """
    rng = gen(7)
    fast = jax.jit(lambda w: dd.cmul_fast(
        dd.cmul_fast(dd.cto_dd(w), dd.cto_dd(w)),
        dd.cmul_fast(dd.cto_dd(w), dd.cto_dd(w))))
    worst = plain = 0.0
    for _ in range(8):
        za = complex(rng.normal(), rng.normal())
        pair = fast(jnp.complex128(za))
        got = mp.mpc(exact(pair[0][0]) + exact(pair[0][1]),
                     exact(pair[1][0]) + exact(pair[1][1]))
        ref = mp.mpc(za) ** 4
        worst = max(worst, float(abs((got - ref) / ref)))
        plain = max(plain, float(abs(
            (mp.mpc(complex(np.complex128(za) ** 4)) - ref) / ref)))
    assert worst < 1e-22, worst
    print(f"  z**4 fast {worst:.2e} against FP64 {plain:.2e}")


def test_complex_layer_under_jit():
    """The complex wrappers must keep the precision of the real ones."""
    rng = gen(6)
    cmul = jax.jit(lambda a, b: dd.cmul(dd.cto_dd(a), dd.cto_dd(b)))
    crec = jax.jit(lambda a: dd.creciprocal(dd.cto_dd(a)))
    pow4 = jax.jit(lambda w: dd.cmul(dd.cmul(dd.cto_dd(w), dd.cto_dd(w)),
                                     dd.cmul(dd.cto_dd(w), dd.cto_dd(w))))
    worst = [0.0, 0.0, 0.0]
    plain4 = 0.0
    for _ in range(12):
        za = complex(rng.normal(), rng.normal())
        zb = complex(rng.normal(), rng.normal())
        for i, (fn, ref) in enumerate((
                (lambda: cmul(jnp.complex128(za), jnp.complex128(zb)),
                 mp.mpc(za) * mp.mpc(zb)),
                (lambda: crec(jnp.complex128(za)), 1 / mp.mpc(za)),
                (lambda: pow4(jnp.complex128(za)), mp.mpc(za) ** 4))):
            pair = fn()
            got = mp.mpc(exact(pair[0][0]) + exact(pair[0][1]),
                         exact(pair[1][0]) + exact(pair[1][1]))
            worst[i] = max(worst[i], float(abs((got - ref) / ref)))
        ref = mp.mpc(za) ** 4
        plain4 = max(plain4, float(abs(
            (mp.mpc(complex(np.complex128(za) ** 4)) - ref) / ref)))
    for w in worst:
        assert w < 1e-30, worst
    print(f"  cmul {worst[0]:.2e}, creciprocal {worst[1]:.2e}, "
          f"z**4 {worst[2]:.2e} against FP64 {plain4:.2e}")


if __name__ == "__main__":
    for fn in (test_two_sum_is_exact_under_jit, test_split_is_exact_and_masked,
               test_dot_beats_fp64_on_a_cancelling_sum, test_sum_dd_matches_mpmath,
               test_mul_and_reciprocal_under_jit, test_complex_layer_under_jit,
               test_mul_fast_is_enough_for_the_sections):
        print(fn.__name__)
        fn()
    print("\nall dd tests pass")
