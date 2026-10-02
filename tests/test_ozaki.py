"""The Ozaki product must reach dd accuracy, also under jax.jit and for sums that cancel.

Run: uv run python -m tests.test_ozaki
"""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import dd, ozaki                               # noqa: E402


def exact(a, b):
    """a @ b in exact rational arithmetic, for FP64 inputs."""
    fa = [[Fraction(float(x)) for x in row] for row in a]
    fb = [[Fraction(float(x)) for x in row] for row in b]
    return [[sum(fa[i][l] * fb[l][j] for l in range(len(fb))) for j in range(len(fb[0]))]
            for i in range(len(fa))]


def relative_error(got, want, scale):
    worst = 0.0
    for i in range(len(want)):
        for j in range(len(want[0])):
            value = Fraction(float(got[0][i, j])) + Fraction(float(got[1][i, j]))
            worst = max(worst, abs(float(value - want[i][j])) / scale)
    return worst


def cases():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 300)) * np.exp(rng.normal(size=(6, 300)) * 3)
    b = rng.normal(size=(300, 4))
    # A row that cancels: the sum of its products is far below the size of its terms.
    b_c = rng.normal(size=(300, 1))
    a_c = rng.normal(size=(1, 300))
    a_c[0, -1] = -(a_c[0, :-1] @ b_c[:-1, 0]) / b_c[-1, 0]
    return [(a, b), (a_c, b_c)]


def test_fp_products_reach_dd():
    fn = jax.jit(ozaki.matmul_fp)
    for a, b in cases():
        want = exact(a, b)
        got = [np.asarray(x) for x in fn(jnp.asarray(a), jnp.asarray(b))]
        scale = float(np.max(np.abs(a)) * np.max(np.abs(b)))
        err = relative_error(got, want, scale)
        assert err < 1e-30, err
        print(f"  {a.shape} @ {b.shape}: error {err:.1e} of max|a| max|b|, under jax.jit")


def test_dd_products_reach_dd():
    rng = np.random.default_rng(6)
    a_hi = rng.normal(size=(5, 200))
    a_lo = a_hi * 2.0 ** -60 * rng.normal(size=a_hi.shape)
    b_hi = rng.normal(size=(200, 3))
    b_lo = b_hi * 2.0 ** -60 * rng.normal(size=b_hi.shape)
    fa = [[Fraction(float(x)) + Fraction(float(y)) for x, y in zip(r, s)]
          for r, s in zip(a_hi, a_lo)]
    fb = [[Fraction(float(x)) + Fraction(float(y)) for x, y in zip(r, s)]
          for r, s in zip(b_hi, b_lo)]
    want = [[sum(fa[i][l] * fb[l][j] for l in range(200)) for j in range(3)] for i in range(5)]
    got = jax.jit(ozaki.matmul)((jnp.asarray(a_hi), jnp.asarray(a_lo)),
                                (jnp.asarray(b_hi), jnp.asarray(b_lo)))
    got = [np.asarray(x) for x in got]
    err = relative_error(got, want, float(np.max(np.abs(a_hi)) * np.max(np.abs(b_hi))))
    assert err < 1e-30, err
    print(f"  dd @ dd: error {err:.1e} of max|a| max|b|, under jax.jit")


def test_long_inner_dimension_is_cut():
    """An inner dimension above 2^15 goes in parts, each exact, added in dd."""
    rng = np.random.default_rng(7)
    k = 40_000
    a = rng.normal(size=(2, k)) * np.exp(rng.normal(size=(2, k)) * 2)
    b = rng.normal(size=(k, 1))
    want = exact(a, b)
    got = [np.asarray(x) for x in jax.jit(ozaki.matmul_fp)(jnp.asarray(a), jnp.asarray(b))]
    err = relative_error(got, want, float(np.max(np.abs(a)) * np.max(np.abs(b))))
    assert err < 1e-30, err
    print(f"  inner dimension {k}: error {err:.1e} of max|a| max|b|, under jax.jit")


if __name__ == "__main__":
    for fn in (test_fp_products_reach_dd, test_dd_products_reach_dd,
               test_long_inner_dimension_is_cut):
        print(fn.__name__)
        fn()
    print("\nall ozaki tests pass")
