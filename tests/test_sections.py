"""Tests of the potential p against an exact reference and against Gamma.

Run: .venv/bin/python -m tests.test_sections
"""
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                           # noqa: E402

from quartic import blocks, sections              # noqa: E402

rng = np.random.default_rng(2026)


def points_on_X(n):
    """Points of the Fermat quartic: fix 3 coordinates, solve for the fourth."""
    z = rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))
    z[:, 3] = (-(z[:, 0] ** 4 + z[:, 1] ** 4 + z[:, 2] ** 4)) ** 0.25
    for _ in range(3):                            # Newton, to the round-off level
        p = (z ** 4).sum(axis=1)
        z[:, 3] -= p / (4 * z[:, 3] ** 3)
    z /= np.abs(z).max(axis=1, keepdims=True)
    return z


def test_p_equals_fubini_study():
    """At theta_FS the model must give exactly (sum_a |z_a|^2)^k."""
    print(f"  {'k':>3} {'n_theta':>8} {'max rel err':>12}")
    for k in (2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 38):
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        z = points_on_X(40)
        got = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(jnp.asarray(z)))
        want = (np.abs(z) ** 2).sum(axis=1) ** k
        err = np.abs(got - want).max() / np.abs(want).max()
        print(f"  {k:>3} {lay.n_theta:>8} {err:>12.3e}")
        assert err < 1e-13, (k, err)
    print("  p = (sum |z_a|^2)^k at theta_FS, to k = 38")


def test_p_is_gamma_invariant():
    """Gamma = (Z_4^3 semidirect S_4) x Z_2, and p must not change under it."""
    lay = sections.layout(12)
    theta = jnp.asarray(rng.normal(size=lay.n_theta))
    z = points_on_X(8)
    base = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(jnp.asarray(z)))
    scale = np.abs(base).max()

    worst_perm = worst_phase = 0.0
    for sigma in blocks.S4:
        moved = np.zeros_like(z)
        for i in range(4):
            moved[:, sigma[i]] = z[:, i]
        got = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(jnp.asarray(moved)))
        worst_perm = max(worst_perm, np.abs(got - base).max() / scale)

    for powers in itertools.product(range(4), repeat=3):
        phase = np.exp(0.5j * np.pi * np.array(powers + (0,)))
        got = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(
            jnp.asarray(z * phase)))
        worst_phase = max(worst_phase, np.abs(got - base).max() / scale)

    conj = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(
        jnp.asarray(np.conj(z))))
    worst_conj = np.abs(conj - base).max() / scale

    assert max(worst_perm, worst_phase, worst_conj) < 1e-13
    print(f"  S_4 {worst_perm:.2e},  Z_4^3 {worst_phase:.2e},  "
          f"conjugation {worst_conj:.2e}")


def test_p_is_real_and_scales():
    """p is a real number, and h -> lambda h gives p -> lambda p.

    At theta_FS every block term is positive, so the sum keeps full precision.
    A random theta makes h indefinite, and then the 64 blocks cancel.
    """
    lay = sections.layout(8)
    z = jnp.asarray(points_on_X(4))
    for tag, theta, tol in (('theta_FS', sections.fubini_study_theta(lay), 2e-15),
                            ('random  ', jnp.asarray(rng.normal(size=lay.n_theta)), 1e-13)):
        base = jax.vmap(lambda w: sections.potential(lay, theta, w))(z)
        assert base.dtype == jnp.float64, base.dtype
        scaled = jax.vmap(lambda w: sections.potential(lay, 3.0 * theta, w))(z)
        err = float(jnp.abs(scaled - 3.0 * base).max() / jnp.abs(base).max())
        assert err < tol, (tag, err)
        print(f"  {tag}: p is float64, and the scale of h passes through to {err:.2e}")


def test_p_matches_a_direct_monomial_sum():
    """Compare against a plain sum over the monomials of degree k, with no blocks."""
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        z = points_on_X(6)
        got = np.array(jax.vmap(lambda w: sections.potential(lay, theta, w))(jnp.asarray(z)))

        exps = [e for e in itertools.product(range(k + 1), repeat=4) if sum(e) == k]
        coeff = np.array([math.factorial(k) //
                          math.prod(math.factorial(i) for i in e) for e in exps],
                         dtype=float)
        mono = np.array([[np.prod(np.abs(w) ** (2 * np.array(e))) for e in exps] for w in z])
        want = mono @ coeff
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e-13, (k, err)
    print("  p agrees with a direct monomial sum at theta_FS, k = 4, 8, 12")


def test_cost_and_timing():
    print(f"  {'k':>3} {'n_theta':>8} {'MACs':>9} {'build':>8} {'1e4 points':>12}")
    for k in (12, 20, 32):
        t0 = time.perf_counter()
        lay = sections.layout(k)
        build = time.perf_counter() - t0
        theta = sections.fubini_study_theta(lay)
        z = jnp.asarray(points_on_X(10_000))
        fn = jax.jit(lambda t, w: jax.vmap(lambda q: sections.potential(lay, t, q))(w))
        fn(theta, z).block_until_ready()
        t0 = time.perf_counter()
        fn(theta, z).block_until_ready()
        print(f"  {k:>3} {lay.n_theta:>8} {lay.structure.macs:>9,} "
              f"{build:>7.1f}s {time.perf_counter() - t0:>11.2f}s")


if __name__ == "__main__":
    for fn in (test_p_equals_fubini_study, test_p_is_gamma_invariant,
               test_p_is_real_and_scales, test_p_matches_a_direct_monomial_sum,
               test_cost_and_timing):
        print(fn.__name__)
        fn()
    print("\nall section tests pass")
