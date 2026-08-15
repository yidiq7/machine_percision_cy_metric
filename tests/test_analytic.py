"""The analytic path must agree with nested autodiff to round-off.

Run: uv run python -m tests.test_analytic
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from MLGeometry import complex_math                         # noqa: E402
from quartic import analytic, data, metric, sections        # noqa: E402

BATCH = data.make(50, 3)


def perturbed_theta(lay, seed):
    """A point away from Fubini-Study, so the test does not sit on a symmetry."""
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * 0.1
    return sections.fubini_study_theta(lay) + jnp.asarray(noise)


def test_phi_matches_complex_hessian():
    worst = 0.0
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        mats = sections.prepare(lay, theta)
        for z in BATCH['points'][:4]:
            got, p_got = analytic.phi_analytic(lay, mats, z)
            want = complex_math.complex_hessian(
                lambda w: sections.log_potential_at(lay, mats, w), z)
            p_want = sections.potential_at(lay, mats, z)
            worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
            assert abs(p_got - p_want) < 1e-13 * abs(p_want), k
    assert worst < 1e-13, worst
    print(f"  max relative difference of phi = {worst:.2e}, k = 4, 8, 12")


def test_residual_matches():
    worst = 0.0
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        got = analytic.weighted_residual_analytic(lay, theta, BATCH)
        want = metric.weighted_residual(lay, theta, BATCH)
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-12, worst
    print(f"  max relative difference of the residual = {worst:.2e}")


def test_jacobian_matches_jacfwd():
    """Jacobi's formula against forward-mode autodiff over theta."""
    worst = 0.0
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        v, dv = analytic.residual_jacobian(lay, theta, BATCH)

        weights = BATCH['mass'] / jnp.sum(BATCH['mass'])
        mean = jnp.sum(weights * v)
        dmean = jnp.einsum('i,ia->a', weights, dv)
        got = jnp.sqrt(weights)[:, None] * (
            dv / mean - jnp.einsum('i,a->ia', v, dmean) / mean ** 2)

        want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH))(theta)
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-12, worst
    print(f"  max relative difference of the Jacobian = {worst:.2e}")


def test_chunking_does_not_change_the_result():
    lay = sections.layout(8)
    theta = perturbed_theta(lay, 8)
    full = analytic.weighted_residual_analytic(lay, theta, BATCH)
    piece = analytic.weighted_residual_analytic(lay, theta, BATCH, chunk=25)
    err = float(jnp.abs(full - piece).max() / jnp.abs(full).max())
    assert err < 1e-14, err
    print(f"  lax.map changes the residual by {err:.2e}")


if __name__ == "__main__":
    for fn in (test_phi_matches_complex_hessian, test_residual_matches,
               test_jacobian_matches_jacfwd,
               test_chunking_does_not_change_the_result):
        print(fn.__name__)
        fn()
    print("\nall analytic tests pass")
