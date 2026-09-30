"""The correction of the potential must agree with autodiff, and with its FS form.

Run: uv run python -m tests.test_correction
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
from quartic import analytic, correction, data, sections    # noqa: E402

analytic.GRAM_DD, analytic.SECTION_DD = False, False
sections.PREPARE_DD = False
BATCH = data.make(50, 3)


def near_fs(lay, seed, size=1e-2):
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * size
    return sections.fubini_study_theta(lay) + jnp.asarray(noise)


def wirtinger(f, z):
    """d f / d z_i of a real function f, as 0.5 (d/dx - i d/dy)."""
    gx = jax.grad(lambda x: f(x + 1j * jnp.imag(z)))(jnp.real(z))
    gy = jax.grad(lambda y: f(jnp.real(z) + 1j * y))(jnp.imag(z))
    return 0.5 * (gx - 1j * gy)


def rel(got, want):
    return float(jnp.abs(got - want).max() / jnp.abs(want).max())


def test_psi_parts_match_autodiff():
    """psi, d psi and ddbar psi against autodiff, for both weights."""
    lay0 = sections.layout(4)
    mats0 = sections.prepare(lay0, near_fs(lay0, 1))
    lay_c = sections.layout(6, normalize=False)
    coeffs = jnp.asarray(np.random.default_rng(2).normal(size=lay_c.n_theta))
    cmats = sections.prepare_fp(lay_c, coeffs)
    worst = 0.0
    for fs_weight, alpha in ((True, 6.0), (False, 1.5)):
        for z in BATCH['points'][:4]:
            if fs_weight:
                state = correction.fs_state(z)

                def pot(w):
                    return jnp.log(jnp.sum(jnp.abs(w) ** 2))
            else:
                state = correction.base_state(lay0, mats0, z)

                def pot(w):
                    return sections.log_potential_at(lay0, mats0, w)

            def psi(w):
                return sections.potential_at(lay_c, cmats, w) * jnp.exp(-alpha * pot(w))

            got = correction.psi_parts(lay_c, cmats, z, alpha, *state)
            worst = max(worst,
                        abs(float(got[0] - psi(z))) / abs(float(psi(z))),
                        rel(got[1], wirtinger(psi, z)),
                        rel(got[2], complex_math.complex_hessian(psi, z)))
    assert worst < 1e-12, worst
    print(f"  psi, d psi and ddbar psi against autodiff: {worst:.2e}")


def test_base_state_matches_autodiff():
    lay0 = sections.layout(8)
    mats0 = sections.prepare(lay0, near_fs(lay0, 3))
    worst = 0.0
    for z in BATCH['points'][:4]:
        got = correction.base_state(lay0, mats0, z)

        def pot(w):
            return sections.log_potential_at(lay0, mats0, w)
        worst = max(worst, abs(float(got[0] - pot(z))) / abs(float(pot(z))),
                    rel(got[1], wirtinger(pot, z)),
                    rel(got[2], complex_math.complex_hessian(pot, z)))
    assert worst < 1e-12, worst
    print(f"  the state of log p_0 against autodiff: {worst:.2e}")


def test_base_weight_at_fs_is_the_fs_weight():
    """At theta_FS, p_0^(k/k_0) = abs(z)^(2k) on X, so the two weights give one psi.

    The equality holds on X only, so the Hessians agree along X and not on C^4:
    compare R ddbar psi R^dag, as the metric uses it.
    """
    lay0 = sections.layout(4)
    mats0 = sections.prepare(lay0, sections.fubini_study_theta(lay0))
    lay_c = sections.layout(6, normalize=False)
    coeffs = jnp.asarray(np.random.default_rng(4).normal(size=lay_c.n_theta))
    cmats = sections.prepare_fp(lay_c, coeffs)
    worst = 0.0
    for z, r in zip(BATCH['points'][:4], BATCH['restriction'][:4]):
        base = correction.psi_parts(lay_c, cmats, z, 1.5,
                                    *correction.base_state(lay0, mats0, z))
        fs = correction.psi_parts(lay_c, cmats, z, 6.0, *correction.fs_state(z))
        along = [(b[0], f[0]), (r @ b[1], r @ f[1]),
                 (r @ b[2] @ jnp.conj(r).T, r @ f[2] @ jnp.conj(r).T)]
        worst = max(worst, *(rel(b, f) for b, f in along))
    assert worst < 1e-13, worst
    print(f"  base weight at theta_FS against the FS weight, along X: {worst:.2e}")


def test_zero_correction_gives_the_base_volume():
    lay0 = sections.layout(6)
    theta0 = near_fs(lay0, 5)
    mats0 = sections.prepare(lay0, theta0)
    lay_c = sections.layout(8, normalize=False)
    cmats = sections.prepare_fp(lay_c, jnp.zeros(lay_c.n_theta))
    state = correction.states(lambda z: correction.base_state(lay0, mats0, z),
                              BATCH['points'], 25)
    worst = 0.0
    for fs_weight in (True, False):
        got = correction.volumes(lay_c, cmats, BATCH, state, 25, 8 / 6, fs_weight)
        want = jax.vmap(lambda z, r, o: analytic.volume_at(lay0, mats0, z, r, o))(
            BATCH['points'], BATCH['restriction'], BATCH['Omega_Omegabar'])
        worst = max(worst, rel(got, want))
    assert worst < 1e-13, worst
    print(f"  v at a correction of 0 against the base: {worst:.2e}")


if __name__ == "__main__":
    for fn in (test_psi_parts_match_autodiff, test_base_state_matches_autodiff,
               test_base_weight_at_fs_is_the_fs_weight,
               test_zero_correction_gives_the_base_volume):
        print(fn.__name__)
        fn()
    print("\nall correction tests pass")
