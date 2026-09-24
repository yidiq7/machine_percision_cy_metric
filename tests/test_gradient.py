"""`grad f` on X must be exact, so the closed forms and the tangents must agree.

`quartic/gradient.py` replaces the stored restriction matrix and
OmegaOmegabar with closed forms, because both change along X and the stored
values cannot be differentiated. These tests hold the closed forms against the
sampler, and the derivative against a central difference.

Run: uv run python -m tests.test_gradient
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, gradient, metric, sections   # noqa: E402

analytic.GRAM_DD = False
analytic.SECTION_DD = False

BATCH = data.make(200, 11)
J, M, FREE = gradient.patch_indices(BATCH)
rng = np.random.default_rng(5)


def perturbed(lay, seed):
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * 1e-2
    return sections.fubini_study_theta(lay) + jnp.asarray(noise)


def test_closed_forms_match_the_sampler():
    """OmegaOmegabar and the restriction must be the ones the sampler made."""
    z = np.asarray(BATCH['points'])
    grad = 4.0 * z ** 3
    omega = 1.0 / np.abs(grad[np.arange(len(M)), M]) ** 2
    err_om = np.abs(omega / np.asarray(BATCH['Omega_Omegabar']) - 1).max()

    worst = 0.0
    stored = np.asarray(BATCH['restriction'])
    for i in range(z.shape[0]):
        got = np.asarray(gradient._restriction(
            jnp.asarray(z[i]), int(M[i]), jnp.asarray(FREE[i])))
        worst = max(worst, np.abs(got - stored[i]).max())
    assert err_om < 1e-14 and worst < 1e-14, (err_om, worst)
    print(f"  OmegaOmegabar to {err_om:.2e}, the restriction to {worst:.2e}")


def test_volume_matches_the_stored_path():
    """`volume_at` must give the v that `metric.volumes` gives."""
    worst = 0.0
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = perturbed(lay, k)
        mats = sections.prepare(lay, theta)
        want = np.asarray(metric.volumes(lay, theta, BATCH))
        got = np.asarray(jax.vmap(
            lambda z_, m_, f_: gradient.volume_at(lay, mats, z_, m_, f_)
        )(BATCH['points'], jnp.asarray(M), jnp.asarray(FREE)))
        worst = max(worst, np.abs(got / want - 1).max())
    assert worst < 1e-13, worst
    print(f"  v from the closed forms matches to {worst:.2e}, k = 4, 8, 12")


def test_tangent_derivative_matches_a_central_difference():
    """A central difference must converge to the autodiff value as the step^2.

    The step runs along a row of the restriction matrix, which is tangent to X,
    so the derivative of the extension equals the derivative on X.
    """
    lay = sections.layout(6)
    theta = perturbed(lay, 6)
    mats = sections.prepare(lay, theta)
    z0 = BATCH['points'][0]
    m0, f0 = int(M[0]), jnp.asarray(FREE[0])

    _, dv, _ = jax.jit(
        lambda z_: gradient.volume_tangents(lay, mats, z_, m0, f0))(z0)
    rest = jax.jit(lambda z_: gradient._restriction(z_, m0, f0))(z0)
    vol = jax.jit(lambda z_: gradient.volume_at(lay, mats, z_, m0, f0))

    errs = []
    for step in (1e-3, 1e-4, 1e-5):
        tangent = rest[0]
        d1 = (vol(z0 + step * tangent) - vol(z0 - step * tangent)) / (2 * step)
        d2 = (vol(z0 + 1j * step * tangent)
              - vol(z0 - 1j * step * tangent)) / (2 * step)
        errs.append(float(jnp.abs(0.5 * (d1 - 1j * d2) - dv[0])
                          / jnp.abs(dv[0])))
    # each step of 10 must buy approximately 100 in the error
    assert errs[0] / errs[1] > 30 and errs[1] / errs[2] > 30, errs
    assert errs[2] < 1e-8, errs
    print(f"  central difference errors {errs[0]:.1e}, {errs[1]:.1e}, "
          f"{errs[2]:.1e}: second order, so the tangent derivative is exact")


def test_ratio_does_not_depend_on_the_point_set():
    """R is a ratio of two means, so it must settle as the sample grows."""
    lay = sections.layout(6)
    theta = perturbed(lay, 6)
    got = []
    for n_pairs, seed in ((500, 31), (2_000, 31), (2_000, 32)):
        batch = data.make(n_pairs, seed)
        got.append(gradient.gradient_report(lay, theta, batch, chunk=1000))
    spread = max(r['ratio'] for r in got) / min(r['ratio'] for r in got) - 1
    assert spread < 0.05, [r['ratio'] for r in got]
    print(f"  R = " + ", ".join(f"{r['ratio']:.4f}" for r in got)
          + f" over 2,000 and 8,000 points and two seeds, spread {spread:.2%}")


def test_the_ratio_is_free_of_the_volume_normalization():
    """Open question 2 does not touch R, because both norms carry the volume."""
    lay = sections.layout(6)
    theta = perturbed(lay, 6)
    a = gradient.gradient_report(lay, theta, BATCH, volume=2.0 * np.pi ** 2)
    b = gradient.gradient_report(lay, theta, BATCH, volume=1.0)
    assert abs(a['ratio'] / b['ratio'] - 1) < 1e-12, (a['ratio'], b['ratio'])
    assert abs(a['l2_f'] / b['l2_f'] - np.sqrt(2.0) * np.pi) < 1e-9
    print(f"  R = {a['ratio']:.6f} for both normalizations, and the norms scale")


def test_dd_is_refused():
    """`gradient_report` is the plain FP64 path, so it refuses the dd flags."""
    lay = sections.layout(4)
    theta = sections.fubini_study_theta(lay)
    try:
        analytic.GRAM_DD = True
        gradient.gradient_report(lay, theta, BATCH)
    except RuntimeError as err:
        assert 'plain FP64' in str(err)
        print("  gradient_report refuses the double-double path")
    else:
        raise AssertionError('gradient_report accepted GRAM_DD')
    finally:
        analytic.GRAM_DD = False


if __name__ == "__main__":
    for fn in (test_closed_forms_match_the_sampler,
               test_volume_matches_the_stored_path,
               test_tangent_derivative_matches_a_central_difference,
               test_ratio_does_not_depend_on_the_point_set,
               test_the_ratio_is_free_of_the_volume_normalization,
               test_dd_is_refused):
        print(fn.__name__)
        fn()
    print("\nall gradient tests pass")
