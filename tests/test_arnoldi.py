"""The V+A basis of a correction level must span the product basis, and be orthonormal.

Run: uv run python -m tests.test_arnoldi
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import analytic, arnoldi, correction, data, sections   # noqa: E402

analytic.GRAM_DD, analytic.SECTION_DD = False, False
sections.PREPARE_DD = False
SAMPLE = data.make(600, 11)
TEST = data.make(300, 12)
K0, K1 = 4, 8
ALPHA = K1 / K0


def base():
    lay0 = sections.layout(K0)
    noise = np.random.default_rng(1).normal(size=lay0.n_theta) * 1e-2
    mats0 = sections.prepare(lay0, sections.fubini_study_theta(lay0) + jnp.asarray(noise))

    def weight(z):
        return jnp.exp(-ALPHA * sections.log_potential_at(lay0, mats0, z))
    return lay0, mats0, weight


def built():
    lay0, mats0, weight = base()
    lay_c = sections.layout(K1, normalize=False)
    z = SAMPLE['points']
    w = jax.vmap(weight)(z)
    basis, report = arnoldi.build_basis(lay_c.structure, np.asarray(z), np.asarray(w),
                                        np.asarray(SAMPLE['mass']), ALPHA)
    return lay0, mats0, weight, lay_c, basis, report


def test_orbits_have_the_right_dimensions():
    *_, report = built()
    for row in report['orbits']:
        assert row['dim_q'] == row['dim_q_expected'], row
        assert row['s_first_dropped'] < 1e-10 * row['s_last_kept'], row
    worst = max(row['s_first_dropped'] / row['s_last_kept'] for row in report['orbits'])
    print(f"  every orbit has its full q space, and the invariant rank gap is at "
          f"least {1 / max(worst, 1e-300):.1e}; cond across orbits "
          f"{report['cond_across_orbits']:.2e}")


def test_basis_is_orthonormal_on_its_sample():
    """The QR over the orbits amplifies the rounding by its condition number."""
    _, _, weight, _, basis, report = built()
    z = SAMPLE['points']
    vals = arnoldi.values(basis, z, jax.vmap(weight)(z))
    mu = SAMPLE['mass'] / jnp.sum(SAMPLE['mass'])
    gram = vals.T @ (mu[:, None] * vals)
    gap = float(jnp.abs(gram - jnp.eye(gram.shape[0])).max())
    bound = 1e-14 * max(1.0, report['cond_across_orbits'])
    assert gap < bound, (gap, bound)
    print(f"  the recurrence reproduces an orthonormal basis on the sample: {gap:.1e}, "
          f"against a bound of {bound:.1e}")


def test_basis_spans_the_product_basis():
    """Each function w sigma^* B_a sigma lies in the span of the V+A basis."""
    _, _, weight, lay_c, basis, _ = built()
    z = TEST['points']
    w = jax.vmap(weight)(z)
    vals = np.asarray(arnoldi.values(basis, z, w))
    worst = 0.0
    for a in range(lay_c.n_theta):
        cmats = sections.prepare_fp(lay_c, jnp.zeros(lay_c.n_theta).at[a].set(1.0))
        col = np.asarray(w * jax.vmap(lambda p: sections.potential_at(lay_c, cmats, p))(z))
        coef, *_ = np.linalg.lstsq(vals, col, rcond=None)
        worst = max(worst, np.linalg.norm(vals @ coef - col) / np.linalg.norm(col))
    assert worst < 1e-10, worst
    print(f"  every product basis function lies in the span, to {worst:.1e}")


def test_restricted_hessian_matches_psi_parts():
    lay0, mats0, weight, lay_c, _, _ = built()
    coeffs = jnp.asarray(np.random.default_rng(3).normal(size=lay_c.n_theta))
    cmats = sections.prepare_fp(lay_c, coeffs)
    worst = 0.0
    for z, r in zip(TEST['points'][:4], TEST['restriction'][:4]):
        def psi(p):
            return (weight(p) * sections.potential_at(lay_c, cmats, p))[None]
        got = arnoldi.restricted_hessian_of(psi, z, r)[0]
        state = correction.base_state(lay0, mats0, z)
        want = r @ correction.psi_parts(lay_c, cmats, z, ALPHA, *state)[2] @ jnp.conj(r).T
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-10, worst
    print(f"  the restricted Hessian matches psi_parts to {worst:.1e}")


def joint(d_max=8):
    lay0, mats0, weight = base()
    lay_c = sections.layout(K1, normalize=False)
    z = SAMPLE['points']
    r2 = jnp.sum(jnp.abs(z) ** 2, axis=1)
    scale = jax.vmap(weight)(z) * r2 ** K1
    lays = {j: sections.layout(j, normalize=False) for j in range(1, K1 + 1)}
    basis, trace = arnoldi.build_joint(
        K1, np.asarray(z), np.asarray(scale),
        np.asarray(SAMPLE['mass'] / jnp.sum(SAMPLE['mass'])), d_max=d_max,
        n_by_degree={j: lay.n_theta for j, lay in lays.items()}, layouts=lays)
    return weight, lay_c, basis, trace


def test_joint_reaches_the_dimension_and_is_orthonormal():
    weight, lay_c, basis, trace = joint()
    assert basis.h.shape[0] == lay_c.n_theta, (basis.h.shape, lay_c.n_theta)
    for row in trace:
        assert row['dim'] == row['want'], row
    z = SAMPLE['points']
    vals = arnoldi.joint_values(basis, z, jax.vmap(weight)(z))
    mu = SAMPLE['mass'] / jnp.sum(SAMPLE['mass'])
    gram = vals.T @ (mu[:, None] * vals)
    gap = float(jnp.abs(gram - jnp.eye(gram.shape[0])).max())
    assert gap < 1e-12, gap
    print(f"  the joint recurrence reaches n = {lay_c.n_theta} with "
          f"{len(basis.entries)} multipliers, highest degree {int(basis.degrees.max())}, "
          f"orthonormal on the sample to {gap:.1e}")


def test_joint_with_seeds_matches_the_build():
    """With d_max = 2 the products fall short, and the seeds fill the gap."""
    weight, lay_c, basis, trace = joint(d_max=2)
    assert basis.h.shape[0] == lay_c.n_theta, (basis.h.shape, lay_c.n_theta)
    assert sum(row['seeded'] for row in trace) > 0, trace
    z = SAMPLE['points']
    vals = arnoldi.joint_values(basis, z, jax.vmap(weight)(z))
    mu = SAMPLE['mass'] / jnp.sum(SAMPLE['mass'])
    gram = vals.T @ (mu[:, None] * vals)
    gap = float(jnp.abs(gram - jnp.eye(gram.shape[0])).max())
    assert gap < 1e-10, gap
    print(f"  with d_max = 2: {sum(row['seeded'] for row in trace)} seeds, worst seed "
          f"ratio {min(row['worst_seed_ratio'] for row in trace):.1e}, orthonormal on "
          f"the sample to {gap:.1e}")


def test_joint_spans_the_product_basis():
    weight, lay_c, basis, _ = joint()
    z = TEST['points']
    w = jax.vmap(weight)(z)
    vals = np.asarray(arnoldi.joint_values(basis, z, w))
    worst = 0.0
    for a in range(lay_c.n_theta):
        cmats = sections.prepare_fp(lay_c, jnp.zeros(lay_c.n_theta).at[a].set(1.0))
        col = np.asarray(w * jax.vmap(lambda p: sections.potential_at(lay_c, cmats, p))(z))
        coef, *_ = np.linalg.lstsq(vals, col, rcond=None)
        worst = max(worst, np.linalg.norm(vals @ coef - col) / np.linalg.norm(col))
    assert worst < 1e-10, worst
    print(f"  every product basis function lies in the joint span, to {worst:.1e}")


if __name__ == "__main__":
    for fn in (test_orbits_have_the_right_dimensions,
               test_basis_is_orthonormal_on_its_sample,
               test_basis_spans_the_product_basis,
               test_restricted_hessian_matches_psi_parts,
               test_joint_reaches_the_dimension_and_is_orthonormal,
               test_joint_with_seeds_matches_the_build,
               test_joint_spans_the_product_basis):
        print(fn.__name__)
        fn()
    print("\nall arnoldi tests pass")
