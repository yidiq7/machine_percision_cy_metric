"""The dd build and the dd evaluation of the joint V+A basis.

Run: uv run python -m tests.test_arnoldi_dd
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import analytic, arnoldi, arnoldi_dd, correction, data, dd, sections  # noqa: E402
import vpa_oracle as vo                                     # noqa: E402

analytic.GRAM_DD, analytic.SECTION_DD = False, False
sections.PREPARE_DD = False
SAMPLE = data.make(1500, 11)
TEST = data.make(300, 12)
K0, K1 = 4, 10
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
    z = SAMPLE['points']
    r2 = jnp.sum(jnp.abs(z) ** 2, axis=1)
    counts = {j: sections.layout(j, normalize=False).n_theta for j in range(1, K1 + 1)}
    basis, h_lo, trace = arnoldi_dd.build_joint_dd(
        K1, np.asarray(z), jax.vmap(weight)(z) * r2 ** K1,
        SAMPLE['mass'] / jnp.sum(SAMPLE['mass']), d_max=12, n_by_degree=counts)
    return lay0, mats0, weight, basis, h_lo, arnoldi_dd.block_inverses(basis, h_lo)


def test_values_match_mpmath():
    """The dd jets give q to dd accuracy, against the recurrence in mpmath."""
    _, _, _, basis, h_lo, uinv = built()
    mp.mp.dps = 50
    worst = 0.0
    for z in TEST['points'][:3]:
        rest = dd.cto_dd(jnp.zeros((2, 4), jnp.complex128).at[0, 1].set(1.0).at[1, 2].set(1.0))
        jets = arnoldi_dd.joint_jets_dd(
            basis, h_lo, arnoldi_dd.vpa_dd.basis_multiplier_jets(basis, dd.cto_dd(z), rest),
            uinv)
        hi_v, lo_v = np.asarray(jets[0][0]), np.asarray(jets[0][1])
        zm = [mp.mpc(complex(c).real, complex(c).imag) for c in np.asarray(z)]
        y_mp = vo.mp_multipliers(basis.entries, zm)
        y_mp = [(y - mp.mpf(a)) / mp.mpf(b) for y, a, b in zip(y_mp, basis.shift, basis.scale)]
        hh = [[mp.mpf(float(basis.h[i, j])) + mp.mpf(float(h_lo[i, j]))
               for j in range(basis.h.shape[0])] for i in range(basis.h.shape[0])]
        n = basis.h.shape[0]
        q = [mp.mpf(0)] * n
        q[0] = 1 / hh[0][0]
        for t in range(1, n):
            grown = y_mp[int(basis.mults[t])] * q[int(basis.parents[t])]
            acc = mp.fsum(hh[s][t] * q[s] for s in range(t))
            q[t] = (grown - acc) / hh[t][t]
        err = [abs(mp.mpf(float(a)) + mp.mpf(float(b)) - x) for a, b, x in zip(hi_v, lo_v, q)]
        worst = max(worst, float(max(err)))
    assert worst < 1e-20, worst
    print(f"  the dd values of all {basis.h.shape[0]} functions match mpmath to {worst:.1e}")


def test_loop_matches_unrolled():
    """The jets from one loop body equal the jets with one copy for each degree."""
    _, _, _, basis, h_lo, uinv = built()
    rest = dd.cto_dd(jnp.zeros((2, 4), jnp.complex128).at[0, 1].set(1.0).at[1, 2].set(1.0))
    worst = 0.0
    for z in TEST['points'][:3]:
        y = arnoldi_dd.vpa_dd.basis_multiplier_jets(basis, dd.cto_dd(z), rest)
        a = jax.tree_util.tree_leaves(jax.jit(
            lambda yy: arnoldi_dd.joint_jets_dd(basis, h_lo, yy, uinv))(y))
        b = jax.tree_util.tree_leaves(jax.jit(
            lambda yy: arnoldi_dd.joint_jets_dd_unrolled(basis, h_lo, yy, uinv))(y))
        for x, w in zip(a[0::2], b[0::2]):
            scale = float(jnp.abs(w).max()) or 1.0
            worst = max(worst, float(jnp.abs(x - w).max()) / scale)
    assert worst < 1e-26, worst
    print(f"  the loop jets match the unrolled jets to {worst:.1e} of the largest entry")


def test_basis_is_orthonormal_on_its_sample():
    _, _, weight, basis, h_lo, _ = built()
    z = SAMPLE['points']
    vals = arnoldi.joint_values(basis, z, jax.vmap(weight)(z))
    mu = SAMPLE['mass'] / jnp.sum(SAMPLE['mass'])
    gap = float(jnp.abs(vals.T @ (mu[:, None] * vals) - jnp.eye(vals.shape[1])).max())
    assert gap < 1e-8, gap
    print(f"  the FP64 values of the dd build are orthonormal on the sample to {gap:.1e}")


def test_hessians_match_fp64():
    """R ddbar G R^dag in dd agrees with FP64 autodiff to the FP64 error of the latter."""
    lay0, mats0, weight, basis, h_lo, uinv = built()
    worst = 0.0
    for z, r in zip(TEST['points'][:3], TEST['restriction'][:3]):
        want = arnoldi.joint_hessians(basis, z, r, weight)
        state = correction.base_state(lay0, mats0, z)
        hi = (dd.cto_dd(state[2]), dd.cto_dd(state[1]))
        scale = weight(z) * jnp.sum(jnp.abs(z) ** 2) ** K1
        got = dd.cto_fp(arnoldi_dd.joint_hessians_dd(basis, h_lo, uinv, dd.cto_dd(z),
                                                     dd.cto_dd(r), hi, ALPHA, scale))
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-8, worst
    print(f"  the dd Hessians match FP64 autodiff to {worst:.1e}")


def test_weight_dd_matches_mpmath():
    """W = (|z|^2 p^(-1/k0))^k1 in dd, under jax.jit, against mpmath at the same dd inputs."""
    mp.mp.dps = 60
    rng = np.random.default_rng(3)
    p_hi = rng.uniform(1e9, 1e11, 6)
    p_lo = p_hi * rng.uniform(-1e-16, 1e-16, 6)
    r_hi = rng.uniform(0.5, 2.0, 6)
    r_lo = r_hi * rng.uniform(-1e-16, 1e-16, 6)
    worst = 0.0
    for k0, k1 in ((28, 36), (28, 48), (12, 20)):
        got = jax.jit(lambda a, b, c, d: arnoldi_dd.vpa_dd.weight_dd(
            (a, b), (c, d), k0, k1))(p_hi, p_lo, r_hi, r_lo)
        for i in range(6):
            p = mp.mpf(float(p_hi[i])) + mp.mpf(float(p_lo[i]))
            r = mp.mpf(float(r_hi[i])) + mp.mpf(float(r_lo[i]))
            want = (r * p ** (mp.mpf(-1) / k0)) ** k1
            have = mp.mpf(float(got[0][i])) + mp.mpf(float(got[1][i]))
            worst = max(worst, float(abs(have / want - 1)))
    assert worst < 1e-29, worst
    print(f"  the dd weight matches mpmath to {worst:.1e}")


if __name__ == "__main__":
    for fn in (test_weight_dd_matches_mpmath, test_loop_matches_unrolled,
               test_values_match_mpmath, test_basis_is_orthonormal_on_its_sample,
               test_hessians_match_fp64):
        print(fn.__name__)
        fn()
    print("\nall arnoldi_dd tests pass")
