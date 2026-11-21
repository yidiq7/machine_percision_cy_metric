"""The end-to-end double-double path must reach double-double precision.

Two facts decide whether stage 2 can reach sigma = 1e-25:

  * one dd Newton step must put `abs(P)/scale` near 1e-32, because an FP64 point
    holds only 1e-16 and that error enters v with an amplification up to 6.6,
  * `v` must agree with mpmath to approximately 1e-31, which needs the exact
    `dd.mul` and not `dd.mul_fast`.

**The reference must sit at the same point.** `phi_exact` rounds an FP64 point
argument, so pass the mpmath values of the dd point. A reference 1e-17 away is
not a reference: that error enters p with the degree k as the amplification, and
it read 3.6e-16 instead of 1.1e-24.

Run: uv run python -m tests.test_dd_metric
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
from mpmath import mp, mpc, mpf                             # noqa: E402
mp.dps = 60

from quartic import analytic, data, dd, dd_metric, metric, sections   # noqa: E402
from quartic.mpref import phi_exact  # noqa: E402

BATCH = data.make(20, 11)


def _point(i):
    """One point, refined in dd, with its indices and its mpmath value."""
    z = BATCH['points'][i]
    m = dd_metric.newton_index(z)
    patch = int(np.argmin(np.abs(np.asarray(z) - 1.0)))
    free = tuple(sorted(set(range(4)) - {patch, m}))
    z_dd, _ = jax.jit(lambda w: dd_metric.refine(w, m, 1))(z)
    exact = [mpc(mpf(float(z_dd[0][0][a])) + mpf(float(z_dd[0][1][a])),
                 mpf(float(z_dd[1][0][a])) + mpf(float(z_dd[1][1][a])))
             for a in range(4)]
    return z_dd, m, free, exact


def test_one_newton_step_reaches_double_double():
    """FP64 holds abs(P)/scale at 1e-16. Newton is quadratic, so one step ends."""
    worst_before = worst_after = 0.0
    for i in range(4):
        z = BATCH['points'][i]
        scale = float(np.abs(np.asarray(z) ** 4).sum())
        worst_before = max(worst_before,
                           float(np.abs((np.asarray(z) ** 4).sum()) / scale))
        z_dd, _, _, _ = _point(i)
        worst_after = max(worst_after, dd_metric.residual_of_P(z_dd))
    assert worst_before > 1e-17, worst_before
    assert worst_after < 1e-30, worst_after
    print(f"  abs(P)/scale: {worst_before:.2e} in FP64, {worst_after:.2e} "
          f"after one dd step")


def test_volume_agrees_with_the_fp64_path():
    """The dd path must give the same v that the FP64 path gives, to FP64."""
    worst = 0.0
    for k in (4, 8):
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        mats = sections.prepare(lay, theta)
        want = np.asarray(metric.volumes(lay, theta, BATCH))
        for i in range(3):
            z_dd, m, free, _ = _point(i)
            got = dd.to_fp(jax.jit(
                lambda zz: dd_metric.volume_dd(lay, mats, zz, m, free))(z_dd))
            worst = max(worst, abs(float(got) / want[i] - 1.0))
    assert worst < 1e-12, worst
    print(f"  v agrees with the FP64 path to {worst:.2e}, k = 4 and 8")


def test_exact_mul_is_needed_for_stage_2():
    """`mul_fast` gives 2^-79, which is 30 times above the 1e-25 target."""
    lay = sections.layout(6)
    theta = sections.fubini_study_theta(lay)
    mats = sections.prepare(lay, theta)
    mats_np = [np.asarray(mm) for mm in mats]
    z_dd, m, free, exact = _point(0)

    got = {}
    try:
        for tag, flag in (('fast', False), ('exact', True)):
            analytic.EXACT_MUL = flag
            phi, _ = jax.jit(lambda w: dd_metric.phi_dd(lay, mats, w))(z_dd)
            ref, _ = phi_exact(lay, mats_np, exact, keep=True)
            scale = max(abs(ref[i][j]) for i in range(4) for j in range(4))
            worst = 0.0
            for i in range(4):
                for j in range(4):
                    val = mpc(mpf(float(phi[0][0][i, j])) + mpf(float(phi[0][1][i, j])),
                              mpf(float(phi[1][0][i, j])) + mpf(float(phi[1][1][i, j])))
                    worst = max(worst, float(abs(val - ref[i][j]) / scale))
            got[tag] = worst
    finally:
        analytic.EXACT_MUL = False

    assert 1e-25 < got['fast'] < 1e-22, got
    assert got['exact'] < 1e-30, got
    print(f"  phi: {got['fast']:.2e} with mul_fast, {got['exact']:.2e} with "
          f"the exact mul")


def test_dd_phi_holds_dd_where_exponents_reach_3():
    """At k = 12 the exponents C_i reach 3, so frame * C_i needs more than 53 bits."""
    lay = sections.layout(12)
    theta = sections.fubini_study_theta(lay)
    mats = sections.prepare(lay, theta)
    mats_np = [np.asarray(mm) for mm in mats]
    worst = 0.0
    try:
        analytic.EXACT_MUL = True
        run = jax.jit(lambda w: dd_metric.phi_dd(lay, mats, w))
        for n in range(2):
            z_dd, _, _, exact = _point(n)
            phi, _ = run(z_dd)
            ref, _ = phi_exact(lay, mats_np, exact, keep=True)
            scale = max(abs(ref[i][j]) for i in range(4) for j in range(4))
            for i in range(4):
                for j in range(4):
                    val = mpc(mpf(float(phi[0][0][i, j])) + mpf(float(phi[0][1][i, j])),
                              mpf(float(phi[1][0][i, j])) + mpf(float(phi[1][1][i, j])))
                    worst = max(worst, float(abs(val - ref[i][j]) / scale))
    finally:
        analytic.EXACT_MUL = False
    assert worst < 1e-29, worst
    print(f"  phi at k = 12 against mpmath: {worst:.2e} of max abs(phi)")


def test_dd_phi_is_hermitian():
    """p = Re(sigma^* H sigma) sees the symmetric part of H only, so phi of the dd H
    must be Hermitian: the FP64 basis gives H an antisymmetric part near 1e-16."""
    lay = sections.layout(12)
    theta = sections.fubini_study_theta(lay)
    theta = theta * (1.0 + 0.3 * np.random.default_rng(3).standard_normal(theta.shape))
    mats = jax.jit(lambda t: sections.prepare_parts(lay, t))(theta)
    for hi, lo in mats:
        assert np.array_equal(np.asarray(hi), np.swapaxes(np.asarray(hi), 1, 2))
        assert np.array_equal(np.asarray(lo), np.swapaxes(np.asarray(lo), 1, 2))
    worst = 0.0
    try:
        analytic.EXACT_MUL = True
        run = jax.jit(lambda w: dd_metric.phi_dd(lay, mats, w))
        for n in range(2):
            z_dd, _, _, _ = _point(n)
            phi, _ = run(z_dd)
            val = [[mpc(mpf(float(phi[0][0][i, j])) + mpf(float(phi[0][1][i, j])),
                        mpf(float(phi[1][0][i, j])) + mpf(float(phi[1][1][i, j])))
                    for j in range(4)] for i in range(4)]
            scale = max(abs(val[i][j]) for i in range(4) for j in range(4))
            worst = max(worst, max(float(abs(val[i][j] - val[j][i].conjugate()) / scale)
                                   for i in range(4) for j in range(4)))
    finally:
        analytic.EXACT_MUL = False
    assert worst < 1e-30, worst
    print(f"  phi of the dd H at k = 12 is Hermitian to {worst:.2e}")


if __name__ == "__main__":
    for fn in (test_one_newton_step_reaches_double_double,
               test_volume_agrees_with_the_fp64_path,
               test_exact_mul_is_needed_for_stage_2,
               test_dd_phi_holds_dd_where_exponents_reach_3,
               test_dd_phi_is_hermitian):
        print(fn.__name__)
        fn()
    print("\nall end-to-end dd tests pass")
