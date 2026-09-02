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


def perturbed_theta(lay, seed, size=1e-3):
    """A point near Fubini-Study, so the test does not sit on a symmetry.

    The step is a fraction of the largest component of theta_FS, which is 1 in
    the convention of `sections.layout`. An absolute step of 0.1 was a small
    perturbation when every component of theta_FS was 1, and it is a large one
    now that the scale is global and the components run down to 1e-18. That put
    the test at a badly conditioned h, where the two Jacobians differ by 3e-10
    for reasons that have nothing to do with either of them.
    """
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * size
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


def test_custom_jvp_matches_autodiff():
    """`lineax` materializes with vmap, so the rule must survive jacfwd."""
    from quartic import analytic as an
    worst = 0.0
    for k in (4, 8):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        got = jax.jacfwd(lambda t: an.weighted_residual_fast(lay, t, BATCH, 25))(theta)
        want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 25))(theta)
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-12, worst
    print(f"  jacfwd through the custom rule differs by {worst:.2e}")


def test_section_dd_matches_the_fp64_path():
    """Stages 1 to 3 in double-double must not change the model.

    `section_vectors_dd` is `section_vectors_closed` with every operation in
    extended precision, so the two must agree to the error of the FP64 one.
    """
    from quartic import dd as qdd
    worst_s = worst_d = worst_phi = 0.0
    for k in (4, 8, 12):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        mats = sections.prepare(lay, theta)
        z = BATCH['points'][0]

        s_fp, d_fp = analytic.section_vectors_closed(lay, z)
        s_dd, d_dd = jax.jit(lambda w: analytic.section_vectors_dd(lay, w))(z)
        for a, b in zip(s_dd, s_fp):
            worst_s = max(worst_s, float(jnp.abs(qdd.cto_fp(a) - b).max()
                                         / jnp.abs(b).max()))
        for a, b in zip(d_dd, d_fp):
            worst_d = max(worst_d, float(jnp.abs(qdd.cto_fp(a) - b).max()
                                         / jnp.abs(b).max()))

        if k > 4:
            continue
        # phi at k = 4 only. The Gram stage in dd compiles slowly, and what the
        # comparison tests does not depend on k.
        try:
            analytic.SECTION_DD = False
            plain, _ = jax.jit(lambda w: analytic.phi_analytic(lay, mats, w))(z)
            analytic.SECTION_DD = True
            extra, _ = jax.jit(lambda w: analytic.phi_analytic(lay, mats, w))(z)
        finally:
            analytic.SECTION_DD = False
        worst_phi = max(worst_phi, float(jnp.abs(plain - extra).max()
                                         / jnp.abs(plain).max()))
    assert worst_s < 1e-14 and worst_d < 1e-14, (worst_s, worst_d)
    assert worst_phi < 1e-13, worst_phi
    print(f"  sig {worst_s:.2e}, dsig {worst_d:.2e}, phi {worst_phi:.2e}")


def test_section_dd_reaches_the_jacobian_path():
    """`SECTION_DD` must reach every caller of the Gram stage, not only phi.

    The first version wired it into `phi_analytic` alone. `volume_and_weights`,
    which the Jacobian goes through, still called `section_vectors` itself, so
    `_gram_parts` took the double-double branch on FP64 arrays and the fit died
    inside `plan_fit` with `Too many indices`. A test of phi cannot see that.
    """
    batch = data.make(20, 7)
    z = batch['points'][0]
    rest = batch['restriction'][0]
    omega = batch['Omega_Omegabar'][0]
    # k = 4 only. The fault is a shape mismatch and does not depend on k, and
    # the Gram stage in dd compiles slowly.
    worst_v = worst_dv = 0.0
    for k in (4,):
        lay = sections.layout(k)
        theta = perturbed_theta(lay, k)
        got = {}
        try:
            for tag, sec in (('plain', False), ('dd', True)):
                analytic.GRAM_DD, analytic.SECTION_DD = True, sec
                got[tag] = jax.jit(
                    lambda t: analytic.volume_jacobian(lay, t, z, rest, omega))(theta)
        finally:
            analytic.GRAM_DD, analytic.SECTION_DD = True, False
        v_a, dv_a = got['plain']
        v_b, dv_b = got['dd']
        worst_v = max(worst_v, float(jnp.abs(v_a - v_b) / jnp.abs(v_a)))
        worst_dv = max(worst_dv, float(jnp.abs(dv_a - dv_b).max()
                                       / jnp.abs(dv_a).max()))
    assert worst_v < 1e-13 and worst_dv < 1e-13, (worst_v, worst_dv)
    print(f"  v {worst_v:.2e}, dv/dtheta {worst_dv:.2e}")


def test_fit_matches_between_the_two_paths():
    """The same optimistix solver must land on the same sigma either way."""
    from quartic import data as qdata, fit as qfit
    batch = qdata.make(250, 5)
    for k in (4, 6):
        lay = sections.layout(k)
        th_b, _ = qfit.fit(lay, batch, chunk=1000, analytic_jac=False)
        th_a, _ = qfit.fit(lay, batch, chunk=1000, analytic_jac=True)
        sig_b = metric.report(lay, th_b, batch)['mape']
        sig_a = metric.report(lay, th_a, batch)['mape']
        rel = abs(sig_a / sig_b - 1)
        # sigma and not theta. cond(J) is large, so the valley is flat and the
        # two paths stop at different points inside it, which is the behaviour
        # measured on the test problem: a large condition number costs theta and
        # not the residual. Eight digits of sigma is the agreement to ask for.
        assert rel < 1e-6, (k, sig_a, sig_b)
        print(f"  k={k}: sigma {sig_b:.10e} against {sig_a:.10e}, rel {rel:.2e}")


def test_jacobian_matches_jacfwd_when_whitened():
    """`volume_jacobian` must carry the whitening, and the plain basis hides it.

    theta parametrizes H_white and H = W H_white W, so dH/dtheta_a is W B_a W
    and not B_a. With W the identity the two agree, so a test in the plain basis
    alone passes while the whitened Jacobian is wrong by a factor of 840.
    """
    from quartic import whiten as qwhiten
    worst = 0.0
    for k in (4, 6, 8):
        lay = qwhiten.whitened_layout(k, n_pairs=1500, seed=4242)
        theta = sections.fubini_study_theta(lay)
        _, dv = analytic.residual_jacobian(lay, theta, BATCH)
        want = jax.jacfwd(lambda t: metric.volumes(lay, t, BATCH, chunk=50))(theta)
        worst = max(worst, float(jnp.abs(dv - want).max() / jnp.abs(want).max()))
    assert worst < 1e-12, worst
    print(f"  max relative difference of the whitened Jacobian = {worst:.2e}")


def test_run_to_limit_never_loses_to_the_default_stop():
    """The convergence test can only stop the fit early, never improve it.

    `rtol` and `atol` enter the termination test of optimistix and nothing else,
    so both runs follow one trajectory. `best_so_far` over every step therefore
    holds an iterate at least as good as the one where the test fires. At k = 28
    the test cost a factor of 6.6 in sigma between two exact reparametrizations.
    """
    from quartic import data as qdata, fit as qfit
    batch = qdata.make(250, 5)
    for k in (4, 6):
        lay = sections.layout(k)
        th_d, st_d = qfit.fit(lay, batch, chunk=1000, max_steps=100)
        th_l, st_l = qfit.fit(lay, batch, chunk=1000, max_steps=100,
                              rtol=0.0, atol=0.0, best_so_far=True)
        loss = lambda t: float(jnp.sum(
            metric.weighted_residual(lay, t, batch) ** 2))
        l_d, l_l = loss(th_d), loss(th_l)
        assert l_l <= l_d * (1 + 1e-9), (k, l_d, l_l)
        print(f"  k={k}: default stops at {st_d['steps']} steps, loss {l_d:.4e}; "
              f"the whole budget gives {l_l:.4e}")


if __name__ == "__main__":
    for fn in (test_phi_matches_complex_hessian, test_residual_matches,
               test_jacobian_matches_jacfwd,
               test_chunking_does_not_change_the_result,
               test_custom_jvp_matches_autodiff,
               test_section_dd_matches_the_fp64_path,
               test_section_dd_reaches_the_jacobian_path,
               test_fit_matches_between_the_two_paths,
               test_jacobian_matches_jacfwd_when_whitened,
               test_run_to_limit_never_loses_to_the_default_stop):
        print(fn.__name__)
        fn()
    print("\nall analytic tests pass")
