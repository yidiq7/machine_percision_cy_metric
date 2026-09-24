"""Tests of the double-double form of `sections.prepare`, and of the report path.

Run on a GPU node: uv run --extra gpu python -m tests.test_prepare
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
from quartic import analytic, data, metric, sections, whiten   # noqa: E402

BATCH = data.make(60, 3)


def perturbed_theta(lay, seed, size=1e-3):
    """A point near Fubini-Study, so that no test sits on a symmetry."""
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * size
    return sections.fubini_study_theta(lay) + jnp.asarray(noise)


def rounded(lay, theta):
    return jax.jit(lambda t: sections._prepare_rounded(lay, t))(theta)


def test_dd_agrees_where_fp64_is_accurate():
    """At small k, H spans few orders of magnitude, so FP64 is nearly exact."""
    worst = 0.0
    for k in (4, 6, 8):
        lay = sections.layout(k)
        for theta in (sections.fubini_study_theta(lay), perturbed_theta(lay, k)):
            for a, b in zip(sections.prepare_fp(lay, theta), rounded(lay, theta)):
                worst = max(worst, float(jnp.abs(a - b).max() / jnp.abs(b).max()))
    assert worst < 2e-15, worst
    print(f"  dd and FP64 forms of H agree to {worst:.2e} of the largest entry")


def test_each_entry_is_rounded_once():
    """Against mpmath at 60 digits, each entry is within one ulp of exact.

    dd holds a sum to approximately 1e-31 of the size of its terms. So an entry
    that is nearly zero by symmetry, near 1e-16 of the block, is good to a few
    ulp of itself only, and that has no effect on r. Those entries must meet an
    absolute bound instead: 1e-29 of the largest entry of the block.
    """
    mp.dps = 60
    lay = sections.layout(8)
    theta = np.asarray(perturbed_theta(lay, 8))
    raw = [mpf(float(t)) * mpf(float(s)) for t, s in zip(theta, lay.scale)]
    got = rounded(lay, jnp.asarray(theta))
    worst = worst_small = 0.0
    for g, mats in zip(lay.groups, got):
        mats = np.asarray(mats)
        d = g.frame.shape[1]
        reps = []
        for i, sl in enumerate(g.slices):
            rep = [[mpf(0)] * d for _ in range(d)]
            for a, idx in enumerate(range(sl.start, sl.stop)):
                for p in range(d):
                    for q in range(d):
                        rep[p][q] += raw[idx] * mpf(float(g.basis[i, a, p, q]))
            reps.append(rep)
        for b in range(g.rot.shape[0]):
            rot, rep = g.rot[b], reps[int(g.orbit_of[b])]
            top = float(np.abs(mats[b]).max())
            for p in range(d):
                for q in range(d):
                    exact = sum(mpf(float(rot[p, u])) * rep[u][v] * mpf(float(rot[q, v]))
                                for u in range(d) for v in range(d))
                    ref = float(exact)
                    err = abs(mats[b, p, q] - ref)
                    if abs(ref) > 1e-12 * top:
                        worst = max(worst, err / np.spacing(abs(ref)))
                    else:
                        worst_small = max(worst_small, err / top)
    assert worst <= 1.0, worst
    assert worst_small < 1e-29, worst_small
    print(f"  every entry above 1e-12 of its block is within {worst:.2f} ulp of "
          f"exact, and every smaller one within {worst_small:.1e} of the block, k = 8")


def test_tangent_is_the_linear_map():
    """The rule of `_prepare_rounded` must give prepare_fp of the tangent."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    tangent = jnp.asarray(np.random.default_rng(1).normal(size=lay.n_theta))
    _, got = jax.jvp(lambda t: sections._prepare_rounded(lay, t), (theta,), (tangent,))
    want = sections.prepare_fp(lay, tangent)
    worst = max(float(jnp.abs(a - b).max()) for a, b in zip(got, want))
    assert worst == 0.0, worst

    # And a Jacobian through the autodiff path, with the flag on and off.
    jac = lambda: jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH))(theta)
    old = sections.PREPARE_DD
    try:
        sections.PREPARE_DD = True
        j_dd = jac()
        sections.PREPARE_DD = False
        j_fp = jac()
    finally:
        sections.PREPARE_DD = old
    rel = float(jnp.abs(j_dd - j_fp).max() / jnp.abs(j_fp).max())
    assert rel < 1e-12, rel
    print(f"  the tangent is prepare_fp exactly, and jacfwd moves by {rel:.2e}")


def test_whitened_layout():
    """The dd form covers W H_white W as well."""
    lay = whiten.whitened_layout(6, n_pairs=1500, seed=4242)
    theta = perturbed_theta(lay, 6)
    worst = max(float(jnp.abs(a - b).max() / jnp.abs(b).max())
                for a, b in zip(sections.prepare_fp(lay, theta), rounded(lay, theta)))
    assert worst < 2e-15, worst
    print(f"  whitened: dd and FP64 forms agree to {worst:.2e}")


def test_report_uses_the_fit_path():
    """`report` must give sigma from `analytic`, and its check from complex_hessian."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    rep = metric.report(lay, theta, BATCH, chunk=100)
    f = np.asarray(analytic.residual_analytic(lay, theta, BATCH))
    w = np.asarray(BATCH['mass']) / np.sum(np.asarray(BATCH['mass']))
    want = float(np.sum(w * np.abs(f)))
    assert abs(rep['mape'] / want - 1) < 1e-14, (rep['mape'], want)
    assert abs(rep['mape_check'] / rep['mape'] - 1) < 1e-10, rep
    print(f"  report sigma {rep['mape']:.12e}, check {rep['mape_check']:.12e}")


def test_parts_split_the_dd_value():
    """hi must be the rounded H exactly, and lo at most half an ulp of hi."""
    lay = sections.layout(8)
    theta = perturbed_theta(lay, 8)
    parts = jax.jit(lambda t: sections.prepare_parts(lay, t))(theta)
    worst = 0.0
    for (hi, lo), ref in zip(parts, rounded(lay, theta)):
        hi, lo = np.asarray(hi), np.asarray(lo)
        assert np.array_equal(hi, np.asarray(ref))
        big = hi != 0
        worst = max(worst, float((np.abs(lo[big]) / np.spacing(np.abs(hi[big]))).max()))
    assert worst <= 0.5, worst
    print(f"  hi is the rounded H, and abs(lo) is at most {worst:.2f} ulp of hi")


def test_low_part_reaches_the_gram_stage():
    """A pair (hi, lo) must act as the single matrix hi + lo.

    lo is made approximately 1e-3 of hi here, so that a Gram stage which drops it
    fails. The factor differs between blocks: a uniform factor only scales H,
    and phi = ddbar log p does not change under a scale of H.
    """
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    hi = sections.prepare_fp(lay, theta)
    lo = tuple(m * (1e-3 * (1.0 + jnp.arange(m.shape[0])) / m.shape[0])[:, None, None]
               for m in hi)
    old = analytic.GRAM_DD
    try:
        analytic.GRAM_DD = True
        z = BATCH['points'][0]
        pair, _ = analytic.phi_analytic(lay, tuple(zip(hi, lo)), z)
        single, _ = analytic.phi_analytic(lay, tuple(a + b for a, b in zip(hi, lo)), z)
        plain, _ = analytic.phi_analytic(lay, hi, z)
    finally:
        analytic.GRAM_DD = old
    rel = float(jnp.abs(pair - single).max() / jnp.abs(single).max())
    moved = float(jnp.abs(plain - single).max() / jnp.abs(single).max())
    assert rel < 1e-13, rel
    assert moved > 1e-6, moved
    print(f"  the pair matches hi + lo to {rel:.2e}, and hi alone differs by {moved:.1e}")


def test_dd_reference_takes_the_pair():
    """`dd_metric.phi_dd` must also act on a pair (hi, lo) as on hi + lo."""
    from quartic import dd, dd_metric
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    hi = sections.prepare_fp(lay, theta)
    lo = tuple(m * (1e-3 * (1.0 + jnp.arange(m.shape[0])) / m.shape[0])[:, None, None]
               for m in hi)
    z = dd.cto_dd(BATCH['points'][0])
    run = jax.jit(lambda m: dd.cto_fp(dd_metric.phi_dd(lay, m, z)[0]))
    pair = run(tuple(zip(hi, lo)))
    single = run(tuple(a + b for a, b in zip(hi, lo)))
    rel = float(jnp.abs(pair - single).max() / jnp.abs(single).max())
    assert rel < 1e-13, rel
    print(f"  the dd reference acts on the pair as on hi + lo, to {rel:.2e}")


def test_jacobian_with_the_low_part():
    """The custom rule must still match autodiff with H_LOW on."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    old = (analytic.GRAM_DD, analytic.H_LOW, sections.PREPARE_DD)
    try:
        analytic.GRAM_DD, analytic.H_LOW, sections.PREPARE_DD = True, True, True
        got = jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, BATCH, 60))(theta)
        want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 60))(theta)
    finally:
        analytic.GRAM_DD, analytic.H_LOW, sections.PREPARE_DD = old
    rel = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    assert rel < 1e-12, rel
    print(f"  with H_LOW the custom rule matches autodiff to {rel:.2e}")


def _with_flags(fn, **flags):
    """Run fn with module flags set, and put them back after it."""
    owners = {'GRAM_DD': analytic, 'SECTION_DD': analytic, 'H_LOW': analytic,
              'PHI_DD': analytic, 'END_DD': analytic, 'PREPARE_DD': sections}
    old = {name: getattr(owners[name], name) for name in flags}
    try:
        for name, value in flags.items():
            setattr(owners[name], name, value)
        return fn()
    finally:
        for name, value in old.items():
            setattr(owners[name], name, value)


def test_phi_dd_agrees_where_fp64_is_accurate():
    """At small k, stage 5 in dd and in FP64 must give the same phi."""
    worst = 0.0
    for k in (6, 8):
        lay = sections.layout(k)
        mats = sections.prepare_fp(lay, perturbed_theta(lay, k))
        for z in BATCH['points'][:4]:
            run = lambda: analytic.phi_analytic(lay, mats, z)[0]
            a = _with_flags(run, GRAM_DD=True, PHI_DD=True)
            b = _with_flags(run, GRAM_DD=True, PHI_DD=False)
            worst = max(worst, float(jnp.abs(a - b).max() / jnp.abs(b).max()))
    assert worst < 1e-14, worst
    print(f"  stage 5 in dd and in FP64 agree to {worst:.2e}, k = 6 and 8")


def test_phi_dd_matches_the_dd_reference():
    """With the same dd inputs, stage 5 in dd must match `dd_metric.phi_dd`."""
    from quartic import dd, dd_metric
    lay = sections.layout(8)
    parts = sections.prepare_parts(lay, perturbed_theta(lay, 8))
    worst = 0.0
    for z in BATCH['points'][:3]:
        got = _with_flags(lambda: analytic.phi_analytic(lay, parts, z)[0],
                          GRAM_DD=True, SECTION_DD=True, PHI_DD=True)
        want = dd.cto_fp(dd_metric.phi_dd(lay, parts, dd.cto_dd(z))[0])
        worst = max(worst, float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert worst < 1e-15, worst
    print(f"  stage 5 in dd matches the dd reference to {worst:.2e}")


def test_jacobian_with_phi_dd():
    """The custom rule must still match autodiff with every dd flag on."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    got = _with_flags(
        lambda: jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, BATCH, 60))(theta),
        GRAM_DD=True, H_LOW=True, PHI_DD=True, PREPARE_DD=True)
    want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 60))(theta)
    rel = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    assert rel < 1e-12, rel
    print(f"  with stage 5 in dd the custom rule matches autodiff to {rel:.2e}")


_END_BATCH = {}


def _end_batch():
    """BATCH with the dd points of `dd_metric.point_extras`, made once."""
    from quartic import dd_metric
    if not _END_BATCH:
        _END_BATCH.update(BATCH)
        _END_BATCH.update(dd_metric.point_extras(BATCH['points']))
    return _END_BATCH


def test_refined_points_lie_on_X():
    """One dd Newton step must put each point on X, and move z_m only."""
    from quartic import dd_metric
    batch = _end_batch()
    z = np.asarray(BATCH['points'])
    hi, lo = np.asarray(batch['points_hi']), np.asarray(batch['points_lo'])
    patch = np.asarray(batch['patch'])
    rows = np.arange(z.shape[0])
    fixed = np.argmin(np.abs(z - 1.0), axis=1)
    res = dd_metric.residual_of_P_batch(batch['points_hi'], batch['points_lo'])
    assert res.max() < 1e-30, res.max()
    assert np.all(hi[rows, fixed] == 1.0) and np.all(lo[rows, fixed] == 0.0)
    for row, f in zip(patch, fixed):
        assert sorted([int(row[0]), int(row[1]), int(row[2]), int(f)]) == [0, 1, 2, 3]
    for i in (1, 2):
        assert np.array_equal(hi[rows, patch[:, i]], z[rows, patch[:, i]])
        assert np.all(lo[rows, patch[:, i]] == 0.0)
    moved = np.abs(hi[rows, patch[:, 0]] - z[rows, patch[:, 0]]) / np.abs(z[rows, patch[:, 0]])
    print(f"  max abs(P)/scale {data.residual_of_P(BATCH['points']).max():.2e} in FP64 "
          f"and {res.max():.2e} after the step; z_m moves by {moved.max():.1e} at most, "
          f"and the FP64 patch coordinate is off 1 by {np.abs(z[rows, fixed] - 1).max():.1e}")


def test_end_dd_matches_the_dd_reference():
    """At the dd point, END_DD must give `dd_metric.volume_dd` to one ulp."""
    from quartic import dd, dd_metric
    lay = sections.layout(8)
    parts = sections.prepare_parts(lay, perturbed_theta(lay, 8))
    batch = _end_batch()
    n = 8
    xs = [x[:n] for x in _with_flags(lambda: analytic.point_inputs(batch), END_DD=True)]
    got = _with_flags(
        lambda: jax.jit(jax.vmap(lambda *a: analytic.volume_at(lay, parts, *a)))(*xs),
        GRAM_DD=True, END_DD=True)

    def ref(z_dd, row):
        return dd.to_fp(dd_metric.volume_dd(lay, parts, z_dd, row[0], (row[1], row[2])))

    def split(hi, lo):
        return ((jnp.real(hi), jnp.real(lo)), (jnp.imag(hi), jnp.imag(lo)))

    patch = batch['patch'][:n]
    want = jax.jit(jax.vmap(lambda h, l, r: ref(split(h, l), r)))(
        batch['points_hi'][:n], batch['points_lo'][:n], patch)
    at_z = jax.jit(jax.vmap(lambda z, r: ref(dd.cto_dd(z), r)))(BATCH['points'][:n], patch)
    got, want, at_z = (np.asarray(a) for a in (got, want, at_z))
    ulp = np.abs(got - want) / np.spacing(np.abs(want))
    point = np.abs(at_z - want) / np.spacing(np.abs(want))
    assert ulp.max() <= 1.0, ulp
    assert point.max() > 1.0, point
    print(f"  END_DD is within {ulp.max():.0f} ulp of the dd reference, and the FP64 "
          f"point moves v by up to {point.max():.0f} ulp, k = 8")


def test_end_dd_agrees_with_the_fp64_path():
    """END_DD must give the v of the stored R and OmegaOmegabar to FP64 accuracy."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    batch = _end_batch()
    got = _with_flags(lambda: analytic.volumes_analytic(lay, theta, batch, 60),
                      GRAM_DD=True, H_LOW=True, PREPARE_DD=True, END_DD=True)
    want = analytic.volumes_analytic(lay, theta, BATCH, 60)
    rel = float(jnp.abs(got / want - 1).max())
    assert rel < 1e-13, rel
    print(f"  END_DD and the FP64 path agree to {rel:.2e}, k = 6")


def test_jacobian_with_end_dd():
    """The custom rule must still match autodiff with the whole path in dd."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    batch = _end_batch()
    got = _with_flags(
        lambda: jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, batch, 60))(theta),
        GRAM_DD=True, H_LOW=True, PHI_DD=True, PREPARE_DD=True, END_DD=True)
    want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 60))(theta)
    rel = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    assert rel < 1e-12, rel
    print(f"  with END_DD the custom rule matches autodiff to {rel:.2e}")


def test_report_with_end_dd():
    """`report` must pass the dd points to the fit path."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    batch = _end_batch()
    flags = dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True, END_DD=True)
    rep = _with_flags(lambda: metric.report(lay, theta, batch, chunk=100), **flags)
    f = np.asarray(_with_flags(lambda: analytic.residual_analytic(lay, theta, batch),
                               **flags))
    w = np.asarray(BATCH['mass']) / np.sum(np.asarray(BATCH['mass']))
    want = float(np.sum(w * np.abs(f)))
    assert abs(rep['mape'] / want - 1) < 1e-14, (rep['mape'], want)
    assert abs(rep['mape_check'] / rep['mape'] - 1) < 1e-10, rep
    print(f"  report sigma {rep['mape']:.12e}, check {rep['mape_check']:.12e}")


if __name__ == '__main__':
    for fn in (test_dd_agrees_where_fp64_is_accurate,
               test_each_entry_is_rounded_once,
               test_tangent_is_the_linear_map,
               test_whitened_layout,
               test_report_uses_the_fit_path,
               test_parts_split_the_dd_value,
               test_low_part_reaches_the_gram_stage,
               test_dd_reference_takes_the_pair,
               test_jacobian_with_the_low_part,
               test_phi_dd_agrees_where_fp64_is_accurate,
               test_phi_dd_matches_the_dd_reference,
               test_jacobian_with_phi_dd,
               test_refined_points_lie_on_X,
               test_end_dd_matches_the_dd_reference,
               test_end_dd_agrees_with_the_fp64_path,
               test_jacobian_with_end_dd,
               test_report_with_end_dd):
        print(fn.__name__)
        fn()
    print('\nall prepare tests pass')
