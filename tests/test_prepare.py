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
              'PHI_DD': analytic, 'END_DD': analytic, 'REMAT_W': analytic,
              'W_CENTERED': analytic, 'J_ROWS': analytic, 'F_DD': analytic,
              'PREPARE_DD': sections}
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


def test_remat_w_gives_the_same_jacobian():
    """W from FP64 sections in the tangent pass must give the same J."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)

    def jac(batch):
        return jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, batch, 60))(theta)

    rel = []
    for batch, flags in ((BATCH, {}),
                         (_end_batch(), dict(GRAM_DD=True, H_LOW=True,
                                             PREPARE_DD=True, END_DD=True))):
        want = _with_flags(lambda: jac(batch), **flags)
        got = _with_flags(lambda: jac(batch), REMAT_W=True, **flags)
        rel.append(float(jnp.abs(got - want).max() / jnp.abs(want).max()))
    assert rel[0] < 1e-14 and rel[1] < 1e-13, rel
    print(f"  REMAT_W moves J by {rel[0]:.2e} in FP64 and {rel[1]:.2e} with END_DD")


def test_remat_w_frees_the_linearization():
    """The LM solve must hold far less for each fit point with REMAT_W."""
    from quartic import fit
    lay = sections.layout(12)
    batch = data.make(1000, 5)

    def per_point():
        low, high = ({k: v[:n] for k, v in batch.items()} for n in (1000, 4000))
        a = fit.solve_memory(lay, low, 250, max_steps=5)
        b = fit.solve_memory(lay, high, 250, max_steps=5)
        return (b - a) / 3000 * 1e6

    old = per_point()
    new = _with_flags(per_point, REMAT_W=True)
    j_kb = lay.n_theta * 8 / 1e3
    assert new < old / 2, (old, new)
    print(f"  kB for each fit point at k = 12: {old:.1f} without REMAT_W, {new:.1f} "
          f"with it; one row of J is {j_kb:.2f}")


def _dd_gap(x, y):
    """The largest entry of x - y for two dd pairs, formed in dd."""
    from quartic import dd
    return float(jnp.abs(dd.to_fp(dd.add(x, dd.neg(y)))).max())


def _with_base(base, fn):
    """Run fn with `sections.H_BASE` set to base, and clear it after."""
    old = sections.H_BASE
    try:
        sections.H_BASE = base
        return fn()
    finally:
        sections.H_BASE = old


def test_base_adds_in_dd():
    """The base of theta_a plus the H of theta_b must be the H of the dd sum."""
    from quartic import dd
    lay = sections.layout(8)
    a = perturbed_theta(lay, 8)
    b = jnp.asarray(np.random.default_rng(3).normal(size=lay.n_theta)) * 1e-9
    base = sections.base_matrices(lay, a)
    got = _with_base(base, lambda: sections.prepare_parts(lay, b))
    scaled = dd.mul(dd.add(dd.to_dd(a), dd.to_dd(b)), dd.to_dd(jnp.asarray(lay.scale)))
    want = [dd.two_sum(*sections._block_matrices_dd(g, scaled, False)) for g in lay.groups]
    rounded = sections.prepare_parts(lay, a + b)
    worst = lost = 0.0
    for g, w, r in zip(got, want, rounded):
        top = float(jnp.abs(w[0]).max())
        worst = max(worst, _dd_gap(g, w) / top)
        lost = max(lost, _dd_gap(r, w) / top)
    assert worst < 1e-30 and lost > 1e-25, (worst, lost)
    print(f"  base plus correction matches the dd sum to {worst:.1e}; an FP64 "
          f"theta a + b is off by {lost:.1e}")


def test_base_keeps_the_tangent_and_the_fp64_path():
    """The tangent must not see the base, and the FP64 path must add it once."""
    lay = sections.layout(6)
    a = perturbed_theta(lay, 6)
    b = jnp.asarray(np.random.default_rng(4).normal(size=lay.n_theta)) * 1e-6
    tangent = jnp.asarray(np.random.default_rng(5).normal(size=lay.n_theta))
    base = sections.base_matrices(lay, a)
    _, tan = _with_base(base, lambda: jax.jvp(
        lambda t: sections.prepare_parts(lay, t), (b,), (tangent,)))
    for (t_hi, t_lo), want in zip(tan, sections.prepare_fp(lay, tangent)):
        assert float(jnp.abs(t_hi - want).max()) == 0.0 and float(jnp.abs(t_lo).max()) == 0.0
    fp = _with_base(base, lambda: sections.prepare(lay, b))
    rel = max(float(jnp.abs(m - w).max() / jnp.abs(w).max())
              for m, w in zip(fp, sections.prepare_fp(lay, a + b)))
    assert rel < 1e-15, rel
    print(f"  the tangent is prepare_fp exactly, and the FP64 path adds the base to {rel:.1e}")


def test_residual_at_the_base():
    """A correction of 0 on the base of theta must give the residual and J of theta."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    zero = jnp.zeros(lay.n_theta)
    flags = dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True)

    def run(t):
        return analytic.weighted_residual_fast(lay, t, BATCH, 60)

    want = _with_flags(lambda: run(theta), **flags)
    j_want = _with_flags(lambda: jax.jacfwd(run)(theta), **flags)
    base = sections.base_matrices(lay, theta)
    got = _with_base(base, lambda: _with_flags(lambda: run(zero), **flags))
    j_got = _with_base(base, lambda: _with_flags(lambda: jax.jacfwd(run)(zero), **flags))
    rel = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    rel_j = float(jnp.abs(j_got - j_want).max() / jnp.abs(j_want).max())
    assert rel < 1e-14 and rel_j < 1e-13, (rel, rel_j)
    print(f"  at a correction of 0, r matches to {rel:.1e} and J to {rel_j:.1e}")


def test_base_with_a_whitened_correction():
    """A plain base and a whitened correction must add as H matrices."""
    from quartic import dd
    lay_p = sections.layout(6)
    lay_w = whiten.whitened_layout(6, n_pairs=1500, seed=4242)
    a = perturbed_theta(lay_p, 6)
    w = jnp.asarray(np.random.default_rng(6).normal(size=lay_w.n_theta)) * 1e-6
    base = sections.base_matrices(lay_p, a)
    got = _with_base(base, lambda: sections.prepare_parts(lay_w, w))
    worst = 0.0
    for g, bp, cw in zip(got, base, sections.base_matrices(lay_w, w)):
        total = dd.add(bp, cw)
        worst = max(worst, _dd_gap(g, total) / float(jnp.abs(total[0]).max()))
    fp = sum(float(jnp.abs((gh + gl) - (p + q)).max() / jnp.abs(p).max())
             for (gh, gl), p, q in zip(got, sections.prepare_fp(lay_p, a),
                                       sections.prepare_fp(lay_w, w)))
    assert worst < 1e-30 and fp < 1e-14, (worst, fp)
    print(f"  plain base plus whitened correction: {worst:.1e} against the dd sum, "
          f"{fp:.1e} against FP64")


def test_w_centered_gives_the_same_jacobian():
    """W from the centered vectors must give the J of the expanded form, and of autodiff."""
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)

    def jac(batch):
        return jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, batch, 60))(theta)

    want = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 60))(theta)
    rel = {}
    for tag, batch, flags in (
            ('fp64', BATCH, {}),
            ('end_dd', _end_batch(), dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True,
                                          END_DD=True)),
            ('end_dd, remat', _end_batch(), dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True,
                                                 END_DD=True, REMAT_W=True))):
        old = _with_flags(lambda: jac(batch), **flags)
        new = _with_flags(lambda: jac(batch), W_CENTERED=True, **flags)
        top = float(jnp.abs(want).max())
        rel[tag] = (float(jnp.abs(new - old).max()) / top, float(jnp.abs(new - want).max()) / top)
    assert all(a < 1e-13 and b < 1e-12 for a, b in rel.values()), rel
    print("  centered W against the expanded W, and against autodiff: " + '; '.join(
        f"{tag} {a:.1e}, {b:.1e}" for tag, (a, b) in rel.items()))


def test_j_rows_gives_the_same_jacobian():
    """J from rows of tau and beta must give the J of the tangent map, in every basis."""
    flags = dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True, END_DD=True)
    batch = _end_batch()
    plain = sections.layout(6)
    white = whiten.whitened_layout(6, n_pairs=1500, seed=4242)

    def jac(lay, theta):
        return jax.jacfwd(lambda t: analytic.weighted_residual_fast(lay, t, batch, 60))(theta)

    rel = {}
    for tag, lay in (('plain', plain), ('whitened', white)):
        theta = perturbed_theta(lay, 6)
        want = _with_flags(lambda: jac(lay, theta), **flags)
        got = _with_flags(lambda: jac(lay, theta), J_ROWS=True, **flags)
        rel[tag] = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    base = sections.base_matrices(plain, perturbed_theta(plain, 7))
    corr = jnp.asarray(np.random.default_rng(8).normal(size=white.n_theta)) * 1e-6
    want = _with_base(base, lambda: _with_flags(lambda: jac(white, corr), **flags))
    got = _with_base(base, lambda: _with_flags(lambda: jac(white, corr), J_ROWS=True, **flags))
    rel['whitened correction on a base'] = float(jnp.abs(got - want).max()
                                                 / jnp.abs(want).max())
    auto = jax.jacfwd(lambda t: metric.weighted_residual(plain, t, BATCH, 60))(
        perturbed_theta(plain, 6))
    rows = _with_flags(lambda: jac(plain, perturbed_theta(plain, 6)), J_ROWS=True, **flags)
    rel['plain against autodiff'] = float(jnp.abs(rows - auto).max() / jnp.abs(auto).max())
    assert all(val < 1e-12 for val in rel.values()), rel
    print("  J_ROWS against the tangent map: " + '; '.join(
        f"{tag} {val:.1e}" for tag, val in rel.items()))


def test_saved_whitening_repeats_the_layout():
    """A layout from saved W must have the fingerprint and the H of the original."""
    import tempfile
    lay = whiten.whitened_layout(6, n_pairs=1500, seed=4242)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'w.npz'
        whiten.save_whitening(lay, path)
        again = whiten.load_layout(6, path)
    theta = perturbed_theta(lay, 6)
    worst = max(float(jnp.abs(a - b).max())
                for a, b in zip(sections.prepare_fp(lay, theta), sections.prepare_fp(again, theta)))
    assert again.fingerprint() == lay.fingerprint() and worst == 0.0, (worst,)
    print(f"  the saved whitening gives the fingerprint {again.fingerprint()} and the same H")


def test_f_dd_rounds_f_once():
    """With F_DD, f must match the dd reference far below the FP64 rounding of v."""
    from quartic import dd_metric
    lay = sections.layout(6)
    theta = perturbed_theta(lay, 6)
    batch = _end_batch()
    flags = dict(GRAM_DD=True, H_LOW=True, PREPARE_DD=True, END_DD=True)

    def f_of():
        return analytic.residual_analytic(lay, theta, batch, 60)

    f_dd = np.asarray(_with_flags(f_of, F_DD=True, **flags))
    f_fp = np.asarray(_with_flags(f_of, **flags))
    parts = _with_flags(lambda: analytic.point_matrices(lay, theta), **flags)

    def ref(h, l, row):
        z_dd = ((jnp.real(h), jnp.real(l)), (jnp.imag(h), jnp.imag(l)))
        return dd_metric.volume_dd(lay, parts, z_dd, row[0], (row[1], row[2]))

    v_ref = jax.jit(jax.vmap(ref))(batch['points_hi'], batch['points_lo'], batch['patch'])
    weights = batch['mass'] / jnp.sum(batch['mass'])
    f_ref = np.asarray(analytic.residual_from_volumes(v_ref, weights))
    gap_dd = float(np.abs(f_dd - f_ref).max())
    gap_fp = float(np.abs(f_fp - f_ref).max())
    jac = _with_flags(lambda: jax.jacfwd(
        lambda t: analytic.weighted_residual_fast(lay, t, batch, 60))(theta), F_DD=True,
        J_ROWS=True, **flags)
    auto = jax.jacfwd(lambda t: metric.weighted_residual(lay, t, BATCH, 60))(theta)
    rel = float(jnp.abs(jac - auto).max() / jnp.abs(auto).max())
    assert gap_dd < 1e-20 and gap_fp > 1e-18 and rel < 1e-12, (gap_dd, gap_fp, rel)
    print(f"  f against the dd reference: {gap_dd:.1e} with F_DD, {gap_fp:.1e} with v "
          f"rounded first; J with F_DD against autodiff {rel:.1e}")


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
               test_report_with_end_dd,
               test_remat_w_gives_the_same_jacobian,
               test_remat_w_frees_the_linearization,
               test_base_adds_in_dd,
               test_base_keeps_the_tangent_and_the_fp64_path,
               test_residual_at_the_base,
               test_base_with_a_whitened_correction,
               test_w_centered_gives_the_same_jacobian,
               test_j_rows_gives_the_same_jacobian,
               test_saved_whitening_repeats_the_layout,
               test_f_dd_rounds_f_once):
        print(fn.__name__)
        fn()
    print('\nall prepare tests pass')
