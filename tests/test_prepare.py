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


if __name__ == '__main__':
    for fn in (test_dd_agrees_where_fp64_is_accurate,
               test_each_entry_is_rounded_once,
               test_tangent_is_the_linear_map,
               test_whitened_layout,
               test_report_uses_the_fit_path):
        print(fn.__name__)
        fn()
    print('\nall prepare tests pass')
