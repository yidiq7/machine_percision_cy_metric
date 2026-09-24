"""`grad f` in double-double must agree with the FP64 autodiff where FP64 is enough.

Run on a GPU node: uv run --extra gpu python -m tests.test_gradient_dd
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd, dd_metric, gradient, gradient_dd, sections   # noqa: E402

# The FP64 references must be plain FP64; `_dd_flags` turns the dd path on.
analytic.GRAM_DD = False
analytic.SECTION_DD = False

BATCH = data.make(40, 17)
BATCH.update(dd_metric.point_extras(BATCH['points']))


def perturbed(lay, seed):
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * 1e-2
    return sections.fubini_study_theta(lay) + jnp.asarray(noise)


def _point(i):
    hi, lo = BATCH['points_hi'][i], BATCH['points_lo'][i]
    row = BATCH['patch'][i]
    z_dd = ((jnp.real(hi), jnp.real(lo)), (jnp.imag(hi), jnp.imag(lo)))
    return z_dd, row[0], (row[1], row[2])


def _dd_flags(fn):
    old = (analytic.END_DD, analytic.GRAM_DD)
    try:
        analytic.END_DD, analytic.GRAM_DD = True, True
        return fn()
    finally:
        analytic.END_DD, analytic.GRAM_DD = old


def test_moved_point_lies_on_X():
    """The moved point must lie on X, with t_i moved by h and nothing else but z_m."""
    eps = 2.0 ** -14
    worst_p, worst_t = 0.0, 0.0
    for n in range(6):
        z_dd, m, free = _point(n)
        for i in range(2):
            for rotate in (False, True):
                zp = gradient_dd.moved_point(z_dd, m, free, i, eps, rotate)
                worst_p = max(worst_p, dd_metric.residual_of_P(zp))
                step = dd.cto_fp(dd.csub(zp, z_dd))
                want = np.zeros(4, complex)
                want[int(free[i])] = 1j * eps if rotate else eps
                want[int(m)] = step[int(m)]
                worst_t = max(worst_t, float(np.abs(np.asarray(step) - want).max()))
    assert worst_p < 1e-30 and worst_t == 0.0, (worst_p, worst_t)
    print(f"  moved points: abs(P)/scale at most {worst_p:.1e}, and only t_i and z_m move")


def test_gradient_matches_autodiff():
    """At k = 6, the dd difference must match the FP64 jvp of `gradient.py`."""
    lay = sections.layout(6)
    theta = perturbed(lay, 6)
    parts = sections.prepare_parts(lay, theta)
    mats = sections.prepare_fp(lay, theta)
    n = 4
    hi, lo, rows = BATCH['points_hi'][:n], BATCH['points_lo'][:n], BATCH['patch'][:n]
    want = jax.vmap(lambda z, row: gradient.volume_tangents(
        lay, mats, z, row[0], row[1:])[1])(BATCH['points'][:n], rows)
    worst = {}
    for eps in (2.0 ** -14, 2.0 ** -20):
        def one(h, l, row, eps=eps):
            z_dd = ((jnp.real(h), jnp.real(l)), (jnp.imag(h), jnp.imag(l)))
            return gradient_dd.volume_gradient(lay, parts, z_dd, row[0], (row[1], row[2]),
                                               eps)[1]
        got = _dd_flags(lambda: jax.jit(jax.vmap(one))(hi, lo, rows))
        worst[eps] = float((jnp.abs(got - want).max(axis=1)
                            / jnp.abs(want).max(axis=1)).max())
    assert worst[2.0 ** -20] < 1e-9, worst
    print("  dv/dt against the FP64 jvp: " + ', '.join(
        f"{w:.1e} at eps = 2^{int(np.log2(e))}" for e, w in worst.items()))


def test_report_matches_fp64():
    """At k = 6, the dd report must give the norms of `gradient.gradient_report`."""
    lay = sections.layout(6)
    theta = perturbed(lay, 6)
    got = _dd_flags(lambda: gradient_dd.gradient_report_dd(lay, theta, BATCH,
                                                           eps=2.0 ** -20))
    want = gradient.gradient_report(lay, theta, BATCH)
    rel_f = abs(got['l2_f'] / want['l2_f'] - 1)
    rel_df = abs(got['l2_grad_f'] / want['l2_grad_f'] - 1)
    assert rel_f < 1e-12 and rel_df < 1e-8, (rel_f, rel_df)
    print(f"  norm(f) to {rel_f:.1e} and norm(grad f) to {rel_df:.1e}; "
          f"R = {got['ratio']:.6f} against {want['ratio']:.6f}")


if __name__ == '__main__':
    for fn in (test_moved_point_lies_on_X,
               test_gradient_matches_autodiff,
               test_report_matches_fp64):
        print(fn.__name__)
        fn()
    print('\nall gradient dd tests pass')
