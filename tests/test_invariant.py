"""Tests of `quartic.invariant`: the base in the monomial basis, exactly invariant.

The blocks of an orbit must hold the same dd numbers at permuted positions, each
block must be unchanged by its stabilizer, p0 must take the same value at the 24
images of a point to the working precision, and the dd path under jit must agree.

Run on a GPU node: uv run --extra gpu python -m tests.test_invariant
"""
import itertools
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import analytic, blocks, correction, data, dd, dd_metric, invariant, sections  # noqa: E402

PERMS = list(itertools.permutations(range(4)))


def base(k, seed=1, size=1e-3):
    """lay, theta near Fubini-Study, and the dd blocks of the per-point path."""
    lay = sections.layout(k)
    noise = np.random.default_rng(seed).normal(size=lay.n_theta) * size
    theta = sections.fubini_study_theta(lay) + jnp.asarray(noise)
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    sections.PREPARE_DD = True
    parts = jax.jit(lambda t: analytic.point_matrices(lay, t))(theta)
    return lay, theta, parts


def test_blocks_are_exact_copies():
    """Each block of an orbit holds the dd numbers of the first one at permuted places."""
    for k in (8, 14):
        lay, _, parts = base(k)
        lay_m, parts_m = invariant.monomial_base(lay, parts)
        checked = 0
        for g, (hi, lo) in zip(lay_m.groups, parts_m):
            hi, lo = np.asarray(hi), np.asarray(lo)
            exps = [tuple(int(x) for x in e) for e in np.asarray(g.exps)]
            index = {e: i for i, e in enumerate(exps)}
            res = [tuple(int(x) for x in r) for r in np.asarray(g.residues)]
            for b, r in enumerate(res):
                for s in PERMS:
                    target = blocks.act(s, r)
                    if target not in res:
                        continue
                    bt = res.index(target)
                    pi = np.array([index[blocks.act(s, e)] for e in exps])
                    assert np.array_equal(hi[bt][np.ix_(pi, pi)], hi[b]), (k, r, s)
                    assert np.array_equal(lo[bt][np.ix_(pi, pi)], lo[b]), (k, r, s)
                    checked += 1
        print(f"  k = {k}: {checked} block pairs equal bit for bit under S_4")


def test_potential_is_invariant_in_mpmath():
    """p0 at the 24 images of random points of CP^3 agrees to the working precision."""
    mp.mp.dps = 50
    rng = np.random.default_rng(7)
    for k in (8, 14):
        lay, _, parts = base(k)
        lay_m, parts_m = invariant.monomial_base(lay, parts)
        worst = 0.0
        for _ in range(3):
            z = [mp.mpc(*rng.normal(size=2)) for _ in range(4)]
            p = invariant.potential_mp(lay_m, parts_m, z)
            for s in PERMS:
                q = invariant.potential_mp(lay_m, parts_m, [z[s[a]] for a in range(4)])
                worst = max(worst, float(abs(q / p - 1)))
        assert worst < 1e-40, (k, worst)
        print(f"  k = {k}: max over S_4 of abs(p0(sigma z) / p0(z) - 1) = {worst:.1e}")


def _p_dd(lay, parts, points):
    extras = dd_metric.point_extras(points)
    xs = (extras['points_hi'], (extras['points_hi'], extras['points_lo'], extras['patch']))
    fn = jax.jit(jax.vmap(lambda z, e: correction.base_state_dd(lay, parts, z, e,
                                                              with_p=True)[2]))
    return fn(*xs)


def test_dd_path_is_invariant_and_close_to_the_frame_base():
    """Under jit: the monomial base is invariant to dd, and within 1e-12 of the old base."""
    k = 14
    lay, _, parts = base(k)
    lay_m, parts_m = invariant.monomial_base(lay, parts)
    points = np.asarray(data.make(20, 11)['points'])[:40]
    p_new = _p_dd(lay_m, parts_m, points)
    p_old = _p_dd(lay, parts, points)
    gap_old = float(np.max(np.abs(dd.to_fp(dd.add(p_new, dd.neg(p_old)))
                                  / dd.to_fp(p_old))))
    worst_new, worst_old = 0.0, 0.0
    for s in PERMS[1:]:
        perm_pts = points[:, list(s)]
        q_new = _p_dd(lay_m, parts_m, perm_pts)
        q_old = _p_dd(lay, parts, perm_pts)
        worst_new = max(worst_new, float(np.max(np.abs(
            dd.to_fp(dd.add(q_new, dd.neg(p_new))) / dd.to_fp(p_new)))))
        worst_old = max(worst_old, float(np.max(np.abs(
            dd.to_fp(dd.add(q_old, dd.neg(p_old))) / dd.to_fp(p_old)))))
    assert worst_new < 1e-29, worst_new
    assert gap_old < 1e-12, gap_old
    print(f"  k = {k}, dd under jit: new base over S_4 {worst_new:.1e}, old base "
          f"{worst_old:.1e}; new against old {gap_old:.1e}")


if __name__ == "__main__":
    for fn in (test_blocks_are_exact_copies, test_potential_is_invariant_in_mpmath,
               test_dd_path_is_invariant_and_close_to_the_frame_base):
        print(fn.__name__)
        fn()
    print("\nall invariant-base tests pass")
