"""Tests of the block decomposition against character theory and against u.

Run: .venv/bin/python -m tests.test_blocks
"""
import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notes'))

from quartic import blocks                       # noqa: E402
import symmetry_param_counts_s4 as counts        # noqa: E402

rng = np.random.default_rng(11)


def compose(t, s):
    """(t o s)[i] = t[s[i]], which matches blocks.act(t, blocks.act(s, v))."""
    return tuple(t[s[i]] for i in range(4))


def monomial_values(m, u):
    """Every degree-m monomial in u, in the order of monomial_exponents."""
    return np.array([np.prod(u ** np.array(e)) for e in blocks.monomial_exponents(m)])


def test_action_is_a_left_action():
    v = (3, 1, 0, 2)
    for t, s in itertools.product(blocks.S4, repeat=2):
        assert blocks.act(t, blocks.act(s, v)) == blocks.act(compose(t, s), v)
    print("  action composes correctly")


def test_relations_vanish_on_X():
    """Every column of relation_columns must be zero where sum_a u_a = 0."""
    worst = 0.0
    for m in range(1, 10):
        rel = blocks.relation_columns(m)
        assert rel.shape == (comb(m + 3, 3), comb(m + 2, 3)), m
        for _ in range(5):
            u = rng.normal(size=4) + 1j * rng.normal(size=4)
            u -= u.mean()                       # now sum_a u_a = 0
            vals = monomial_values(m, u)
            worst = max(worst, np.abs(rel.T @ vals).max() / np.abs(vals).max())
    assert worst < 1e-12, worst
    print(f"  relations vanish on X to {worst:.2e}, for m = 1..9")


def test_frame_dimension_and_orthonormality():
    for m in range(0, 10):
        frame = blocks.section_frame(m)
        assert frame.shape == (comb(m + 3, 3), comb(m + 2, 2)), m
        err = np.abs(frame.T @ frame - np.eye(frame.shape[1])).max()
        assert err < 1e-13, (m, err)
    print("  frames have dimension C(m+2,2) and are orthonormal to 1e-13")


def test_rho_is_an_orthogonal_representation():
    worst_rep = worst_orth = 0.0
    for m in range(0, 10):
        frame = blocks.section_frame(m)
        mats = {s: blocks.rho(m, s, frame) for s in blocks.S4}
        eye = np.eye(frame.shape[1])
        for s, r_mat in mats.items():
            worst_orth = max(worst_orth, np.abs(r_mat.T @ r_mat - eye).max())
        for t, s in itertools.product(blocks.S4, repeat=2):
            worst_rep = max(worst_rep,
                            np.abs(mats[t] @ mats[s] - mats[compose(t, s)]).max())
    assert worst_orth < 1e-13 and worst_rep < 1e-13, (worst_orth, worst_rep)
    print(f"  max |rho^T rho - I| = {worst_orth:.2e}, "
          f"max |rho(t)rho(s) - rho(t o s)| = {worst_rep:.2e}, m = 0..9")


def test_counts_match_character_theory():
    print(f"  {'k':>3} {'blocks':>7} {'sections':>9} {'2k^2+2':>8} "
          f"{'n_theta':>8} {'chars':>7} {'MACs':>9}")
    for k in (2, 4, 6, 8, 10, 12, 16, 20):
        st = blocks.build(k)
        expected, _ = counts.n_theta(k)
        print(f"  {k:>3} {len(st.blocks):>7} {st.n_sections:>9} {2 * k * k + 2:>8} "
              f"{st.n_theta:>8} {expected:>7} {st.macs:>9,}")
        assert st.n_sections == 2 * k * k + 2, k
        assert st.n_theta == expected, (k, st.n_theta, expected)
        assert st.macs == counts.block_stats(k)[1], k
    print("  n_theta agrees with character theory to k = 20")


def test_basis_is_invariant():
    """Rule 1: h must stay in the invariant subspace, so the basis must be exact."""
    worst = 0.0
    for k in (8, 12, 20, 32, 38):
        st = blocks.build(k)
        for orb in st.orbits:
            for s in orb.stabilizer:
                r_mat = blocks.rho(orb.m, s, st.frames[orb.m])
                for b_mat in orb.basis:
                    worst = max(worst, np.abs(r_mat @ b_mat @ r_mat.T - b_mat).max())
    assert worst < 1e-13, worst
    print(f"  max |rho B rho^T - B| over the stabilizers = {worst:.2e}, to k = 38")


def test_blocks_cover_every_residue():
    for k in (12, 20, 32, 38):
        st = blocks.build(k)
        assert len(st.blocks) == 64, (k, len(st.blocks))
        assert len({b.r for b in st.blocks}) == 64
        for b in st.blocks:
            assert tuple(x % 4 for x in b.r) == b.r
            assert (k - sum(b.r)) // 4 == b.m
    print("  64 blocks at k = 12, 20, 32 and 38, and each residue appears once")


def test_high_k_counts():
    for k, n_theta, secs in ((32, 1559, 2050), (36, 2432, 2594), (38, 2987, 2890)):
        st = blocks.build(k)
        assert (st.n_theta, st.n_sections) == (n_theta, secs), k
    print("  n_theta = 1559, 2432 and 2987 at k = 32, 36 and 38")


if __name__ == "__main__":
    for fn in (test_action_is_a_left_action, test_relations_vanish_on_X,
               test_frame_dimension_and_orthonormality,
               test_rho_is_an_orthogonal_representation,
               test_counts_match_character_theory, test_basis_is_invariant,
               test_blocks_cover_every_residue, test_high_k_counts):
        print(fn.__name__)
        fn()
    print("\nall block tests pass")
