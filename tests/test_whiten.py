"""The L^2(X) whitening must be an exact reparametrization of h.

An exact reparametrization moves no critical point, so the fit must reach the
same minimum in the new coordinates. These tests check the three things that
could break it: the round trip of W, the invariant subspace, and the reachable
set of metrics.

Run: uv run python -m tests.test_whiten
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import blocks, data, sections, whiten          # noqa: E402

rng = np.random.default_rng(2026)
GRAM_BATCH = data.make(4_000, 20260901)          # 16,000 points, the Gram sample
TEST_BATCH = data.make(2_000, 20260902)           # 8,000 points, independent
KS = (4, 6, 8, 12)


def whitened(k):
    return whiten.whitened_layout(k, batch=GRAM_BATCH)


def test_gram_is_positive_definite():
    print(f"  {'k':>3} {'orbits':>7} {'cond(G) worst':>14} {'min eig':>11}")
    for k in KS:
        plain = sections.layout(k, normalize=False)
        grams = whiten.gram_matrices(plain, GRAM_BATCH)
        worst = max(np.linalg.cond(g) for g in grams)
        least = min(np.linalg.eigvalsh(g).min() for g in grams)
        assert least > 0, (k, least)
        print(f"  {k:>3} {len(grams):>7} {worst:>14.3e} {least:>11.3e}")
    print("  every orbit Gram is positive definite")


def test_p_equals_fubini_study_after_whitening():
    """The strongest check of the W round trip: an exact target that W cannot move.

    theta_FS is built by undoing W, and the model applies W again. If either
    matrix is wrong the exact power of Fubini-Study does not come back.
    """
    z = np.asarray(TEST_BATCH['points'][:40])
    print(f"  {'k':>3} {'max rel err':>12}")
    for k in KS:
        lay = whitened(k)
        theta = sections.fubini_study_theta(lay)
        got = np.asarray(jax.vmap(
            lambda w: sections.potential(lay, theta, w))(jnp.asarray(z)))
        want = (np.abs(z) ** 2).sum(axis=1) ** k
        err = np.abs(got - want).max() / np.abs(want).max()
        print(f"  {k:>3} {err:>12.3e}")
        assert err < 1e-12, (k, err)
    print("  p = (sum |z_a|^2)^k at theta_FS with whitening, to k = 12")


def test_whitening_keeps_the_invariant_subspace():
    """W must commute with the stabilizer, or h leaves the invariant subspace.

    This is rule 1 of the Output section. The Gram matrix is symmetrized over
    the stabilizer for exactly this reason.
    """
    worst = 0.0
    for k in KS:
        lay = whitened(k)
        theta = jnp.asarray(rng.normal(size=lay.n_theta))
        for g in lay.groups:
            mats = np.asarray(sections.block_matrices(g, theta))
            frame = np.asarray(g.frame)
            for bi in range(g.residues.shape[0]):
                r = tuple(int(x) for x in np.asarray(g.residues[bi]))
                stab = tuple(s for s in blocks.S4 if blocks.act(s, r) == r)
                h = mats[bi]
                for s in stab:
                    rs = blocks.rho(g.m, s, frame)
                    moved = rs @ h @ rs.T
                    worst = max(worst,
                                np.abs(moved - h).max() / np.abs(h).max())
    assert worst < 1e-11, worst
    print(f"  max |rho H rho^T - H| over every block and stabilizer = {worst:.2e}")


def test_the_reachable_set_is_unchanged():
    """W is invertible, so every plain h has a whitened theta with the same p.

    If this fails the reparametrization is not exact and the minimum can move.
    """
    z = jnp.asarray(TEST_BATCH['points'][:60])
    worst = 0.0
    for k in KS:
        plain = sections.layout(k)
        lay = whitened(k)
        theta = rng.normal(size=plain.n_theta)

        # theta_white such that W H_white W equals the plain H, orbit by orbit.
        white = np.zeros(plain.n_theta)
        for oi, (orb, sl) in enumerate(zip(plain.structure.orbits,
                                           plain.structure.param_slices())):
            h_plain = np.einsum('a,aij->ij', theta[sl] * plain.scale[sl],
                                orb.basis)
            inv = lay.whiten_inv[oi]
            white[sl] = np.einsum('aij,ij->a', orb.basis,
                                  inv @ h_plain @ inv) / lay.scale[sl]

        p_plain = jax.vmap(lambda w: sections.potential(
            plain, jnp.asarray(theta), w))(z)
        p_white = jax.vmap(lambda w: sections.potential(
            lay, jnp.asarray(white), w))(z)
        err = float(jnp.abs(p_plain - p_white).max() / jnp.abs(p_plain).max())
        worst = max(worst, err)
        assert err < 1e-11, (k, err)
    print(f"  the same p from plain and whitened coordinates, to {worst:.2e}")


def test_whitened_features_are_orthonormal():
    """The point of the change: the Gram becomes the identity on fresh points."""
    z = np.asarray(TEST_BATCH['points'])
    mass = np.asarray(TEST_BATCH['mass'])
    w = mass / mass.sum()
    nrm2 = (np.abs(z) ** 2).sum(axis=1)
    u = z ** 4
    abs2 = np.abs(z) ** 2

    def worst_cond(lay, use_whiten):
        out = 0.0
        for g in lay.groups:
            exps = np.asarray(g.exps)
            mono = np.prod(u[:, None, :] ** exps[None, :, :], axis=2)
            psi = mono @ np.asarray(g.frame)
            for bi in range(g.residues.shape[0]):
                r = np.asarray(g.residues[bi])
                wt = w * np.prod(abs2 ** r[None, :], axis=1) / nrm2 ** lay.k
                feat = psi
                if use_whiten:
                    rot = np.asarray(g.rot[bi])
                    w_orb = np.asarray(g.whiten[int(g.orbit_of[bi])])
                    feat = psi @ (rot @ w_orb)
                gram = ((feat.conj() * wt[:, None]).T @ feat).real
                ev = np.linalg.eigvalsh(gram)
                out = max(out, ev[-1] / ev[0])
        return out

    print(f"  {'k':>3} {'cond before':>12} {'cond after':>11}")
    for k in KS:
        before = worst_cond(sections.layout(k, normalize=False), False)
        after = worst_cond(whitened(k), True)
        print(f"  {k:>3} {before:>12.3e} {after:>11.4f}")
        assert after < 1.6, (k, after)
    print("  whitened features are orthonormal on an independent sample")


def test_fingerprint_separates_the_conventions():
    for k in (4, 6):
        plain = sections.layout(k)
        lay = whitened(k)
        assert plain.fingerprint() != lay.fingerprint(), k
        assert lay.fingerprint() == whitened(k).fingerprint(), k
    print("  the fingerprint repeats and separates whitened from plain")


if __name__ == "__main__":
    for fn in (test_gram_is_positive_definite,
               test_p_equals_fubini_study_after_whitening,
               test_whitening_keeps_the_invariant_subspace,
               test_the_reachable_set_is_unchanged,
               test_whitened_features_are_orthonormal,
               test_fingerprint_separates_the_conventions):
        print(fn.__name__)
        fn()
    print("\nall whitening tests pass")
