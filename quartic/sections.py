"""Evaluation of the Headrick-Nassar potential p at degree k on the quartic.

p = sum_r |z^r|^2 * (psi_r^dag H_r psi_r), where r runs over the 64 residue
blocks and psi_r holds the block sections in the frame of `blocks`. Every block
with the same m uses the same section vector, so the code evaluates that vector
once for each m.

Two rules come from measurements, and both matter below 1e-13:
  * build every monomial by repeated multiplication, so the exponent is an exact
    integer; a float exponent goes through exp and log and loses 15 digits,
  * keep the three sums p, dp and ddbar-p inside `_quadratic_form`, so that a
    compensated accumulator is a one-function change later.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np

from . import blocks

__all__ = ['Group', 'Layout', 'layout', 'log_potential', 'potential',
           'fubini_study_theta', 'block_matrices', 'prepare', 'potential_at',
           'log_potential_at']


@dataclass(frozen=True)
class Group:
    """Every block that shares one value of m. They share one section vector."""
    m: int
    exps: np.ndarray        # (n_monomials, 4) integer exponents of u
    frame: np.ndarray       # (n_monomials, d) orthonormal frame of the sections
    residues: np.ndarray    # (n_blocks, 4) the residue r of each block
    rot: np.ndarray         # (n_blocks, d, d) orthogonal, from the orbit rep
    orbit_of: np.ndarray    # (n_blocks,) index into `basis`
    basis: np.ndarray       # (n_orbits, n_params_max, d, d) invariant matrices
    slices: tuple           # one slice of theta for each orbit in this group


@dataclass(frozen=True)
class Layout:
    """Static description of the model at degree k."""
    k: int
    groups: tuple
    n_theta: int
    structure: blocks.Structure


def layout(k, struct=None):
    """Group the blocks of degree k by m, ready for a batch matrix-vector product."""
    struct = struct or blocks.build(k)
    slices = struct.param_slices()

    groups = []
    for m in sorted({b.m for b in struct.blocks}):
        members = [b for b in struct.blocks if b.m == m]
        orbit_ids = sorted({b.orbit for b in members})
        local = {o: i for i, o in enumerate(orbit_ids)}
        n_max = max(struct.orbits[o].basis.shape[0] for o in orbit_ids)
        d = struct.frames[m].shape[1]

        basis = np.zeros((len(orbit_ids), n_max, d, d))
        for o in orbit_ids:
            b_o = struct.orbits[o].basis
            basis[local[o], :b_o.shape[0]] = b_o

        groups.append(Group(
            m=m,
            exps=np.array(blocks.monomial_exponents(m), dtype=np.int32),
            frame=struct.frames[m],
            residues=np.array([b.r for b in members], dtype=np.int32),
            rot=np.stack([b.rot for b in members]),
            orbit_of=np.array([local[b.orbit] for b in members]),
            basis=basis,
            slices=tuple(slices[o] for o in orbit_ids),
        ))
    return Layout(k=k, groups=tuple(groups), n_theta=struct.n_theta,
                  structure=struct)


def _integer_powers(x, top):
    """x**0 .. x**top by repeated multiplication, so each exponent is exact."""
    out = [jnp.ones_like(x)]
    for _ in range(top):
        out.append(out[-1] * x)
    return jnp.stack(out, axis=0)


def _monomials(u, exps, m):
    """Every monomial u^E for the rows E of `exps`, with integer exponents."""
    table = _integer_powers(u, m)                       # (m+1, 4)
    cols = table[exps, jnp.arange(4)]                   # (n_monomials, 4)
    return jnp.prod(cols, axis=1)


def _quadratic_form(mats, vec):
    """psi^dag H psi for a stack of real symmetric H, as a real number.

    H is real, so the form splits into the real and the imaginary part of psi.
    That makes the result real with no round-off in the imaginary part. The
    three sums of the model pass through here, so stage 2 changes only this.
    """
    x, y = jnp.real(vec), jnp.imag(vec)
    return (jnp.einsum('i,bij,j->b', x, mats, x)
            + jnp.einsum('i,bij,j->b', y, mats, y))


def block_matrices(group, theta):
    """H_r for every block of a group, from the flat parameter vector."""
    reps = jnp.stack([jnp.einsum('a,aij->ij', theta[s], group.basis[i, :s.stop - s.start])
                      for i, s in enumerate(group.slices)])
    per_block = reps[group.orbit_of]
    return jnp.einsum('bik,bkl,bjl->bij', group.rot, per_block, group.rot)


def prepare(lay, theta):
    """H_r for every block, built once for a whole point set.

    theta -> H costs O(64 d^3) and does not depend on the point, so it must stay
    outside the loop over points.
    """
    return tuple(block_matrices(g, theta) for g in lay.groups)


def potential_at(lay, mats, z):
    """p(z) for one point z, with the block matrices already built."""
    z2 = z * z
    u = z2 * z2                                          # u_a = z_a^4, exact
    abs2 = jnp.real(z) ** 2 + jnp.imag(z) ** 2
    weight_table = _integer_powers(abs2, 3)              # r_a is 0 to 3

    total = jnp.zeros((), dtype=abs2.dtype)
    for g, mat in zip(lay.groups, mats):
        psi = jnp.asarray(g.frame).T @ _monomials(u, jnp.asarray(g.exps), g.m)
        weights = jnp.prod(weight_table[jnp.asarray(g.residues), jnp.arange(4)],
                           axis=1)
        total = total + jnp.dot(weights, _quadratic_form(mat, psi))
    return total


def log_potential_at(lay, mats, z):
    """log p(z). This is the Kahler potential up to a constant factor."""
    return jnp.log(potential_at(lay, mats, z))


def potential(lay, theta, z):
    """p(z) for one point z, a 4-vector of homogeneous coordinates."""
    return potential_at(lay, prepare(lay, theta), z)


def log_potential(lay, theta, z):
    """log p(z). This is the Kahler potential up to a constant factor."""
    return jnp.log(potential(lay, theta, z))


def fubini_study_theta(lay):
    """theta such that p = (sum_a |z_a|^2)^k, which is the k-th power of Fubini-Study.

    The multinomial coefficients are S_4-symmetric, phase diagonal and real, so
    this point is inside the invariant subspace. The assert proves it.
    """
    k = lay.k
    theta = np.zeros(lay.n_theta)
    for orb, sl in zip(lay.structure.orbits, lay.structure.param_slices()):
        frame = lay.structure.frames[orb.m]
        diag = np.array([factorial(k) / np.prod([factorial(int(r + 4 * c))
                                                 for r, c in zip(orb.rep, e)])
                         for e in blocks.monomial_exponents(orb.m)])
        h_rep = frame.T @ np.diag(diag) @ frame
        coeffs = np.einsum('aij,ij->a', orb.basis, h_rep)
        rebuilt = np.einsum('a,aij->ij', coeffs, orb.basis)
        assert np.abs(rebuilt - h_rep).max() < 1e-9 * max(np.abs(h_rep).max(), 1.0)
        theta[sl] = coeffs
    return jnp.asarray(theta)


def batched_log_potential(lay, theta, points):
    """log p at every row of `points`."""
    return jax.vmap(lambda z: log_potential(lay, theta, z))(points)
