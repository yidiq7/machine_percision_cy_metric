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

import hashlib
from dataclasses import dataclass
from functools import partial
from math import factorial, prod

import jax
import jax.numpy as jnp
import numpy as np

from . import blocks, dd

__all__ = ['Group', 'Layout', 'layout', 'log_potential', 'potential',
           'fubini_study_theta', 'block_matrices', 'prepare', 'prepare_fp',
           'prepare_parts',
           'potential_at', 'log_potential_at', 'PREPARE_DD']

# Form each H_r in double-double and round it once to FP64. The FP64 form moves
# the residual at the stalled theta of k = 28 by 38 percent of its norm.
PREPARE_DD = False


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
    whiten: np.ndarray      # (n_orbits, d, d) symmetric; H = W H_white W


@dataclass(frozen=True)
class Layout:
    """Static description of the model at degree k.

    `scale` is **one number for every component**, the largest of `abs(theta_FS)`.
    The raw invariant basis is orthonormal in the Frobenius norm, so theta_FS
    carries the multinomial coefficients: they reach 1.2e14 at k = 28. LM damps
    with lambda*I and tests the step in absolute size, and neither is invariant
    under a change of scale, so a trust-region step of order 1 was a relative
    change of 1e-14 and the fit stopped after 2 steps.

    A scale for each component, `abs(theta_FS)` itself, looks tidier and is
    wrong. It measures how far a basis matrix leans on the Fubini-Study point,
    not how much it moves the model, and a basis vector nearly orthogonal to
    h_FS gets a tiny scale that then amplifies its direction. At k = 28 the
    spread of those scales is 8.6e17 with 56 components under 1e-6 of the
    largest, and cond(J) reached 8.1e30. Measured at theta_FS, one global scale
    gives cond(J) of 2.36e17 at k = 24 against 1.05e25 for a scale for each
    component.

    The scale is part of the reported convention for h.
    """
    k: int
    groups: tuple
    n_theta: int
    structure: blocks.Structure
    scale: np.ndarray
    whiten_inv: tuple = ()   # (d, d) per orbit, G^1/2; empty means no whitening
    gram_id: str = 'none'    # names the Gram sample, for the fingerprint

    def fingerprint(self):
        """Digest of every convention that theta depends on.

        theta means nothing without the basis, the whitening and the scale.
        Save this with the fit and compare it on a restart.
        """
        digest = hashlib.sha256()
        digest.update(self.structure.fingerprint().encode())
        digest.update(self.gram_id.encode())
        for mat in self.whiten_inv:
            digest.update(np.ascontiguousarray(np.round(mat, 8)).tobytes())
        digest.update(np.ascontiguousarray(
            np.round(self.scale / max(self.scale.max(), 1e-300), 12)).tobytes())
        return digest.hexdigest()[:16]


def layout(k, struct=None, normalize=True, whiten=None, gram_id='none'):
    """Group the blocks of degree k by m, ready for a batch matrix-vector product.

    `whiten` is one symmetric matrix for each orbit, as `whiten.whitening_matrices`
    returns. It reparametrizes h and moves no critical point. See `quartic.whiten`.
    """
    struct = struct or blocks.build(k)
    slices = struct.param_slices()
    w_fwd, w_inv = (whiten if whiten is not None else (None, None))

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

        w_group = (np.stack([w_fwd[o] for o in orbit_ids]) if w_fwd is not None
                   else np.stack([np.eye(d) for _ in orbit_ids]))

        groups.append(Group(
            m=m,
            exps=np.array(blocks.monomial_exponents(m), dtype=np.int32),
            frame=struct.frames[m],
            residues=np.array([b.r for b in members], dtype=np.int32),
            rot=np.stack([b.rot for b in members]),
            orbit_of=np.array([local[b.orbit] for b in members]),
            basis=basis,
            slices=tuple(slices[o] for o in orbit_ids),
            whiten=w_group,
        ))
    inv = tuple(w_inv) if w_inv is not None else ()
    lay = Layout(k=k, groups=tuple(groups), n_theta=struct.n_theta,
                 structure=struct, scale=np.ones(struct.n_theta),
                 whiten_inv=inv, gram_id=gram_id)
    if not normalize:
        return lay
    # One global scale, from the largest raw component. See `Layout`.
    raw = np.asarray(fubini_study_theta(lay))
    largest = float(np.abs(raw).max())
    scale = np.full(struct.n_theta, largest if largest > 0 else 1.0)
    return Layout(k=k, groups=lay.groups, n_theta=lay.n_theta,
                  structure=struct, scale=scale, whiten_inv=inv,
                  gram_id=gram_id)


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
    """H_r for every block of a group, from the flat parameter vector.

    `group.whiten` carries the L^2(X) whitening W, so theta parametrizes
    H_white and H = W H_white W. W is the identity when no Gram matrix is given.
    """
    reps = jnp.stack([jnp.einsum('a,aij->ij', theta[s], group.basis[i, :s.stop - s.start])
                      for i, s in enumerate(group.slices)])
    w = jnp.asarray(group.whiten)
    reps = jnp.einsum('oik,okl,olj->oij', w, reps, w)
    per_block = reps[group.orbit_of]
    return jnp.einsum('bik,bkl,bjl->bij', group.rot, per_block, group.rot)


def prepare(lay, theta):
    """H_r for every block, built once for a whole point set.

    theta -> H costs O(64 d^3) and does not depend on the point, so it must stay
    outside the loop over points. `lay.scale` enters here, so a tangent gets the
    same factor and the chain rule of the reparametrization is automatic.

    `PREPARE_DD` selects the form in double-double. H spans many orders of
    magnitude and `rot` is dense, so in FP64 a small entry of H_r can carry an
    absolute error of order eps * max abs(H).
    """
    if PREPARE_DD:
        return _prepare_rounded(lay, theta)
    return prepare_fp(lay, theta)


def prepare_fp(lay, theta):
    """`prepare` in plain FP64. H is linear in theta, so this is its tangent map too."""
    scaled = theta * jnp.asarray(lay.scale)
    return tuple(block_matrices(g, scaled) for g in lay.groups)


def _conjugate_dd(a, h):
    """a h a^T in double-double, for a stack of FP64 matrices a and dd matrices h."""
    t = dd.dot_dd_fp((h[0][:, :, None, :], h[1][:, :, None, :]),
                     a[:, None, :, :], axis=-1)                  # h a^T
    t_t = (jnp.swapaxes(t[0], 1, 2)[:, None, :, :],
           jnp.swapaxes(t[1], 1, 2)[:, None, :, :])
    return dd.dot_dd_fp(t_t, a[:, :, None, :], axis=-1)         # a (h a^T)


def _block_matrices_dd(group, scaled, whitened):
    """`block_matrices` in double-double, from theta * scale as a dd pair."""
    basis = jnp.asarray(group.basis)
    his, los = [], []
    for i, sl in enumerate(group.slices):
        terms = jnp.moveaxis(basis[i, :sl.stop - sl.start], 0, -1)   # (d, d, n)
        hi, lo = dd.dot_dd_fp((scaled[0][sl], scaled[1][sl]), terms, axis=-1)
        his.append(hi)
        los.append(lo)
    reps = (jnp.stack(his), jnp.stack(los))
    if whitened:
        # W is symmetric, so W H W is W H W^T.
        reps = _conjugate_dd(jnp.asarray(group.whiten), reps)
    per_block = (reps[0][group.orbit_of], reps[1][group.orbit_of])
    return _conjugate_dd(jnp.asarray(group.rot), per_block)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _prepare_rounded(lay, theta):
    """`prepare` in double-double, with each entry of each H_r rounded once."""
    scale = jnp.asarray(lay.scale, dtype=theta.dtype)
    scaled = dd.mul(dd.to_dd(theta), dd.to_dd(scale))       # exact product
    whitened = bool(lay.whiten_inv)
    return tuple(dd.to_fp(_block_matrices_dd(g, scaled, whitened))
                 for g in lay.groups)


@_prepare_rounded.defjvp
def _prepare_rounded_jvp(lay, primals, tangents):
    # H is linear in theta, so the tangent is the FP64 map of the tangent.
    # `dd.split` is a bitcast with no derivative, so autodiff must not see it.
    (theta,), (dtheta,) = primals, tangents
    return _prepare_rounded(lay, theta), prepare_fp(lay, dtheta)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def prepare_parts(lay, theta):
    """H_r in double-double, as one (hi, lo) pair for each group.

    hi is the rounded form that `PREPARE_DD` gives, and lo is what that rounding
    leaves. The rounding alone moves f by 4e-13 at the noisiest points of the
    stall at k = 28, so the per-point path can carry lo into its dd products.
    """
    scale = jnp.asarray(lay.scale, dtype=theta.dtype)
    scaled = dd.mul(dd.to_dd(theta), dd.to_dd(scale))
    whitened = bool(lay.whiten_inv)
    out = []
    for g in lay.groups:
        # Renormalize, so that the high part is the rounded H. `dd.two_sum`
        # holds barriers; XLA can fold the plain form (hi - (hi + lo)) + lo to 0.
        out.append(dd.two_sum(*_block_matrices_dd(g, scaled, whitened)))
    return tuple(out)


@prepare_parts.defjvp
def _prepare_parts_jvp(lay, primals, tangents):
    # The whole tangent goes to the high part, and the low part has none.
    (theta,), (dtheta,) = primals, tangents
    tan = prepare_fp(lay, dtheta)
    return prepare_parts(lay, theta), tuple((t, jnp.zeros_like(t)) for t in tan)


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
    for oi, (orb, sl) in enumerate(zip(lay.structure.orbits,
                                       lay.structure.param_slices())):
        frame = lay.structure.frames[orb.m]
        # sum_a A_a = k exactly, so the multinomial is an integer. Keep it in
        # Python ints: np.prod wraps int64 and gives 3.049e18 for the 5.839e19
        # of A = (20, 4, 0, 0), which makes p wrong by 95 at k = 28.
        diag = np.array([
            float(factorial(k) // prod(factorial(int(r + 4 * c))
                                       for r, c in zip(orb.rep, e)))
            for e in blocks.monomial_exponents(orb.m)])
        h_rep = frame.T @ np.diag(diag) @ frame
        if lay.whiten_inv:
            # theta parametrizes H_white with H = W H_white W, so undo W here.
            inv = lay.whiten_inv[oi]
            h_rep = inv @ h_rep @ inv
        coeffs = np.einsum('aij,ij->a', orb.basis, h_rep)
        rebuilt = np.einsum('a,aij->ij', coeffs, orb.basis)
        assert np.abs(rebuilt - h_rep).max() < 1e-9 * max(np.abs(h_rep).max(), 1.0)
        theta[sl] = coeffs
    return jnp.asarray(theta / lay.scale)


def batched_log_potential(lay, theta, points):
    """log p at every row of `points`."""
    return jax.vmap(lambda z: log_potential(lay, theta, z))(points)
