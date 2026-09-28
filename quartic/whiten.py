"""L^2(X) whitening of the section basis, one matrix for each S_4 orbit.

`blocks.section_frame` is orthonormal in the coefficients of the monomials, and
not in L^2(X). The Gram matrix of the frame sections on X therefore has a
condition number of 1.2e2, 2.2e3 and 1.2e4 at k = 12, 16 and 20, and it grows by
1.9 for each unit of k. That conditioning enters the fit twice, because h is
quadratic in the sections.

The cure is a change of coordinates that moves no critical point: put
`W = G^-1/2` and parametrize `H = W H_white W`. Three facts make it cheap and
exact:

  * `G` for any block of an orbit is `rot G_rep rot^T`, so one matrix for each
    orbit is enough,
  * `G_rep` commutes with rho(s) for s in the stabilizer, so W does too, and the
    invariant subspace is preserved exactly,
  * `H` is built once for each fit step, outside the loop over points, so the
    whitening costs nothing for each point.

`G` comes from a point sample. A sampled Gram is enough, because the error of a
sum of positive rank-one terms is multiplicative and not additive: measured
across independent samples, the whitened condition number is 1.07 to 1.19 at
k = 12 to 20, against 1.2e2 to 1.2e4 before.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import blocks

__all__ = ['gram_matrices', 'whitening_matrices', 'whitened_layout', 'save_whitening',
           'load_layout',
           'GRAM_SEED', 'GRAM_PAIRS']

# The sample that defines the convention. Report both with h.
GRAM_SEED = 20260901
GRAM_PAIRS = 50_000


def _orbit_blocks(lay):
    """For each global orbit, the group and the index of its representative block.

    `blocks.build` gives the representative block the identity rotation, so it
    is the one whose residue equals `orbit.rep`.
    """
    out = {}
    for gi, g in enumerate(lay.groups):
        for bi in range(g.residues.shape[0]):
            r = tuple(int(x) for x in np.asarray(g.residues[bi]))
            for oi, orb in enumerate(lay.structure.orbits):
                if orb.rep == r:
                    out[oi] = (gi, bi)
    assert len(out) == len(lay.structure.orbits)
    return out


def gram_matrices(lay, batch, chunk=20_000):
    """Real symmetric L^2(X) Gram of the frame sections, one for each orbit.

    The inner product is `int_X |z^r|^2 u^C conj(u^C') / |z|^(2k) dmu`, with the
    sample weights of `batch` for dmu. The result is symmetrized over the
    stabilizer, so it commutes with rho(s) to round-off and not to sample noise.
    """
    z = jnp.asarray(batch['points'])
    mass = jnp.asarray(batch['mass'])
    weight = mass / jnp.sum(mass)
    where = _orbit_blocks(lay)

    grams = {}
    for oi, (gi, bi) in where.items():
        g = lay.groups[gi]
        exps = jnp.asarray(g.exps)
        frame = jnp.asarray(g.frame)
        r = jnp.asarray(g.residues[bi])
        d = frame.shape[1]

        total = jnp.zeros((d, d))
        for start in range(0, z.shape[0], chunk):
            zc = z[start:start + chunk]
            wc = weight[start:start + chunk]
            u = zc ** 4
            abs2 = jnp.real(zc) ** 2 + jnp.imag(zc) ** 2
            # integer exponents only; a float exponent goes through exp and log
            mono = jnp.prod(u[:, None, :] ** exps[None, :, :], axis=2)
            psi = mono @ frame
            wt = wc * jnp.prod(abs2 ** r[None, :], axis=1) / jnp.sum(
                abs2, axis=1) ** lay.k
            total = total + jnp.real((jnp.conj(psi) * wt[:, None]).T @ psi)

        # Sample noise breaks the commutation with the stabilizer by about 1
        # percent. Averaging over the stabilizer restores it exactly, so the
        # invariant subspace survives the whitening.
        orb = lay.structure.orbits[oi]
        frame_np = np.asarray(g.frame)
        acc = np.zeros((d, d))
        for s in orb.stabilizer:
            rs = blocks.rho(g.m, s, frame_np)
            acc += rs @ np.asarray(total) @ rs.T
        grams[oi] = acc / len(orb.stabilizer)

    return tuple(grams[oi] for oi in range(len(lay.structure.orbits)))


def whitening_matrices(grams):
    """`(W, W_inv)` for each orbit, with `W = G^-1/2` and `W_inv = G^1/2`.

    Both are real symmetric, so they commute with the stabilizer exactly when G
    does, and `H = W H_white W` stays in the invariant subspace.
    """
    w, w_inv = [], []
    for gram in grams:
        gram = 0.5 * (gram + gram.T)
        ev, vec = np.linalg.eigh(gram)
        if ev.min() <= 0:
            raise ValueError(f'the Gram matrix is not positive definite: '
                             f'smallest eigenvalue {ev.min():.3e}')
        w.append(vec @ np.diag(ev ** -0.5) @ vec.T)
        w_inv.append(vec @ np.diag(ev ** 0.5) @ vec.T)
    return tuple(w), tuple(w_inv)


def whitened_layout(k, n_pairs=GRAM_PAIRS, seed=GRAM_SEED, batch=None,
                    chunk=20_000):
    """Layout at degree k with the section basis whitened in L^2(X).

    The Gram sample is part of the convention for h, so `gram_id` records it and
    `Layout.fingerprint` covers the whitening matrices themselves.
    """
    from . import data, sections
    plain = sections.layout(k, normalize=False)
    if batch is None:
        batch = data.make(n_pairs, seed)
        tag = f'seed{seed}-pairs{n_pairs}'
    else:
        tag = f'given-{int(batch["points"].shape[0])}pts'
    grams = gram_matrices(plain, batch, chunk=chunk)
    return sections.layout(k, struct=plain.structure,
                           whiten=whitening_matrices(grams), gram_id=tag)


def save_whitening(lay, path):
    """Save W and W^-1 of a whitened layout. W moves between processes by approximately
    2.5e-9 at k = 32, because the GPU Gram carries noise of 1e-16 and cond(G) is 4e7."""
    arrays = {f'w{i}': np.asarray(m) for i, m in enumerate(lay.whiten_fwd)}
    arrays.update({f'winv{i}': np.asarray(m) for i, m in enumerate(lay.whiten_inv)})
    np.savez(path, gram_id=np.array(lay.gram_id), **arrays)


def load_layout(k, path):
    """The whitened layout at degree k with the W and W^-1 of `save_whitening`."""
    from . import blocks, sections
    data = np.load(path)
    n = len([key for key in data.files if key.startswith('winv')])
    w = tuple(data[f'w{i}'] for i in range(n))
    w_inv = tuple(data[f'winv{i}'] for i in range(n))
    return sections.layout(k, struct=blocks.build(k), whiten=(w, w_inv),
                           gram_id=str(data['gram_id']))
