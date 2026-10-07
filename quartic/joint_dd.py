"""One orthonormal basis of several correction levels, built and applied in dd.

The functions of two levels with the weights p0^(-k1/k0) and p0^(-k2/k0) almost
overlap on X, so their Hessians side by side give a J that FP64 cannot resolve.
A pivoted Gram-Schmidt in dd on a build sample, level after level, makes one
orthonormal set: joint function t is (G_sel(t) - sum over s < t of H_st Gt_s) /
H_tt for one function G_sel(t) of a level. At a point, the Hessians of the joint
functions come in blocks from those of the level functions, as the blocks of a
degree in `arnoldi_dd.joint_jets_dd`: the part of the earlier joint functions is
one `ozaki` product, and the block is the rest times U^-1 from mpmath.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from . import arnoldi_dd, dd, ozaki

__all__ = ['JointMap', 'build', 'transform', 'save', 'load', 'effective_weights']


@dataclass(frozen=True)
class JointMap:
    sel: np.ndarray       # (n,) the index of G_sel(t) among the level functions in order
    level: np.ndarray     # (n,) the level of G_sel(t)
    h: np.ndarray         # (n, n) H, upper triangular, with its low part in h_lo
    h_lo: np.ndarray
    blocks: tuple         # (a, b) ranges of joint functions
    uinv: tuple           # (hi, lo) of U^-1 for the block H[a:b, a:b]
    ratio: np.ndarray     # (n,) H_tt over the norm of G_sel(t): the new part of each pick
    gamma: tuple = ()     # (hi, lo) of <Gt_t, 1> on the build sample: psi_N = sum gamma_t Gt_t
    gap: float = 0.0      # the norm of psi_N - 1 over the norm of 1 on the build sample


def build(values, mu, tol=1e-15, block=256, verbose=False):
    """The joint map of levels whose dd values on a build sample are `values`.

    `values` is a list of dd pairs (N, n_l), one for each level in order, and `mu`
    the FP64 measure of the build points. The first level comes first, so every
    prefix of the levels spans the same functions as those levels alone. A
    function whose new part is below `tol` of its norm does not enter.
    """
    n_pts = values[0][0].shape[0]
    room = sum(v[0].shape[1] for v in values)
    cols = (jnp.zeros((n_pts, room)), jnp.zeros((n_pts, room)))
    weight = jnp.asarray(mu)
    ncur, offset = 0, 0
    sel, level, vecs, norms, ratios = [], [], [], [], []
    start = time.perf_counter()
    for lev, cand in enumerate(values):
        size = jnp.sqrt(jnp.sum(weight[:, None] * cand[0] ** 2, axis=0))
        picks, cols, ncur = arnoldi_dd._pivoted_dd(cols, ncur, weight, cand, size,
                                                   cand[0].shape[1], tol)
        for idx, vec, norm, ratio in picks:
            sel.append(offset + idx)
            level.append(lev)
            vecs.append(vec)
            norms.append(norm)
            ratios.append(ratio)
        if verbose:
            got = np.array([r for r in ratios[-len(picks):]]) if picks else np.zeros(0)
            hist = {f'1e-{e}': int(np.sum((got < 10.0 ** -e) & (got >= 10.0 ** -(e + 1))))
                    for e in range(0, 32)}
            print(f'    level {lev}: {len(picks)} of {cand[0].shape[1]} functions enter, '
                  f'{time.perf_counter() - start:.0f}s; new parts by decade '
                  f'{ {k: v for k, v in hist.items() if v} }', flush=True)
        offset += cand[0].shape[1]
    n = ncur
    h, h_lo = np.zeros((n, n)), np.zeros((n, n))
    for t in range(n):
        h[:t, t], h_lo[:t, t] = vecs[t][0][:t], vecs[t][1][:t]
        h[t, t], h_lo[t, t] = norms[t]
    blocks = tuple((a, min(a + block, n)) for a in range(0, n, block))
    uinv = tuple(arnoldi_dd.upper_inverse(h, h_lo, a, b) for a, b in blocks)
    # The joint functions are orthonormal in dd on the build sample, so the projection
    # of 1 onto them is the inner products, and psi_N - 1 is orthogonal to all of them.
    # The inner products run over blocks of the built columns, so that no temporary has
    # the size of all the room.
    pieces = []
    for a in range(0, n, arnoldi_dd.COL_BLOCK):
        b = min(a + arnoldi_dd.COL_BLOCK, n)
        piece = dd.dot_dd_fp((cols[0][:, a:b], cols[1][:, a:b]), weight[:, None], axis=0)
        pieces.append((np.asarray(piece[0]), np.asarray(piece[1])))
    gamma = (np.concatenate([x[0] for x in pieces]), np.concatenate([x[1] for x in pieces]))
    proj = (jnp.asarray(gamma[0]), jnp.asarray(gamma[1]))
    sq = dd.sum_dd(*dd.mul(proj, proj))
    rest = dd.add(dd.sum_fp(weight), dd.neg(sq))
    gap = float(np.sqrt(max(float(dd.to_fp(rest)), 0.0) / float(jnp.sum(weight))))
    return JointMap(sel=np.array(sel), level=np.array(level), h=h, h_lo=h_lo,
                    blocks=blocks, uinv=uinv, ratio=np.array(ratios), gamma=gamma, gap=gap)


def save(path, jm):
    """The arrays of a joint map, so that another process can use the same map."""
    sizes = np.array([b - a for a, b in jm.blocks])
    np.savez(path, sel=jm.sel, level=jm.level, h=jm.h, h_lo=jm.h_lo,
             blocks=np.array(jm.blocks), ratio=jm.ratio, gamma_hi=jm.gamma[0],
             gamma_lo=jm.gamma[1], gap=jm.gap,
             uinv_hi=np.concatenate([u[0].ravel() for u in jm.uinv]),
             uinv_lo=np.concatenate([u[1].ravel() for u in jm.uinv]), sizes=sizes)


def load(path):
    """A joint map from `save`."""
    arr = np.load(path)
    sizes = arr['sizes']
    ends = np.cumsum(sizes ** 2)
    starts = ends - sizes ** 2
    uinv = tuple((arr['uinv_hi'][a:b].reshape(m, m), arr['uinv_lo'][a:b].reshape(m, m))
                 for a, b, m in zip(starts, ends, sizes))
    return JointMap(sel=arr['sel'], level=arr['level'], h=arr['h'], h_lo=arr['h_lo'],
                    blocks=tuple((int(a), int(b)) for a, b in arr['blocks']), uinv=uinv,
                    ratio=arr['ratio'], gamma=(arr['gamma_hi'], arr['gamma_lo']),
                    gap=float(arr['gap']))


def _flat(p):
    """A complex dd array (n, 2, 2) as a dd pair of 8 real rows (8, n)."""
    parts = [p[c][i] for c in range(2) for i in range(2)]                # re hi, re lo, ...
    rows = [jnp.stack([x[:, a, b] for a in range(2) for b in range(2)]) for x in parts]
    return (jnp.concatenate([rows[0], rows[2]]), jnp.concatenate([rows[1], rows[3]]))


def _unflat(rows, n):
    """The inverse of `_flat`."""
    def back(x):
        return jnp.stack([jnp.stack([x[2 * a + b] for b in range(2)], axis=-1)
                          for a in range(2)], axis=-2)
    hi, lo = rows
    return (back(hi[:4]), back(lo[:4])), (back(hi[4:]), back(lo[4:]))


def transform(p, jm):
    """The Hessians of the joint functions, complex dd (n, 2, 2), at one point.

    `p` holds the Hessians of the level functions of every level in order, complex
    dd (n_total, 2, 2), and `jm` comes from `build`. One loop body serves every
    block: the earlier joint functions enter through all of H[:, block], because
    H is upper triangular and the entries of the later functions are still zero.
    """
    rows = _flat(p)
    n = len(jm.sel)
    width = jm.blocks[0][1] - jm.blocks[0][0]
    count = len(jm.blocks)
    size = count * width
    pad = size - n
    sel = jtu.tree_map(lambda x: jnp.pad(x[:, jnp.asarray(jm.sel)], ((0, 0), (0, pad))), rows)
    h = tuple(jnp.pad(jnp.asarray(x), ((0, pad), (0, pad))) for x in (jm.h, jm.h_lo))

    def stacked(i):
        return jnp.stack([jnp.pad(jnp.asarray(u[i]), ((0, width - u[i].shape[0]),
                                                       (0, width - u[i].shape[1])))
                          for u in jm.uinv])
    uinv = (stacked(0), stacked(1))
    done = (jnp.zeros((8, size)), jnp.zeros((8, size)))

    def body(i, done):
        a = i * width
        rest = tuple(jax.lax.dynamic_slice(x, (0, a), (8, width)) for x in sel)
        cols = tuple(jax.lax.dynamic_slice(x, (0, a), (size, width)) for x in h)
        rest = dd.add(rest, dd.neg(ozaki.matmul(done, cols)))
        piece = ozaki.matmul(rest, (uinv[0][i], uinv[1][i]))
        return tuple(jax.lax.dynamic_update_slice(d, x, (0, a)) for d, x in zip(done, piece))
    done = jax.lax.fori_loop(0, count, body, done)
    return _unflat(tuple(x[:, :n] for x in done), n)


def effective_weights(jm, w, sizes):
    """Weights on the level functions whose sum equals sum_t w_t Gt_t, in dd.

    The joint functions are G_sel H^-1, because G_sel(t) = sum over s <= t of
    H_st Gt_s with H upper triangular. So sum_t w_t Gt_t = sum_t what_t G_sel(t)
    with what = H^-1 w, by back substitution over the blocks of `jm`. `w` is a dd
    pair over the joint functions, and `sizes` are the numbers of functions of each
    level. The result is one dd pair for each level, zero at the functions that do
    not enter.
    """
    n = len(jm.sel)
    h = (jnp.asarray(jm.h), jnp.asarray(jm.h_lo))
    out = (jnp.zeros(n), jnp.zeros(n))
    w = (jnp.asarray(w[0]), jnp.asarray(w[1]))
    for (a, b), (u_hi, u_lo) in reversed(list(zip(jm.blocks, jm.uinv))):
        rhs = (w[0][a:b, None], w[1][a:b, None])
        if b < n:
            later = ozaki.matmul((h[0][a:b, b:], h[1][a:b, b:]), (out[0][b:, None],
                                                                   out[1][b:, None]))
            rhs = dd.add(rhs, dd.neg(later))
        piece = ozaki.matmul((jnp.asarray(u_hi), jnp.asarray(u_lo)), rhs)
        out = (out[0].at[a:b].set(piece[0][:, 0]), out[1].at[a:b].set(piece[1][:, 0]))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    hi, lo = np.asarray(out[0]), np.asarray(out[1])
    levels = []
    for i, size in enumerate(sizes):
        take = np.nonzero(jm.level == i)[0]
        lh, ll = np.zeros(size), np.zeros(size)
        lh[jm.sel[take] - offsets[i]] = hi[take]
        ll[jm.sel[take] - offsets[i]] = lo[take]
        levels.append((lh, ll))
    return levels
