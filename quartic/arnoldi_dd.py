"""The joint V+A basis of a level, built and evaluated in dd.

In FP64 the error of the recurrence grows by a factor of 2.8 to 3.2 for each
degree, so the functions of degree 28 and more are noise, and the coefficients h
that a build in FP64 finds there are noise too. Here the build and the
evaluation run in dd, with the matrix products of `ozaki`. The multipliers come
in dd from their monomials, and every function is a jet (q, D_k q, D_k Dbar_l q)
along the rows of the restriction matrix, as in `vpa_dd`.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from . import analytic, arnoldi, dd, ozaki, vpa_dd

__all__ = ['multiplier_values_dd', 'build_joint_dd', 'upper_inverse', 'block_inverses',
           'joint_jets_dd', 'joint_hessians_dd']


def _sqrt_dd(x):
    """sqrt of a positive dd value, by one Newton step on the FP64 root."""
    s = dd.to_dd(jnp.sqrt(x[0]))
    corr = dd.mul(dd.add(x, dd.neg(dd.mul(s, s))), dd.to_dd(0.5 / s[0]))
    return dd.add(s, corr)


def multiplier_values_dd(entries, z, chunk=500):
    """Re(sum over pi of z_pi^A conj(z_pi)^B) / |z|^(2d) in dd at FP64 points z (N, 4).

    Returns a dd pair of shape (N, E), in blocks of `chunk` points.
    """
    def one(p):
        return jtu.tree_map(lambda a: a[0], _multiplier_values_dd(entries, p[None]))
    return jax.jit(lambda x: jax.lax.map(one, x, batch_size=chunk))(jnp.asarray(z))


def _multiplier_values_dd(entries, z):
    """`multiplier_values_dd` for one block of points, as one array program."""
    a_exp = np.stack([np.asarray(a) for _, a, _ in entries])
    b_exp = np.stack([np.asarray(b) for _, _, b in entries])
    degs = np.array([d for d, _, _ in entries])
    n_e, n_p = len(entries), arnoldi.PERMS.shape[0]
    ea = np.zeros((n_e, n_p, 4), np.int64)
    eb = np.zeros((n_e, n_p, 4), np.int64)
    for p, perm in enumerate(arnoldi.PERMS):
        ea[:, p, perm] = a_exp
        eb[:, p, perm] = b_exp
    top = int(max(ea.max(), eb.max()))
    z_dd = dd.cto_dd(z)                                                   # (N, 4)
    pw = [dd.cto_dd(jnp.ones(z.shape, jnp.complex128))]
    for _ in range(top):
        pw.append(dd.cmul(pw[-1], z_dd))
    pw = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=1), *pw)            # (N, top+1, 4)
    cols = jnp.arange(4)

    def monomial(exps):
        part = jtu.tree_map(lambda a: a[:, jnp.asarray(exps), cols], pw)  # (N, E, P, 4)
        out = analytic._cindex(part, Ellipsis, 0)
        for c in range(1, 4):
            out = dd.cmul(out, analytic._cindex(part, Ellipsis, c))
        return out
    m = dd.cmul(monomial(ea), (lambda v: (v[0], dd.neg(v[1])))(monomial(eb)))
    s = dd.sum_dd(*m[0], axis=-1)                                         # (N, E), real
    r2 = dd.add(dd.sum_dd(*dd.mul(z_dd[0], z_dd[0]), axis=-1),
                dd.sum_dd(*dd.mul(z_dd[1], z_dd[1]), axis=-1))     # (N,)
    inv = dd.reciprocal(r2)
    pows = [dd.to_dd(jnp.ones_like(r2[0]))]
    for _ in range(int(degs.max())):
        pows.append(dd.mul(pows[-1], inv))
    stack = (jnp.stack([p[0] for p in pows], axis=1), jnp.stack([p[1] for p in pows], axis=1))
    rho = jtu.tree_map(lambda a: a[:, jnp.asarray(degs)], stack)          # (N, E)
    return dd.mul(s, rho)


# The projections on the columns built so far run over blocks of this many columns,
# and only over the blocks that hold a column: the Ozaki slices of all the room of a
# joint build of 8 levels of degree up to 40 on 60,000 points need 40 GB.
COL_BLOCK = 2048


def _blocks_for(ncur):
    """The number of column blocks that hold the ncur columns built so far."""
    return max(1, -(-int(ncur) // COL_BLOCK))


def _cols_t_matmul(cols, b, nb):
    """cols^T @ b over the first nb column blocks, with zeros for the rest of the room."""
    room = cols[0].shape[1]
    parts = []
    for c in range(nb):
        blk = jtu.tree_map(lambda a: a[:, c * COL_BLOCK:(c + 1) * COL_BLOCK].T, cols)
        parts.append(ozaki.matmul(blk, b))
    top = jtu.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)
    pad = room - top[0].shape[0]
    return jtu.tree_map(lambda a: jnp.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1)), top)


def _cols_matmul(cols, x, nb):
    """cols @ x over the first nb column blocks, where the later columns are zero."""
    acc = None
    for c in range(nb):
        sl = slice(c * COL_BLOCK, (c + 1) * COL_BLOCK)
        piece = ozaki.matmul(jtu.tree_map(lambda a: a[:, sl], cols),
                             jtu.tree_map(lambda a: a[sl], x))
        acc = piece if acc is None else dd.add(acc, piece)
    return acc


@partial(jax.jit, static_argnames=('nb',))
def _pick_dd(cols, coef, cand, weight, alive, idx, ncur, nb=None):
    """One step of the pivoted Gram-Schmidt in dd, on arrays of fixed shape.

    `nb` column blocks of `cols` hold the columns built so far.
    """
    take = lambda tree, i, ax: jtu.tree_map(lambda a: jnp.take(a, i, axis=ax), tree)
    c = take(cand, idx, 1)                                                # (N,)
    wc = dd.mul_dd_fp(c, weight)
    extra = _cols_t_matmul(cols, jtu.tree_map(lambda a: a[:, None], wc), nb)
    extra = jtu.tree_map(lambda a: a[:, 0], extra)                        # (room,)
    proj_old = _cols_matmul(cols, jtu.tree_map(lambda a: a[:, None], extra), nb)
    col = dd.add(c, dd.neg(jtu.tree_map(lambda a: a[:, 0], proj_old)))
    norm = _sqrt_dd(dd.dot_dd_fp(dd.mul(col, col), weight, axis=0))
    inv = dd.reciprocal(norm)
    newcol = dd.mul(col, inv)
    vec = dd.add(take(coef, idx, 1), extra)
    cols = jtu.tree_map(lambda a, x: jax.lax.dynamic_update_slice(a, x[:, None], (0, ncur)),
                        cols, newcol)
    wn = dd.mul_dd_fp(newcol, weight)
    proj = ozaki.matmul(jtu.tree_map(lambda a: a[None, :], wn), cand)     # (1, C)
    proj = jtu.tree_map(lambda a: jnp.where(alive, a[0], 0.0), proj)
    outer = dd.mul(jtu.tree_map(lambda a: a[:, None], newcol),
                          jtu.tree_map(lambda a: a[None, :], proj))
    cand = dd.add(cand, dd.neg(outer))
    coef = jtu.tree_map(lambda a, x: jax.lax.dynamic_update_slice(a, x[None], (ncur, 0)),
                        coef, proj)
    rest = jnp.sqrt(jnp.sum(weight[:, None] * cand[0] ** 2, axis=0))
    return cols, coef, cand, vec, norm, rest


def _pivoted_dd(cols, ncur, weight, cand, size, budget, tol):
    """`arnoldi._pivoted` in dd: cols, cand and the coefficients are dd pairs."""
    w = weight[:, None]
    nb = _blocks_for(ncur)
    coef = _cols_t_matmul(cols, dd.mul_dd_fp(cand, w), nb)
    cand = dd.add(cand, dd.neg(_cols_matmul(cols, coef, nb)))
    again = _cols_t_matmul(cols, dd.mul_dd_fp(cand, w), nb)
    cand = dd.add(cand, dd.neg(_cols_matmul(cols, again, nb)))
    coef = dd.add(coef, again)
    size_np = np.asarray(size)
    rest = jnp.sqrt(jnp.sum(w * cand[0] ** 2, axis=0))
    alive, picks = np.ones(cand[0].shape[1], bool), []
    while alive.any() and len(picks) < budget:
        rest_np = np.asarray(rest)
        idx = int(np.argmax(np.where(alive, rest_np / size_np, -np.inf)))
        if rest_np[idx] <= tol * size_np[idx]:
            break
        alive[idx] = False
        cols, coef, cand, vec, norm, rest = _pick_dd(cols, coef, cand, weight,
                                                     jnp.asarray(alive), idx, ncur,
                                                     nb=_blocks_for(ncur))
        vec = tuple(np.asarray(v)[:ncur] for v in vec)
        norm = tuple(float(v) for v in norm)
        picks.append((idx, vec, norm, norm[0] / size_np[idx]))
        ncur += 1
    return picks, cols, ncur


def build_joint_dd(k, z, scale, mu, d_max=12, tol=1e-9, n_by_degree=None, verbose=False,
                   center=True, entries=None):
    """`arnoldi.build_joint` in dd, with no seeds. h comes as h + h_lo.

    The products must reach the HN count at every degree, so take d_max large
    enough; 10 or more reached it at k = 24 with no seed. `entries` limits the
    multipliers, for example to those that a build in FP64 uses: the candidates in
    dd at degree 36 with the whole dictionary of degree 12 need more than 100 GB.
    """
    entries = arnoldi.dictionary(d_max) if entries is None else list(entries)
    z = jnp.asarray(z)
    mvals = multiplier_values_dd(entries, z)                              # (N, E) dd
    weight = jnp.asarray(mu) * jnp.asarray(scale) ** 2
    shift = scale_y = None
    if center:
        total = jnp.sum(weight)
        mean = (weight @ mvals[0]) / total
        std = jnp.sqrt((weight @ (mvals[0] - mean) ** 2) / total)
        keep = np.nonzero(np.asarray(std) > 1e-12 * np.asarray(jnp.abs(mean)).max())[0]
        entries = [entries[i] for i in keep]
        shift, scale_y = np.asarray(mean)[keep], np.asarray(std)[keep]
        inv = dd.reciprocal(dd.to_dd(jnp.asarray(scale_y)))
        mvals = jtu.tree_map(lambda a: a[:, keep], mvals)
        mvals = dd.mul(dd.add(mvals, dd.to_dd(-jnp.broadcast_to(
            jnp.asarray(shift), mvals[0].shape))), jtu.tree_map(lambda a: a[None], inv))
    degs = np.array([d for d, _, _ in entries])
    norm0 = float(jnp.sqrt(jnp.sum(weight)))
    room = 1 if n_by_degree is None else n_by_degree[k]
    first = dd.reciprocal(dd.to_dd(jnp.full(z.shape[0], norm0)))
    cols = tuple(jnp.zeros((z.shape[0], room)).at[:, 0].set(x) for x in first)
    ncur = 1
    parents, mults, degrees = [-1], [-1], [0]
    coefs, norms = [(np.zeros(0), np.zeros(0))], [(norm0, 0.0)]
    trace, start = [], time.perf_counter()
    for target in range(1, k + 1):
        want = None if n_by_degree is None else n_by_degree[target]
        before = ncur
        pairs = [(int(e), t) for e in np.nonzero(degs <= target)[0]
                 for t in range(len(degrees)) if degrees[t] == target - degs[e]]
        worst = 1.0
        if pairs:
            es, ts = (jnp.asarray([pair[i] for pair in pairs]) for i in (0, 1))
            cand = dd.mul(jtu.tree_map(lambda a: a[:, es], mvals),
                                 jtu.tree_map(lambda a: a[:, ts], cols))
            size = jnp.sqrt(jnp.sum(weight[:, None] * cand[0] ** 2, axis=0))
            budget = len(pairs) if want is None else max(want - ncur, 0)
            picks, cols, ncur = _pivoted_dd(cols, ncur, weight, cand, size, budget, tol)
            for idx, coef, norm, ratio in picks:
                parents.append(pairs[idx][1])
                mults.append(pairs[idx][0])
                degrees.append(target)
                coefs.append(coef)
                norms.append(norm)
                worst = min(worst, ratio)
        trace.append({'degree': target, 'want': want, 'dim': int(ncur),
                      'from_products': int(ncur - before), 'worst_product_ratio': worst,
                      'candidates': len(pairs), 'seconds': time.perf_counter() - start})
        if verbose:
            print(f"    degree {target}: {ncur} of {want}, {len(pairs)} candidates, worst "
                  f"ratio {worst:.1e}, {trace[-1]['seconds']:.0f}s", flush=True)
        if want is not None and ncur < want:
            raise ValueError(f'the products reach {ncur} of {want} at degree {target}; '
                             'take a larger d_max')
    n = ncur
    h, h_lo = np.zeros((n, n)), np.zeros((n, n))
    for t in range(n):
        h[:t, t], h_lo[:t, t] = coefs[t]
        h[t, t], h_lo[t, t] = norms[t]
    used = sorted(set(m for m in mults if m >= 0))
    remap = {m: i for i, m in enumerate(used)}
    basis = arnoldi.JointBasis(
        k=int(k), entries=tuple(entries[m] for m in used), parents=np.array(parents),
        mults=np.array([remap[m] if m >= 0 else -1 for m in mults]),
        degrees=np.array(degrees), h=h, alpha=0.0, seed_slot=-np.ones(n, np.int64),
        seeds=(), layouts=(), shift=None if shift is None else shift[used],
        scale=None if scale_y is None else scale_y[used])
    return basis, h_lo, trace


def upper_inverse(h, h_lo, a, b, dps=40):
    """U^-1 in dd for the upper triangular block U = (h + h_lo)[a:b, a:b], by mpmath."""
    import mpmath as mp
    mp.mp.dps = dps
    u = [[mp.mpf(float(h[i, k])) + mp.mpf(float(h_lo[i, k])) for k in range(a, b)]
         for i in range(a, b)]
    m = b - a
    x = [[mp.mpf(0)] * m for _ in range(m)]
    for col in range(m):
        x[col][col] = 1 / u[col][col]
        for row in range(col - 1, -1, -1):
            acc = mp.fsum(u[row][k] * x[k][col] for k in range(row + 1, col + 1))
            x[row][col] = -acc / u[row][row]
    hi = np.array([[float(v) for v in r] for r in x])
    lo = np.array([[float(v - mp.mpf(float(v))) for v in r] for r in x])
    return hi, lo


def block_inverses(basis, h_lo, dps=40):
    """U^-1 in dd for the upper triangular block U of h inside each degree, by mpmath.

    q_t h_tt + sum over s < t in the degree of h_st q_s is what remains of the
    candidate after the lower degrees, so the block of a degree is rest U^-1.
    """
    degrees = np.asarray(basis.degrees)
    out = []
    for j in sorted(set(degrees.tolist()) - {0}):
        idx = np.nonzero(degrees == j)[0]
        out.append(upper_inverse(basis.h, h_lo, int(idx[0]), int(idx[-1]) + 1, dps))
    return tuple(out)


def _flat(jets):
    """A jet of shape (m,) as 13 real dd rows (13, m)."""
    v, d, hh = jets
    rows = [v] + [jtu.tree_map(lambda a: a[:, c], part) for part in d for c in range(2)]
    rows += [jtu.tree_map(lambda a: a[:, c, e], part) for part in hh
             for c in range(2) for e in range(2)]
    return tuple(jnp.stack([r[i] for r in rows]) for i in range(2))


def _unflat(mat):
    """The inverse of `_flat`."""
    v = (mat[0][0], mat[1][0])
    d = tuple(tuple(jnp.stack([mat[i][1 + 2 * part + c] for c in range(2)], axis=-1)
                    for i in range(2)) for part in range(2))
    hh = tuple(tuple(jnp.stack([jnp.stack([mat[i][5 + 4 * part + 2 * c + e]
                                           for e in range(2)], axis=-1)
                                for c in range(2)], axis=-2)
                     for i in range(2)) for part in range(2))
    return v, d, hh


def _degree_blocks(degrees):
    """The ranges (a, b) of the functions of each degree above 0, in order."""
    out = []
    for j in sorted(set(degrees.tolist()) - {0}):
        idx = np.nonzero(degrees == j)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        assert np.all(idx == np.arange(a, b)), 'the functions of a degree must be contiguous'
        out.append((a, b))
    return out


def stack_inverses(basis, uinv):
    """The U^-1 of `block_inverses` as two arrays (blocks, width, width), with zeros."""
    sizes = [b - a for a, b in _degree_blocks(np.asarray(basis.degrees))]
    width = max(sizes)
    out = []
    for i in range(2):
        arr = np.zeros((len(sizes), width, width))
        for k, (m, u) in enumerate(zip(sizes, uinv)):
            arr[k, :m, :m] = u[i]
        out.append(arr)
    return tuple(out)


def joint_jets_dd(basis, h_lo, y_jets, uinv, groups=4, algebra=None):
    """The jets of every q_t in dd at one point, from the jets of the multipliers.

    The functions of each degree form one block: their products y q_parent come
    from lower degrees, the part of the lower degrees is one `ozaki` product, and
    the block is that rest times U^-1 from `block_inverses`, a second product. The
    blocks go in `groups` runs of one loop body each, padded to the widest block of
    the run, and the earlier functions enter through h[:L, block] with L the end of
    the run: h is upper triangular and the later functions are still zero. `uinv`
    is the tuple of `block_inverses` or the pair of `stack_inverses`. `algebra` sets
    the form of a jet, for example `tjets.T36`; the default is the 13-row jet of
    `vpa_dd.jet_mul`.
    """
    degrees = np.asarray(basis.degrees)
    n = len(degrees)
    blocks = _degree_blocks(degrees)
    starts = np.array([a for a, _ in blocks])
    sizes = np.array([b - a for a, b in blocks])
    top = int(sizes.max())
    size = n + top
    if isinstance(uinv[0], tuple):
        uinv = stack_inverses(basis, uinv)
    u_hi, u_lo = (jnp.asarray(x) for x in uinv)
    h = tuple(jnp.pad(jnp.asarray(x), ((0, top), (0, top))) for x in (basis.h, h_lo))
    inv_h = dd.reciprocal((h[0][:1, 0], h[1][:1, 0]))
    if algebra is None:
        flat, unflat, jmul, n_rows = _flat, _unflat, vpa_dd.jet_mul, 13
        zero_d = dd.cto_dd(jnp.zeros((1, 2), jnp.complex128))
        zero_h = dd.cto_dd(jnp.zeros((1, 2, 2), jnp.complex128))
        first = _flat((inv_h, zero_d, zero_h))
    else:
        flat, unflat, jmul, n_rows = algebra.flat, algebra.unflat, algebra.mul, algebra.rows
        first = flat(algebra.first(inv_h))
    done = tuple(jnp.zeros((n_rows, size)).at[:, :1].set(x) for x in first)
    take = lambda tree, i: jtu.tree_map(lambda x: x[i], tree)
    parents = np.maximum(np.asarray(basis.parents), 0)
    mults = np.maximum(np.asarray(basis.mults), 0)
    for run in np.array_split(np.arange(len(blocks)), min(groups, len(blocks))):
        width = int(sizes[run].max())
        span = int(starts[run[-1]]) + width
        idx = np.minimum(starts[run][:, None] + np.arange(width)[None], n - 1)
        valid = jnp.asarray((np.arange(width)[None] < sizes[run][:, None]).astype(np.float64))
        par, mul = jnp.asarray(parents[idx]), jnp.asarray(mults[idx])
        a_of, u_of = jnp.asarray(starts[run]), (u_hi[run, :width, :width],
                                                u_lo[run, :width, :width])
        cols_all = tuple(x[:span] for x in h)

        def body(k, part, width=width, span=span, valid=valid, par=par, mul=mul,
                 a_of=a_of, u_of=u_of, cols_all=cols_all):
            a = a_of[k]
            grown = jmul(take(y_jets, mul[k]), take(unflat(part), par[k]))
            cols = tuple(jax.lax.dynamic_slice(x, (0, a), (span, width)) for x in cols_all)
            rest = dd.add(flat(grown), dd.neg(ozaki.matmul(part, cols)))
            piece = ozaki.matmul(rest, (u_of[0][k], u_of[1][k]))
            return tuple(jax.lax.dynamic_update_slice(d, x * valid[k][None], (0, a))
                         for d, x in zip(part, piece))
        part = jax.lax.fori_loop(0, len(run), body, tuple(x[:, :span] for x in done))
        done = tuple(jax.lax.dynamic_update_slice(d, x, (0, 0)) for d, x in zip(done, part))
    return unflat(tuple(x[:, :n] for x in done))


def joint_jets_dd_unrolled(basis, h_lo, y_jets, uinv):
    """`joint_jets_dd` with one copy of the products for each degree, as a reference."""
    degrees = np.asarray(basis.degrees)
    h = (jnp.asarray(basis.h), jnp.asarray(h_lo))
    inv_h = dd.reciprocal((jnp.diag(h[0][:1, :1]), jnp.diag(h[1][:1, :1])))
    parents = np.maximum(np.asarray(basis.parents), 0)
    mults = np.maximum(np.asarray(basis.mults), 0)

    def flat(jets):
        """A jet of shape (m,) as 13 real dd rows (13, m)."""
        v, d, hh = jets
        rows = [v] + [jtu.tree_map(lambda a: a[:, c], part) for part in d for c in range(2)]
        rows += [jtu.tree_map(lambda a: a[:, c, e], part) for part in hh
                 for c in range(2) for e in range(2)]
        return tuple(jnp.stack([r[i] for r in rows]) for i in range(2))

    def unflat(mat):
        """The inverse of `flat`."""
        v = (mat[0][0], mat[1][0])
        d = tuple(tuple(jnp.stack([mat[i][1 + 2 * part + c] for c in range(2)], axis=-1)
                        for i in range(2)) for part in range(2))
        hh = tuple(tuple(jnp.stack([jnp.stack([mat[i][5 + 4 * part + 2 * c + e]
                                               for e in range(2)], axis=-1)
                                    for c in range(2)], axis=-2)
                         for i in range(2)) for part in range(2))
        return v, d, hh

    zero_d = dd.cto_dd(jnp.zeros((1, 2), jnp.complex128))
    zero_h = dd.cto_dd(jnp.zeros((1, 2, 2), jnp.complex128))
    blocks = [flat((inv_h, zero_d, zero_h))]
    for block, j in enumerate(sorted(set(degrees.tolist()) - {0})):
        idx = np.nonzero(degrees == j)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        assert np.all(idx == np.arange(a, b)), 'the functions of a degree must be contiguous'
        done = tuple(jnp.concatenate([blk[i] for blk in blocks], axis=1) for i in range(2))
        take = lambda tree, i: jtu.tree_map(lambda x: x[i], tree)
        grown = vpa_dd.jet_mul(take(y_jets, jnp.asarray(mults[a:b])),
                               take(unflat(done), jnp.asarray(parents[a:b])))
        lower = ozaki.matmul(done, (h[0][:a, a:b], h[1][:a, a:b]))         # (13, m)
        rest = dd.add(flat(grown), dd.neg(lower))
        blocks.append(ozaki.matmul(rest, tuple(jnp.asarray(x) for x in uinv[block])))
    done = tuple(jnp.concatenate([blk[i] for blk in blocks], axis=1) for i in range(2))
    return unflat(done)


def joint_hessians_dd(basis, h_lo, uinv, z_dd, rest, hi, alpha, weight):
    """R ddbar(W q_t) R^dag for every function of the level in complex dd, (n, 2, 2).

    `rest` is the dd restriction matrix, `hi` the dd state of the base and `weight`
    the value of W = w |z|^(2k), a factor common to every function, in FP64 or dd.
    """
    q_v, q_d, q_h = joint_jets_dd(basis, h_lo,
                                  vpa_dd.basis_multiplier_jets(basis, z_dd, rest), uinv)
    d_l, h_l = vpa_dd.weight_jets(hi, z_dd, rest, alpha, basis.k)
    d_lb = analytic._cindex(d_l, None)
    h_lb = analytic._cindex(h_l, None)
    lead = dd.cadd(dd.cadd(q_h, vpa_dd._outer(q_d, d_lb)), vpa_dd._outer(d_lb, q_d))
    rest_term = vpa_dd._creal(dd.cadd(h_lb, analytic._cindex(vpa_dd._outer(d_l, d_l), None)),
                              vpa_dd._ex(q_v, 0, 0))
    total = dd.cadd(lead, rest_term)
    if isinstance(weight, tuple):
        return vpa_dd._creal(total, weight)
    return analytic._cscale(total, weight)
