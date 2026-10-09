"""Correction levels on a trained level fit, with the weights of the trained potential.

The first stage is a fit of `vpa_dd_levels.py`: Phi_1 = log p0 + psi_1, with psi_1 a
sum of joint functions of its levels. A level of degree k1 of the second stage takes
the weight |z|^(2 k1) exp(-(k1/k0) Phi_1) in place of |z|^(2 k1) p0^(-k1/k0). Two
levels then differ by the factor exp(psi_1 / k0), so the joint basis of the second
stage also holds products of psi_1 with the level functions. psi_1, its gradient and
its Hessian come in dd at each point from the jets of the first-stage levels.

A constant C, the mean of psi_1 on the fit set, comes out of the weights: it scales
each level by a constant and leaves its span alone, and exp(-(k1/k0)(psi_1 - C))
then comes from a short series.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/vpa_dd_stage2.py --k0 12 \\
        --theta runs_explore/f2r/theta_k0_12.npy --stage1 runs_explore/s1 \\
        --stage1-k1 24 20 16 --k1 36 35 34 33 32 31 30 29 --load runs_explore/f2r \\
        --joint-tol 1e-15 --exact-mul --out runs_explore/s2
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
import jax.tree_util as jtu                                 # noqa: E402

import correction_fit as cf                                 # noqa: E402
import seam_levels as sl                                    # noqa: E402
import vpa_dd_fit as vdf                                    # noqa: E402
import vpa_dd_levels as vl                                  # noqa: E402
import vpa_fit as vf                                        # noqa: E402
from quartic import (analytic, arnoldi_dd, correction, dd, dd_metric,  # noqa: E402
                     joint_dd, ozaki, sections, vpa_dd)

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def flat13(jets):
    """The jets (v, d, h) of n real functions as a dd pair of 13 real rows (13, n)."""
    v, d, h = jets

    def rows(i):
        out = [v[i]]
        out += [d[c][i][..., k] for c in range(2) for k in range(2)]
        out += [h[c][i][..., a, b] for c in range(2) for a in range(2) for b in range(2)]
        return jnp.stack(out)
    return rows(0), rows(1)


def unflat13(rows):
    """The jets (v, d, h) from 13 rows of `flat13`, with the first axis removed."""
    def part(x, idx, shape):
        return jnp.stack([x[i] for i in idx], axis=-1).reshape(x.shape[1:] + shape)
    out_v = (rows[0][0], rows[1][0])
    out_d = tuple(tuple(part(x, [1 + 2 * c + k for k in range(2)], (2,)) for x in rows)
                  for c in range(2))
    out_h = tuple(tuple(part(x, [5 + 4 * c + 2 * a + b for a in range(2) for b in range(2)],
                             (2, 2)) for x in rows) for c in range(2))
    return out_v, out_d, out_h


def transform_rows(rows, jm):
    """`joint_dd.transform` on any number of real rows: (R, n_levels) -> (R, n_joint)."""
    count_rows = rows[0].shape[0]
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
    done = (jnp.zeros((count_rows, size)), jnp.zeros((count_rows, size)))

    def body(i, done):
        a = i * width
        rest = tuple(jax.lax.dynamic_slice(x, (0, a), (count_rows, width)) for x in sel)
        cols = tuple(jax.lax.dynamic_slice(x, (0, a), (size, width)) for x in h)
        rest = dd.add(rest, dd.neg(ozaki.matmul(done, cols)))
        piece = ozaki.matmul(rest, (uinv[0][i], uinv[1][i]))
        return tuple(jax.lax.dynamic_update_slice(d, x, (0, a)) for d, x in zip(done, piece))
    done = jax.lax.fori_loop(0, count, body, done)
    return tuple(x[:, :n] for x in done)


def level_jets(basis, h_lo, uinv, z_dd, rest_dd, d_l, h_l, weight):
    """The jets (W q, D(W q), D Dbar(W q)) of every function of a level, in dd.

    `weight` is W at the point, and d_l, h_l are D log W and D Dbar log W.
    """
    q_v, q_d, q_h = arnoldi_dd.joint_jets_dd(
        basis, h_lo, vpa_dd.basis_multiplier_jets(basis, z_dd, rest_dd), uinv)
    d_lb = analytic._cindex(d_l, None)
    h_lb = analytic._cindex(h_l, None)
    grad = dd.cadd(q_d, vpa_dd._creal(d_lb, vpa_dd._ex(q_v, 0)))
    lead = dd.cadd(dd.cadd(q_h, vpa_dd._outer(q_d, d_lb)), vpa_dd._outer(d_lb, q_d))
    rest_term = vpa_dd._creal(dd.cadd(h_lb, analytic._cindex(vpa_dd._outer(d_l, d_l), None)),
                              vpa_dd._ex(q_v, 0, 0))
    hess = dd.cadd(lead, rest_term)
    return (dd.mul(q_v, weight), vpa_dd._creal(grad, vpa_dd._ex(weight, 0)),
            vpa_dd._creal(hess, vpa_dd._ex(weight, 0, 0)))


def exp_small(x):
    """exp(x) in dd for a real dd x with abs(x) below 1e-3: 9 terms of the series."""
    one = dd.to_dd(jnp.ones_like(x[0]))
    out = one
    for n in range(9, 0, -1):
        inv_n = dd.reciprocal(dd.to_dd(jnp.full_like(x[0], float(n))))
        out = dd.add(one, dd.mul(dd.mul(x, inv_n), out))
    return out


@dataclasses.dataclass
class Stage1:
    levels: list
    jm: joint_dd.JointMap
    weights: tuple          # (hi, lo) of psi_1 = sum_t weights_t Gt_t over the joint functions
    shift: float = 0.0      # the constant C that comes out of the weights of the second stage


def stage1_weights(c1, jm, prefix):
    """psi_1 as dd weights on the joint functions, with the psi_N columns folded in."""
    n = len(jm.sel)
    g_hi, g_lo = (np.asarray(x) for x in jm.gamma)
    acc = dd.to_dd(jnp.asarray(c1[:n]))
    for i, size in enumerate(prefix):
        ci = float(c1[n + i])
        if ci == 0.0:
            continue
        keep = np.arange(n) < size
        part = (jnp.asarray(np.where(keep, g_hi, 0.0)), jnp.asarray(np.where(keep, g_lo, 0.0)))
        acc = dd.add(acc, dd.mul_dd_fp(part, jnp.asarray(ci)))
    return tuple(np.asarray(x) for x in acc)


def stage1_arrays(s1):
    hs = vl.level_arrays(s1.levels)[0]
    jm = s1.jm
    ja = (jnp.asarray(jm.h), jnp.asarray(jm.h_lo),
          tuple(tuple(jnp.asarray(x) for x in inv) for inv in jm.uinv))
    return hs, ja, tuple(jnp.asarray(x) for x in s1.weights), jnp.asarray(s1.shift)


def psi1_jets(s1, arrays, z_dd, rest_dd, hi, p_dd, r2):
    """(psi_1 - C, D psi_1, D Dbar psi_1) at one point, in dd."""
    hs, ja, wts, shift = arrays
    jm = dataclasses.replace(s1.jm, h=ja[0], h_lo=ja[1], uinv=ja[2])
    rows = []
    for lev, (h, hl, uh, ul) in zip(s1.levels, hs):
        basis = dataclasses.replace(lev.basis, h=h)
        d_l, h_l = vpa_dd.weight_jets(hi, z_dd, rest_dd, lev.alpha, lev.k1)
        weight = vpa_dd.weight_dd(p_dd, r2, lev.k0, lev.k1)
        rows.append(flat13(level_jets(basis, hl, (uh, ul), z_dd, rest_dd, d_l, h_l, weight)))
    rows = tuple(jnp.concatenate([r[i] for r in rows], axis=1) for i in range(2))
    joint = transform_rows(rows, jm)
    psi = dd.add(dd.dot_dd_fp(joint, wts[0], axis=1), dd.dot_dd_fp(joint, wts[1], axis=1))
    v, d, h = unflat13(psi)
    return dd.add(v, dd.to_dd(-shift)), d, h


def stage2_weight(lev, hi, z_dd, rest_dd, p_dd, r2, psi):
    """W, D log W and D Dbar log W of a second-stage level, W = |z|^(2 k1) exp(-a Phi_1)."""
    v, d, h = psi
    a = vpa_dd.alpha_dd(lev.alpha)
    d_l, h_l = vpa_dd.weight_jets(hi, z_dd, rest_dd, lev.alpha, lev.k1)
    d_l = dd.csub(d_l, vpa_dd._creal(d, vpa_dd._ex(a, 0)))
    h_l = dd.csub(h_l, vpa_dd._creal(h, vpa_dd._ex(a, 0, 0)))
    weight = dd.mul(vpa_dd.weight_dd(p_dd, r2, lev.k0, lev.k1), exp_small(dd.neg(dd.mul(a, v))))
    return weight, d_l, h_l


def point_state(lay0, parts0, s1, levels, a, arrays1, hs, joint=None):
    """The base and psi_1 at one point, and the dd Hessians of every second-stage column."""
    z, _, extra = a
    state, hi, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
    z_dd, m, free = analytic._dd_point(extra)
    rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
    r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
    psi = psi1_jets(s1, arrays1, z_dd, rest_dd, hi, p_dd, r2)
    ps = []
    for lev, (h, hl, uh, ul) in zip(levels, hs):
        basis = dataclasses.replace(lev.basis, h=h)
        weight, d_l, h_l = stage2_weight(lev, hi, z_dd, rest_dd, p_dd, r2, psi)
        ps.append(level_jets(basis, hl, (uh, ul), z_dd, rest_dd, d_l, h_l, weight)[2])
    if joint is None:
        cols = jtu.tree_map(lambda *xs: jnp.concatenate(xs), *ps)
    else:
        jm, _, sizes = joint
        cols = joint_dd.transform(jtu.tree_map(lambda *xs: jnp.concatenate(xs), *ps), jm)
        n = cols[0][0].shape[0]
        extra_cols = []
        for size in sizes:
            mask = (jnp.arange(n) < size)[:, None, None]
            g_hi, g_lo = (jnp.where(mask, jnp.asarray(x)[:, None, None], 0.0) for x in jm.gamma)
            extra_cols.append(tuple(dd.add(dd.dot_dd_fp(comp, g_hi, axis=0),
                                           dd.dot_dd_fp(comp, g_lo, axis=0)) for comp in cols))
        cols = jtu.tree_map(lambda x, *ys: jnp.concatenate([x] + [y[None] for y in ys]),
                            cols, *extra_cols)
    return state, hi, (z_dd, m, free), cols, psi


def psi1_program(lay0, parts0, s1, chunk):
    """psi_1 - C and its jets at a batch of points, rounded to FP64, and p0 is not needed."""
    def one(a, arrays1):
        z, _, extra = a
        _, hi, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
        z_dd, m, free = analytic._dd_point(extra)
        rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
        r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
        v, d, h = psi1_jets(s1, arrays1, z_dd, rest_dd, hi, p_dd, r2)
        return v, dd.cto_fp(d), dd.cto_fp(h)
    run = jax.jit(lambda x, arrays1: jax.lax.map(lambda a: one(a, arrays1), x, batch_size=chunk))
    return lambda batch: run(vf.point_inputs(batch), stage1_arrays(s1))


def level_values(lay0, parts0, s1, levels, batch, chunk):
    """W q of every second-stage function in dd at the dd points of `batch`."""
    def one(a, arrays1, hs):
        z, _, extra = a
        _, hi, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
        z_dd, m, free = analytic._dd_point(extra)
        rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
        r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
        psi = psi1_jets(s1, arrays1, z_dd, rest_dd, hi, p_dd, r2)
        out = []
        for lev, (h, hl, uh, ul) in zip(levels, hs):
            basis = dataclasses.replace(lev.basis, h=h)
            jets = arnoldi_dd.joint_jets_dd(
                basis, hl, vpa_dd.basis_multiplier_jets(basis, z_dd, rest_dd), (uh, ul))
            weight = stage2_weight(lev, hi, z_dd, rest_dd, p_dd, r2, psi)[0]
            out.append(dd.mul(jets[0], weight))
        return tuple(jnp.concatenate([v[i] for v in out]) for i in range(2))
    hs, _ = vl.level_arrays(levels)
    vals = jax.jit(lambda x, arrays1, hs: jax.lax.map(lambda a: one(a, arrays1, hs), x,
                                                      batch_size=chunk))(
        vf.point_inputs(batch), stage1_arrays(s1), hs)
    sizes = np.cumsum([0] + [int(lev.basis.h.shape[0]) for lev in levels])
    return [tuple(v[:, a:b] for v in vals) for a, b in zip(sizes[:-1], sizes[1:])]


def fit_rows(lay0, parts0, s1, levels, batch, chunk, joint):
    """v of Phi_1 as a dd pair and the rows of dv/dc of the second-stage columns."""
    def one(a, arrays1, hs, ja):
        state, hi, (z_dd, m, free), p, psi = point_state(lay0, parts0, s1, levels, a, arrays1,
                                                         hs, vl._with_arrays(joint, ja))
        v0 = analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True, delta=psi[2])
        rest = a[1]
        g1 = rest @ state[2] @ jnp.conj(rest).T + dd.cto_fp(psi[2])
        ginv = jnp.linalg.inv(g1)
        rows = jnp.real(jnp.einsum('kl,blk->b', ginv, dd.cto_fp(p)))
        return v0, dd.to_fp(v0) * rows
    hs, ja = vl.level_arrays(levels, joint)
    return jax.jit(lambda x, arrays1, hs, ja: jax.lax.map(lambda a: one(a, arrays1, hs, ja), x,
                                                          batch_size=chunk))(
        vf.point_inputs(batch), stage1_arrays(s1), hs, ja)


def residual_program(lay0, parts0, s1, levels, chunk, joint):
    """v in dd at a block of points for Phi_1 plus each column of second-stage c."""
    def one(a, cc, arrays1, hs, ja):
        _, hi, (z_dd, m, free), p, psi = point_state(lay0, parts0, s1, levels, a, arrays1, hs,
                                                     vl._with_arrays(joint, ja))

        def volume_of(c):
            delta = tuple(dd.dot_dd_fp(comp, c[:, None, None], axis=0) for comp in p)
            return analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True,
                                                delta=dd.cadd(psi[2], delta))
        return jax.vmap(volume_of)(cc.T)
    program = jax.jit(lambda x, cc, arrays1, hs, ja: jax.lax.map(
        lambda a: one(a, cc, arrays1, hs, ja), x, batch_size=chunk))
    hs, ja = vl.level_arrays(levels, joint)
    arrays1 = stage1_arrays(s1)
    return lambda x, cc: program(x, cc, arrays1, hs, ja)


def check(lay0, parts0, s1, levels, batch, args):
    """Three tests of the jets at a few points, against independent forms.

    1. D Dbar psi_1 from the 13-row path against the Hessian columns of
       `vpa_dd_levels.point_hessians` times the first-stage coefficients.
    2. The jets of the second-stage functions with the weight of Phi_1 against the
       product rule on the jets of the p0-weighted functions and of exp(-a psi_1).
    3. D psi_1 and the Levi form of psi_1 against differences of psi_1 along X.
    """
    stage1_joint = (s1.jm, None, tuple(np.cumsum([int(np.sum(s1.jm.level == i))
                                                  for i in range(len(s1.levels))])))
    c1 = jnp.asarray(args.c1_full)
    hs1, ja1 = vl.level_arrays(s1.levels, stage1_joint)
    hs2, _ = vl.level_arrays(levels)
    arrays1 = stage1_arrays(s1)

    def one(a):
        z, _, extra = a
        _, hi, p_dd = correction.base_state_dd(lay0, parts0, z, extra, with_p=True)
        z_dd, m, free = analytic._dd_point(extra)
        rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
        r2 = vpa_dd._radius_jets(z_dd, rest_dd)[0]
        psi = psi1_jets(s1, arrays1, z_dd, rest_dd, hi, p_dd, r2)
        cols = vl.point_hessians(lay0, parts0, s1.levels, hs1, a,
                                 vl._with_arrays(stage1_joint, ja1))[3]
        ref = tuple(dd.dot_dd_fp(comp, c1[:, None, None], axis=0) for comp in cols)
        # Test 2 on the first level of the second stage.
        lev, (h, hl, uh, ul) = levels[0], hs2[0]
        basis = dataclasses.replace(lev.basis, h=h)
        weight, d_l, h_l = stage2_weight(lev, hi, z_dd, rest_dd, p_dd, r2, psi)
        new = level_jets(basis, hl, (uh, ul), z_dd, rest_dd, d_l, h_l, weight)
        d0, h0 = vpa_dd.weight_jets(hi, z_dd, rest_dd, lev.alpha, lev.k1)
        w0 = vpa_dd.weight_dd(p_dd, r2, lev.k0, lev.k1)
        old = level_jets(basis, hl, (uh, ul), z_dd, rest_dd, d0, h0, w0)
        al = vpa_dd.alpha_dd(lev.alpha)
        e_val = exp_small(dd.neg(dd.mul(al, psi[0])))
        e_jet = (e_val, vpa_dd._creal(psi[1], vpa_dd._ex(dd.neg(dd.mul(al, e_val)), 0)),
                 vpa_dd._creal(dd.csub(vpa_dd._creal(vpa_dd._outer(psi[1], psi[1]),
                                                     vpa_dd._ex(dd.mul(al, al), 0, 0)),
                                       vpa_dd._creal(psi[2], vpa_dd._ex(al, 0, 0))),
                               vpa_dd._ex(e_val, 0, 0)))
        e_b = jtu.tree_map(lambda x: x[None], e_jet)
        prod = vpa_dd.jet_mul(old, e_b)
        return (dd.cto_fp(psi[2]), dd.cto_fp(ref), dd.to_fp(psi[0]),
                [dd.to_fp(new[0]), dd.cto_fp(new[1]), dd.cto_fp(new[2])],
                [dd.to_fp(prod[0]), dd.cto_fp(prod[1]), dd.cto_fp(prod[2])])
    take = jtu.tree_map(lambda x: x[:args.check], vf.point_inputs(batch))
    out = jax.jit(lambda x: jax.lax.map(one, x, batch_size=8))(take)
    h_new, h_ref = np.asarray(out[0]), np.asarray(out[1])
    scale = np.abs(h_ref).max(axis=(1, 2))
    say(f'check 1: D Dbar psi_1, 13-row path against the Hessian columns: largest '
        f'difference over the largest entry {np.max(np.abs(h_new - h_ref).max(axis=(1, 2)) / scale):.2e}')
    for name, i in (('value', 0), ('gradient', 1), ('Hessian', 2)):
        a_, b_ = np.asarray(out[3][i]), np.asarray(out[4][i])
        print(f'    check 2, {name}: largest difference {np.abs(a_ - b_).max():.2e} against '
              f'largest entry {np.abs(b_).max():.2e}', flush=True)
    psi_v = np.asarray(out[2])
    say(f'psi_1 - C at the check points: {psi_v.min():.3e} to {psi_v.max():.3e}')
    return h_new


def finite_differences(lay0, parts0, s1, batch, args):
    """Test 3: D psi_1 and its Levi form against differences of psi_1 along X."""
    n = args.check
    z_hi = np.asarray(batch['points_hi'][:n])
    z_lo = np.asarray(batch['points_lo'][:n])
    patch = np.asarray(batch['patch'][:n])
    steps = (2.0 ** -9, 2.0 ** -8)
    dirs = np.array([[1, 0], [1j, 0], [0, 1], [0, 1j]], dtype=complex)

    def move(zh, zl, pt, v, s):
        z_dd = ((jnp.real(zh), jnp.real(zl)), (jnp.imag(zh), jnp.imag(zl)))
        m, free = pt[0], (pt[1], pt[2])
        rest, _ = dd_metric._restriction_dd(z_dd, m, free)
        total = z_dd
        zero = jnp.zeros_like(s)
        for i in range(2):
            c = ((jnp.real(v[i]) * s, zero), (jnp.imag(v[i]) * s, zero))
            total = dd.cadd(total, dd.cmul(analytic._cindex(rest, i), c))
        for _ in range(4):
            total = dd_metric.newton_step(total, m)
        return total[0][0] + 1j * total[1][0], total[0][1] + 1j * total[1][1]
    plan = [(None, 0.0)] + [(d, s * sign) for d in range(4) for s in steps for sign in (1, -1)]
    pts_hi, pts_lo, pats = [], [], []
    for d, s in plan:
        v = jnp.asarray(dirs[d] if d is not None else np.zeros(2, complex))
        mh, ml = jax.vmap(lambda a, b, c: move(a, b, c, v, jnp.asarray(s)))(
            jnp.asarray(z_hi), jnp.asarray(z_lo), jnp.asarray(patch))
        pts_hi.append(np.asarray(mh))
        pts_lo.append(np.asarray(ml))
        pats.append(patch)
    moved = {'points_hi': jnp.asarray(np.concatenate(pts_hi)),
             'points_lo': jnp.asarray(np.concatenate(pts_lo)),
             'patch': jnp.asarray(np.concatenate(pats)),
             'restriction': jnp.asarray(np.concatenate([np.asarray(batch['restriction'][:n])]
                                                       * len(plan)))}
    v, d, h = psi1_program(lay0, parts0, s1, 8)(moved)
    # psi_1 - C is small, so its high part alone holds it to 1e-16 of itself.
    v = np.asarray(v[0]).reshape(len(plan), n)
    d0, h0 = np.asarray(d)[:n], np.asarray(h)[:n]
    worst_d = 0.0
    for k in range(4):
        base = 1 + 4 * k
        first = [(v[base + 2 * j] - v[base + 2 * j + 1]) / (2 * steps[j]) for j in range(2)]
        second = [(v[base + 2 * j] + v[base + 2 * j + 1] - 2 * v[0]) / steps[j] ** 2
                  for j in range(2)]
        rich_d = (4 * first[0] - first[1]) / 3
        rich_h = (4 * second[0] - second[1]) / 3
        vec = dirs[k]
        # For a real function, the derivative along t + s v is 2 Re(v . D psi), and the
        # second derivative is 2 Re(v^T D D psi v) + 2 vbar^T (D Dbar psi) v.
        lev_form = np.real(np.einsum('nij,i,j->n', h0, vec, np.conj(vec)))
        worst_d = max(worst_d, float(np.max(np.abs(rich_d - 2 * np.real(d0 @ vec))
                                            / (np.abs(d0).max(axis=1) + 1e-300))))
        if k in (0, 2):
            # Directions k and k + 1 are v and i v, so the Levi form is the sum of the two
            # second derivatives over 4.
            nxt = base + 4
            second_iv = [(v[nxt + 2 * j] + v[nxt + 2 * j + 1] - 2 * v[0]) / steps[j] ** 2
                         for j in range(2)]
            levi = (rich_h + (4 * second_iv[0] - second_iv[1]) / 3) / 4
            err = np.abs(levi - lev_form) / (np.abs(h0).max(axis=(1, 2)) + 1e-300)
            say(f'check 3, direction {k}: Levi form of psi_1 from differences against the '
                f'jets, largest relative difference {err.max():.2e}')
    say(f'check 3: D psi_1 from central differences against the jets, largest relative '
        f'difference {worst_d:.2e}')


def evaluate_saved(lay0, parts0, s1, levels, joint, hold, args, out_dir, tag):
    """v of saved coefficients on a hold-out slice, or the seam test in two charts."""
    coeffs = np.load(args.coeffs)
    program = residual_program(lay0, parts0, s1, levels, args.chunk, joint)
    if args.eval_only:
        t1 = time.perf_counter()
        hi, lo = vl.volumes(program, coeffs, hold, args.block)
        say(f'{len(hi)} hold-out points from {args.eval_offset} in dd in '
            f'{time.perf_counter() - t1:.0f}s')
        size = np.abs(np.asarray(hold['points_hi']))
        np.savez(out_dir / f'vpa_levels_v_{tag}_{args.eval_offset}.npz', hi=hi, lo=lo,
                 mass=np.asarray(hold['mass']), ratio=size.min(axis=1) / size.max(axis=1))
        mass = hold['mass'] / jnp.sum(hold['mass'])
        f = np.asarray(analytic.residual_from_volumes((jnp.asarray(hi), jnp.asarray(lo)),
                                                      mass[:, None]))
        for i in range(f.shape[1]):
            got = vf.summary(f[:, i], hold)
            say(f"column {i}, this slice alone: sigma {got['mape']:.4e}, rms {got['rms']:.3e}, "
                f"max {got['max']:.3e}")
    if args.seam:
        batch = cf.load_points(args.seam, 20263813)
        second = sl.other_chart(batch)
        mass = batch['mass'] / jnp.sum(batch['mass'])
        w = np.asarray(mass)
        fs = []
        for name, points in (('first', batch), ('second', second)):
            t1 = time.perf_counter()
            hi, lo = vl.volumes(program, coeffs, points, args.block)
            fs.append(np.asarray(analytic.residual_from_volumes(
                (jnp.asarray(hi), jnp.asarray(lo)), mass[:, None])))
            say(f'{name} chart: {args.seam} points in dd in {time.perf_counter() - t1:.0f}s')
        for i in range(coeffs.shape[1]):
            fa, fb = fs[0][:, i], fs[1][:, i]
            d = fb - fa
            print(f'column {i}: sigma {np.sum(w * np.abs(fa)):.4e} in the first chart and '
                  f'{np.sum(w * np.abs(fb)):.4e} in the second; the change of f has mean abs '
                  f'{np.sum(w * np.abs(d)):.3e}, rms {math.sqrt(np.sum(w * d * d)):.3e}, max '
                  f'{np.abs(d).max():.3e} and weighted mean {np.sum(w * d):.3e}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--stage1', type=str, required=True,
                    help='the output directory of the first-stage run of vpa_dd_levels.py')
    ap.add_argument('--stage1-k1', type=int, nargs='+', required=True)
    ap.add_argument('--stage1-col', type=int, default=-1,
                    help='the column of the saved first-stage coefficients')
    ap.add_argument('--k1', type=int, nargs='+', required=True)
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument('--n-fit', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--n-eval', type=int, default=400_000)
    ap.add_argument('--eval-seed', type=int, default=20261813)
    ap.add_argument('--n-build', type=int, default=60_000)
    ap.add_argument('--n-build-level', type=int, default=20_000)
    ap.add_argument('--build-seed', type=int, default=20262001)
    ap.add_argument('--n-check', type=int, default=20_000)
    ap.add_argument('--d-max', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--block', type=int, default=65_536)
    ap.add_argument('--null-degree', type=int, default=9)
    ap.add_argument('--joint-tol', type=float, default=1e-15)
    ap.add_argument('--joint-block', type=int, default=256)
    ap.add_argument('--prefixes', type=int, nargs='+', default=None)
    ap.add_argument('--exact-mul', action='store_true')
    ap.add_argument('--invariant-base', action='store_true',
                    help='the base in the monomial basis, exactly invariant under S_4')
    ap.add_argument('--refine', type=int, default=3)
    ap.add_argument('--check', type=int, default=0,
                    help='only run the three tests of the jets at this many points')
    ap.add_argument('--load-joint', type=str, default=None,
                    help='a saved second-stage joint map, in place of a new build')
    ap.add_argument('--coeffs', type=str, default=None,
                    help='saved second-stage coefficients, for --eval-only and --seam')
    ap.add_argument('--record', type=str, default=None,
                    help='the json of a second-stage fit: its constant C of psi_1 is reused, '
                         'because the saved coefficients hold only for that C')
    ap.add_argument('--eval-only', action='store_true',
                    help='save v in dd on the hold-out points --eval-offset to '
                         '--eval-offset + --n-eval of one sample of --eval-total points')
    ap.add_argument('--eval-offset', type=int, default=0)
    ap.add_argument('--eval-total', type=int, default=0)
    ap.add_argument('--seam', type=int, default=0,
                    help='the seam test: f at this many points of seed 20263813 in two charts')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    args.gram, args.entries_from = False, None

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay0.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis at k0')
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    analytic.EXACT_MUL = args.exact_mul
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    # With --invariant-base the dd paths take the base in the monomial basis, exactly
    # invariant under S_4; the level bases keep the frame form of lay0.
    lay0d = lay0
    if args.invariant_base:
        from quartic import invariant
        lay0d, parts0 = invariant.monomial_base(lay0, parts0)
    counts = vf.hn_counts(max(args.k1 + args.stage1_k1))

    def sample(n, seed):
        batch = vl.data.make(math.ceil(n / 4), seed)
        return {key: val[:n] for key, val in batch.items()}
    evaluate = args.eval_only or args.seam > 0
    if evaluate and (args.load_joint is None or args.coeffs is None or args.record is None):
        raise SystemExit('--eval-only and --seam need --load-joint, --coeffs and --record')
    fit_set = None if evaluate else cf.load_points(args.n_fit, args.seed)
    hold = None
    if args.eval_offset or args.eval_total:
        # The sampler makes a new sample for each size and orders it by patch, so every
        # slice must come from the same whole sample.
        whole = cf.load_points(args.eval_total or args.eval_offset + args.n_eval,
                               args.eval_seed)
        hold = {key: val[args.eval_offset:args.eval_offset + args.n_eval]
                for key, val in whole.items()}
        del whole
    elif not args.seam:
        hold = cf.load_points(args.n_eval, args.eval_seed)
    build = sample(args.n_build_level, args.build_seed)
    check_pts = sample(args.n_check, args.build_seed + 1000)
    say(f'{args.n_fit} fit points of seed {args.seed}, {args.n_eval} hold-out points of seed '
        f'{args.eval_seed}, {args.n_build} joint build points of seed {args.build_seed}')

    class LoadArgs:
        load = args.stage1
    s1_tag = f'k0_{args.k0}_' + '_'.join(str(k) for k in args.stage1_k1)
    s1_levels = [vl.build_level(args.k0, k1, lay0, mats0, None, None, counts, LoadArgs,
                                out_dir)[0] for k1 in args.stage1_k1]
    jm1 = joint_dd.load(Path(args.stage1) / f'joint_map_{s1_tag}.npz')
    prefix1 = np.cumsum([int(np.sum(jm1.level == i)) for i in range(len(s1_levels))])
    c1_all = np.load(Path(args.stage1) / f'vpa_levels_coeffs_{s1_tag}.npy')
    c1 = c1_all[:, args.stage1_col]
    args.c1_full = c1
    s1 = Stage1(s1_levels, jm1, stage1_weights(c1, jm1, prefix1))
    say(f'first stage {args.stage1_k1} from {args.stage1}: {len(jm1.sel)} joint functions, '
        f'column {args.stage1_col}, norm of c {np.linalg.norm(c1):.3e}')

    if args.record is not None:
        s1.shift = float(json.loads(Path(args.record).read_text())['stage1_shift'])
        spread = float('nan')
        say(f'the constant C of psi_1 from {args.record}: {s1.shift!r}')
    else:
        probe = jtu.tree_map(lambda x: x[:4096], fit_set)
        v1 = np.asarray(psi1_program(lay0d, parts0, s1, args.chunk)(probe)[0][0])
        w = np.asarray(probe['mass']) / float(np.sum(probe['mass']))
        s1.shift = float(np.sum(w * v1))
        spread = float(np.max(np.abs(v1 - s1.shift)))
        top = max(Fraction(k, args.k0) for k in args.k1)
        say(f'psi_1 on 4096 fit points: mean {s1.shift:.6e}, largest distance from the mean '
            f'{spread:.3e}, times the largest k1/k0 {float(top) * spread:.3e}')
        if float(top) * spread > 1e-3:
            raise SystemExit('psi_1 - C is too large for the series of exp; take a general exp')

    levels = []
    for k1 in args.k1:
        lev, _ = vl.build_level(args.k0, k1, lay0, mats0, build, check_pts, counts, args,
                                out_dir)
        levels.append(lev)
    if args.check:
        check(lay0d, parts0, s1, levels, fit_set, args)
        finite_differences(lay0d, parts0, s1, fit_set, args)
        return

    tag = f"k0_{args.k0}_s1_{'_'.join(str(k) for k in args.stage1_k1)}_s2_" + \
        '_'.join(str(k) for k in args.k1)
    t1 = time.perf_counter()
    if args.load_joint is not None:
        jm = joint_dd.load(args.load_joint)
        say(f'joint map loaded from {args.load_joint}')
    else:
        jbuild = cf.load_points(args.n_build, args.build_seed)
        values = level_values(lay0d, parts0, s1, levels, jbuild, args.chunk)
        # Host copies, so that the joint build holds the values of one level at a time.
        values = [tuple(np.asarray(a) for a in v) for v in values]
        say(f'dd values of every second-stage level at {args.n_build} build points in '
            f'{time.perf_counter() - t1:.0f}s')
        t1 = time.perf_counter()
        jm = joint_dd.build(values, jbuild['mass'] / jnp.sum(jbuild['mass']),
                            tol=args.joint_tol, block=args.joint_block, verbose=True)
        del values
        joint_dd.save(out_dir / f'joint_map_{tag}.npz', jm)
    counts_in = [int(np.sum(jm.level == i)) for i in range(len(levels))]
    prefix = np.cumsum(counts_in)
    b0s = [int(np.argmax(np.abs(jm.gamma[0][:n]))) for n in prefix]
    joint = (jm, b0s[-1], tuple(int(x) for x in prefix))
    say(f'joint map in {time.perf_counter() - t1:.0f}s: {counts_in} functions of the levels '
        f'enter, smallest new part {jm.ratio.min():.3e}; psi_N - 1 has the norm '
        f'{jm.gap:.3e} on the build points')
    if evaluate:
        evaluate_saved(lay0d, parts0, s1, levels, joint, hold, args, out_dir, tag)
        return
    chosen = args.prefixes or list(range(1, len(levels) + 1))
    n_joint = int(prefix[-1])
    plans = [('+'.join(str(lev.k1) for lev in levels[:i]) + ' stage 2',
              np.array([t for t in range(prefix[i - 1]) if t != b0s[i - 1]]
                       + [n_joint + i - 1]), n_joint + i - 1) for i in chosen]
    total = n_joint + len(levels)
    record = {'args': {k: v for k, v in vars(args).items() if k != 'c1_full'},
              'stage1_shift': s1.shift, 'stage1_spread': spread,
              'joint': {'entered': counts_in, 'ratio_min': float(jm.ratio.min())}}

    t1 = time.perf_counter()
    v0, lrows = fit_rows(lay0d, parts0, s1, levels, fit_set, args.chunk, joint)
    mass = fit_set['mass'] / jnp.sum(fit_set['mass'])
    # The sampled mean, as in vpa_dd_levels.py: `vpa_dd_levels.factor` takes its derivative.
    f0 = analytic.residual_from_volumes(v0, mass)
    m0 = float(dd.to_fp(dd.dot_dd_fp(v0, mass, axis=0)))
    r0 = jnp.sqrt(mass) * f0
    say(f'rows of J of {total} columns at {args.n_fit} points in {time.perf_counter() - t1:.0f}s')

    order = sorted(range(len(plans)), key=lambda i: len(plans[i][1]))
    steps, coeffs, labels = {}, [np.zeros(total)], ['stage 1']
    for i in order:
        name, cols, psi_col = plans[i]
        t1 = time.perf_counter()
        if len(cols) == total:
            rows_sub, lrows = lrows, None
        else:
            rows_sub = lrows[:, jnp.asarray(cols)]
            if i == order[-1]:
                lrows = None
        q, r, norms = vl.factor(rows_sub, f0, mass, m0)
        del rows_sub
        spec, sols, predicted, step = vl.solve(q, r, norms, r0, [0.0])
        del q, r
        say(f"{name}: {len(cols)} columns, cond {spec['cond_scaled']:.3e}, share outside the "
            f"range of J {spec['outside']:.3e}, linear model loss {predicted[0]:.4e}")
        c = np.zeros(total)
        c[cols] = sols[:, 0]
        print(f'    norm of c {np.linalg.norm(c):.3e}, coefficient of psi_N {c[psi_col]:.6f}',
              flush=True)
        coeffs.append(c)
        labels.append(name)
        steps[name] = (cols, step)
        record.setdefault('solutions', []).append({'label': name, 'spectrum': spec,
                                                   'predicted_loss': predicted[0]})
    del lrows
    coeffs = np.stack(coeffs, axis=1)
    program = residual_program(lay0d, parts0, s1, levels, args.chunk, joint)

    def residuals(cc, batch):
        hi, lo = vl.volumes(program, cc, batch, args.block)
        m = batch['mass'] / jnp.sum(batch['mass'])
        # A copy, because the refinements write into it.
        return np.array(analytic.residual_from_volumes(
            (jnp.asarray(hi), jnp.asarray(lo)), m[:, None]))
    t1 = time.perf_counter()
    f_fit = residuals(coeffs, fit_set)
    say(f'the fit set in dd in {time.perf_counter() - t1:.0f}s')
    best = np.array([vf.summary(f_fit[:, i], fit_set)['loss'] for i in range(len(labels))])
    for i, label in enumerate(labels):
        got = vf.summary(f_fit[:, i], fit_set)
        print(f"    {label}: fit set sigma {got['mape']:.4e}, loss {got['loss']:.4e}", flush=True)
    root = jnp.sqrt(mass)
    for it in range(1, args.refine + 1):
        trial = coeffs.copy()
        for i, label in enumerate(labels[1:], start=1):
            cols, step = steps[label]
            trial[cols, i] += step(root * jnp.asarray(f_fit[:, i]), 0.0)
        f_new = residuals(trial, fit_set)
        say(f'refinement {it}')
        for i, label in enumerate(labels[1:], start=1):
            got = vf.summary(f_new[:, i], fit_set)
            kept = got['loss'] < best[i]
            if kept:
                coeffs[:, i], f_fit[:, i], best[i] = trial[:, i], f_new[:, i], got['loss']
            print(f"    {label}: fit set sigma {got['mape']:.4e}, loss {got['loss']:.4e}"
                  f"{'' if kept else ', not kept'}", flush=True)
    np.save(out_dir / f'vpa_stage2_coeffs_{tag}.npy', coeffs)
    t1 = time.perf_counter()
    f_hold = residuals(coeffs, hold)
    say(f'the hold-out in dd in {time.perf_counter() - t1:.0f}s')
    np.save(out_dir / f'vpa_stage2_f_hold_{tag}.npy', f_hold)
    record['hold'] = {}
    for i, label in enumerate(labels):
        got = vf.summary(f_hold[:, i], hold)
        record['hold'][label] = got
        say(f"{label}: hold-out sigma {got['mape']:.4e}, rms {got['rms']:.3e}, "
            f"max {got['max']:.3e}")
    (out_dir / f'vpa_stage2_{tag}.json').write_text(json.dumps(record, indent=1, default=str))


if __name__ == '__main__':
    main()
