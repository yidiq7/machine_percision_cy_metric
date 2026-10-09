"""The Taylor model of f on a batch of boxes on the GPU, for the strict bound of sup abs(f).

The design is docs/superpowers/specs/2026-10-09-gpu-taylor-certificate-design.md; the
arb reference of the same steps is scripts/tm_ref.py (--affine). All boxes of a batch
share one reference center c_r and its local forms (scripts/tm_local_forms.py).

1. Host: each box has a center p on X (t = (y_0, y_1), y_2 from P = 0), the frame
   E = dy/dt, the unit normal n and 1/c1 = 1/(grad P . n), all exact dd numbers.
2. GPU, models in tau alone (order n_y): h from the fixed point h -> h - P(y(h)) / c1 with
   a contraction bound, y(tau) = p + E tau + h n, the monomials (t - c_r)^alpha y_2^e.
3. Composition W = sum over e of Y_e Q_e Y_e^H with the local forms, up to the degree n
   in tau and in sigma; the parts above n go to the remainder through bounds.
4. The base: r = abs(y)^2, p~ = p0 r^(-k0), log p0 = k0 log r + log p~.
5. The level sum: x = p0^(-1/k0) = r^(-1) p~^(-1/k0), xi = x / x_c - 1 with
   x_c = x0 (1 + zeta), and psi = sum over k of F_k xi^k with
   F_k = (1 + zeta)^k sum over m >= k of C(m, k) zeta^(m - k) M~_m, formed per box on the
   local forms; F_k for k > K enter only through a bound.
6. Phi = log p0 + psi, g = d_tau d_sigma Phi (Cauchy), det g, v, f = v / vbar - 1.

Every rounding of the GPU goes into a remainder by the bounds of `quartic.tmc` and
`quartic.ddgemm`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import cupy as cp
import mpmath as mp

from . import tmc, ddgemm

U2 = tmc.U2
SAFE = tmc.SAFE
mp.mp.dps = 60


def mp_dd(z):
    """An mpmath complex as dd parts [re_hi, re_lo, im_hi, im_lo]."""
    z = mp.mpc(z)
    rh, ih = float(z.real), float(z.imag)
    return [rh, float(z.real - rh), ih, float(z.imag - ih)]


def dd_mp(v):
    return mp.mpc(mp.mpf(v[0]) + mp.mpf(v[1]), mp.mpf(v[2]) + mp.mpf(v[3]))


@dataclass
class Frames:
    """Exact dd numbers of the boxes, each (..., 4, B) on the GPU, and host values."""
    p: cp.ndarray          # (3, 4, B)
    e: cp.ndarray          # (2, 3, 4, B): e[i] = dy/dt_i
    n: cp.ndarray          # (3, 4, B)
    c1inv: cp.ndarray      # (4, B)
    vol: cp.ndarray        # (4, B): 1 / (abs(det U)^2 vbar), U = [E, n]
    vol_err: cp.ndarray    # (B,): a bound of the error of vol relative to its value
    host: list             # mpmath values of each box, for checks
    ha: cp.ndarray = None  # (3, 4, B): c1inv P(p), c1inv grad.E_0, c1inv grad.E_1
    a6: cp.ndarray = None  # (3, 4, B): 6 p_k^2
    a4: cp.ndarray = None  # (3, 4, B): 4 p_k
    h_err: cp.ndarray = None   # (6, B): the errors of ha (3), abs(1 - c1inv c1), and the
                               # largest errors of a6 and a4


def make_frames(centers, k0):
    """The frames of boxes with centers t = (t_0, t_1), each given as two dd lists."""
    vbar = 512 * mp.pi ** 4 * k0 ** 2 / mp.gamma(mp.mpf(1) / 4) ** 8
    cols = {k: [] for k in ('p', 'e', 'n', 'c1inv', 'vol', 'ha', 'a6', 'a4')}
    errs, host, herr = [], [], []
    for t in centers:
        t0, t1 = dd_mp(t[0]), dd_mp(t[1])
        y2 = (-(1 + t0 ** 4 + t1 ** 4)) ** (mp.mpf(1) / 4)
        p = [mp.mpc(t0), mp.mpc(t1), dd_mp(mp_dd(y2))]
        e = [[mp.mpc(1), mp.mpc(0), dd_mp(mp_dd(-p[0] ** 3 / p[2] ** 3))],
             [mp.mpc(0), mp.mpc(1), dd_mp(mp_dd(-p[1] ** 3 / p[2] ** 3))]]
        grad = [4 * c ** 3 for c in p]
        norm = mp.sqrt(sum(abs(c) ** 2 for c in grad))
        n = [dd_mp(mp_dd(mp.conj(c) / norm)) for c in grad]
        c1 = sum(g * m for g, m in zip(grad, n))
        c1inv = dd_mp(mp_dd(1 / c1))
        u = mp.matrix([[e[0][i], e[1][i], n[i]] for i in range(3)])
        w = 1 / (abs(mp.det(u)) ** 2 * vbar)
        wd = dd_mp(mp_dd(w))
        errs.append(float(abs(wd - w) / abs(w)) + 1e-40)
        cols['p'].append([mp_dd(c) for c in p])
        cols['e'].append([[mp_dd(c) for c in row] for row in e])
        cols['n'].append([mp_dd(c) for c in n])
        cols['c1inv'].append(mp_dd(c1inv))
        cols['vol'].append(mp_dd(wd))
        # The map of implicit_h about p: T(h) = kappa h - A(tau) - c1inv N(E tau + h n).
        ha = [c1inv * (sum(c ** 4 for c in p) + 1),
              c1inv * sum(g * x for g, x in zip(grad, e[0])),
              c1inv * sum(g * x for g, x in zip(grad, e[1]))]
        a6 = [6 * c ** 2 for c in p]
        a4 = [4 * c for c in p]
        err = lambda v: float(abs(dd_mp(mp_dd(v)) - v)) * (1 + 1e-9) + 1e-55
        cols['ha'].append([mp_dd(v) for v in ha])
        cols['a6'].append([mp_dd(v) for v in a6])
        cols['a4'].append([mp_dd(v) for v in a4])
        herr.append([err(v) for v in ha] + [float(abs(1 - c1inv * c1)) * (1 + 1e-9) + 1e-55,
                                            max(err(v) for v in a6), max(err(v) for v in a4)])
        host.append(dict(p=p, e=e, n=n, c1inv=c1inv, w=wd))
    to = lambda key, axes: cp.asarray(np.moveaxis(np.array(cols[key]), 0, -1))
    return Frames(p=to('p', None), e=to('e', None), n=to('n', None), c1inv=to('c1inv', None),
                  vol=to('vol', None), vol_err=cp.asarray(np.array(errs)), host=host,
                  ha=to('ha', None), a6=to('a6', None), a4=to('a4', None),
                  h_err=cp.asarray(np.array(herr).T))


def neg(k):
    return -k


def tm_const(sp, value, delta):
    return tmc.const(sp, value, delta)


def lin_tm(sp, p, e0, e1, delta):
    """p + e0 tau_0 + e1 tau_1 with dd numbers (4, B): exact, with no rounding."""
    c = tmc.zeros(sp, delta.shape[0])
    c[:, 0, :] = p
    c[:, sp.index[(1, 0)], :] = e0
    c[:, sp.index[(0, 1)], :] = e1
    return tmc.TM(c, cp.zeros_like(delta), delta)


def p_value(sp, ys):
    total = None
    for y in ys:
        y2 = tmc.mul(sp, y, y)
        y4 = tmc.mul(sp, y2, y2)
        total = y4 if total is None else tmc.add(sp, total, y4)
    return tmc.add_const(sp, total, tmc.cconst(1, total.delta.shape[0]))


def implicit_h(sp, fr, delta):
    """The models of y(tau) = p + E tau + h n on X, with a strict remainder of h.

    T(h) = h - c1inv P(y(h)). If T maps the ball abs(h - P_h) <= rho into itself with a
    contraction factor lam < 1, the root of each tau lies in that ball. With w = E tau + h n,
    T(h) = kappa h - A(tau) - c1inv N(w): kappa = 1 - c1inv c1, A = c1inv (P(p) + grad.E tau)
    and N(w) = sum over k of w_k^2 (6 p_k^2 + w_k (4 p_k + w_k)). The host gives A, 6 p_k^2
    and 4 p_k as dd numbers with their errors; kappa h goes to the error. So the rounding
    acts only on the small terms of N. Returns ys, rho, lam and the model of h.
    """
    b = delta.shape[0]
    zero = cp.zeros((4, b))
    lin0 = [lin_tm(sp, zero, fr.e[0, k], fr.e[1, k], delta) for k in range(3)]
    a_tau = lin_tm(sp, fr.ha[0], fr.ha[1], fr.ha[2], delta)

    def t_map(h):
        ws = [tmc.add(sp, lin0[k], tmc.scale(sp, h, fr.n[k])) for k in range(3)]
        nw = None
        for k in range(3):
            inner = tmc.add_const(sp, tmc.mul(sp, ws[k], tmc.add_const(sp, ws[k], fr.a4[k])),
                                  fr.a6[k])
            term = tmc.mul(sp, tmc.mul(sp, ws[k], ws[k]), inner)
            nw = term if nw is None else tmc.add(sp, nw, term)
        return tmc.neg(tmc.add(sp, a_tau, tmc.scale(sp, nw, fr.c1inv))), ws

    h = tmc.TM(tmc.zeros(sp, b), cp.zeros(b), delta)
    for _ in range(sp.n + 2):
        h, _ = t_map(h)
        h = tmc.TM(h.c, cp.zeros(b), delta)
    th, ws = t_map(h)
    bw = [tmc.bound(sp, w) for w in ws]
    iabs = tmc.abs_ub(fr.c1inv[:, None, :])[0]
    he = fr.h_err
    # The terms that T(h) above leaves out or rounds on the host.
    eps = (he[0] + (he[1] + he[2]) * delta + he[3] * tmc.bound(sp, h)
           + iabs * sum(he[4] * x ** 2 + he[5] * x ** 3 for x in bw)) * SAFE
    bd = (tmc.bound(sp, tmc.sub(sp, th, h)) + eps) * SAFE
    lin = [lin_tm(sp, fr.p[k], fr.e[0, k], fr.e[1, k], delta) for k in range(3)]
    ys = [tmc.add(sp, lin[k], tmc.scale(sp, h, fr.n[k])) for k in range(3)]
    dp = None
    for k in range(3):
        cube = tmc.mul(sp, tmc.mul(sp, ys[k], ys[k]), ys[k])
        term = tmc.scale(sp, cube, 4 * fr.n[k])
        dp = term if dp is None else tmc.add(sp, dp, term)
    lam0 = tmc.bound(sp, tmc.add_const(sp, tmc.scale(sp, dp, -fr.c1inv), tmc.cconst(1, b)))
    by = [tmc.bound(sp, y) for y in ys]
    nabs = [tmc.abs_ub(fr.n[k][:, None, :])[0] for k in range(3)]
    ok = lam0 < 1
    r = cp.where(ok, 2 * bd / cp.where(ok, 1 - lam0, 1.0), cp.inf) * SAFE
    second = sum(12 * nabs[k] ** 2 * (by[k] + r * nabs[k]) ** 2 for k in range(3))
    lam = (lam0 + r * second * iabs) * SAFE
    good = ok & (lam < 1) & ((bd + lam * r) * SAFE <= r)
    rho = cp.where(good, r, cp.inf)
    h = tmc.TM(h.c, rho, delta)
    ys = [tmc.add(sp, lin[k], tmc.scale(sp, h, fr.n[k])) for k in range(3)]
    return ys, rho, lam, h


def offset_models(sp, fr, h, ref, delta):
    """The models of t_i - ref_i (i = 0, 1) and of y_2 - p_2, from h and the frames.

    The constant p_i - ref_i comes from the host, rounded to dd with its error, so no
    difference of two near values rounds on the GPU.
    """
    b = delta.shape[0]
    out = []
    for i in range(2):
        vals = [hb['p'][i] - ref[i] for hb in fr.host]
        dp = _to_dd(vals)
        err = cp.asarray(np.array([float(abs(dd_mp(mp_dd(v)) - v)) * (1 + 1e-9) + 1e-60
                                   for v in vals]))
        x = tmc.add(sp, lin_tm(sp, dp, fr.e[0, i], fr.e[1, i], delta), tmc.scale(sp, h, fr.n[i]))
        out.append(tmc.TM(x.c, (x.rem + err) * SAFE, x.delta))
    v = tmc.add(sp, lin_tm(sp, cp.zeros((4, b)), fr.e[0, 2], fr.e[1, 2], delta),
                tmc.scale(sp, h, fr.n[2]))
    return out, v


def conj_tm(x):
    c = x.c.copy()
    c[2:] = -c[2:]
    return tmc.TM(c, x.rem.copy(), x.delta)


def monomial_mats(sp2, sp_low, base_t, y2, top):
    """The coefficients of y_2^e (t - c_r)^alpha, alpha of degree <= top, for e = 0..3,
    or for e = 0 only if y2 is None.

    base_t: the models of t_i - c_r,i (offset_models). Returns, for each e: Y (B, 4,
    S_low, n_top) for the degrees <= sp_low.n in tau, and (n_top, B) arrays of the bound
    of that part, of the bound of the rest of the polynomial (degrees above sp_low.n),
    and of the remainder.
    """
    from .tm_forms import Poly2
    p2 = Poly2(top)
    b = base_t[0].delta.shape[0]
    # The monomials of one degree come from those of the degree below in one product,
    # with the monomials stacked along the batch axis.
    mono = {(0, 0): tmc.const(sp2, tmc.cconst(1, b), base_t[0].delta)}
    for d in range(1, top + 1):
        layer = [(d - j, j) for j in range(d + 1)]
        srcs = [mono[(a - 1, c)] if a else mono[(0, c - 1)] for a, c in layer]
        facs = [base_t[0] if a else base_t[1] for a, c in layer]
        prod = _batched_mul(sp2, srcs, facs)
        for m, x in zip(layer, prod):
            mono[m] = x
    n_e = 1 if y2 is None else 4
    if y2 is not None:
        y2e = [None, y2, tmc.mul(sp2, y2, y2)]
        y2e.append(tmc.mul(sp2, y2e[2], y2))
    low = sp_low.size
    deg = cp.asarray(sp2.deg)
    pw = base_t[0].delta[None, :] ** deg[:, None]             # (S2, B)
    is_low = cp.asarray((sp2.deg <= sp_low.n).astype(np.float64))[:, None]
    out = []
    monos = [mono[m] for m in p2.mons]
    for e in range(n_e):
        tms = monos if e == 0 else _batched_mul(sp2, [y2e[e]] * len(monos), monos)
        cs = cp.stack([t.c for t in tms], axis=2)             # (4, S2, n_top, B)
        a = tmc.abs_ub(cs.reshape(4, sp2.size, -1)).reshape(sp2.size, len(tms), b)
        w = a * pw[:, None, :]
        b_low = (cp.sum(w * is_low[:, :, None], axis=0)) * SAFE
        b_high = (cp.sum(w * (1 - is_low)[:, :, None], axis=0)) * SAFE
        rem = cp.stack([t.rem for t in tms])                  # (n_top, B)
        y = cp.moveaxis(cs[:, :low], -1, 0)                   # (B, 4, low, n_top)
        onehot = cp.asarray((sp2.deg[:, None] == np.arange(sp2.n + 1)[None, :])
                            .astype(np.float64))              # (S2, ny + 1)
        gdeg = cp.einsum('sab,sd->adb', w, onehot) * SAFE     # (n_top, ny + 1, B)
        out.append((cp.ascontiguousarray(y), b_low, b_high, rem, gdeg))
    return out


def _batched_mul(sp, xs, ys):
    """The products xs[i] ys[i] of lists of models, in one call with the batches stacked."""
    b = xs[0].delta.shape[0]
    cat = lambda ms: tmc.TM(cp.concatenate([m.c for m in ms], axis=2),
                            cp.concatenate([m.rem for m in ms]),
                            cp.concatenate([m.delta for m in ms]))
    z = tmc.mul(sp, cat(xs), cat(ys))
    return [tmc.TM(z.c[:, :, i * b:(i + 1) * b], z.rem[i * b:(i + 1) * b],
                   z.delta[i * b:(i + 1) * b]) for i in range(len(xs))]


def mats_at(mats, sp_k):
    """The monomial matrices for the lower order sp_k.n: the rows of degree <= sp_k.n, and
    the bounds split at that degree."""
    out = []
    for y, _, _, r, gdeg in mats:
        cut = sp_k.n + 1
        b_low = cp.sum(gdeg[:, :cut], axis=1) * SAFE
        b_high = cp.sum(gdeg[:, cut:], axis=1) * SAFE
        out.append((cp.ascontiguousarray(y[:, :, :sp_k.size]), b_low, b_high, r))
    return out


def compose(sp4, sp_low, mats, q, q_rad, q_abs, delta, td=False):
    """The model in four variables of sum over e of Y_e Q_e Y_e^H, with all remainders.

    The sum runs over the blocks of q; mats[e] belongs to q[e].

    q[e]: (nq, 4, n, n) with nq = 1 (shared) or B; q_rad[e], q_abs[e]: (nq, n, n) FP64.
    The true form differs from the polynomial of total degree <= n by the entries of W
    of higher total degree (their exact sizes; a majorant from abs(Q) is far too large,
    because those entries cancel strongly), by the parts of degree above sp_low.n of
    the monomial models, by their remainders, by the radii of Q, and by the rounding of
    the two products (2.1 gamma abs(Y) abs(Q) abs(Y)^H) and of the sums over e. With td,
    the products take the triple-double accumulators, and their rounding is
    2.01 (lin + cub) abs(Y) abs(Q) abs(Y)^H with (lin, cub) of `ddgemm.gamma_td`.
    """
    n = q[0].shape[-1]
    b = delta.shape[0]
    w = None
    rem = cp.zeros(b)
    parts = {k: cp.zeros(b) for k in ('round', 'high', 'models', 'radius', 'sums', 'abs')}
    g = 2.01 * sum(ddgemm.gamma_td(n)) if td else 2.1 * ddgemm.gamma(n)
    quad = lambda u, m, v: cp.matmul(cp.matmul(u.T[:, None, :], m), v.T[:, :, None])[:, 0, 0]
    for e in range(len(q)):
        y, b_low, b_high, r = mats[e][:4]
        y = cp.ascontiguousarray(y[:, :, :, :n])
        b_low, b_high, r = b_low[:n], b_high[:n], r[:n]
        t = ddgemm.zgemm(y, q[e], mode=0, td=td)
        part = ddgemm.zgemm(t, y, mode=1, td=td)
        if w is not None:
            sums = 4 * U2 * (_bnorm(sp_low, w, delta) + _bnorm(sp_low, part, delta))
            parts['sums'] += sums
            rem += sums
        w = part if w is None else _add_mats(w, part)
        qa = cp.broadcast_to(q_abs[e], (b, n, n))
        qr = cp.broadcast_to(q_rad[e], (b, n, n))
        full = b_low + b_high + r
        ab = quad(b_low, qa, b_low)
        terms = {'round': g * ab, 'high': 2 * quad(b_high, qa, b_low) + quad(b_high, qa, b_high),
                 'models': 2 * quad(r, qa, b_low + b_high) + quad(r, qa, r),
                 'radius': quad(full, qr, full), 'abs': ab}
        for k, v in terms.items():
            parts[k] += v
        rem += (terms['round'] + terms['high'] + terms['models'] + terms['radius']) * SAFE
    # The entries of total degree above n go to the remainder with their exact sizes.
    out = _from_bidegree(sp4, sp_low, w, rem * SAFE, delta)
    out.parts = parts
    return out


def _add_mats(x, y):
    """x + y for complex dd arrays (B, 4, ...)."""
    shape = x.shape
    xf = cp.ascontiguousarray(cp.moveaxis(x, 1, 0)).reshape(4, -1)
    yf = cp.ascontiguousarray(cp.moveaxis(y, 1, 0)).reshape(4, -1)
    z = tmc._axpy(xf, yf, 1.0)
    return cp.ascontiguousarray(cp.moveaxis(z.reshape((4, shape[0]) + shape[2:]), 0, 1))


def _bnorm(sp_low, w, delta):
    """sum of abs(w_ij) delta^(deg i + deg j) for each box."""
    a = ddgemm.cabs(w)                                         # (B, S, S)
    d = cp.asarray(sp_low.deg)
    pw = delta[:, None] ** d[None, :]
    return cp.einsum('bij,bi,bj->b', a, pw, pw) * SAFE


def _bnorm_low(sp4, sp_low, w, delta):
    """sum of abs(w_ij) delta^(deg i + deg j) over the entries of total degree <= n."""
    rows, cols = tmc.split_index(sp4, sp_low)
    a = ddgemm.cabs(w)[:, cp.asarray(rows), cp.asarray(cols)]                    # (B, S4)
    d = cp.asarray(sp4.deg)
    return cp.sum(a * delta[:, None] ** d[None, :], axis=1) * SAFE


def _from_bidegree(sp4, sp_low, w, rem, delta, tail_done=False):
    rows, cols = tmc.split_index(sp4, sp_low)
    rows, cols = cp.asarray(rows), cp.asarray(cols)
    c = cp.moveaxis(w[:, :, rows, cols], 0, -1)                # (4, S4, B)
    if tail_done:
        return tmc.TM(cp.ascontiguousarray(c), rem * SAFE, delta)
    a = ddgemm.cabs(w)
    d = sp_low.deg
    mask = cp.asarray(((d[:, None] + d[None, :]) > sp4.n).astype(np.float64))
    dd = cp.asarray(d[:, None] + d[None, :])
    tail = cp.sum(a * mask[None] * delta[:, None, None] ** dd[None], axis=(1, 2)) * SAFE
    return tmc.TM(cp.ascontiguousarray(c), (rem + tail) * SAFE, delta)


def reexpand(local, zeta_host, k_max, top, tag='e', td=False):
    """Per box: the matrices of F_k = (1 + zeta)^k sum over m >= k of C(m, k) zeta^(m-k) M~_m.

    For each e this is one product of the weights (B, k_max + 1, top + 1), formed in
    mpmath and rounded to dd (2 u^2 each), with the stacked M~_m (top + 1, n^2) by
    `ddgemm.zgemm`. tag 'j' takes the R_j of tm_forms.delta_split in place of the Q_e;
    td takes the triple-double product, whose rounding is (lin + cub) abs(w) abs(M~).
    Returns, for each k <= k_max, a list over e of (q (B, 4, n, n),
    rad (B, n, n), abs (B, n, n)); rad holds the radii of the M~_m, the rounding of the
    weights and that of the product.
    """
    b = len(zeta_host)
    nm = top + 1
    w = np.zeros((b, 4, k_max + 1, nm))
    wabs = np.zeros((b, k_max + 1, nm))
    for i, z in enumerate(zeta_host):
        for k in range(k_max + 1):
            for m in range(k, nm):
                v = (1 + z) ** k * math.comb(m, k) * z ** (m - k)
                w[i, :, k, m] = mp_dd(v)
                wabs[i, k, m] = float(abs(v)) * (1 + 1e-12)
    wd = cp.asarray(w)
    wa = cp.asarray(wabs)
    g = sum(ddgemm.gamma_td(nm)) if td else ddgemm.gamma(nm)
    out = [[None] * 4 for _ in range(k_max + 1)]
    for e in range(4):
        n = local[f'mt0_{tag}{e}'].shape[-1]
        mt = cp.stack([local[f'mt{m}_{tag}{e}'].reshape(4, n * n) for m in range(nm)], axis=1)
        q = ddgemm.zgemm(wd, mt[None], mode=0, td=td)                   # (B, 4, K+1, n*n)
        ma = cp.stack([local[f'mt{m}_{tag}{e}_abs'].reshape(n * n) for m in range(nm)])
        mr = cp.stack([local[f'mt{m}_{tag}{e}_rad'].reshape(n * n) for m in range(nm)])
        absq = cp.matmul(wa, ma) * SAFE                                  # (B, K+1, n*n)
        rad = (cp.matmul(wa, mr) + (g + 2 * U2) * absq) * SAFE
        for k in range(k_max + 1):
            out[k][e] = (cp.ascontiguousarray(q[:, :, k].reshape(b, 4, n, n)),
                         rad[:, k].reshape(b, n, n), absq[:, k].reshape(b, n, n))
    return out


def local_on_gpu(path):
    """The local forms of one reference center on the GPU."""
    z = np.load(path)
    out = {}
    for key in z.files:
        out[key] = cp.asarray(z[key]) if z[key].ndim >= 2 else z[key]
    return out


def _series_order(eta, n, eps=1e-30):
    """The number of terms with eta^(k+1) below eps for the largest eta of the batch.

    The series of r and p~ give x, and psi takes the error of x about 2e3 times, so eps
    is 1e-30 and not the 1e-24 that the Cauchy estimate alone would allow.
    """
    e = float(cp.max(eta))
    if not 0 < e < 0.5:
        return n
    return int(min(n, max(4, math.ceil(math.log(eps) / math.log(e)) - 1)))


def _host_dd(x):
    """A CuPy dd array (4, B) as a list of mpmath complex numbers."""
    h = cp.asnumpy(x)
    return [dd_mp(h[:, i]) for i in range(h.shape[1])]


def _to_dd(vals):
    return cp.asarray(np.array([mp_dd(v) for v in vals]).T)


def _power(sp, x, m, cache):
    if m not in cache:
        half = _power(sp, x, m // 2, cache)
        cache[m] = (tmc.mul(sp, half, half) if m % 2 == 0
                    else tmc.mul(sp, _power(sp, x, m - 1, cache), x))
    return cache[m]


def form_abs(mats, q_abs, bdl=None):
    """A bound of a form on the polydisc from abs(Q): the sum of full^T abs(Q) full over e.

    With bdl (the bound of Delta), the blocks are the R_j and block j has the factor bdl^j.
    """
    out = None
    for e in range(4):
        _, b_low, b_high, r = mats[0 if bdl is not None else e][:4]
        n = q_abs[e].shape[-1]
        full = (b_low + b_high + r)[:n]
        v = cp.einsum('ab,bac,cb->b', full, q_abs[e], full)
        if bdl is not None:
            v = v * bdl ** e
        out = v if out is None else out + v
    return out * SAFE


def reexpand_abs(local, zeta_host, k, top, tag='e'):
    """abs bounds of the matrices of F_k only, (B, n, n) for each e."""
    b = len(zeta_host)
    ms = list(range(k, top + 1))
    wa = [cp.asarray(np.array([float(abs((1 + z) ** k * math.comb(m, k) * z ** (m - k)))
                               for z in zeta_host]) * (1 + 1e-12)) for m in ms]
    out = []
    for e in range(4):
        acc = None
        for w, m in zip(wa, ms):
            a = local[f'mt{m}_{tag}{e}_abs'][None] * w[:, None, None]
            acc = a if acc is None else acc + a
        out.append(acc * SAFE)
    return out


def delta_model(sp4, sp2, fr, v, local, delta):
    """Delta = abs(y_2)^2 - s2 = (abs(p_2)^2 - s2) + conj(p_2) v + p_2 conj(v) + v conj(v).

    v = y_2 - p_2 has no constant from p_2, and abs(p_2)^2 - s2 comes from the host, so
    Delta has no rounding of the size of abs(y_2)^2.
    """
    b = delta.shape[0]
    s2 = dd_mp(np.asarray(local['s2'])).real
    vals = [abs(hb['p'][2]) ** 2 - s2 for hb in fr.host]
    d0 = _to_dd(vals)
    err = cp.asarray(np.array([float(abs(dd_mp(mp_dd(x)) - x)) * (1 + 1e-9) + 1e-60
                               for x in vals]))
    one = tmc.const(sp2, tmc.cconst(1, b), delta)
    vc = conj_tm(v)
    p2c = fr.p[2].copy()
    p2c[2:] = -p2c[2:]
    t1 = tmc.tensor(sp4, sp2, tmc.scale(sp2, v, p2c), one)
    t2 = tmc.tensor(sp4, sp2, one, tmc.scale(sp2, vc, fr.p[2]))
    t3 = tmc.tensor(sp4, sp2, v, vc)
    x = tmc.add_const(sp4, tmc.add(sp4, tmc.add(sp4, t1, t2), t3), d0)
    return tmc.TM(x.c, (x.rem + err) * SAFE, x.delta)


def delta_form(sp4, sp_low, mats, q, q_rad, q_abs, delta, dl, bdl, td=False):
    """sum over j of Delta^j W_j with W_j = Y R_j Y^H, by Horner's rule in Delta.

    The parts of the remainder are those of the W_j, with the factor bdl^j.
    """
    ws = [compose(sp4, sp_low, mats[:1], [q[j]], [q_rad[j]], [q_abs[j]], delta, td)
          for j in range(4)]
    s = ws[3]
    for j in (2, 1, 0):
        s = tmc.add(sp4, ws[j], tmc.mul(sp4, dl, s))
    s.parts = {key: sum(ws[j].parts[key] * bdl ** j for j in range(4)) for key in ws[0].parts}
    return s


def box_f(fr, local, delta_box, n, ny, k_max=20, a=None, k_full=3, td=False):
    """The model of f on a batch of boxes that share the reference center of `local`.

    td: the compositions with the forms take the triple-double products of ddgemm.

    Returns f (on delta_f), the radii (box, f, Phi) and the parts of the remainder.
    """
    k0, top = int(local['k0']), int(local['top'])
    a = 2.0 / n if a is None else a
    b = len(delta_box)
    d = cp.asarray(np.asarray(delta_box, float))
    d_f = d * (1 + a)
    d_phi = d_f * (1 + a)
    sp2, sp_low, sp4 = _space(2, ny), _space(2, n), _space(4, n)
    info = {}
    clock = _Clock()
    # The forms are sums over e of abs(y_2)^(2e) Q_e, or over j of Delta^j R_j.
    mode_j = 's2' in local
    tag = 'j' if mode_j else 'e'
    ys, rho, lam, hm = implicit_h(sp2, fr, d_phi)
    info['rho_h'], info['lam_h'] = rho, lam
    yc = [conj_tm(y) for y in ys]
    cref = [dd_mp(cp.asnumpy(cp.asarray(local['c']))[i]) for i in range(2)]
    base_t, v2 = offset_models(sp2, fr, hm, cref, d_phi)
    clock('h')
    mats = monomial_mats(sp2, sp_low, base_t, None if mode_j else ys[2], top)
    dls, bdl = {}, None
    if mode_j:
        dls[n] = delta_model(sp4, sp2, fr, v2, local, d_phi)
        bdl = tmc.bound(sp4, dls[n])
        info['bound_delta'] = bdl
    clock('monomials')

    def form(sp4k, splk, mk, q, rr, aa):
        if not mode_j:
            return compose(sp4k, splk, mk, q, rr, aa, d_phi, td)
        if sp4k.n not in dls:
            dls[sp4k.n] = tmc.truncate(sp4k, sp4, dls[n])
        return delta_form(sp4k, splk, mk, q, rr, aa, d_phi, dls[sp4k.n], bdl, td)

    # The base and abs(y)^2.
    qb = [local[f'base_{tag}{e}'][None] for e in range(4)]
    rb = [local[f'base_{tag}{e}_rad'][None] for e in range(4)]
    ab = [local[f'base_{tag}{e}_abs'][None] for e in range(4)]
    p0 = form(sp4, sp_low, mats, qb, rb, ab)
    one = tmc.cconst(1, b)
    r2 = tmc.const(sp4, one, d_phi)
    for k in range(3):
        r2 = tmc.add(sp4, r2, tmc.tensor(sp4, sp2, ys[k], yc[k]))
    # With any exact s, 1/r2 = s (1 + yr)^-1 and log r2 = log(1 + yr) - log s, yr = s r2 - 1.
    s_r = _to_dd([1 / v for v in _host_dd(r2.c[:, 0, :])])
    yr = tmc.add_const(sp4, tmc.scale(sp4, r2, s_r), -one)
    eta_r = tmc.bound(sp4, yr)
    pw_r = tmc.powers(sp4, yr, _series_order(eta_r, n))
    ir2 = tmc.scale(sp4, tmc.power_series(sp4, pw_r, -1, eta_r), s_r)
    log_r2 = tmc.log1p_series(sp4, pw_r, eta_r)
    cache = {1: ir2}
    pt = tmc.mul(sp4, p0, _power(sp4, ir2, k0, cache))
    s_p = _to_dd([1 / v for v in _host_dd(pt.c[:, 0, :])])
    yp = tmc.add_const(sp4, tmc.scale(sp4, pt, s_p), -one)
    eta_p = tmc.bound(sp4, yp)
    pw_p = tmc.powers(sp4, yp, _series_order(eta_p, n))
    xh = tmc.power_series(sp4, pw_p, Fraction(-1, k0), eta_p)
    log_pt = tmc.log1p_series(sp4, pw_p, eta_p)
    info.update(eta_r=eta_r, eta_p=eta_p, rem_p0=p0.rem / tmc.abs_ub(p0.c[:, :1, :])[0],
                parts_p0=p0.parts)

    # x = r2^-1 p~^(-1/k0) = ir2 xh s_p^(1/k0); x_c = x0 (1 + zeta) with an exact dd zeta.
    x0 = dd_mp(np.asarray(local['x0']))
    sp_host = _host_dd(s_p)
    xc_raw = [v.real ** (mp.mpf(1) / k0) * w for v, w in
              zip(sp_host, _host_dd(tmc.mul(sp4, ir2, xh).c[:, 0, :]))]
    zeta = [dd_mp(mp_dd(x / x0 - 1)) for x in xc_raw]
    kappa = [v.real ** (mp.mpf(1) / k0) / (x0 * (1 + z)) for v, z in zip(sp_host, zeta)]
    kappa_dd = _to_dd(kappa)
    kap_err = cp.asarray(np.array([float(abs(dd_mp(mp_dd(k)) - k)) for k in kappa]) + 1e-300)
    xw = tmc.mul(sp4, ir2, xh)
    xi = tmc.add_const(sp4, tmc.scale(sp4, xw, kappa_dd), -one)
    xi.rem = (xi.rem + kap_err * tmc.bound(sp4, xw)) * SAFE
    bxi = tmc.bound(sp4, xi)

    clock('base')
    # F_k for k <= k_max as models; the rest through a bound.
    fq = reexpand(local, zeta, k_max, top, tag, td)
    clock('reexpand')
    # F_k enters psi as xi^k F_k, and xi is of degree 1 up to a rounding, so F_k needs
    # only the order n - (k - k_full) for k > k_full; the rest goes to its remainder.
    orders = [n if k <= k_full else max(n - (k - k_full), 4) for k in range(k_max + 1)]
    fk = []
    for k in range(k_max + 1):
        q = [x[0] for x in fq[k]]
        rr = [x[1] for x in fq[k]]
        aa = [x[2] for x in fq[k]]
        sp4k, splk = _space(4, orders[k]), _space(2, orders[k])
        mk = mats if orders[k] == n else mats_at(mats, splk)
        fk.append(form(sp4k, splk, mk, q, rr, aa))
    tail = cp.zeros(b)
    for k in range(k_max + 1, top + 1):
        tail += form_abs(mats, reexpand_abs(local, zeta, k, top, tag), bdl) * bxi ** k
    clock('forms')
    xis = {n: xi}
    for m in set(orders):
        if m not in xis:
            xis[m] = tmc.truncate(_space(4, m), sp4, xi)
    s = fk[k_max]
    for k in range(k_max - 1, -1, -1):
        spk = _space(4, orders[k])
        if orders[k + 1] < orders[k]:
            s = tmc.embed(spk, s)
        s = tmc.add(spk, fk[k], tmc.mul(spk, xis[orders[k]], s))
    psi = tmc.TM(s.c, (s.rem + tail * SAFE) * SAFE, s.delta)
    clock('level sum')
    phi = tmc.add(sp4, tmc.add(sp4, tmc.scale(sp4, log_r2, tmc.cconst(k0, b)), log_pt), psi)
    info.update(parts_f0=fk[0].parts, value_f0=tmc.abs_ub(fk[0].c[:, :1, :])[0])
    info.update(rem_f0=fk[0].rem, rem_psi=psi.rem, rem_logpt=log_pt.rem, rem_logr2=log_r2.rem,
                rem_phi=phi.rem, bxi=bxi, tail=tail, zeta=zeta)

    # The metric, the volume and f.
    g = {(i, j): tmc.cauchy_mixed(sp4, phi, i, 2 + j, d_f) for i in range(2) for j in range(2)}
    det = tmc.sub(sp4, tmc.mul(sp4, g[0, 0], g[1, 1]), tmc.mul(sp4, g[0, 1], g[1, 0]))
    dp = None
    for k in range(3):
        cube = tmc.mul(sp2, tmc.mul(sp2, ys[k], ys[k]), ys[k])
        term = tmc.scale(sp2, cube, 4 * fr.n[k])
        dp = term if dp is None else tmc.add(sp2, dp, term)
    dp = tmc.restrict(dp, d_f)
    q4 = tmc.tensor(sp4, sp2, dp, conj_tm(dp))
    dq = tmc.mul(sp4, det, q4)
    v = tmc.scale(sp4, dq, fr.vol)
    v.rem = (v.rem + fr.vol_err * tmc.abs_ub(fr.vol[:, None, :])[0] * tmc.bound(sp4, dq)) * SAFE
    f = tmc.add_const(sp4, v, -one)
    clock('metric')
    info.update(rem_g=g[0, 0].rem, rem_f=f.rem, times=clock.times)
    return f, (d, d_f, d_phi), info, dict(p0=p0, f0=fk[0], ys=ys, sp4=sp4, sp2=sp2,
                                          delta=dls.get(n))


_SPACES = {}


def _space(nvars, n):
    """One Space for each (nvars, n), so that its tables are built once."""
    if (nvars, n) not in _SPACES:
        _SPACES[nvars, n] = tmc.Space(nvars, n)
    return _SPACES[nvars, n]


class _Clock:
    """Wall time of the stages, with a device synchronization at each mark."""

    def __init__(self):
        import time
        self._time = time.perf_counter
        cp.cuda.Device().synchronize()
        self.t = self._time()
        self.times = {}

    def __call__(self, name):
        cp.cuda.Device().synchronize()
        now = self._time()
        self.times[name] = now - self.t
        self.t = now


def sup_bound(sp4, f, d):
    """abs(f) <= abs(f(c)) + B(f - f(c)) on the box delta d + the remainder of f."""
    c0 = tmc.abs_ub(f.c[:, :1, :])[0]
    var = f.c.copy()
    var[:, 0, :] = 0
    pb = tmc.poly_bound(sp4, tmc.TM(var, f.rem, d), d)
    return (c0 + pb + f.rem) * SAFE, c0, pb
