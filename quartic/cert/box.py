"""The Taylor model of f on a batch of boxes on the GPU, for the strict bound of sup abs(f).

The design is docs/superpowers/specs/2026-10-09-gpu-taylor-certificate-design.md. All
boxes of a batch share one reference center c_r and its local forms
(scripts/tm_local_forms.py), stored in td, each form as sum over j of Delta^j R_j with
Delta = abs(y_2)^2 - s2.

1. Host: each box has a center p on X (t = (y_0, y_1), y_2 from P = 0), the frame
   E = dy/dt, the unit normal n and 1/c1 = 1/(grad P . n), all exact dd numbers.
2. GPU, models in tau alone (order n_y), in td: h from the fixed point
   h -> h - P(y(h)) / c1 with a contraction bound, y(tau) = p + E tau + h n, and the
   monomials (t - c_r)^alpha.
3. Composition W_j = Y R_j Y^H with the local forms, up to the degree n in tau and in
   sigma, summed by Horner's rule in Delta; the parts above n go to the remainder through
   bounds.
4. The base: r = abs(y)^2, p~ = p0 r^(-k0), log p0 = k0 log r + log p~.
5. The level sum: x = p0^(-1/k0) = r^(-1) p~^(-1/k0), xi = x / x_c - 1 with
   x_c = x0 (1 + zeta), and psi = sum over k of F_k xi^k with
   F_k = (1 + zeta)^k sum over m >= k of C(m, k) zeta^(m - k) M~_m, formed per box on the
   local forms, for every k up to the top level.
6. Phi = log p0 + psi, g = d_tau d_sigma Phi (Cauchy), det g, v, f = v / vbar - 1.

Every round-off of the GPU goes into a remainder by the bounds of `tmgpu.tm`,
`tmgpu.tmtd` and `tmgpu.gemm`. Each FP64 operation on a bound in this module uses `tmgpu.mag`:
sums, products, quotients and powers of bounds round up, and lower bounds round down. No
constant factor covers a round-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import cupy as cp
import mpmath as mp

from tmgpu import gemm, mag, tm, tmtd
from tmgpu.host import (dd_frac, dd_mp, down, exact, frac_dd, mp_dd, precision,
                        to_td as ball_td, upper)
from tmgpu.tools import conj, gpu_dd, host_dd, power, series_order, space

from .forms import Poly2
from .frame import frame, kappa

# The series of r and p~ stop at eta^(k+1) < SERIES_EPS: psi takes the error of x about
# 2e3 times, and the Cauchy estimate of g about 1e5 times.
SERIES_EPS = 1e-34
# Every host number that enters a bound comes from exact rationals (dd_frac, frac_dd) or
# from ball arithmetic (tmgpu.host), so no bound depends on the precision of mpmath.
# mpmath only picks free numbers (s_r, s_p, zeta); each is rounded to dd and then exact.
# F_k for k > K_FULL enters psi as xi^k F_k, so it needs only the order n - (k - K_FULL).
K_FULL = 3
# The models in tau alone (h, the offsets, the monomials) take the order of the box plus this.
NY_EXTRA = 6


@dataclass
class Frames:
    """Exact dd numbers of the boxes, each (..., 4, B) on the GPU, and host values."""
    p: cp.ndarray          # (3, 4, B)
    e: cp.ndarray          # (2, 3, 4, B): e[i] = dy/dt_i
    n: cp.ndarray          # (3, 4, B)
    c1inv: cp.ndarray      # (4, B)
    vol: cp.ndarray        # (4, B): 1 / (abs(det U)^2 vbar), U = [E, n]
    vol_err: cp.ndarray    # (B,): a bound of the error of vol relative to the true vol
    host: list             # p, e, n in mpmath and p as dd lists, for each box
    ha_td: cp.ndarray      # (3, 6, B): c1inv P(p), c1inv grad.E_0, c1inv grad.E_1
    a6_td: cp.ndarray      # (3, 6, B): 6 p_k^2
    a4_td: cp.ndarray      # (3, 6, B): 4 p_k
    kappa_td: cp.ndarray   # (6, B): kappa = 1 - c1inv c1
    h_err_td: cp.ndarray   # (6, B): the errors of ha_td (3) and kappa_td, and the
                           # largest errors of a6_td and a4_td


def make_frames(centers, k0):
    """The frames of boxes with centers t = (t_0, t_1), each given as two dd lists.

    Every number and every error bound comes from `quartic.cert.frame.frame` in ball
    arithmetic, so the boxes and the cover (quartic.cert.cover) use the same frame.
    """
    keys = ('p', 'e', 'n', 'c1inv', 'vol', 'ha_td', 'a6_td', 'a4_td', 'kappa_td')
    cols = {k: [] for k in keys}
    errs, host, herr_td = [], [], []
    for t in centers:
        f = frame(t, k0)
        for k in keys:
            cols[k].append(f[k])
        errs.append(f['vol_err'])
        herr_td.append(f['h_err_td'])
        host.append(dict(p=[dd_mp(v) for v in f['p']], e=[[dd_mp(v) for v in row] for row in f['e']],
                         n=[dd_mp(v) for v in f['n']], p_dd=[list(v) for v in f['p']]))
    to = lambda key: cp.asarray(np.moveaxis(np.array(cols[key], dtype=np.float64), 0, -1))
    return Frames(p=to('p'), e=to('e'), n=to('n'), c1inv=to('c1inv'), vol=to('vol'),
                  vol_err=cp.asarray(np.array(errs)), host=host, ha_td=to('ha_td'),
                  a6_td=to('a6_td'), a4_td=to('a4_td'), kappa_td=to('kappa_td'),
                  h_err_td=cp.asarray(np.array(herr_td).T))


def lin_tm(sp, p, e0, e1, delta):
    """p + e0 tau_0 + e1 tau_1 with dd numbers (4, B): exact, with no rounding."""
    c = tm.zeros(sp, delta.shape[0])
    c[:, 0, :] = p
    c[:, sp.index[(1, 0)], :] = e0
    c[:, sp.index[(0, 1)], :] = e1
    return tm.TM(c, cp.zeros_like(delta), delta)


def offset_consts(fr, ref):
    """p_i - ref_i (i = 0, 1) of each box from the exact rationals of the dd numbers, as td
    arrays (6, B), and a bound of each error."""
    out = []
    for i in range(2):
        rq = dd_frac(ref[i])
        vals = [frac_dd(*(a - c for a, c in zip(dd_frac(hb['p_dd'][i]), rq)), limbs=3)
                for hb in fr.host]
        out.append((cp.asarray(np.array([v for v, _ in vals]).T),
                    cp.asarray(np.array([e for _, e in vals]))))
    return out


def mats_at(mats, sp_k):
    """The monomial matrices for the lower order sp_k.n: the rows of degree <= sp_k.n, and
    the bounds split at that degree."""
    y, _, _, r, gdeg = mats
    cut = sp_k.n + 1
    b_low = mag.sum(gdeg[:, :cut], axis=1)
    b_high = mag.sum(gdeg[:, cut:], axis=1)
    return cp.ascontiguousarray(y[:, :, :sp_k.size]), b_low, b_high, r


def _from_bidegree(sp4, sp_low, w, rem, delta):
    rows, cols = tm.split_index(sp4, sp_low)
    rows, cols = cp.asarray(rows), cp.asarray(cols)
    c = cp.moveaxis(w[:, :, rows, cols], 0, -1)                # (4, S4, B)
    a = tm.abs_ub(w, axis=1)
    d = sp_low.deg
    high = cp.asarray((d[:, None] + d[None, :]) > sp4.n)
    dd = cp.asarray(d[:, None] + d[None, :])
    # delta^(deg_i + deg_j) (B, S_low, S_low): an exact gather of the upward powers of delta.
    pw = cp.moveaxis(mag.powers(delta, 2 * sp_low.n)[dd], -1, 0)
    tail = mag.sum(cp.where(high[None], mag.mul(a, pw), 0.0), axis=(1, 2))
    return tm.TM(cp.ascontiguousarray(c), mag.add(rem, tail), delta)


def reexpand_weights(zeta_dd, top):
    """The weights (1 + zeta)^k C(m, k) zeta^(m - k), k <= m <= top, of `reexpand_td` in ball
    arithmetic, from the exact dd zeta of each box: the weights rounded to td
    (B, 6, top + 1, top + 1), and bounds of their errors and of their moduli
    (B, top + 1, top + 1)."""
    from flint import acb
    b, nm = len(zeta_dd), top + 1
    w = np.zeros((b, 6, nm, nm))
    werr = np.zeros((b, nm, nm))
    wabs = np.zeros((b, nm, nm))
    with precision():
        for i, zd in enumerate(zeta_dd):
            z = exact(zd)
            zp, op = [acb(1)], [acb(1)]
            for _ in range(top):
                zp.append(zp[-1] * z)
            for _ in range(top):
                op.append(op[-1] * (1 + z))
            for k in range(nm):
                for m in range(k, nm):
                    v = op[k] * zp[m - k] * math.comb(m, k)
                    w[i, :, k, m], werr[i, k, m] = ball_td(v)
                    wabs[i, k, m] = upper(v)
    return w, werr, wabs


def local_on_gpu(path):
    """The local forms of one reference center on the GPU."""
    z = np.load(path)
    out = {}
    for key in z.files:
        out[key] = cp.asarray(z[key]) if z[key].ndim >= 2 else z[key]
    return out


def delta_const(fr, s2_dd):
    """abs(p_2)^2 - s2 of each box from the exact rationals of the dd numbers, as a dd
    array (4, B), and a bound of each error (B,)."""
    s2 = dd_frac(s2_dd)[0]
    vals = []
    for hb in fr.host:
        re, im = dd_frac(hb['p_dd'][2])
        vals.append(frac_dd(re * re + im * im - s2, Fraction(0)))
    return (cp.asarray(np.array([v for v, _ in vals]).T),
            cp.asarray(np.array([e for _, e in vals])))


def delta_model(sp4, sp2, fr, v, local, delta):
    """Delta = abs(y_2)^2 - s2 = (abs(p_2)^2 - s2) + conj(p_2) v + p_2 conj(v) + v conj(v).

    v = y_2 - p_2 has no constant from p_2, and abs(p_2)^2 - s2 comes from the host as an
    exact rational, rounded to dd with its error, so Delta has no round-off of the size of
    abs(y_2)^2.
    """
    b = delta.shape[0]
    d0, err = delta_const(fr, np.asarray(local['s2']))
    one = tm.const(sp2, tm.cconst(1, b), delta)
    vc = conj(v)
    p2c = fr.p[2].copy()
    p2c[2:] = -p2c[2:]
    t1 = tm.tensor(sp4, sp2, tm.scale(sp2, v, p2c), one)
    t2 = tm.tensor(sp4, sp2, one, tm.scale(sp2, vc, fr.p[2]))
    t3 = tm.tensor(sp4, sp2, v, vc)
    x = tm.add_const(sp4, tm.add(sp4, tm.add(sp4, t1, t2), t3), d0)
    return tm.TM(x.c, mag.add(x.rem, err), x.delta)


# ---- h, the offsets, the monomials, the re-expansion and the compositions run in td: near
# the edge of a reference cell the forms cancel by up to 1e6, and dd errors grow as much.

def lin_tmt(sp, p6, e0, e1, delta):
    """p + e0 tau_0 + e1 tau_1 as a td model, from td numbers (6, B): exact."""
    c = cp.zeros((6, sp.size, delta.shape[0]))
    c[:, 0, :] = p6
    c[:, sp.index[(1, 0)], :] = e0
    c[:, sp.index[(0, 1)], :] = e1
    return tmtd.TMT(c, cp.zeros_like(delta), delta)


def implicit_h_td(sp, fr, delta, hmax, parts=None):
    """The models of y(tau) = p + E tau + h n on X, with a strict remainder of h, in td.

    T(h) = h - c1inv P(y(h)). If T maps the ball abs(h - P_h) <= rho into itself with a
    contraction factor lam < 1, the root of each tau lies in that ball. With w = E tau + h n,
    T(h) = kappa h - A(tau) - c1inv N(w): kappa = 1 - c1inv c1, A = c1inv (P(p) + grad.E tau)
    and N(w) = sum over k of w_k^2 (6 p_k^2 + w_k (4 p_k + w_k)). The host gives kappa, A,
    6 p_k^2 and 4 p_k as td numbers with their errors, so the round-off acts only on the
    small terms of N. The residual of the td model of h, with those errors, gives rho; the
    contraction factors come from the dd models, as bounds.

    With hmax (B,), a bound of abs(h) for every point of X over the cell of the box (from
    the cover), T must also contract on the ball of radius R = hmax + B(h) + rho around
    the model, which holds both that point and the root: then they are the same point.
    A box whose bound lam_big of T' on that ball is not below 1 gets rho = inf. Returns the
    dd models ys, rho, lam, the td model of h, its dd model, and lam_big. With a dict
    `parts`, it gets the arrays residual (the bound of T(h) - h), host (the host errors
    eps), lam0 and rho.

    The trial radius r = 2 bd / (1 - lam0) is a free number. Each check forms its left side
    upward from that r, so a radius that passes the checks is a valid rho.
    """
    b = delta.shape[0]
    T6 = tmtd.tconst
    zero6 = cp.zeros((6, b))
    lin0 = [lin_tmt(sp, zero6, T6(fr.e[0, k], b), T6(fr.e[1, k], b), delta) for k in range(3)]
    a_tau = lin_tmt(sp, fr.ha_td[0], fr.ha_td[1], fr.ha_td[2], delta)
    n6 = [T6(fr.n[k], b) for k in range(3)]
    c1inv6 = T6(fr.c1inv, b)

    def t_map(h):
        ws = [tmtd.add(sp, lin0[k], tmtd.scale(sp, h, n6[k])) for k in range(3)]
        nw = None
        for k in range(3):
            inner = tmtd.add_const(sp, tmtd.mul(sp, ws[k], tmtd.add_const(sp, ws[k], fr.a4_td[k])),
                                   fr.a6_td[k])
            term = tmtd.mul(sp, tmtd.mul(sp, ws[k], ws[k]), inner)
            nw = term if nw is None else tmtd.add(sp, nw, term)
        rest = tmtd.add(sp, a_tau, tmtd.scale(sp, nw, c1inv6))
        return tmtd.sub(sp, tmtd.scale(sp, h, fr.kappa_td), rest), ws

    h = tmtd.TMT(cp.zeros((6, sp.size, b)), cp.zeros(b), delta)
    for _ in range(sp.n + 2):
        h, _ = t_map(h)
        h = tmtd.TMT(h.c, cp.zeros(b), delta)
    th, ws = t_map(h)
    bw = [tmtd.bound(sp, w) for w in ws]
    iabs = tm.abs_ub(fr.c1inv[:, None, :])[0]
    he = fr.h_err_td
    cubic = [mag.add(mag.mul(he[4], mag.powi(x, 2)), mag.mul(he[5], mag.powi(x, 3))) for x in bw]
    eps = mag.add(he[0], mag.mul(mag.add(he[1], he[2]), delta),
                  mag.mul(he[3], tmtd.bound(sp, h)), mag.mul(iabs, mag.add(*cubic)))
    residual = tmtd.bound(sp, tmtd.sub(sp, th, h))
    bd = mag.add(residual, eps)
    h_dd = tmtd.to_dd(sp, h)
    lin = [lin_tm(sp, fr.p[k], fr.e[0, k], fr.e[1, k], delta) for k in range(3)]
    ys = [tm.add(sp, lin[k], tm.scale(sp, h_dd, fr.n[k])) for k in range(3)]
    dp = None
    for k in range(3):
        cube = tm.mul(sp, tm.mul(sp, ys[k], ys[k]), ys[k])
        term = tm.scale(sp, cube, 4 * fr.n[k])
        dp = term if dp is None else tm.add(sp, dp, term)
    lam0 = tm.bound(sp, tm.add_const(sp, tm.scale(sp, dp, -fr.c1inv), tm.cconst(1, b)))
    by = [tm.bound(sp, y) for y in ys]
    nabs = [tm.abs_ub(fr.n[k][:, None, :])[0] for k in range(3)]
    ok = lam0 < 1
    # A free trial radius in round to nearest; scripts/cert_steps.py copies this formula.
    r = cp.where(ok, 2 * bd / cp.where(ok, 1 - lam0, 1.0), cp.inf)
    lam, good, lam_big = _h_checks(r, bd, lam0, iabs, nabs, by, hmax, tm.bound(sp, h_dd))
    rho = cp.where((lam_big < 1) & good, r, cp.inf)
    if parts is not None:
        parts.update(residual=residual, host=eps, lam0=lam0, rho=rho)
    h_td = tmtd.TMT(h.c, rho, delta)
    h_dd = tm.TM(h_dd.c, mag.add(h_dd.rem, rho), delta)
    ys = [tm.add(sp, lin[k], tm.scale(sp, h_dd, fr.n[k])) for k in range(3)]
    return ys, rho, lam, h_td, h_dd, lam_big


def _h_checks(r, bd, lam0, iabs, nabs, by, hmax, bh):
    """The checks of implicit_h_td for the trial radius r (B,), each left side upward.

    lam bounds T' on the ball of radius r, and lam_big on the ball of radius hmax + bh + r,
    with lam0 >= abs(T') at the model, iabs >= abs(c1inv), nabs[k] >= abs(n_k) and
    by[k] >= B(y_k). Returns lam, the mask good of lam0 < 1, lam < 1 and bd + lam r <= r,
    and lam_big.
    """
    def second(rad):
        """An upper bound of the sum over k of 12 abs(n_k)^2 (B(y_k) + rad abs(n_k))^2."""
        return mag.add(*[mag.mul(mag.mul(12.0, mag.powi(nabs[k], 2)),
                                 mag.powi(mag.fma(rad, nabs[k], by[k]), 2)) for k in range(3)])

    lam = mag.add(lam0, mag.mul(mag.mul(r, second(r)), iabs))
    good = (lam0 < 1) & (lam < 1) & (mag.fma(lam, r, bd) <= r)
    big = mag.add(cp.asarray(hmax), bh, r)
    lam_big = mag.add(lam0, mag.mul(mag.mul(big, second(big)), iabs))
    return lam, good, lam_big


def offset_models_td(sp, fr, h_td, h_dd, ref, delta):
    """The td models of t_i - ref_i (i = 0, 1) and the dd model of v = y_2 - p_2.

    The constant p_i - ref_i comes from the host, rounded to td with its error, so no
    difference of two near values rounds on the GPU. ref: the two dd lists of c_r.
    """
    b = delta.shape[0]
    out = []
    for i, (dp6, err) in enumerate(offset_consts(fr, ref)):
        x = tmtd.add(sp, lin_tmt(sp, dp6, tmtd.tconst(fr.e[0, i], b),
                                 tmtd.tconst(fr.e[1, i], b), delta),
                     tmtd.scale(sp, h_td, tmtd.tconst(fr.n[i], b)))
        out.append(tmtd.TMT(x.c, mag.add(x.rem, err), x.delta))
    v = tm.add(sp, lin_tm(sp, cp.zeros((4, b)), fr.e[0, 2], fr.e[1, 2], delta),
                tm.scale(sp, h_dd, fr.n[2]))
    return out, v


def _batched_mul_td(sp, xs, ys):
    """The products xs[i] ys[i] of lists of td models, in one call."""
    b = xs[0].delta.shape[0]
    cat = lambda ms: tmtd.TMT(cp.concatenate([m.c for m in ms], axis=2),
                              cp.concatenate([m.rem for m in ms]),
                              cp.concatenate([m.delta for m in ms]))
    z = tmtd.mul(sp, cat(xs), cat(ys))
    return [tmtd.TMT(z.c[:, :, i * b:(i + 1) * b], z.rem[i * b:(i + 1) * b],
                     z.delta[i * b:(i + 1) * b]) for i in range(len(xs))]


def monomial_mats_td(sp2, sp_low, base_t, top):
    """The td coefficients of the monomials (t - c_r)^alpha, alpha of degree <= top.

    base_t: the td models of t_i - c_r,i (offset_models_td). Returns (Y, b_low, b_high,
    rem, gdeg): Y (B, 6, S_low, n_top) for the degrees <= sp_low.n in tau; (n_top, B)
    bounds of that part, of the rest of the polynomial and of the remainder; gdeg, the
    bound of each degree.
    """
    p2 = Poly2(top)
    b = base_t[0].delta.shape[0]
    one = cp.zeros((6, b))
    one[0] = 1.0
    mono = {(0, 0): tmtd.const(sp2, one, base_t[0].delta)}
    for d in range(1, top + 1):
        layer = [(d - j, j) for j in range(d + 1)]
        srcs = [mono[(a - 1, c)] if a else mono[(0, c - 1)] for a, c in layer]
        facs = [base_t[0] if a else base_t[1] for a, c in layer]
        for m, x in zip(layer, _batched_mul_td(sp2, srcs, facs)):
            mono[m] = x
    # The monomials of a Space come in degree order, so the first sp_low.size of them are
    # those of degree <= sp_low.n.
    low = sp_low.size
    # delta^deg (S2, B): an exact gather of the upward powers of delta.
    pw = mag.powers(base_t[0].delta, sp2.n)[cp.asarray(sp2.deg)]
    tms = [mono[m] for m in p2.mons]
    cs = cp.stack([t.c for t in tms], axis=2)                 # (6, S2, n_top, B)
    a = tmtd.abs_ub(cs.reshape(6, sp2.size, -1)).reshape(sp2.size, len(tms), b)
    w = mag.mul(a, pw[:, None, :])
    b_low = mag.sum(w[:low], axis=0)
    b_high = mag.sum(w[low:], axis=0)
    rem = cp.stack([t.rem for t in tms])
    y = cp.moveaxis(cs[:, :low], -1, 0)                       # (B, 6, low, n_top)
    # The upward sum of w over the monomials of each degree, (n_top, n_y + 1, B).
    gdeg = cp.ascontiguousarray(cp.moveaxis(tm.degree_sums(sp2, w), 0, 1))
    return cp.ascontiguousarray(y), b_low, b_high, rem, gdeg


def reexpand_td(local, zeta_dd, top):
    """Per box: the td matrices of F_k = (1 + zeta)^k sum over m >= k of C(m, k)
    zeta^(m - k) M~_m, for k <= top and each block j, with rad and abs.

    For each j this is one product of the td weights of `reexpand_weights` with the stacked
    td local forms. The error of the product has the bound that the kernel of
    `gemm.zgemm_tt` sums as it runs. With w the weights and M the stored forms,
    w M - w_true M_true = (w - w_true) M + w_true (M - M_true). The modulus bound ma of M_true
    also bounds the stored M. This follows from `forms.to_td_complex`. There ma is up of
    abs(M_true) plus the radius, and abs(M) <= abs(midpoint) + radius. So
    rad = wa mr + we ma + qerr needs no term we mr."""
    b = len(zeta_dd)
    nm = top + 1
    wd, we, wa = (cp.asarray(x) for x in reexpand_weights(zeta_dd, top))
    out = [[None] * 4 for _ in range(nm)]
    for e in range(4):
        n = local[f'mt0_j{e}'].shape[-1]
        mt = cp.stack([local[f'mt{m}_j{e}'].reshape(6, n * n) for m in range(nm)], axis=1)
        q, qerr = gemm.zgemm_tt(wd, mt[None], mode=0, out_td=True)  # (B, 6, K+1, n*n)
        ma = cp.stack([local[f'mt{m}_j{e}_abs'].reshape(n * n) for m in range(nm)])
        mr = cp.stack([local[f'mt{m}_j{e}_rad'].reshape(n * n) for m in range(nm)])
        absq = mag.matmul(wa, ma)                                    # (B, K+1, n*n)
        rad = mag.add(mag.matmul(wa, mr), mag.matmul(we, ma), qerr)
        for k in range(nm):
            out[k][e] = (cp.ascontiguousarray(q[:, :, k].reshape(b, 6, n, n)),
                         rad[:, k].reshape(b, n, n), absq[:, k].reshape(b, n, n))
        del q, qerr
    return out


def compose_td(sp4, sp_low, mats, q, q_rad, q_abs, delta):
    """The model in four variables of Y Q Y^H for one block, with all remainders.

    T = Y Q in td, W = T Y^H to dd. The true form differs from the polynomial of total
    degree <= n by the entries of W of higher total degree (their exact sizes; a majorant
    from abs(Q) is far too large, because those entries cancel strongly), by the parts of
    degree above sp_low.n of the monomial models, by their remainders, by the radii of Q,
    and by the round-off of the two products.

    The part round is the bound that the kernels of the two products sum as they run: that of
    T enters W through abs(Y), that of W directly, each weighted by delta^deg.

    The cross terms act on the true Q. Each monomial is Y_low + Y_high + E: the low part, the
    high part and the remainder of its model. The true form minus Y_low Q_stored Y_low^H then
    has two kinds of terms. The terms with Y_high or E act on Q_true, and the parts high and
    models bound them with q_abs. The other term is Y_low (Q_true - Q_stored) Y_low^H. The
    part radius covers it, with full in place of b_low on both sides. So q_abs must bound
    abs(Q_true) only. The factor 2 of the cross terms needs q_abs[b, a] >= abs(Q_true[a, b])
    as well. Each true form is Hermitian, or (F_k) a combination of Hermitian M~_m. The array
    q_abs is a sum of their modulus bounds. So it bounds abs(Q_true[a, b]) and
    abs(Q_true[b, a]).
    """
    n = q.shape[-1]
    b = delta.shape[0]
    y, b_low, b_high, r = mats[:4]
    y = cp.ascontiguousarray(y[:, :, :, :n])
    b_low, b_high, r = b_low[:n], b_high[:n], r[:n]
    t, terr = gemm.zgemm_tt(y, q, mode=0, out_td=True)
    w, werr = gemm.zgemm_tt(t, y, mode=1, out_td=False)
    del t
    # delta^deg (B, low): an exact gather of the upward powers of delta.
    pw = mag.powers(delta, sp_low.n)[cp.asarray(sp_low.deg)].T
    pw_row = pw[:, None, :]                                   # (B, 1, low)
    # round = (pw terr) b_low + (pw werr) pw for each box; pw_row @ terr and pw_row @ werr have
    # one output for each column, so their reads of terr and werr coalesce.
    t_part = mag.matmul(mag.matmul(pw_row, terr), b_low.T[:, :, None])  # (B, 1, nf) @ (B, nf, 1)
    w_part = mag.matmul(mag.matmul(pw_row, werr), pw[:, :, None])       # (B, 1, low) @ (B, low, 1)
    rnd = mag.add(t_part, w_part)[:, 0, 0]
    # One contiguous copy of each block, so that mag.matmul makes no copy in each quad.
    q_abs, q_rad = cp.ascontiguousarray(q_abs), cp.ascontiguousarray(q_rad)

    def quad(u, m, v):
        """An upper bound of the sum over i, j of u[i, b] m[b, i, j] v[j, b], shape (B,):
        (B, 1, nf) @ (1 or B, nf, nf), then @ (B, nf, 1); a batch of 1 goes to all boxes."""
        return mag.matmul(mag.matmul(u.T[:, None, :], m), v.T[:, :, None])[:, 0, 0]

    full = mag.add(b_low, b_high, r)
    ab = quad(b_low, q_abs, b_low)
    parts = {'round': rnd,
             'high': mag.add(mag.mul(2.0, quad(b_high, q_abs, b_low)),
                             quad(b_high, q_abs, b_high)),
             'models': mag.add(mag.mul(2.0, quad(r, q_abs, mag.add(b_low, b_high))),
                               quad(r, q_abs, r)),
             'radius': quad(full, q_rad, full), 'sums': cp.zeros(b), 'abs': ab}
    rem = mag.add(parts['round'], parts['high'], parts['models'], parts['radius'])
    out = _from_bidegree(sp4, sp_low, w, rem, delta)
    out.parts = parts
    return out


def delta_form_td(sp4, sp_low, mats, q, q_rad, q_abs, delta, dl, bdl):
    """sum over j of Delta^j W_j with W_j = Y R_j Y^H (compose_td), by Horner's rule in Delta.

    The parts of the remainder are those of the W_j, with the factor bdl^j.
    """
    ws = [compose_td(sp4, sp_low, mats, q[j], q_rad[j], q_abs[j], delta) for j in range(4)]
    s = ws[3]
    for j in (2, 1, 0):
        s = tm.add(sp4, ws[j], tm.mul(sp4, dl, s))
    s.parts = {key: mag.horner(cp.stack([ws[j].parts[key] for j in range(4)]), bdl)
               for key in ws[0].parts}
    return s


def _check_local(local):
    """Stop with a clear message if the local forms are not in Delta and in td."""
    if int(local.get('parts', 0)) != 6 or 's2' not in local:
        raise ValueError('the box needs local forms in Delta and in td, as '
                         'scripts/tm_local_forms.py writes them')


@mp.workprec(103)
def box_f(fr, local, delta_box, n, ny, a, hmax, grad=False, keep=None):
    """The model of f on a batch of boxes that share the reference center of `local`.

    a: the gap fraction of the Cauchy estimate, d_f = d (1 + a), d_Phi = d_f (1 + a).
    hmax: the bound of abs(h) over the cell of each box, for the check in implicit_h_td.
    grad: also bound the gradient of f on the box (`gradient_bounds`), in info.
    keep: a dict or None. With a dict, box_f stores in it references to the intermediate
    models and free numbers that it makes (see the keys at the end of the function).
    mpmath picks the free dd numbers at 103 bits, the precision of every run, so the bits
    of the result do not depend on the precision of the caller.

    Returns f (on delta_f), the radii (box, f, Phi) and the parts of the remainder.
    """
    k0, top = int(local['k0']), int(local['top'])
    b = len(delta_box)
    d = cp.asarray(np.asarray(delta_box, float))
    d_f = d * (1 + a)
    d_phi = d_f * (1 + a)
    sp2, sp_low, sp4 = space(2, ny), space(2, n), space(4, n)
    info = {}
    clock = _Clock()
    # Each form is a sum over j of Delta^j R_j, with the R_j of the local forms in td.
    _check_local(local)
    cref = [list(v) for v in cp.asnumpy(local['c'])]
    h_parts = {}
    ys, rho, lam, h_td, hm, lam_big = implicit_h_td(sp2, fr, d_phi, hmax, h_parts)
    base_t, v2 = offset_models_td(sp2, fr, h_td, hm, cref, d_phi)
    info['rho_h'], info['lam_h'], info['lam_big'] = rho, lam, lam_big
    info['h_parts'] = h_parts
    yc = [conj(y) for y in ys]
    clock('h')
    mats = monomial_mats_td(sp2, sp_low, base_t, top)
    dls = {n: delta_model(sp4, sp2, fr, v2, local, d_phi)}
    bdl = tm.bound(sp4, dls[n])
    info['bound_delta'] = bdl
    clock('monomials')

    def form(sp4k, splk, mk, q, rr, aa):
        if sp4k.n not in dls:
            dls[sp4k.n] = tm.truncate(sp4k, sp4, dls[n])
        return delta_form_td(sp4k, splk, mk, q, rr, aa, d_phi, dls[sp4k.n], bdl)

    # The base and abs(y)^2.
    qb = [local[f'base_j{e}'][None] for e in range(4)]
    rb = [local[f'base_j{e}_rad'][None] for e in range(4)]
    ab = [local[f'base_j{e}_abs'][None] for e in range(4)]
    p0 = form(sp4, sp_low, mats, qb, rb, ab)
    one = tm.cconst(1, b)
    r2 = tm.const(sp4, one, d_phi)
    for k in range(3):
        r2 = tm.add(sp4, r2, tm.tensor(sp4, sp2, ys[k], yc[k]))
    # With any exact s, 1/r2 = s (1 + yr)^-1 and log r2 = log(1 + yr) - log s, yr = s r2 - 1.
    s_r = gpu_dd([1 / v for v in host_dd(r2.c[:, 0, :])])
    yr = tm.add_const(sp4, tm.scale(sp4, r2, s_r), -one)
    eta_r = tm.bound(sp4, yr)
    pw_r = tm.powers(sp4, yr, series_order(eta_r, n, SERIES_EPS))
    ir2 = tm.scale(sp4, tm.power_series(sp4, pw_r, -1, eta_r), s_r)
    log_r2 = tm.log1p_series(sp4, pw_r, eta_r)
    cache = {1: ir2}
    pt = tm.mul(sp4, p0, power(sp4, ir2, k0, cache))
    # A real s_p, so that p~^(-1/k0) = xh s_p^(1/k0) holds with the real root below.
    s_p = gpu_dd([mp.mpc(1 / v.real) for v in host_dd(pt.c[:, 0, :])])
    yp = tm.add_const(sp4, tm.scale(sp4, pt, s_p), -one)
    eta_p = tm.bound(sp4, yp)
    pw_p = tm.powers(sp4, yp, series_order(eta_p, n, SERIES_EPS))
    xh = tm.power_series(sp4, pw_p, Fraction(-1, k0), eta_p)
    log_pt = tm.log1p_series(sp4, pw_p, eta_p)
    info.update(eta_r=eta_r, eta_p=eta_p, rem_p0=mag.div(p0.rem, tm.abs_ub(p0.c[:, :1, :])[0]),
                parts_p0=p0.parts)

    # x = r2^-1 p~^(-1/k0) = ir2 xh s_p^(1/k0); x_c = x0 (1 + zeta) with an exact dd zeta.
    x0 = dd_mp(np.asarray(local['x0']))
    sp_host = host_dd(s_p)
    xc_raw = [v.real ** (mp.mpf(1) / k0) * w for v, w in
              zip(sp_host, host_dd(tm.mul(sp4, ir2, xh).c[:, 0, :]))]
    # zeta is a free dd number near x_c / x0 - 1; from here on its dd value is exact, and
    # kappa and the weights of the re-expansion both take that value in ball arithmetic.
    zeta = [mp_dd(x / x0 - 1) for x in xc_raw]
    # kappa = s_p^(1/k0) / (x0 (1 + zeta)) in ball arithmetic.
    x0_dd = list(np.asarray(local['x0']))
    sp_dd = cp.asnumpy(s_p)
    kap = [kappa(list(sp_dd[:, i]), k0, x0_dd, z) for i, z in enumerate(zeta)]
    kappa_dd = cp.asarray(np.array([k[0] for k in kap]).T)
    kerrs = [k[1] for k in kap]
    # A float goes into the array as it is; a conversion of another type can round.
    if not all(isinstance(e, float) for e in kerrs):
        raise TypeError(f'the error bounds of kappa must be floats: {[type(e) for e in kerrs]}')
    kap_err = cp.asarray(np.array(kerrs))
    xw = tm.mul(sp4, ir2, xh)
    xi = tm.add_const(sp4, tm.scale(sp4, xw, kappa_dd), -one)
    xi_kap = mag.mul(kap_err, tm.bound(sp4, xw))
    xi.rem = mag.add(xi.rem, xi_kap)
    bxi = tm.bound(sp4, xi)

    clock('base')
    fq = reexpand_td(local, zeta, top)
    clock('reexpand')
    # xi is of degree 1 up to a rounding, so the part of F_k above its order goes to its
    # remainder; the order is at least 4 for a small n.
    orders = [n if k <= K_FULL else max(n - (k - K_FULL), 4) for k in range(top + 1)]
    fk = []
    for k in range(top + 1):
        q = [x[0] for x in fq[k]]
        rr = [x[1] for x in fq[k]]
        aa = [x[2] for x in fq[k]]
        sp4k, splk = space(4, orders[k]), space(2, orders[k])
        mk = mats if orders[k] == n else mats_at(mats, splk)
        fk.append(form(sp4k, splk, mk, q, rr, aa))
    clock('forms')
    xis = {n: xi}
    for m in set(orders):
        if m not in xis:
            xis[m] = tm.truncate(space(4, m), sp4, xi)
    # The parts of the remainder of psi, carried through Horner's rule: the remainder of
    # s enters each later step times the bound of xi.
    split = {key: cp.zeros(b) for key in ('F_k', 'xi', 'xi cut', 'tails', 'rounding')}
    split['F_k'] = mag.add(split['F_k'], fk[top].rem)
    s = fk[top]
    for k in range(top - 1, -1, -1):
        spk = space(4, orders[k])
        if orders[k + 1] < orders[k]:
            s = tm.embed(spk, s)
        xk = xis[orders[k]]
        prod = tm.mul(spk, xk, s)
        mp_ = prod.parts
        s_next = tm.add(spk, fk[k], prod)
        for key in split:
            split[key] = mag.mul(split[key], mp_['bx'])
        yb = mag.add(mp_['by'], s.rem)
        split['xi'] = mag.add(split['xi'], mag.mul(xi.rem, yb))
        split['xi cut'] = mag.add(split['xi cut'], mag.mul(mag.add(xk.rem, -xi.rem), yb))
        split['tails'] = mag.add(split['tails'], mp_['tail'])
        # The round-off of the sum fk[k] + prod: its remainder less fk[k].rem and prod.rem.
        add_round = mag.sub_down(mag.sub_down(s_next.rem, fk[k].rem), prod.rem)
        split['rounding'] = mag.add(split['rounding'], mp_['round'], add_round)
        split['F_k'] = mag.add(split['F_k'], fk[k].rem)
        s = s_next
    info['psi_parts'] = split
    info['xi_parts'] = dict(xi=xi.rem, xw=xw.rem, ir2=ir2.rem, xh=xh.rem, log_r2=log_r2.rem,
                            yr=yr.rem, yp=yp.rem,
                            pt=mag.div(pt.rem, tm.abs_ub(pt.c[:, :1, :])[0]))
    info['fk_rem'] = cp.stack([mag.mul(fk[k].rem, mag.powi(bxi, k)) for k in range(top + 1)])
    psi = tm.TM(s.c, s.rem, s.delta)
    clock('level sum')
    phi = tm.add(sp4, tm.add(sp4, tm.scale(sp4, log_r2, tm.cconst(k0, b)), log_pt), psi)
    info.update(parts_f0=fk[0].parts, value_f0=tm.abs_ub(fk[0].c[:, :1, :])[0])
    info.update(rem_f0=fk[0].rem, rem_psi=psi.rem, rem_logpt=log_pt.rem, rem_logr2=log_r2.rem,
                rem_phi=phi.rem, bxi=bxi, zeta=zeta)

    # The metric, the volume and f.
    g = {(i, j): tm.cauchy_mixed(sp4, phi, i, 2 + j, d_f) for i in range(2) for j in range(2)}
    det = tm.sub(sp4, tm.mul(sp4, g[0, 0], g[1, 1]), tm.mul(sp4, g[0, 1], g[1, 0]))
    dp = None
    for k in range(3):
        cube = tm.mul(sp2, tm.mul(sp2, ys[k], ys[k]), ys[k])
        term = tm.scale(sp2, cube, 4 * fr.n[k])
        dp = term if dp is None else tm.add(sp2, dp, term)
    dp = tm.restrict(dp, d_f)
    q4 = tm.tensor(sp4, sp2, dp, conj(dp))
    dq = tm.mul(sp4, det, q4)
    v = tm.scale(sp4, dq, fr.vol)
    v_vol = mag.mul(mag.mul(_vol_factor(fr.vol_err), tm.abs_ub(fr.vol[:, None, :])[0]),
                    tm.bound(sp4, dq))
    v.rem = mag.add(v.rem, v_vol)
    f = tm.add_const(sp4, v, -one)
    if grad:
        info.update(gradient_bounds(sp4, f, g, det, d, k0, keep))
    clock('metric')
    info.update(rem_g=g[0, 0].rem, rem_f=f.rem, times=clock.times)
    if keep is not None:
        keep.update(d=d, d_f=d_f, d_phi=d_phi, h_td=h_td, h_dd=hm, ys=ys, delta=dls[n], p0=p0,
                    r2=r2, ir2=ir2, log_r2=log_r2, pt=pt, xh=xh, log_pt=log_pt, xw=xw, xi=xi,
                    fk=fk, psi=psi, phi=phi, g=g, det=det, q4=q4, v=v, f=f, orders=orders,
                    s_r=s_r, s_p=s_p, zeta=zeta, kappa=kappa_dd, kap_err=kap_err,
                    xi_kap=xi_kap, v_vol=v_vol)
    return f, (d, d_f, d_phi), info, dict(p0=p0, f0=fk[0], ys=ys, sp4=sp4, sp2=sp2,
                                          delta=dls[n])


def gradient_bounds(sp4, f, g, det, d, k0, keep=None):
    """Bounds of the gradient of f on the box of radius d, for f, g and det g models on d_f > d.

    In the tangent coordinates tau (variables 0, 1; sigma = conj(tau) are 2, 3), f is real
    on the real points, so abs(d f)^2 = 4 sum_i abs(df/dtau_i)^2 in the real coordinates of
    tau, and abs(df/dtau_i) <= B of the model of `tm.deriv1` (Cauchy from d_f to d). In
    the metric of volume 1, G = c Re(g) with c = 1 / (sqrt(2) pi k0), abs(grad f)^2 =
    4 (df)^H g^-1 df / c <= 4 abs(df)^2 / (c lam_min), and lam_min >= det g / tr g with
    det g >= Re det(center) - (its variation and remainder) on the box; g_00 > 0 there as
    well makes g positive definite. A box without that proof gets an infinite bound.
    With a dict `keep`, it gets dfs, df_bounds, det_low, g00_low and tr_up.
    The upper bounds round up and the lower bounds (lam_low, c) round down.
    """
    dfs = [tm.deriv1(sp4, f, i, d) for i in range(2)]
    b = [tm.bound(sp4, x) for x in dfs]
    # The bound of abs(df/dtau).
    grad_tau = mag.sqrt(mag.add(mag.mul(b[0], b[0]), mag.mul(b[1], b[1])))
    det_low = _real_low(sp4, det, d)
    tr_up = mag.add(tm.poly_bound(sp4, g[0, 0], d), g[0, 0].rem,
                    tm.poly_bound(sp4, g[1, 1], d), g[1, 1].rem)
    g00_low = _real_low(sp4, g[0, 0], d)
    lam_low = _lam_low(det_low, g00_low, tr_up)
    grad_g = _grad_g(grad_tau, metric_c_low(k0), lam_low)
    if keep is not None:
        keep.update(dfs=dfs, df_bounds=b, det_low=det_low, g00_low=g00_low, tr_up=tr_up)
    return dict(grad_tau=grad_tau, grad_g=grad_g, lam_low=lam_low)


def _vol_factor(vol_err):
    """vol_err / (1 - vol_err) with a downward denominator and an upward quotient; +inf for
    vol_err >= 1. vol_err is relative to the true vol, so abs(vol) <= abs(vol in dd) /
    (1 - vol_err), and this factor times abs(vol in dd) bounds the error of vol in dd."""
    return mag.div(vol_err, mag.sub_down(1.0, vol_err))


def _lam_low(det_low, g00_low, tr_up):
    """A lower bound of lam_min(g): RD(det_low / tr_up) where det_low > 0 and g00_low > 0
    (then g is positive definite on the box and lam_min >= det g / tr g), else 0."""
    ok = (det_low > 0) & (g00_low > 0)
    return cp.where(ok, mag.div_down(det_low, tr_up), 0.0)


def _grad_g(grad_tau, c, lam_low):
    """The bound 2 grad_tau / sqrt(c lam_low) of abs(grad f) in G: the divisor rounds down and
    the quotient rounds up, for a lower bound c of the factor of G; +inf where lam_low is not
    above 0."""
    pos = lam_low > 0
    root = mag.sqrt_down(mag.mul_down(c, cp.where(pos, lam_low, 1.0)))
    return cp.where(pos, mag.div(mag.mul(2.0, grad_tau), root), cp.inf)


def metric_c_low(k0):
    """A lower bound (a float) of c = 1 / (sqrt(2) pi k0), the factor of the metric of
    volume 1, from ball arithmetic."""
    from flint import arb
    with precision():
        return down(1 / (arb(2).sqrt() * arb.pi() * int(k0)))


def _real_low(sp4, x, d):
    """A lower bound of the real part of a model on the polydisc d: Re c0 - abs(Im c0) -
    B(x - c0) - rem. Re c0 = a0 + a1 rounds down, the sum of the three subtracted magnitudes
    rounds up, and the difference rounds down."""
    var = x.c.copy()
    var[:, 0, :] = 0
    pb = tm.poly_bound(sp4, tm.TM(var, x.rem, d), d)
    a0, a1, b0, b1 = x.c[0, 0], x.c[1, 0], x.c[2, 0], x.c[3, 0]
    # abs(b0 + b1) is at most the larger abs of the upward and of the downward sum.
    im_up = cp.maximum(cp.abs(mag.add(b0, b1)), cp.abs(mag.add_down(b0, b1)))
    return mag.sub_down(mag.add_down(a0, a1), mag.add(im_up, pb, x.rem))


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


