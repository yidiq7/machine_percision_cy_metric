"""A reference Taylor model of f on one box in arb: tangent coordinates and the xi expansion.

The box is the polydisc abs(tau_i) <= delta in the coordinates of the tangent plane of X
at the center p, in the affine chart z_3 = 1 with y = (z_0, z_1, z_2):

    y(tau) = p + E tau + h(tau) n,    P(y) = y_0^4 + y_1^4 + y_2^4 + 1 = 0,

with n the unit normal conj(grad P(p)) / abs(grad P(p)) and E = dy/dt at p, so that
tau = t - c to first order. h comes from a fixed point in Taylor-model arithmetic, with
a strict bound from the contraction. The conjugate side takes sigma = conj(tau) as
independent variables.

Each form is the sum over e of y_2^e conj(y_2)^e Q_e(y_0, y_1, conj(y_0), conj(y_1)), with the
global polynomials Q_e of `quartic.tm_forms`. The potential is

    Phi = k0 log abs(y)^2 + log p~ + abs(y)^(-2 top) sum over k of F_k xi^k,

with p~ = p0 abs(y)^(-2 k0), x = p~^(-1/k0), xi = x / x_c - 1 and
F_k = sum over L of C(L, k) x_c^L M_L, formed exactly in arb before any remainder. So the
levels cancel in the coefficients and not in the remainders. g = d_tau d_sigma Phi comes
from the exact derivative of the polynomial part and a Cauchy estimate for the rest.
v = det(g) abs(dP/dh)^2 / abs(det U)^2 with U = [E, n], and f = v / vbar - 1.

For each center, order and radius delta, the script reports the bound of abs(f - f(c)) on
the polydisc delta and the remainder, with its parts. Phi lives on delta (1 + a)^2 and f on
delta (1 + a). It also checks the models of p0 and F_0 against direct values at a point of
the box, f(c) against the dd f of the export, and f at that point against `tm_box.py`.

Inputs: an export of `level_oracle.py --invariant-base` and the forms of
`level_matrices.py`. Usage, on CPU cores:
    JAX_PLATFORMS=cpu uv run --extra gpu --with python-flint python -u scripts/tm_ref.py \\
        --data runs_tm/r20i_data.npz --forms runs_tm/r20_forms.npz --orders 16 20 24 \\
        --deltas 0.005 0.01 0.02 0.03 0.04 0.06 --procs 6
"""
import argparse
import math
import multiprocessing
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import flint
from flint import acb, acb_mat, arb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import tm_box as tb                                         # noqa: E402
from quartic import tm_forms                                # noqa: E402

G = {}
SLACK = 1 + 1e-12


def exact(x):
    """The midpoint of a ball as an exact acb."""
    return acb(x.real.mid(), x.imag.mid())


def frame(p):
    """n, the frame E = dy/dt of the tangent plane, det [E, n] and c1 = dP/dh at p, exact.

    With E = dy/dt, tau = t - c to first order, so a polydisc in tau matches one in t.
    """
    grad = [4 * c ** 3 for c in p]
    norm = sum((abs(c) ** 2 for c in grad), arb(0)).sqrt()
    n = [exact(c.conjugate() / norm) for c in grad]
    vecs = []
    for k in range(2):
        v = [acb(0)] * 3
        v[k] = acb(1)
        v[2] = exact(-(p[k] ** 3) / p[2] ** 3)
        vecs.append(v)
    u = acb_mat(3, 3, [[vecs[0][i], vecs[1][i], n[i]][j] for i in range(3) for j in range(3)])
    c1 = exact(sum((g * m for g, m in zip(grad, n)), acb(0)))
    return vecs, n, u, c1


def tm_forms_conj_t(m):
    """The conjugate transpose of an acb_mat."""
    return acb_mat(m.ncols(), m.nrows(), [m[i, j].conjugate() for j in range(m.ncols())
                                           for i in range(m.nrows())])


def var2(sp2, i, deltas):
    c = np.array([acb(0)] * sp2.size, dtype=object)
    e = [0, 0]
    e[i] = 1
    c[sp2.index[tuple(e)]] = acb(1)
    return tb.TM(sp2, c, np.zeros(len(deltas)), deltas)


def conj_tm(x):
    return tb.TM(x.sp, np.array([c.conjugate() for c in x.c], dtype=object), x.rem.copy(),
                 x.deltas)


def on(x, deltas):
    """The same model with its bounds taken on smaller radii; the remainder stays valid."""
    return tb.TM(x.sp, x.c, x.rem.copy(), deltas)


def p_value(ys):
    total = None
    for y in ys:
        y2 = y * y
        y4 = y2 * y2
        total = y4 if total is None else total + y4
    return total.add_const(1)


def implicit_h(sp2, p, vecs, n, c1, deltas):
    """y(tau) = p + E tau + h n on X: the Taylor polynomial of h and a strict remainder.

    T(h) = h - P(y(h)) / c1. If T maps the ball abs(h - P_h) <= rho into itself with a
    contraction factor lam < 1, the root h(tau) of each tau lies in that ball.
    """
    t = [var2(sp2, 0, deltas), var2(sp2, 1, deltas)]
    lin = [tb.TM.const(sp2, p[k], deltas) + t[0].scale(vecs[0][k]) + t[1].scale(vecs[1][k])
           for k in range(3)]
    inv = 1 / c1
    h = tb.TM.const(sp2, 0, deltas)
    for _ in range(sp2.n + 2):
        ys = [lin[k] + h.scale(n[k]) for k in range(3)]
        h = h - p_value(ys).scale(inv)
        h = tb.TM(sp2, h.c, np.zeros(len(deltas)), deltas)
    ys = [lin[k] + h.scale(n[k]) for k in range(3)]
    bd = p_value(ys).scale(inv).bound()
    dp = None
    for k in range(3):
        term = (ys[k] * ys[k] * ys[k]).scale(4 * n[k])
        dp = term if dp is None else dp + term
    lam0 = dp.scale(-inv).add_const(1).bound()
    by = [y.bound() for y in ys]
    nabs = [tb.ub(x) for x in n]
    iabs = tb.ub(inv)
    rho = np.full(len(deltas), np.inf)
    lam = np.full(len(deltas), np.inf)
    for i in range(len(deltas)):
        if not lam0[i] < 1:
            continue
        r = 2 * bd[i] / (1 - lam0[i]) * SLACK
        second = sum(12 * nabs[k] ** 2 * (by[k][i] + r * nabs[k]) ** 2 for k in range(3))
        l_ = (lam0[i] + r * second * iabs) * SLACK
        if l_ < 1 and (bd[i] + l_ * r) * SLACK <= r:
            rho[i], lam[i] = r, l_
    return lin, tb.TM(sp2, h.c, rho, deltas), lam


def coef_matrix(tms, conj=False):
    """(S, m) with column a the coefficients of tms[a]; conj=True gives conj, transposed."""
    s = tms[0].sp.size
    if conj:
        return acb_mat(len(tms), s, [tms[a].c[i].conjugate() for a in range(len(tms))
                                     for i in range(s)])
    return acb_mat(s, len(tms), [tms[a].c[i] for i in range(s) for a in range(len(tms))])


def abs_np(m):
    """Upper bounds of abs of the entries of an arb_mat, as a NumPy array."""
    return np.array([[tb.ub(m[i, j]) for j in range(m.ncols())] for i in range(m.nrows())])


def shift_matrix(p2, c):
    """T with sum over a of q_a t^a = sum over a' of (T q)_a' (t - c)^a', for t^a in p2."""
    n = p2.size
    vals = []
    for ap in p2.mons:
        for a in p2.mons:
            if ap[0] <= a[0] and ap[1] <= a[1]:
                vals.append(math.comb(a[0], ap[0]) * math.comb(a[1], ap[1])
                            * c[0] ** (a[0] - ap[0]) * c[1] ** (a[1] - ap[1]))
            else:
                vals.append(acb(0))
    return acb_mat(n, n, vals)


def form_tm(sp4, sp2, ys, q, absq, deltas):
    """The model of sum over e of Y_e Q_e Y_e^H, with the parts that the remainders add.

    ys[e] = (Y, Yh, b, r): the coefficient matrix of the models y_2^e y^alpha, its conjugate
    transpose, and their bounds and remainders (n_p, len(deltas)).
    """
    w = None
    rem = np.zeros(len(deltas))
    absb = np.zeros(len(deltas))
    n_p = q[0].nrows()
    for e in range(4):
        y_e, yh_e, b, r = ys[e]
        part = y_e * (q[e] if isinstance(q[e], acb_mat) else acb_mat(q[e])) * yh_e
        w = part if w is None else w + part
        b, r = b[:n_p], r[:n_p]
        rem += (2 * np.einsum('ad,ab,bd->d', r, absq[e], b)
                + np.einsum('ad,ab,bd->d', r, absq[e], r))
        absb += np.einsum('ad,ab,bd->d', b, absq[e], b)
    c = np.array([acb(0)] * sp4.size, dtype=object)
    for k, (a0, a1, b0, b1) in enumerate(sp4.mons):
        c[k] = w[sp2.index[(a0, a1)], sp2.index[(b0, b1)]]
    big = np.zeros(2 * sp2.n + 1)
    for i in range(sp2.size):
        di = sp2.deg[i]
        for j in range(sp2.size):
            if di + sp2.deg[j] > sp4.n:
                big[di + sp2.deg[j]] += tb.ub(w[i, j])
    tail = np.array([float(np.sum(big * dl ** np.arange(big.size))) for dl in deltas])
    out = tb.TM(sp4, c, (rem + tail) * SLACK, deltas)
    # The dd rounding of the two matrix products on the GPU would add about this much.
    out.round_dd = 3 * (n_p + 20) * 8 * 2.0 ** -106 * absb
    out.abs_ratio = absb / tb.ub(c[0])
    return out


def powers(y, k):
    out = [tb.TM.const(y.sp, 1, y.deltas), y]
    for _ in range(2, k + 1):
        out.append(out[-1] * y)
    return out


def fraction_arb(q):
    return arb(q.numerator) / q.denominator


def binomial(pw, a, eta):
    """(1 + y)^a from the powers of y, with the Lagrange bound of the rest, for real a."""
    k = len(pw) - 1
    a = Fraction(a)
    co = [Fraction(1)]
    for j in range(k + 1):
        co.append(co[-1] * (a - j) / (j + 1))
    total = None
    for j in range(k + 1):
        term = pw[j].scale(acb(fraction_arb(co[j])))
        total = term if total is None else total + term
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rest = abs(float(co[k + 1])) * eta ** (k + 1) * (1 - eta) ** (float(a) - k - 1)
    rest = np.where(eta < 1, rest * SLACK, np.inf)
    return tb.TM(total.sp, total.c, total.rem + rest, total.deltas)


def log1p(pw, eta):
    """log(1 + y) from the powers of y, with the bound eta^(k+1) / ((k + 1)(1 - eta))."""
    k = len(pw) - 1
    total = None
    for j in range(1, k + 1):
        term = pw[j].scale(acb(fraction_arb(Fraction((-1) ** (j + 1), j))))
        total = term if total is None else total + term
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rest = eta ** (k + 1) / ((k + 1) * (1 - eta))
    rest = np.where(eta < 1, rest * SLACK, np.inf)
    return tb.TM(total.sp, total.c, total.rem + rest, total.deltas)


def mixed(x, va, vb, small):
    """d^2 / dv_a dv_b of x on the radii `small`, with Cauchy's estimate for the rest."""
    sp = x.sp
    c = np.array([acb(0)] * sp.size, dtype=object)
    for k, e in enumerate(sp.mons):
        if e[va] and e[vb]:
            f = list(e)
            f[va] -= 1
            f[vb] -= 1
            c[sp.index[tuple(f)]] += x.c[k] * (e[va] * e[vb])
    gap = (x.deltas - small) * (1 - 1e-14)
    return tb.TM(sp, c, x.rem / gap ** 2 * SLACK, small)


def point_on_x(p, vecs, n, tau, h):
    """The point y(tau) of X near y = p + E tau + h n, by Newton's method in h."""
    for _ in range(12):
        y = [p[k] + vecs[0][k] * tau[0] + vecs[1][k] * tau[1] + h * n[k] for k in range(3)]
        val = sum((c ** 4 for c in y), acb(0)) + 1
        der = sum((4 * c ** 3 * m for c, m in zip(y, n)), acb(0))
        h = exact(h - val / der)
    return [p[k] + vecs[0][k] * tau[0] + vecs[1][k] * tau[1] + h * n[k] for k in range(3)]


def load(args):
    tb.load(args)
    flint.ctx.prec = args.prec
    data = np.load(args.data)
    forms = np.load(args.forms)
    t0 = time.perf_counter()
    cf = tm_forms.chart_forms_arb(data, forms, args.prec)
    G.update(cf)
    G['abs_base'] = [abs_np(m) for m in cf['base']]
    G['abs_M'] = {k1: [abs_np(m) for m in cf[f'M{k1}']] for k1 in cf['levels']}
    G['seconds_forms'] = time.perf_counter() - t0
    G['a'] = args.a
    G['extra_y'] = args.extra_y
    G['shift'] = args.shift


def run_box(job):
    k, n, base_deltas = job
    flint.ctx.prec = G['prec']
    t_start = time.perf_counter()
    times = {}
    k0, top, levels = G['k0'], G['top'], G['levels']
    a = G['a'] if G['a'] > 0 else 2.0 / n
    dl = np.asarray(base_deltas, float)
    d_f = dl * (1 + a)
    d_phi = d_f * (1 + a)
    # The models in tau alone go to a higher order, so that their truncation, which the
    # absolute-value form multiplies, stays far below the tail of the composed form.
    sp2, sp4 = tb.Space(2, n + G['extra_y']), tb.Space(4, n)
    sp2.table()
    sp4.table()
    times['tables'] = time.perf_counter() - t_start

    t1 = time.perf_counter()
    z = to_region(D_z(k))
    p = [exact(z[i] / z[3]) for i in range(3)]
    vecs, nrm, u, c1 = frame(p)
    lin, h, lam = implicit_h(sp2, p, vecs, nrm, c1, d_phi)
    ys = [lin[i] + h.scale(nrm[i]) for i in range(3)]
    ysc = [conj_tm(y) for y in ys]
    times['h'] = time.perf_counter() - t1

    t1 = time.perf_counter()
    p2 = tm_forms.Poly2(top)
    # With --shift the forms move to the center c, and the monomials are those of t - c.
    base_t = ys[:2]
    if G['shift']:
        base_t = [ys[i].add_const(-p[i]) for i in range(2)]
    mono = {(0, 0): tb.TM.const(sp2, 1, d_phi)}
    for e in p2.mons[1:]:
        mono[e] = (mono[(e[0] - 1, e[1])] * base_t[0] if e[0]
                   else mono[(0, e[1] - 1)] * base_t[1])
    y2e = [tb.TM.const(sp2, 1, d_phi), ys[2]]
    y2e.append(ys[2] * ys[2])
    y2e.append(y2e[2] * ys[2])
    tms = [[mono[m] if e == 0 else y2e[e] * mono[m] for m in p2.mons] for e in range(4)]
    mats = {}
    for n_p in sorted({tm_forms.Poly2(k0).size, p2.size}):
        mats[n_p] = []
        for e in range(4):
            part = tms[e][:n_p]
            b = np.array([x.poly_bound() for x in part])
            r = np.array([x.rem for x in part])
            mats[n_p].append((coef_matrix(part), coef_matrix(part, conj=True), b, r))
    times['monomials'] = time.perf_counter() - t1

    t1 = time.perf_counter()
    nb = G['base'][0].nrows()
    base_q, base_abs = G['base'], G['abs_base']
    if G['shift']:
        tb_ = shift_matrix(tm_forms.Poly2(k0), p[:2])
        tbh = tm_forms_conj_t(tb_)
        base_q = [tb_ * acb_mat(m) * tbh for m in G['base']]
        base_abs = [abs_np(m) for m in base_q]
        tl_ = shift_matrix(p2, p[:2])
        tlh = tm_forms_conj_t(tl_)
    p0 = form_tm(sp4, sp2, mats[nb], base_q, base_abs, d_phi)
    r2 = tb.TM.const(sp4, 1, d_phi)
    for i in range(3):
        r2 = r2 + tb.tensor(sp4, sp2, ys[i], ysc[i], d_phi)
    xc = exact(r2.c[0] * (p0.c[0].log() * (-arb(1) / k0)).exp())
    fk = []
    for kk in range(top + 1):
        q, aq = [], []
        for e in range(4):
            qe, ae = None, np.zeros_like(G['abs_M'][levels[0]][e])
            for k1 in levels:
                if kk > k1:
                    continue
                w = arb(math.comb(k1, kk)) * xc.real ** k1
                term = G[f'M{k1}'][e] * w
                qe = term if qe is None else qe + term
                ae = ae + tb.ub(w) * G['abs_M'][k1][e]
            if G['shift']:
                qe = tl_ * acb_mat(qe) * tlh
                ae = abs_np(qe)
            q.append(qe)
            aq.append(ae * SLACK)
        fk.append(form_tm(sp4, sp2, mats[p2.size], q, aq, d_phi))
    times['forms'] = time.perf_counter() - t1

    t1 = time.perf_counter()
    mr = exact(r2.c[0])
    yr = r2.scale(1 / mr).add_const(-1)
    eta_r = yr.bound()
    pw = powers(yr, n)
    ir2 = binomial(pw, -1, eta_r).scale(1 / mr)
    log_r2 = log1p(pw, eta_r)
    ipow = {1: ir2}

    def ipower(m):
        if m not in ipow:
            ipow[m] = ipower(m // 2) * ipower(m // 2) if m % 2 == 0 else ipower(m - 1) * ir2
        return ipow[m]
    ipower(k0)
    ipower(top)
    pt = p0 * ipow[k0]
    mp = exact(pt.c[0])
    yp = pt.scale(1 / mp).add_const(-1)
    eta_p = yp.bound()
    pwp = powers(yp, n)
    xm = binomial(pwp, Fraction(-1, k0), eta_p).scale((mp.log() * (-arb(1) / k0)).exp())
    log_pt = log1p(pwp, eta_p)
    xi = xm.scale(1 / xc).add_const(-1)
    s = fk[top]
    for kk in range(top - 1, -1, -1):
        s = fk[kk] + xi * s
    psi = ipow[top] * s
    phi = log_r2.scale(k0) + log_pt + psi
    times['potential'] = time.perf_counter() - t1

    t1 = time.perf_counter()
    g = {(i, j): mixed(phi, i, 2 + j, d_f) for i in range(2) for j in range(2)}
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    dp = None
    for i in range(3):
        term = (ys[i] * ys[i] * ys[i]).scale(4 * nrm[i])
        dp = term if dp is None else dp + term
    dp_f = on(dp, d_f)
    q4 = tb.tensor(sp4, sp2, dp_f, conj_tm(dp_f), d_f)
    vbar = 512 * arb.pi() ** 4 * k0 ** 2 / arb(0.25).gamma() ** 8
    f = (det * q4).scale(1 / (abs(u.det()) ** 2 * vbar)).add_const(-1)
    times['metric'] = time.perf_counter() - t1
    bxi = xi.bound()
    bir = ipow[top].bound()
    round_psi = sum(fk[kk].round_dd * bxi ** kk for kk in range(top + 1)) * bir
    round_phi = p0.round_dd / tb.ub(p0.c[0]) + round_psi
    round_g = round_phi / (d_phi - d_f) ** 2

    t1 = time.perf_counter()
    small = float(dl.min())
    tau = [acb(0.6 * small, 0.3 * small), acb(-0.2 * small, 0.5 * small)]
    point4 = tau + [x.conjugate() for x in tau]
    ystar = point_on_x(p, vecs, nrm, tau, h.at(tau))
    zstar = ystar + [acb(1)]
    check_p = tb.ub(p0.at(point4) - tb.direct_p(zstar)) / tb.ub(p0.c[0])
    r2star = 1 + sum((abs(c) ** 2 for c in ystar), arb(0))
    f0_direct = acb(0)
    for lv in tb.D['levels']:
        f0_direct += xc ** lv[0] * tb.direct_form(lv, zstar) * r2star ** (top - lv[0])
    check_f0 = tb.ub(fk[0].at(point4) - f0_direct) / tb.ub(fk[0].c[0])
    f_star = point_f(zstar, k)
    gap_star = abs(float((f.at(point4)).real) - f_star)
    times['checks'] = time.perf_counter() - t1

    f_c = f.c[0]
    var = f.add_const(-f_c)
    jac = tangent_jacobian(p, u)
    f_dd = tb.D['v1'][k] / vbar - 1
    return {'point': k, 'ratio': tb.D['ratio'][k], 'n': n, 'ny': sp2.n, 'a': a,
            'deltas': dl.tolist(),
            'f_center': float(f_c.real), 'f_dd': float(f_dd.real), 'center_gap': tb.ub(f_c - f_dd),
            'poly': tb.poly_bound(var.graded(), dl).tolist(), 'rem': f.rem.tolist(),
            'rem_h': h.rem.tolist(), 'lam': lam.tolist(), 'rem_p0': (p0.rem / tb.ub(p0.c[0])).tolist(),
            'rem_f0': fk[0].rem.tolist(), 'rem_psi': psi.rem.tolist(), 'rem_logpt': log_pt.rem.tolist(),
            'rem_logr2': log_r2.rem.tolist(), 'rem_phi': phi.rem.tolist(),
            'rem_g': g[0, 0].rem.tolist(), 'eta_r': eta_r.tolist(), 'eta_p': eta_p.tolist(),
            'xi': bxi.tolist(), 'check_p': check_p, 'check_f0': check_f0,
            'abs_p0': p0.abs_ratio.tolist(), 'abs_f0': fk[0].abs_ratio.tolist(),
            'round_p0': (p0.round_dd / tb.ub(p0.c[0])).tolist(), 'round_psi': round_psi.tolist(),
            'round_g': round_g.tolist(),
            'f_star': f_star, 'gap_star': gap_star, 'rem_star': float(f.rem[int(np.argmin(dl))]),
            'jacobian_rows': jac, 'times': times, 'seconds': time.perf_counter() - t_start}


def D_z(k):
    return tb.D['z'][k]


def to_region(z):
    """The S_4 image with abs(z_1) <= abs(z_0) <= abs(z_2) <= abs(z_3); f is the same there."""
    order = sorted(range(4), key=lambda i: tb.ub(z[i]))
    return [z[order[1]], z[order[0]], z[order[2]], z[order[3]]]


def point_f(zstar, k):
    """f at a point of X from `tm_box.py` at order 2, in the chart z_3 = 1, m = 2."""
    tb.D['z'].append(zstar)
    tb.D['patch'].append([2, 0, 1])
    tb.D['ratio'].append(0.0)
    tb.D['v1'].append(acb(0))
    res = tb.run_box((len(tb.D['z']) - 1, 2, [1e-4]))
    return res['f_center']


def tangent_jacobian(p, u):
    """The largest row sum of abs(d tau / d t) at p, with t = (y_0, y_1) and y_2 dependent."""
    inv = u.inv()
    dy2 = [-(p[i] ** 3) / p[2] ** 3 for i in range(2)]
    rows = []
    for r in range(2):
        rows.append(sum(tb.ub(inv[r, j] + inv[r, 2] * dy2[j]) for j in range(2)))
    return max(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--forms', type=str, required=True)
    ap.add_argument('--col', type=int, default=3, help='the column of the export v')
    ap.add_argument('--points', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--orders', type=int, nargs='+', default=[16, 20, 24])
    ap.add_argument('--deltas', type=float, nargs='+',
                    default=[0.005, 0.01, 0.02, 0.03, 0.04, 0.06])
    ap.add_argument('--a', type=float, default=0.0,
                    help='the radius steps 1 + a between box, f and Phi; 0 means 2 / order')
    ap.add_argument('--extra-y', type=int, default=10,
                    help='the order of the models in tau alone above the order of the box')
    ap.add_argument('--shift', action='store_true',
                    help='move the chart forms to the center in arb before the composition')
    ap.add_argument('--prec', type=int, default=192)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()
    t0 = time.perf_counter()
    G['prec'] = args.prec
    load(args)
    print(f'loaded: k0 {G["k0"]}, levels {G["levels"]}, chart forms in {G["seconds_forms"]:.0f}s, '
          f'{len(tb.D["z"])} points with min/max abs(z) {tb.D["ratio"]}, '
          f'{time.perf_counter() - t0:.0f}s', flush=True)
    jobs = [(k, n, args.deltas) for n in args.orders for k in args.points]
    results = []
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        for res in pool.imap_unordered(run_box, jobs):
            results.append(res)
            print(f'point {res["point"]} (rho {res["ratio"]:.3f}), order {res["n"]} (tau alone '
                  f'{res["ny"]}), a {res["a"]:.3f}: '
                  f'f(c) {res["f_center"]:+.4e}, dd {res["f_dd"]:+.4e}, gap {res["center_gap"]:.1e}; '
                  f'{res["seconds"]:.0f}s (' + ', '.join(f'{key} {val:.0f}s'
                                                       for key, val in res['times'].items())
                  + ')', flush=True)
            print(f'    checks: p0 {res["check_p"]:.1e}, F_0 {res["check_f0"]:.1e} relative; f at a '
                  f'point {res["f_star"]:+.4e}, model gap {res["gap_star"]:.1e} against the '
                  f'remainder {res["rem_star"]:.1e}; max row sum of d tau / d t '
                  f'{res["jacobian_rows"]:.3f}', flush=True)
            for i, d in enumerate(res['deltas']):
                print(f'    delta {d:.4f}: abs(f - f(c)) <= {res["poly"][i]:.2e} + remainder '
                      f'{res["rem"][i]:.2e} | h {res["rem_h"][i]:.1e} (lam {res["lam"][i]:.2f}), '
                      f'p0 {res["rem_p0"][i]:.1e} rel, F_0 {res["rem_f0"][i]:.1e}, psi '
                      f'{res["rem_psi"][i]:.1e}, log p~ {res["rem_logpt"][i]:.1e}, log r2 '
                      f'{res["rem_logr2"][i]:.1e}, Phi {res["rem_phi"][i]:.1e}, g_00 '
                      f'{res["rem_g"][i]:.1e}; eta r2 {res["eta_r"][i]:.3f}, p~ '
                      f'{res["eta_p"][i]:.3f}, xi {res["xi"][i]:.1e}', flush=True)
                print(f'        dd estimate: abs form over value p0 {res["abs_p0"][i]:.1e}, F_0 '
                      f'{res["abs_f0"][i]:.1e}; rounding p0 {res["round_p0"][i]:.1e} rel, psi '
                      f'{res["round_psi"][i]:.1e}, g {res["round_g"][i]:.1e}', flush=True)
    if args.out:
        np.save(args.out, np.array(results, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
