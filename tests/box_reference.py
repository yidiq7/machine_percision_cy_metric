"""An independent reference in arb for the steps 1 to 12 of the box code of the strict bound.

The values start from the two files of the certified potential (f6t_data.npz and
f6t_forms.npz) and from the definitions of notes/strict_bound_proof.tex. This module does
not use the local forms, the shift, the split in Delta or the re-expansion in zeta of the
box code. Each value is an acb ball at 320 bits. The steps 10 to 12 (the metric, f and its
gradient on the slice) take g_t, f and df/dt from scripts/tm_box.py, which the caller
loads and passes in.

* The chart is z_3 = 1 with t = (z_0, z_1), and z_2 is the dependent coordinate:
  P(y) = y_0^4 + y_1^4 + y_2^4 + 1. Over tau the point of X is
  y(tau) = p + E tau + h(tau) n, with E[i][k] = dy_k / dt_i of the frame of the box.
* Polarization: a function of y and conj(y) takes y(tau) and yb(sigma) =
  conj(y(conj(sigma))) in their places. On the slice sigma = conj(tau) it is the real
  function of the document.
* p0 and N_L (Definitions "the base p0" and "the level forms N_L") take w^r wb^r in place
  of abs(w^r)^2, and u(w)^T C u(wb) in place of u(w)^T C conj(u(w)).

The module imports only the standard library, numpy and python-flint.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import acb, acb_mat, arb, ctx

ctx.prec = 320

# The sum of a few FP64 numbers is exact at this precision for all finite exponents.
SUM_PREC = 2200
# The Krawczyk steps for h: the largest count, and the largest final radius.
H_STEPS = 60
H_RAD_MAX = 1e-80
S4 = list(itertools.permutations(range(4)))


def _exact_real(values):
    """The exact sum of FP64 numbers as an arb."""
    with ctx.workprec(SUM_PREC):
        out = arb(0)
        for v in values:
            out = out + arb(float(v))
    return out


def exact(limbs) -> acb:
    """The exact sum of 4 dd limbs (re_hi, re_lo, im_hi, im_lo) or of 6 td limbs
    (re0, re1, re2, im0, im1, im2) as an acb."""
    v = np.asarray(limbs, dtype=float).ravel()
    if v.size not in (4, 6):
        raise ValueError(f'a dd or td number has 4 or 6 limbs, not {v.size}')
    half = v.size // 2
    with ctx.workprec(SUM_PREC):
        out = acb(_exact_real(v[:half]), _exact_real(v[half:]))
    if np.all(np.isfinite(v)) and not out.is_exact():
        raise ValueError(f'the sum of the limbs {v.tolist()} is not exact')
    return out


def _as_acb(x) -> acb:
    """An acb from an acb, an arb, a Python complex or a real number."""
    if isinstance(x, acb):
        return x
    if isinstance(x, complex):
        return acb(x.real, x.imag)
    return acb(x)


def poly_at(coefs, mons, x) -> acb:
    """sum over s of c_s prod over v of x_v^mons[s, v], with c_s the exact sum of the
    limbs coefs[:, s] (4 or 6 limbs)."""
    c = np.asarray(coefs, dtype=float)
    m = np.asarray(mons, dtype=np.int64)
    if c.ndim != 2 or m.ndim != 2 or c.shape[1] != m.shape[0] or m.shape[1] != len(x):
        raise ValueError(f'coefs {c.shape}, mons {m.shape} and {len(x)} variables do not agree')
    if m.size and m.min() < 0:
        raise ValueError('an exponent is negative')
    pw = []
    for v, xv in enumerate(x):
        row = [acb(1)]
        for _ in range(int(m[:, v].max()) if m.shape[0] else 0):
            row.append(row[-1] * xv)
        pw.append(row)
    total = acb(0)
    for s in range(m.shape[0]):
        term = exact(c[:, s])
        for v in range(m.shape[1]):
            e = int(m[s, v])
            if e:
                term = term * pw[v][e]
        total += term
    return total


def ppow(x, a) -> acb:
    """The principal power exp(a log x) for a rational a."""
    a = Fraction(a)
    return (arb(a.numerator) / a.denominator * plog(x)).exp()


def plog(x) -> acb:
    """The principal logarithm, with the cut on the negative real axis."""
    return _as_acb(x).log()


@dataclass
class Potential:
    """The base p0 and the level forms N_L of the certified potential.

    base: for each group, (exps, blocks) with blocks a list of (r, C);
    levels: for each level, (L, groups, orbits) with groups {g: (exps, Phi_g^T)} and
    orbits a list of (g, r, C_o). Each C is an acb_mat with real entries.
    """
    k0: int
    top: int
    base: list
    levels: list


def load(data_path, forms_path) -> Potential:
    """p0 from data_path by Definition "the base p0", and N_L from forms_path by
    Definition "the level forms N_L"."""
    data, forms = np.load(data_path), np.load(forms_path)
    base = []
    for i in range(int(data['n_groups'])):
        frame = np.asarray(data[f'g{i}_frame'])
        if not np.array_equal(frame, np.eye(frame.shape[0])):
            raise ValueError(f'base group {i}: the frame is not the identity')
        exps = [tuple(int(a) for a in e) for e in data[f'g{i}_exps']]
        hhi, hlo = np.asarray(data[f'g{i}_hhi']), np.asarray(data[f'g{i}_hlo'])
        n = len(exps)
        blocks = []
        for b, r in enumerate(np.asarray(data[f'g{i}_res']).tolist()):
            if hhi[b].shape != (n, n) or hlo[b].shape != (n, n):
                raise ValueError(f'base group {i}, block {b}: shape {hhi[b].shape}, {n} exponents')
            # Each entry is the exact sum hhi + hlo of two FP64 numbers.
            ent = [_exact_real((x, y)) for x, y in zip(hhi[b].ravel(), hlo[b].ravel())]
            blocks.append((tuple(int(a) for a in r), acb_mat(n, n, ent)))
        base.append((exps, blocks))
    levels = []
    for L in sorted(int(x) for x in forms['k1']):
        groups = {}
        for g in range(int(forms[f'L{L}_n_groups'])):
            exps = [tuple(int(a) for a in e) for e in forms[f'L{L}_g{g}_exps']]
            phi = np.asarray(forms[f'L{L}_g{g}_frame'], dtype=float)
            if phi.shape[0] != len(exps):
                raise ValueError(f'level {L}, group {g}: frame {phi.shape}, {len(exps)} exponents')
            # psi_g(w) = Phi_g^T u(w), so the stored matrix is the transpose of Phi_g.
            groups[g] = (exps, acb_mat(phi.shape[1], phi.shape[0],
                                       [arb(float(x)) for x in phi.T.ravel()]))
        orbits = []
        for o in range(int(forms[f'L{L}_n_orbits'])):
            # Each entry of C_o is a decimal text; arb holds its exact rational in a ball.
            cs = np.asarray(forms[f'L{L}_o{o}_C'])
            g = int(forms[f'L{L}_o{o}_group'])
            if cs.shape != (groups[g][1].nrows(),) * 2:
                raise ValueError(f'level {L}, orbit {o}: C_o {cs.shape}')
            orbits.append((g, tuple(int(a) for a in forms[f'L{L}_o{o}_r']),
                           acb_mat(cs.shape[0], cs.shape[1], [arb(str(x)) for x in cs.ravel()])))
        levels.append((L, groups, orbits))
    return Potential(k0=int(data['k0']), top=max(L for L, _, _ in levels), base=base,
                     levels=levels)


def _fourth_powers(w, emax):
    """pw[a][e] = (w_a^4)^e for e <= emax."""
    out = []
    for a in range(4):
        w4 = w[a] ** 4
        row = [acb(1)]
        for _ in range(emax):
            row.append(row[-1] * w4)
        out.append(row)
    return out


def _u(pw, exps):
    """The column ((w^4)^E)_E over the exponents E of exps."""
    vals = []
    for e in exps:
        v = acb(1)
        for a in range(4):
            if e[a]:
                v = v * pw[a][e[a]]
        vals.append(v)
    return acb_mat(len(vals), 1, vals)


def _weight(q, r):
    """prod over a of q_a^r_a, with q_a = w_a wb_a."""
    v = acb(1)
    for a in range(4):
        if r[a]:
            v = v * q[a] ** r[a]
    return v


def _bilinear(u, c, ub):
    """u^T C ub for acb columns u, ub."""
    return (u.transpose() * (c * ub))[0, 0]


def forms_polar(pot: Potential, z, zb) -> tuple:
    """The polarized p0 and N_L at (z, zb): w^r wb^r in place of abs(w^r)^2 and
    u(w)^T C u(wb) in place of u(w)^T C conj(u(w)). Returns (p0, {L: N_L})."""
    z, zb = [_as_acb(v) for v in z], [_as_acb(v) for v in zb]
    if len(z) != 4 or len(zb) != 4:
        raise ValueError('z and zb need 4 homogeneous coordinates')
    emax = pot.top // 4
    pw, pwb = _fourth_powers(z, emax), _fourth_powers(zb, emax)
    q = [z[a] * zb[a] for a in range(4)]
    p0 = acb(0)
    for exps, blocks in pot.base:
        u, ub = _u(pw, exps), _u(pwb, exps)
        for r, c in blocks:
            p0 += _weight(q, r) * _bilinear(u, c, ub)
    nl = {}
    for L, groups, orbits in pot.levels:
        total = acb(0)
        for s in S4:
            # w = (z_s(0), ..., z_s(3)), and wb takes the same order.
            idx = [s[a] for a in range(4)]
            pws, pwbs = [pw[i] for i in idx], [pwb[i] for i in idx]
            qs = [q[i] for i in idx]
            psi = {g: (ft * _u(pws, exps), ft * _u(pwbs, exps))
                   for g, (exps, ft) in groups.items()}
            for g, r, c in orbits:
                total += _weight(qs, r) * _bilinear(psi[g][0], c, psi[g][1])
        nl[L] = total
    return p0, nl


@dataclass
class Box:
    """The data of a box: the dd numbers of its frame (p, e[i][k] = dy_k / dt_i, n, c1inv,
    vol), the free dd numbers s_r, s_p, zeta and the dd kappa of the GPU, and c_r, x0, s2
    of the file of local forms, each as an exact acb (or a list of them)."""
    p: list
    e: list
    n: list
    c1inv: acb
    vol: acb
    s_r: acb
    s_p: acb
    zeta: acb
    kappa: acb
    c_r: list
    x0: acb
    s2: acb
    k0: int
    top: int

    @classmethod
    def from_npz(cls, z) -> Box:
        """The box from an export file (a path or the result of np.load), with each dd
        number as 4 limbs (re_hi, re_lo, im_hi, im_lo).

        s_p, x0 and s2 must be real (imaginary limbs 0), and s_p and x0 must be positive:
        x = r2^-1 s_p^(1/k0) (s_p pt)^(-1/k0) = p0^(-1/k0) needs a real positive s_p. s_r
        can be complex: box_f takes 1 / r2 of a complex dd number, and log(s_r r2) and
        s_r (1 + yr)^-1 hold for each nonzero s_r. A wrong shape, or a number that is not
        real or not positive, raises ValueError.
        """
        if isinstance(z, (str, Path)):
            z = np.load(z)
        shapes = dict(fr_p=(3, 4), fr_e=(2, 3, 4), fr_n=(3, 4), fr_c1inv=(4,), fr_vol=(4,),
                      s_r=(4,), s_p=(4,), zeta=(4,), kappa=(4,), c_r=(2, 4), x0=(4,), s2=(4,))
        for key, shape in shapes.items():
            if np.shape(z[key]) != shape:
                raise ValueError(f'{key}: shape {np.shape(z[key])}, not {shape}')
        for key in ('s_p', 'x0', 's2'):
            if np.any(np.asarray(z[key], dtype=float)[2:] != 0):
                raise ValueError(f'{key}: the imaginary limbs {np.asarray(z[key])[2:].tolist()} '
                                 f'are not 0')
        for key in ('s_p', 'x0'):
            if not exact(z[key]).real > 0:
                raise ValueError(f'{key}: {np.asarray(z[key]).tolist()} is not positive')
        fe = np.asarray(z['fr_e'])
        return cls(p=[exact(v) for v in z['fr_p']],
                   e=[[exact(fe[i, k]) for k in range(3)] for i in range(2)],
                   n=[exact(v) for v in z['fr_n']], c1inv=exact(z['fr_c1inv']),
                   vol=exact(z['fr_vol']), s_r=exact(z['s_r']), s_p=exact(z['s_p']),
                   zeta=exact(z['zeta']), kappa=exact(z['kappa']),
                   c_r=[exact(v) for v in z['c_r']], x0=exact(z['x0']), s2=exact(z['s2']),
                   k0=int(z['k0']), top=int(z['top']))


def _y(box: Box, tau, h):
    """y_k = p_k + E[0][k] tau_0 + E[1][k] tau_1 + h n_k for k = 0, 1, 2."""
    return [box.p[k] + box.e[0][k] * tau[0] + box.e[1][k] * tau[1] + h * box.n[k]
            for k in range(3)]


def quartic(y) -> acb:
    """P(y) = y_0^4 + y_1^4 + y_2^4 + 1 in the chart z_3 = 1."""
    return y[0] ** 4 + y[1] ** 4 + y[2] ** 4 + 1


def _rad(x: acb) -> float:
    """The larger of the radii of the real and the imaginary part, rounded up."""
    return max(float(x.real.rad()), float(x.imag.rad())) * (1 + 1e-12)


def solve_h(box: Box, tau, h0, rad=1e-3) -> acb:
    """The unique root h of P(p + E tau + h n) = 0 in the disc abs(h - h0) <= rad.

    The first Krawczyk step takes the square of half-width rad about h0, which holds the
    disc; K inside that square proves one root there and no other. Each next step takes
    the square of half-width 4 rad(K) about mid(K), which holds the root. The steps stop
    when the radius does not fall by a factor of 2. A step with K not inside its square, a
    root off the disc, or a final radius not below 1e-80 raises ValueError.
    """
    tau = [_as_acb(t) for t in tau]
    h0 = _as_acb(h0).mid()

    def f(h):
        return quartic(_y(box, tau, h))

    def df(h):
        y = _y(box, tau, h)
        return sum((4 * y[k] ** 3 * box.n[k] for k in range(3)), acb(0))

    mid, half = h0, arb(rad)
    best, best_rad = None, math.inf
    for _ in range(H_STEPS):
        x = acb(arb(mid.real, half), arb(mid.imag, half))
        slope = df(mid)
        if not slope.is_finite() or slope.contains(0):
            raise ValueError(f'dP/dh at {mid} holds 0')
        # An exact preconditioner keeps the Krawczyk form valid.
        y = (1 / slope).mid()
        k = mid - y * f(mid) + (1 - y * df(x)) * (x - mid)
        if not x.contains_interior(k):
            raise ValueError(f'a Krawczyk step does not prove a root near {mid} '
                             f'(half-width {float(half):.3e})')
        new = _rad(k)
        if new < best_rad:
            best, prev, best_rad = k, best_rad, new
        else:
            prev = best_rad
        if not new < 0.5 * prev:
            break
        mid, half = k.mid(), arb(4 * new)
    if best is None or not best_rad < H_RAD_MAX:
        raise ValueError(f'the radius of h is {best_rad:.3e}, not below {H_RAD_MAX:.0e}')
    if not abs(best - h0).upper() <= rad:
        raise ValueError('the root lies off the disc abs(h - h0) <= rad')
    return best


def polar_point(box: Box, tau, sigma, h_tau, h_conj_sigma) -> tuple:
    """z = (y(tau), 1) and zb = (yb(sigma), 1) with yb(sigma) = conj(y(conj(sigma))), with
    the roots h at tau and at conj(sigma) of solve_h."""
    tau = [_as_acb(t) for t in tau]
    sc = [_as_acb(s).conjugate() for s in sigma]
    z = _y(box, tau, _as_acb(h_tau)) + [acb(1)]
    zb = [v.conjugate() for v in _y(box, sc, _as_acb(h_conj_sigma))] + [acb(1)]
    return z, zb


def step_values(pot: Potential, box: Box, z, zb) -> dict:
    """The polarized values of the steps 2 to 9 at (z, zb), with principal branches:

    delta = z_2 zb_2 - s2; p0; r2 = 1 + sum over k < 3 of z_k zb_k; ir2 = 1 / r2;
    log_r2 = log(s_r r2); pt = p0 r2^-k0; xh = (s_p pt)^(-1/k0); log_pt = log(s_p pt);
    xw = ir2 xh; xi = kappa xw - 1 with the exact kappa = s_p^(1/k0) / x_c and
    x_c = x0 (1 + zeta); fk<k> = sum over L of C(L, k) x_c^L N_L for k = 0, ..., top;
    psi = sum over L of N_L x^L with x = x_c (1 + xi) = r2^-1 s_p^(1/k0) (s_p pt)^(-1/k0);
    phi = k0 log_r2 + log_pt + psi.
    """
    k0, top = pot.k0, pot.top
    if box.k0 != k0 or box.top != top:
        raise ValueError(f'the box has k0 {box.k0}, top {box.top}; the potential {k0}, {top}')
    z, zb = [_as_acb(v) for v in z], [_as_acb(v) for v in zb]
    p0, nl = forms_polar(pot, z, zb)
    out = {'delta': z[2] * zb[2] - box.s2, 'p0': p0}
    r2 = 1 + z[0] * zb[0] + z[1] * zb[1] + z[2] * zb[2]
    ir2 = 1 / r2
    pt = p0 / r2 ** k0
    xh = ppow(box.s_p * pt, Fraction(-1, k0))
    out.update(r2=r2, ir2=ir2, log_r2=plog(box.s_r * r2), pt=pt, xh=xh,
               log_pt=plog(box.s_p * pt), xw=ir2 * xh)
    xc = box.x0 * (1 + box.zeta)
    kappa = ppow(box.s_p, Fraction(1, k0)) / xc
    xi = kappa * out['xw'] - 1
    out['xi'] = xi
    for k in range(top + 1):
        out[f'fk{k}'] = sum((math.comb(L, k) * xc ** L * nl[L] for L in nl if L >= k), acb(0))
    x = xc * (1 + xi)
    out['psi'] = sum((nl[L] * x ** L for L in nl), acb(0))
    out['phi'] = k0 * out['log_r2'] + out['log_pt'] + out['psi']
    return out


# The order and the radius of tm_box.run_box for the steps 10 to 12. The coefficients of
# degree 0 and 1 of its models are exact Taylor coefficients for each order >= 1.
TM_ORDER = 2
TM_RADIUS = 1e-4
TM_KEYS = ('z', 'patch', 'ratio', 'v1')


def tm_center(z, tb, order=TM_ORDER) -> dict:
    """The result of tb.run_box at the point z = (y_0, y_1, y_2, 1) of X, with tb the module
    scripts/tm_box.py after tm_box.load at 320 bits.

    The chart is that of the frame: patch [2, 0, 1], so z_3 = 1, t = (z_0, z_1) and m = 2.
    The point takes one new slot of tb.D, and the call removes it again.
    """
    if ctx.prec != 320:
        raise ValueError(f'the arb precision is {ctx.prec}, not 320')
    if len(z) == 4 and not _as_acb(z[3]) == 1:
        raise ValueError('the point is not in the chart z_3 = 1')
    sizes = {len(tb.D[key]) for key in TM_KEYS}
    if len(sizes) != 1:
        raise ValueError(f'the lists {TM_KEYS} of tm_box.D have the sizes {sizes}')
    k = sizes.pop()
    tb.D['z'].append([_as_acb(v) for v in z[:3]] + [acb(1)])
    tb.D['patch'].append([2, 0, 1])
    tb.D['ratio'].append(0.0)
    tb.D['v1'].append(acb(0))
    try:
        return tb.run_box((k, order, [TM_RADIUS]))
    finally:
        for key in TM_KEYS:
            del tb.D[key][k:]


def metric_from_center(box: Box, z, res) -> dict:
    """The values of the steps 10 to 12 at the point z = y(tau) of the slice, from the
    result `res` of tm_center at z.

    J[a][i] = dt_a / dtau_i = E[i][a] + n_a dh/dtau_i, with dh/dtau_i = -(grad P . E_i) /
    (grad P . n) and grad P = (4 y_k^3). The values: g<i><j> of g_tau = J^T g_t conj(J);
    det and tr of g_tau; lam_min, the smallest eigenvalue of the Hermitian g_tau;
    q4 = abs(grad P . n)^2; f, and v = f + 1; df<i> = sum over a of df/dt_a J[a][i];
    grad_G = sqrt((4 / c) conj(d_t f)^T g_t^-1 d_t f) with c = 1 / (sqrt(2) pi k0).
    """
    y = [_as_acb(v) for v in z[:3]]
    gt = [[res['g_center'][a, b] for b in range(2)] for a in range(2)]
    ft = list(res['df_center'])
    f = res['f_acb']
    grad = [4 * v ** 3 for v in y]
    dn = sum((g * m for g, m in zip(grad, box.n)), acb(0))
    dh = [-sum((g * e for g, e in zip(grad, box.e[i])), acb(0)) / dn for i in range(2)]
    jac = [[box.e[i][a] + box.n[a] * dh[i] for i in range(2)] for a in range(2)]
    gtau = [[sum((jac[a][i] * gt[a][b] * jac[b][j].conjugate()
                  for a in range(2) for b in range(2)), acb(0)) for j in range(2)]
            for i in range(2)]
    out = {f'g{i}{j}': gtau[i][j] for i in range(2) for j in range(2)}
    out['det'] = gtau[0][0] * gtau[1][1] - gtau[0][1] * gtau[1][0]
    out['tr'] = gtau[0][0] + gtau[1][1]
    # The true g_tau is Hermitian, so its diagonal is real and g_10 = conj(g_01).
    da, db = gtau[0][0].real, gtau[1][1].real
    out['lam_min'] = acb((da + db) / 2 - (((da - db) / 2) ** 2 + abs(gtau[0][1]) ** 2).sqrt())
    out['q4'] = acb(abs(dn) ** 2)
    out['f'], out['v'] = f, f + 1
    for i in range(2):
        out[f'df{i}'] = sum((ft[a] * jac[a][i] for a in range(2)), acb(0))
    det_t = gt[0][0] * gt[1][1] - gt[0][1] * gt[1][0]
    inv = [[gt[1][1] / det_t, -gt[0][1] / det_t], [-gt[1][0] / det_t, gt[0][0] / det_t]]
    quad = sum((ft[j].conjugate() * inv[j][i] * ft[i] for i in range(2) for j in range(2)),
               acb(0))
    c = 1 / (arb(2).sqrt() * arb.pi() * box.k0)
    out['grad_G'] = acb((4 * quad.real / c).sqrt())
    return out


def metric_values(box: Box, z, tb) -> dict:
    """The values of metric_from_center at the point z = y(tau) of the slice, with
    tb.run_box at the order TM_ORDER and the radius TM_RADIUS."""
    if tb.D['k0'] != box.k0:
        raise ValueError(f'tm_box has k0 {tb.D["k0"]}, the box {box.k0}')
    return metric_from_center(box, z, tm_center(z, tb))
