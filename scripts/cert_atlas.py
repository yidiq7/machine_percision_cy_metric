"""The bounds of the atlas of notes/draft-main.tex that need no potential, and their
combination with the records of a run.

docs/superpowers/specs/2026-10-24-geometry-bounds-design.md has the formulas. The chart
b = (3, 2) has z = (t_0, t_1, w, 1) with w^4 = -(1 + t_0^4 + t_1^4). A chart a = (i, j) has
the coordinates (z_k / z_i, z_l / z_i), with {k < l} the other two indices, and the domain
D(i, j) = {abs(z_i) >= 0.99 abs(z_x) for x != i, abs(z_j) >= 0.98 abs(z_x) for x not in
{i, j}}, with no window of the argument (a superset of each chart of the draft). The phases of
t_0 and t_1 and the conjugation keep each domain and each norm, so the boxes need only
arg t_0 in [0, pi/4] and arg t_1 in [0, pi/2]. No value depends on the branch of w.

Stages:
* fs: lam0 and Lam0, the smallest and the largest eigenvalue of the FS metric of volume 1
  (FS / (pi sqrt 2)) in the chart b, on D(3, 2).
* transitions: for each chart a != (3, 2), bounds of the sups c~1_0 and c~2_0 on D(3, 2) and D(a) of the
  real norms of Q0_a D^n F_a [Q0_b^-1 .] (n = 2, 3) with the FS forms Q0 at the same point, F_a
  the coordinates of a as functions of t. Since FS is one metric, abs(Q0_a w) = abs(Q0_b DF_a^-1
  w), so each is at most sqrt(lam_max) lam_min^(-n/2) abs(DF_a^-1 D^n F_a) with the FS
  eigenvalues of chart b at t. A floor from random points of all charts lets a chart with a
  smaller sup stop early, so each result bounds max(its sup, the floor) and their maximum bounds
  the maximum over the charts, which is what combine takes.
* combine: the constants of the draft from the records of a run (all parts) and the results
  of fs and transitions.

The bound of a box of side s and center c comes from Taylor jets at c (exact) and on the box
(arb balls), so that the growth of the balls enters only the term of s^2:
* E_n = DF^-1 D^n F (n = 2, 3), holomorphic in t: abs(E(x)) <= abs(E(c)) + abs(DE(c)) s
  + sup abs(D^2 E) s^2 / 2 on the box, since abs(x - c)_2 <= s; jets of F of degree 5.
* the FS eigenvalues: the ball value comes from the closed form `fs_closed`, which takes no
  root of a ball; and the potential log(1 + t.sigma + w(t) w~(sigma)) in the four variables
  (t, sigma), sigma = conj(t) on the slice, as jets of degree 4. g(x) - g(c) is at most the
  linear terms at c plus the terms of degree 2 on the box, each with abs(x_i - c_i) <= s /
  sqrt 2, and Weyl's inequality moves each eigenvalue by at most abs(g(x) - g(c))_F.
The bound is the better of this form and the ball value on the box. A branch-and-bound halves
a box in each coordinate until its bound is within --tol of the best value at a point of the
domain, or its side is at most R / 2^--min-depth. The result is the largest (or smallest)
bound of a box.

Usage, on CPU cores:
    uv run --extra gpu --locked python -u scripts/cert_atlas.py fs --procs 48 --out A.npz
    uv run --extra gpu --locked python -u scripts/cert_atlas.py transitions --procs 48 --out T.npz
    uv run --extra gpu --locked python -u scripts/cert_atlas.py combine --runs P0.npz P1.npz \\
        --fs A.npz --transitions T.npz
"""
import argparse
import itertools
import math
import multiprocessing
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import acb, arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmgpu.host import down, up                              # noqa: E402

ctx.prec = 128
# The first boxes have the side R / 8; R = 33/32 > 1 / 0.99 holds every coordinate of D(3, 2).
R = 33 / 32
F99, F98 = Fraction(99, 100), Fraction(49, 50)
CHART_B = (3, 2)
CHARTS = [(i, j) for i in range(4) for j in range(4) if i != j and (i, j) != CHART_B]


# ---------------------------------------------------------------- Taylor jets


def compositions(total, k):
    if k == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest


class Space:
    """The monomials of degree <= order in n variables, and the pairs of a product."""

    def __init__(self, n, order):
        self.n, self.order = n, order
        self.mons = [m for d in range(order + 1) for m in compositions(d, n)]
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.pairs = [(a, b, self.idx[tuple(x + y for x, y in zip(ma, mb))])
                      for a, ma in enumerate(self.mons) for b, mb in enumerate(self.mons)
                      if sum(ma) + sum(mb) <= order]


SP2, SP4 = Space(2, 5), Space(4, 4)
SPACES2 = {k: Space(2, k) for k in range(5)}
SPACES2[5] = SP2


class Jet:
    """A holomorphic function about a point, to the order of its Space: the Taylor coefficients
    as acb balls."""

    def __init__(self, sp, c):
        self.sp, self.c = sp, c

    @staticmethod
    def const(sp, z):
        return Jet(sp, [acb(z)] + [acb(0)] * (len(sp.mons) - 1))

    @staticmethod
    def var(sp, k, z):
        j = Jet.const(sp, z)
        j.c[1 + k] = acb(1)
        return j

    def __add__(self, o):
        if not isinstance(o, Jet):
            o = Jet.const(self.sp, o)
        return Jet(self.sp, [a + b for a, b in zip(self.c, o.c)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(self.sp, [-a for a in self.c])

    def __sub__(self, o):
        return self + (-o if isinstance(o, Jet) else acb(-o))

    def __mul__(self, o):
        if not isinstance(o, Jet):
            return Jet(self.sp, [a * o for a in self.c])
        out = [acb(0)] * len(self.c)
        for a, b, k in self.sp.pairs:
            out[k] += self.c[a] * o.c[b]
        return Jet(self.sp, out)

    def rel(self):
        """y = x / x(0) - 1, the jet with no constant term."""
        x0 = self.c[0]
        return Jet(self.sp, [acb(0)] + [v / x0 for v in self.c[1:]])


def trunc(x, order):
    """The jet to a lower order (the monomials come in the order of the degree)."""
    sp = SPACES2[order]
    return Jet(sp, x.c[:len(sp.mons)])


def deriv(x, k):
    """d/dt_k of a jet of two variables, as a jet of one order less."""
    sp = SPACES2[x.sp.order - 1]
    out = []
    for m in sp.mons:
        e = list(m)
        e[k] += 1
        out.append(x.c[x.sp.idx[tuple(e)]] * (m[k] + 1))
    return Jet(sp, out)


def series(x, value, coeffs):
    """value times the sum of coeffs[k] y^k (k to the order) with y = x / x(0) - 1, by Horner's
    rule."""
    y = x.rel()
    s = Jet.const(x.sp, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        s = s * y + c
    return s * value


def _binomial(a, n):
    """The coefficients C(a, k), k = 0..n, of (1 + y)^a for a Fraction a, as acb."""
    out, c = [], Fraction(1)
    for k in range(n + 1):
        out.append(acb(arb(c.numerator) / c.denominator))
        c = c * (a - k) / (k + 1)
    return out


def inv(x):
    return series(x, 1 / x.c[0], [acb((-1) ** k) for k in range(x.sp.order + 1)])


def log_jet(x):
    return series(x, acb(1), [acb(0)] + [acb((-1) ** (k + 1)) / k
                                         for k in range(1, x.sp.order + 1)]) + x.c[0].log()


def root4(v):
    """A fourth root of the ball v, continuous on it: the principal root of v, or
    e^(i pi/4) times the principal root of -v when v is nearer the negative real axis."""
    if float(v.real.mid()) >= 0:
        return v.root(4)
    return acb(arb(1) / 4).exp_pi_i() * (-v).root(4)


def root4_jet(v, root):
    """The fourth root of the jet v with the value root at its base point."""
    return series(v, root, _binomial(Fraction(1, 4), v.sp.order))


# ---------------------------------------------------------------- the boxes


def sq(x):
    """x * x: python-flint gives NaN for x ** 2 of a ball that holds 0."""
    return x * x


def abs2(x):
    """abs(x)^2 as the real part of x conj(x)."""
    return (x * x.conjugate()).real


def root_up(x):
    """sqrt of the upper end of the ball x, for a true value not below 0: its upper end is an
    upper bound of the true root (the lower end of x can be below 0 for a ball that holds 0)."""
    return arb(up(x)).sqrt()


def wof(t0, t1):
    """w with w^4 = -(1 + t_0^4 + t_1^4), for acb balls."""
    q0, q1 = t0 * t0, t1 * t1
    return root4(-(1 + q0 * q0 + q1 * q1))


def ball(lo, s):
    """t_0, t_1 as acb balls of the box [lo, lo + s]^4 of (Re t_0, Im t_0, Re t_1, Im t_1)."""
    x = [arb(lo[k] + s / 2, s / 2) for k in range(4)]
    return acb(x[0], x[1]), acb(x[2], x[3])


def _range(x):
    """(lower, upper) floats of an arb x >= 0; a lower end below 0 (or NaN) gives 0."""
    lo = down(x)
    return (lo if lo > 0 else 0.0), up(x)


def moduli(t0, t1):
    """The ranges (lower, upper) of abs(z) for z = (t_0, t_1, w, 1), from abs(w)^4 =
    abs(1 + t_0^4 + t_1^4), so that a ball that holds w = 0 gives no NaN."""
    q0, q1 = t0 * t0, t1 * t1
    lo_u, hi_u = _range(abs(1 + q0 * q0 + q1 * q1))
    w = (down(arb(lo_u).sqrt().sqrt()) if lo_u > 0 else 0.0, up(arb(hi_u).sqrt().sqrt()))
    return [_range(abs(t0)), _range(abs(t1)), w, (1.0, 1.0)]


def _below(a, k, b):
    """True only if the float a is below k times the float b, in exact arithmetic."""
    return math.isfinite(a) and math.isfinite(b) and Fraction(a) < k * Fraction(b)


def may_meet(m, chart):
    """False only if the ranges m of the moduli prove that no point is in D(chart)."""
    i, j = chart
    for x in range(4):
        if x != i and _below(m[i][1], F99, m[x][0]):
            return False
        if x not in chart and _below(m[j][1], F98, m[x][0]):
            return False
    return True


def inside(m, chart):
    """True only if the ranges m prove that their point is in D(chart)."""
    i, j = chart
    ok = lambda a, k, b: math.isfinite(b) and not _below(a, k, b)
    return all(ok(m[i][0], F99, m[x][1]) for x in range(4) if x != i) \
        and all(ok(m[j][0], F98, m[x][1]) for x in range(4) if x not in chart)


def in_wedge(lo, s):
    """The box can hold a point with Im t_0 <= Re t_0 (arg t_0 <= pi/4)."""
    return lo[1] <= lo[0] + s


# ---------------------------------------------------------------- the values


def fs_matrix(t0, t1, w):
    """The Hermitian matrix of the FS metric in the chart b (before the division by pi sqrt 2)."""
    y = [t0, t1, w]
    r0, r1 = t0 / w, t1 / w
    jac = [[acb(1), acb(0)], [acb(0), acb(1)], [-r0 * r0 * r0, -r1 * r1 * r1]]
    s = 1 + abs2(t0) + abs2(t1) + abs2(w)
    h = [[(1 if m == q else 0) / s - y[m].conjugate() * y[q] / (s * s) for q in range(3)]
         for m in range(3)]
    return [[sum((jac[m][i] * h[m][q] * jac[q][j].conjugate() for m in range(3) for q in range(3)),
                 acb(0)) for j in range(2)] for i in range(2)]


FS_SCALE = arb.pi() * arb(2).sqrt()


def eigs(g):
    """The two eigenvalues of g / (pi sqrt 2) for an acb matrix g: the lower end of the first and
    the upper end of the second are bounds (root_up)."""
    a, d = g[0][0].real, g[1][1].real
    mid = (a + d) / 2
    rad = root_up(sq((a - d) / 2) + abs2(g[0][1]))
    return (mid - rad) / FS_SCALE, (mid + rad) / FS_SCALE


def fs_eigs(t0, t1, w):
    """The two eigenvalues of the FS metric of volume 1 in the chart b, as in `eigs`."""
    return eigs(fs_matrix(t0, t1, w))


def fs_closed(t0, t1):
    """Bounds (low, up) as floats of the two eigenvalues of the FS metric of volume 1 in chart b
    on the balls t, from the closed form with no root of a ball: with u = 1 + t_0^4 + t_1^4 and
    abs(w)^2 = abs(u)^(1/2), s g = I + a a^H - b b^H / s, s = 1 + abs(t)^2 + abs(u)^(1/2),
    abs(a)^2 = (abs(t_0)^6 + abs(t_1)^6) / abs(u)^(3/2), b_i = conj(t_i) + abs(u)^(1/2) t_i^3 / u,
    abs(a^H b)^2 = abs(sum of conj(t_i^4) + abs(u)^(1/2) abs(t_i)^6 / u)^2 / abs(u)^(3/2).
    lam_min(M) = 2 det / (tr + sqrt(tr^2 - 4 det)) falls with tr and grows with det, and lam_max(M)
    = (tr + sqrt(tr^2 - 4 det)) / 2 grows with tr and falls with det. A second form in the basis
    a / abs(a) has no cancellation near the corner abs(t_0) ~ abs(t_1) ~ abs(w); each bound is the
    better of the two."""
    q = [t0 * t0, t1 * t1]
    p = [x * x for x in q]
    r2 = [abs2(t0), abs2(t1)]
    # r6 and the forms below are those of abs(t_i)^6 = abs(t_i^3)^2.
    r6 = [x * x * x for x in r2]
    u = 1 + p[0] + p[1]
    au = abs(u)
    if not down(au) > 0:
        return -math.inf, math.inf
    su = au.sqrt()
    s = 1 + r2[0] + r2[1] + su
    a2 = (r6[0] + r6[1]) / (au * su)
    b2 = sum((abs2(t.conjugate() + su * t * x / u) for t, x in zip((t0, t1), q)), arb(0))
    c = sum((x.conjugate() + su * r / u for x, r in zip(p, r6)), acb(0))
    cc = abs2(c) / (au * su)
    s_lo, s_hi = down(s), up(s)
    if not s_lo > 0:
        return -math.inf, math.inf
    low, high = -math.inf, math.inf
    tr = 2 + a2 - b2 / s
    det = (1 + a2) * (1 - b2 / s) + cc / s
    tr_hi, det_lo = up(tr), down(det)
    if det_lo > 0:
        t_hi, d_lo = arb(tr_hi), arb(det_lo)
        root = arb(up(t_hi * t_hi - 4 * d_lo)).sqrt() if up(t_hi * t_hi - 4 * d_lo) > 0 else arb(0)
        low = down(2 * d_lo / (t_hi + root) / arb(s_hi) / FS_SCALE)
        high = up((t_hi + root) / 2 / arb(s_lo) / FS_SCALE)
    # The same matrix in the basis a / abs(a), e_2: M_11 = 1 + P - Q / s, M_22 = 1 - R / s,
    # abs(M_12)^2 = X = Q R / s^2, with P = abs(a)^2, Q = abs(a^H b)^2 / P and R = abs(a_0 b_1 -
    # a_1 b_0)^2 / P (Lagrange); then lam = min or max(M_11, M_22) -+ X / (abs(D) + sqrt(D^2 + X)),
    # D = (M_11 - M_22) / 2, with no cancellation where one eigenvalue is much the larger.
    r6s = r6[0] + r6[1]
    if down(r6s) > 0:
        bb = [t.conjugate() + su * t * x / u for t, x in zip((t0, t1), q)]
        qq = abs2(c) / r6s
        rr = abs2(t0 * q[0] * bb[1] - t1 * q[1] * bb[0]) / r6s
        m11, m22 = 1 + a2 - qq / s, 1 - rr / s
        x_hi = arb(up(qq * rr / (s * s)))
        dd = (m11 - m22) / 2
        d_lo = arb(max(down(abs(dd)), 0.0))
        shift = x_hi / (d_lo + (d_lo * d_lo + x_hi).sqrt())
        lam_lo = arb(min(down(m11), down(m22))) - shift
        lam_hi = arb(max(up(m11), up(m22))) + shift
        if down(lam_lo) > 0:
            low = max(low, down(lam_lo / arb(s_hi) / FS_SCALE))
        high = min(high, up(lam_hi / arb(s_lo) / FS_SCALE))
    return low, high


def fs_jets(t0, t1):
    """The jets of degree 2 of the matrix g_ij = d_t_i d_sigma_j phi about (t, conj t), with
    phi = log(1 + t.sigma + w(t) w~(sigma)) and w~(sigma) = conj(w(conj sigma))."""
    T = [Jet.var(SP4, k, z) for k, z in enumerate((t0, t1, t0.conjugate(), t1.conjugate()))]
    q = [x * x for x in T]
    u = 1 + q[0] * q[0] + q[1] * q[1]
    us = 1 + q[2] * q[2] + q[3] * q[3]
    w0 = root4(-u.c[0])
    w = root4_jet(-u, w0)
    ws = root4_jet(-us, w0.conjugate())
    phi = log_jet(1 + T[0] * T[2] + T[1] * T[3] + w * ws)
    sp2 = Space(4, 2)
    out = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            c = []
            for m in sp2.mons:
                e = list(m)
                e[i] += 1
                e[2 + j] += 1
                c.append(phi.c[SP4.idx[tuple(e)]] * ((m[i] + 1) * (m[2 + j] + 1)))
            out[i][j] = c
    return sp2, out


def chart_jets(t0, t1, chart):
    """The two coordinates of chart a = (i, j) as jets of degree 5 about the point t of b."""
    z = [Jet.var(SP2, 0, t0), Jet.var(SP2, 1, t1), None, Jet.const(SP2, 1)]
    u = 1 + z[0] * z[0] * z[0] * z[0] + z[1] * z[1] * z[1] * z[1]
    z[2] = root4_jet(-u, root4(-u.c[0]))
    i, j = chart
    k, l = (x for x in range(4) if x not in chart)
    zi = inv(z[i])
    return [z[k] * zi, z[l] * zi]


def e_tensor(comps, n):
    """The jets (order 5 - n) of E_n = DF^-1 D^n F, the entries E[(m,) + ks] for the ordered
    tuples ks: with this tensor the FS form of chart a is that of chart b, abs(Q0_a w) =
    abs(Q0_b DF^-1 w)."""
    order = 5 - n
    df = [[trunc(deriv(f, k), order) for k in range(2)] for f in comps]
    di = inv(df[0][0] * df[1][1] - df[0][1] * df[1][0])
    dinv = [[df[1][1] * di, -(df[0][1] * di)], [-(df[1][0] * di), df[0][0] * di]]
    out = {}
    for ks in itertools.product(range(2), repeat=n):
        dn = []
        for f in comps:
            for k in ks:
                f = deriv(f, k)
            dn.append(f)
        for m in range(2):
            out[(m,) + ks] = dinv[m][0] * dn[0] + dinv[m][1] * dn[1]
    return out


def tensor_squares(ent):
    """The squares of the complex norms of E, DE and D^2 E at the base point of the jets of
    the entries of E: the sums of k! m! abs(c_m)^2 over the monomials m of degree k."""
    q = [arb(0), arb(0), arb(0)]
    for e in ent.values():
        for idx, m in enumerate(e.sp.mons):
            k = sum(m)
            if k <= 2:
                q[k] += math.factorial(k) * math.prod(math.factorial(x) for x in m) * abs2(e.c[idx])
    return q


def fs_factor(lam_lo, lam_hi, n):
    """An upper bound of sqrt(lam_max) / lam_min^(n/2) from the floats lam_lo <= lam_min and
    lam_hi >= lam_max; inf if lam_lo is not above 0."""
    if not lam_lo > 0 or math.isnan(lam_hi):
        return arb(math.inf)
    low = arb(lam_lo) if n == 2 else arb(lam_lo) * arb(lam_lo).sqrt()
    return arb(lam_hi).sqrt() / low


def square_norm(comps, n):
    """The square of the complex norm of D^n F from the jets: the sum over the components and
    the monomials m of degree n of n! m! abs(c_m)^2."""
    sp = comps[0].sp
    out = arb(0)
    for f in comps:
        for k, m in enumerate(sp.mons):
            if sum(m) == n:
                out += math.factorial(n) * math.prod(math.factorial(x) for x in m) * abs2(f.c[k])
    return out


def real_factor(n):
    """2^(n/2) as an arb: the real norm of a complex n-linear map over its complex norm."""
    return arb(2) if n == 2 else 2 * arb(2).sqrt()


def value(kind, t0, t1):
    """The value of the quantity `kind` at the acb balls t, a ball whose lower end (fs, 0) or
    upper end (else) is a bound: ('fs', 0 or 1) the FS eigenvalues; ('d2' or 'd3', chart) the
    real norm of D^n F_a; ('c1' or 'c2', chart) the real norm of Q0_b E_n [Q0_b^-1 .] with the
    FS form Q0_b of chart b at t (c_bound, with S from the midpoint of the ball)."""
    if kind[0] == 'fs':
        return arb(fs_closed(t0, t1)[kind[1]])
    if kind[0] in ('d2', 'd3'):
        n = 2 if kind[0] == 'd2' else 3
        return real_factor(n) * root_up(square_norm(chart_jets(t0, t1, kind[1]), n))
    mid = lambda z: acb(z.real.mid(), z.imag.mid())
    return c_bound(kind, t0, t1, mid(t0), mid(t1))


def domains(kind):
    return [CHART_B] if kind[0] == 'fs' else [CHART_B, kind[1]]


def is_min(kind):
    return kind == ('fs', 0)


def fs_change(t0, t1, c0, c1, s):
    """(g(c), E) for the box (balls t) of side s and center c: g(c) the FS matrix at the center
    (before the division by pi sqrt 2), and E an arb whose upper end bounds abs(g(x) - g(c))_F on
    the box: the linear terms of the jets at c and the terms of degree 2 on the box, with
    abs(x_i - c_i) <= s / sqrt 2 in each of (t, sigma)."""
    sp2, gc = fs_jets(c0, c1)
    _, gb = fs_jets(t0, t1)
    h = arb(s) / arb(2).sqrt()
    err = arb(0)
    for i in range(2):
        for j in range(2):
            lin = sum((abs(gc[i][j][k]) for k, m in enumerate(sp2.mons) if sum(m) == 1), arb(0))
            rem = sum((abs(gb[i][j][k]) for k, m in enumerate(sp2.mons) if sum(m) == 2), arb(0))
            err += sq(lin * h + rem * h * h)
    return [[gc[i][j][0] for j in range(2)] for i in range(2)], root_up(err)


def sqrtm2(h):
    """The Hermitian square root of a positive Hermitian 2x2 matrix h (acb):
    (h + sqrt(det) I) / sqrt(tr + 2 sqrt(det))."""
    det = (h[0][0] * h[1][1] - h[0][1] * h[1][0]).real
    sd = det.sqrt()
    den = ((h[0][0] + h[1][1]).real + 2 * sd).sqrt()
    return [[(h[0][0] + sd) / den, h[0][1] / den], [h[1][0] / den, (h[1][1] + sd) / den]]


def inv2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]


def hermitian_form(g):
    """H with G(u, u) = u^H H u for the FS matrix g of fs_matrix: H = conj(g) / (pi sqrt 2)."""
    return [[g[i][j].conjugate() / FS_SCALE for j in range(2)] for i in range(2)]


def normalized(ent, qm, sm, n):
    """The jets of E' = Q E[S ., ..., S .] for the entries ent of E (keys (m,) + ks), with the
    constant matrices Q (on the output index) and S (on each of the n input indices)."""
    out = {}
    for m in range(2):
        for ks in itertools.product(range(2), repeat=n):
            tot = None
            for m2 in range(2):
                for src in itertools.product(range(2), repeat=n):
                    c = qm[m][m2]
                    for p, q in zip(src, ks):
                        c = c * sm[p][q]
                    term = ent[(m2,) + src] * c
                    tot = term if tot is None else tot + term
            out[(m,) + ks] = tot
    return out


def congruence(sm, h):
    """S^H h S for 2x2 acb matrices."""
    smh = [[sm[j][i].conjugate() for j in range(2)] for i in range(2)]
    return [[sum((smh[i][a] * h[a][b] * sm[b][j] for a in range(2) for b in range(2)), acb(0))
             for j in range(2)] for i in range(2)]


def n_bounds(h_lo, h_hi, sm, nmat, shift):
    """Bounds (lo, hi) of the eigenvalues of N = S^H H S on the box, the better of lam(H) (the
    floats h_lo, h_hi) times the squares of the extreme singular values of S, and the eigenvalues
    of the matrix nmat moved by shift (Weyl's inequality, shift >= abs(N - nmat)_F)."""
    sf2 = sum((abs2(sm[i][j]) for i in range(2) for j in range(2)), arb(0))
    qf2 = sum((abs2(x) for x in [d for row in inv2(sm) for d in row]), arb(0))
    lo = down(arb(h_lo) / arb(up(qf2))) if h_lo > 0 else -math.inf
    hi = up(arb(h_hi) * sf2)
    e_lo, e_hi = eigs([[x * FS_SCALE for x in row] for row in nmat])
    if not math.isnan(down(e_lo - shift)):
        lo = max(lo, down(e_lo - shift))
    if not math.isnan(up(e_hi + shift)):
        hi = min(hi, up(e_hi + shift))
    return lo, hi


def c_bound(kind, t0, t1, c0, c1, s=None):
    """The bound of c~1_0 or c~2_0 on the ball t: with S = Q_c^-1 from the FS form at the center c,
    abs(Q_x E[Q_x^-1 .]) <= lam_max(N)^(1/2) lam_min(N)^(-n/2) abs(Q_c E[S .]), N = S^H H_x S.
    With s, the Taylor form of the box of side s; without, the ball value."""
    n = 2 if kind[0] == 'c1' else 3
    hc = hermitian_form(fs_matrix(c0, c1, wof(c0, c1)))
    qm = [[acb(x.mid()) for x in row] for row in sqrtm2(hc)]
    sm = inv2(qm)
    h_lo, h_hi = fs_closed(t0, t1)
    if s is None:
        nmat = congruence(sm, hermitian_form(fs_matrix(t0, t1, wof(t0, t1))))
        lo, hi = n_bounds(h_lo, h_hi, sm, nmat, arb(0))
        q = tensor_squares(normalized(e_tensor(chart_jets(t0, t1, kind[1]), n), qm, sm, n))
        e_up = root_up(q[0])
    else:
        gc, err = fs_change(t0, t1, c0, c1, s)
        sf2 = sum((abs2(sm[i][j]) for i in range(2) for j in range(2)), arb(0))
        lo, hi = n_bounds(h_lo, h_hi, sm, congruence(sm, hermitian_form(gc)), sf2 * err / FS_SCALE)
        qc = tensor_squares(normalized(e_tensor(chart_jets(c0, c1, kind[1]), n), qm, sm, n))
        # On the box, D^2 E' = Q D^2 E [S .]: the norm inequality, as the explicit product sums the
        # radii of the ball through the products with S.
        qb = tensor_squares(e_tensor(chart_jets(t0, t1, kind[1]), n))
        qf = root_up(sum((abs2(x) for row in qm for x in row), arb(0)))
        sf = root_up(sum((abs2(x) for row in sm for x in row), arb(0)))
        sfn = sf * sf if n == 2 else sf * sf * sf
        rem = qf * sfn * root_up(qb[2]) * (arb(s) * s / 2)
        e_up = root_up(qc[0]) + root_up(qc[1]) * s + rem
    return real_factor(n) * fs_factor(lo, hi, n) * e_up


def taylor_bound(kind, t0, t1, c0, c1, s):
    """The second-order bound of the quantity on the box (balls t) of side s and center c: an
    arb whose lower end (fs, 0) or upper end (else) bounds the quantity on the box."""
    if kind[0] == 'fs':
        g0, err = fs_change(t0, t1, c0, c1, s)
        lam = eigs(g0)[kind[1]]
        shift = err / FS_SCALE
        return lam - shift if kind[1] == 0 else lam + shift
    if kind[0] in ('d2', 'd3'):
        n = 2 if kind[0] == 'd2' else 3
        jc = chart_jets(c0, c1, kind[1])
        jb = chart_jets(t0, t1, kind[1])
        norm = lambda js, k: root_up(square_norm(js, k))
        bound = norm(jc, n) + norm(jc, n + 1) * s + norm(jb, n + 2) * (arb(s) * s / 2)
        return real_factor(n) * bound
    return c_bound(kind, t0, t1, c0, c1, s)


# ---------------------------------------------------------------- the search


def children(lo, s):
    h = s / 2
    return [([lo[k] + h * ((n >> k) & 1) for k in range(4)], h) for n in range(16)]


def first_boxes():
    """The boxes of side R / 8 of the region arg t_0 <= pi/4 in the first quadrants."""
    s = R / 8
    out = []
    for n in range(8 ** 4):
        lo = [s * ((n // 8 ** k) % 8) for k in range(4)]
        if in_wedge(lo, s):
            out.append((lo, s))
    return out


def step(job):
    """None if the box cannot meet the domain, else (its bound, the value at its center if the
    box goes on and its center is in the domain, else None). The bound is the ball value, or the
    better of it and taylor_bound when the ball value does not finish the box (with best and tol
    of the round); a NaN bound counts as no bound."""
    kind, lo, s, best, tol = job
    t0, t1 = ball(lo, s)
    if not all(may_meet(moduli(t0, t1), ch) for ch in domains(kind)):
        return None
    small = is_min(kind)
    pick = down if small else up
    done = (lambda b: b >= best * (1 - tol)) if small else (lambda b: b <= best * (1 + tol))
    c = [lo[k] + s / 2 for k in range(4)]
    c0, c1 = acb(c[0], c[1]), acb(c[2], c[3])
    b = pick(value(kind, t0, t1))
    if math.isnan(b) or not done(b):
        bt = pick(taylor_bound(kind, t0, t1, c0, c1, s))
        if not math.isnan(bt):
            b = bt if math.isnan(b) else (max(b, bt) if small else min(b, bt))
    if math.isnan(b):
        b = -math.inf if small else math.inf
    point = None
    if not done(b) and c[1] <= c[0] and all(inside(moduli(c0, c1), ch) for ch in domains(kind)):
        qc = value(kind, c0, c1)
        point = up(qc) if small else down(qc)
    return b, point


def sample_best(kind, pool, count=4096, seed=20261024):
    """The best value at random points of the domain of `kind` (a value at a point)."""
    rnd = np.random.default_rng(seed)
    pts = rnd.uniform(0, R, size=(count, 4))
    pts = pts[pts[:, 1] <= pts[:, 0]]
    vals = [v for v in pool.map(_point, [(kind, list(x)) for x in pts], chunksize=16)
            if v is not None]
    if not vals:
        return math.inf if is_min(kind) else 0.0
    return min(vals) if is_min(kind) else max(vals)


def _point(job):
    kind, x = job
    c0, c1 = acb(x[0], x[1]), acb(x[2], x[3])
    if not all(inside(moduli(c0, c1), ch) for ch in domains(kind)):
        return None
    v = value(kind, c0, c1)
    return up(v) if is_min(kind) else down(v)


def run_kind(kind, pool, tol, min_side, boxes=None, verbose=False, floor=None):
    """The branch-and-bound in rounds of one side, so that each round knows the best point
    value of all boxes so far; boxes (default: first_boxes()) are the boxes to start from. With
    floor (a value at a point of another kind of the same maximum), a box is done once its bound
    is within tol of the larger of best and floor: the result then bounds the maximum of the
    kinds, not this kind alone."""
    small = is_min(kind)
    best = math.inf if small else (0.0 if floor is None else floor)
    top = math.inf if small else -math.inf
    level, n, at_min = (first_boxes() if boxes is None else list(boxes)), 0, 0
    while level:
        chunk = max(1, min(16, len(level) // (4 * pool._processes)))
        res = pool.map(step, [(kind, lo, s, best, tol) for lo, s in level], chunksize=chunk)
        pts = [r[1] for r in res if r is not None and r[1] is not None]
        if pts:
            best = min(best, min(pts)) if small else max(best, max(pts))
        nxt = []
        for (lo, s), r in zip(level, res):
            if r is None:
                continue
            n += 1
            b = r[0]
            done = b >= best * (1 - tol) if small else b <= best * (1 + tol)
            if done or s <= min_side:
                at_min += not done
                top = min(top, b) if small else max(top, b)
            else:
                nxt.extend(children(lo, s))
        if verbose:
            print(f'    {kind} side R/{round(R / level[0][1])}: {len(level)} boxes, {len(nxt)} '
                  f'next; best {best:.6g}, bound so far {top:.6g}', flush=True)
        level = nxt
    return dict(bound=top, best=best, boxes=n, at_min=at_min)


def stage(args, kinds, name):
    t0 = time.perf_counter()
    out = {}
    with multiprocessing.Pool(args.procs) as pool:
        floors = {}
        if name == 'transitions':
            # The stage needs only the largest value of each kind over all 11 charts.
            for kind in [(k, ch) for ch in CHARTS for k in args.kinds]:
                b = sample_best(kind, pool)
                floors[kind[0]] = max(floors.get(kind[0], 0.0), b)
                print(f'{name} sample {kind}: best point value {b:.6g}', flush=True)
            for path in args.floor_from or []:
                z = np.load(path)
                for key in args.kinds:
                    vals = [float(z[k]) for k in z.files if k.startswith(key + '_') and k.endswith('_best')]
                    if vals:
                        floors[key] = max(floors.get(key, 0.0), max(vals))
            print(f'{name}: floors {floors}', flush=True)
        for kind in kinds:
            t1 = time.perf_counter()
            r = run_kind(kind, pool, args.tol, R / 2 ** args.min_depth, verbose=True,
                         floor=floors.get(kind[0]))
            key = kind[0] if kind[0] == 'fs' else f'{kind[0]}_{kind[1][0]}{kind[1][1]}'
            if kind[0] == 'fs':
                key = 'lam0' if kind[1] == 0 else 'Lam0'
            for k, v in r.items():
                out[f'{key}_{k}'] = v
            print(f'{name} {key}: bound {r["bound"]:.10g}, best point value {r["best"]:.10g}, '
                  f'{r["boxes"]} boxes, {r["at_min"]} at the smallest side; '
                  f'{time.perf_counter() - t1:.0f}s', flush=True)
    np.savez(args.out, tol=args.tol, min_depth=args.min_depth, **out)
    print(f'{name}: {time.perf_counter() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- the combination


def combine(args):
    from quartic.cert import records as rs
    rec, target, commit, _ = rs.load(args.runs)
    col = {c: k for k, c in enumerate(rs.COLUMNS)}
    top = lambda key: float(np.max(rec[:, col[key]]))
    if np.isnan(rec[:, 8:]).any():
        raise SystemExit('a record holds NaN')
    fs = np.load(args.fs)
    tr = {}
    for path in args.transitions:
        z = np.load(path)
        tr.update({k: z[k] for k in z.files})
    def largest(key):
        """The largest bound of a kind over the 11 charts, or None if a chart has none."""
        vals = [tr.get(f'{key}_{i}{j}_bound') for i, j in CHARTS]
        return None if any(v is None for v in vals) else max(float(v) for v in vals)
    routes = {k: largest(k) for k in ('c1', 'c2', 'd2', 'd3')}
    if (routes['c1'] is None and routes['d2'] is None) or (routes['c2'] is None and routes['d3'] is None):
        raise SystemExit(f'the files of the stage transitions miss charts: {routes}')
    x = lambda v: arb(float(v))
    c_up, mu_min = x(top('mu_up')), x(float(np.min(rec[:, col['mu_low']])))
    lam = mu_min * x(fs['lam0_bound'])
    big = c_up * x(fs['Lam0_bound'])
    # With the FS form: abs(Q_a v) <= sqrt(C) abs(Q0_a v) and abs(Q_b^-1 u) <= mu^(-1/2) abs(Q0_b^-1 u).
    # With the norms: abs(Q_a) <= Lambda^(1/2) and abs(Q_b^-1) <= lambda^(-1/2). Each is a bound.
    c1s, c2s = [], []
    if routes['c1'] is not None:
        c1s.append(c_up.sqrt() / mu_min * x(routes['c1']))
    if routes['d2'] is not None:
        c1s.append(big.sqrt() / lam * x(routes['d2']))
    if routes['c2'] is not None:
        c2s.append(c_up.sqrt() / (mu_min * mu_min.sqrt()) * x(routes['c2']))
    if routes['d3'] is not None:
        c2s.append(big.sqrt() / (lam * lam.sqrt()) * x(routes['d3']))
    c1 = min(c1s, key=up)
    c2 = min(c2s, key=up)
    e2, e3 = routes['c1'], routes['c2']
    k_t = x(top('k_t'))
    kt = k_t + 2 * c1
    kappa = x(top('ric')) / 3
    sup_f = x(top('bound'))
    grad = x(top('grad_g'))
    pi = arb.pi()
    conn = arb(3).sqrt() * (pi / 4 - (1 / arb(3).sqrt().sqrt()).atan()) + arb(1) / 4 \
        + pi / 8 + arb(2).log() / 4
    norm = (pi * arb(2).sqrt()).sqrt()
    diam_m = c_up.sqrt() * (2 * conn + 4 * pi / (3 * arb(3).sqrt())) / norm
    diam_g = c_up.sqrt() * 2 * conn / norm
    gap_root = pi / diam_g - 3 * kappa * diam_g / (4 * pi)
    rows = [
        ('sup abs(f), f = eta - 1', up(sup_f)),
        ('sup abs(1 - 1/eta) (the f of the draft)', up(sup_f / (1 - sup_f))),
        ('sup abs(grad f)_G', up(grad)),
        ('C: omega <= C omega_0', up(c_up)),
        ('mu: omega >= mu omega_0', down(mu_min)),
        ('kappa: Ric >= -3 kappa g', up(kappa)),
        ('lambda: lambda I <= G_a', down(lam)),
        ('Lambda: G_a <= Lambda I', up(big)),
        ('K~ (K in the chart of F + 2 c~1)', up(kt)),
        ('K in the chart of F', up(k_t)),
        ('c~1', up(c1)),
        ('c~2', up(c2)),
        ('[f o psi_a^-1]_{1,0;V_a} <= sqrt(Lambda) sup abs(grad f)_G', up(big.sqrt() * grad)),
        ('diam(M, omega) <= sqrt(C) diam(M, omega_0)', up(diam_m)),
        ('diam(M/G, omega) <= sqrt(C) diam(M/G, omega_0)', up(diam_g)),
        ('lambda_1,G >= (pi/D - 3 kappa D / (4 pi))^2, D = diam(M/G)', down(gap_root * gap_root)),
        ('FS: lam0', float(fs['lam0_bound'])), ('FS: Lam0', float(fs['Lam0_bound'])),
        ('c~1_0 (the FS form; nan if not computed)', math.nan if e2 is None else e2),
        ('c~2_0 (the FS form; nan if not computed)', math.nan if e3 is None else e3),
        ('sup abs(D^2 F_a), real (nan if not computed)', math.nan if routes['d2'] is None else routes['d2']),
        ('sup abs(D^3 F_a), real (nan if not computed)', math.nan if routes['d3'] is None else routes['d3']),
    ]
    print(f'{len(rec)} records, target {target:.1e}, code {commit}', flush=True)
    for name, v in rows:
        print(f'{name:62s} {v:.6e}', flush=True)
    if args.out:
        np.savez(args.out, names=np.array([r[0] for r in rows]),
                 values=np.array([r[1] for r in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    for name in ('fs', 'transitions'):
        p = sub.add_parser(name)
        p.add_argument('--procs', type=int, default=8)
        p.add_argument('--tol', type=float, default=0.05)
        p.add_argument('--min-depth', type=int, default=9,
                       help='the smallest side of a box is R / 2^min_depth')
        p.add_argument('--out', required=True)
        if name == 'transitions':
            p.add_argument('--charts', nargs='*', default=None,
                           help='only these charts, as digit pairs ij (default: all 11)')
            p.add_argument('--floor-from', nargs='*', default=None,
                           help='files of this stage whose best point values also enter the floors')
            p.add_argument('--kinds', nargs='+', default=['c1', 'c2'], choices=['c1', 'c2', 'd2', 'd3'],
                           help='c1 c2: the transitions normalized by FS at the point; d2 d3: the real '
                                'norms of D^2 F and D^3 F, for c~1 <= Lambda^(1/2) lambda^-1 sup abs(D^2 F)')
    c = sub.add_parser('combine')
    c.add_argument('--runs', nargs='+', required=True)
    c.add_argument('--fs', required=True)
    c.add_argument('--transitions', nargs='+', required=True,
                   help='the files of the stage transitions; together they must hold all 11 charts')
    c.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.stage == 'combine':
        return combine(args)
    if Path(args.out).exists():
        raise SystemExit(f'{args.out} exists; choose a new file')
    if args.stage == 'fs':
        return stage(args, [('fs', 0), ('fs', 1)], 'fs')
    charts = CHARTS if not args.charts else [tuple(int(x) for x in c) for c in args.charts]
    return stage(args, [(kind, ch) for ch in charts for kind in args.kinds], 'transitions')


if __name__ == '__main__':
    sys.exit(main())
