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
* transitions: for each chart a != (3, 2), the sups on D(3, 2) and D(a) of the real norms of
  D^2 F_a and D^3 F_a, with F_a the coordinates of a as functions of t.
* combine: the constants of the draft from the records of a run (all parts) and the results
  of fs and transitions.

The bound of a box is the better of two arb enclosures over the box: the ball value, and the
centered form q(c) + grad q(box) . (x - c), with the partials from dual numbers. A
branch-and-bound halves a box in each coordinate until its bound is within --tol of the best
value at a point of the domain, or its side is at most R / 2^--min-depth. The result is the
largest (or smallest) bound of a box.

Usage, on CPU cores:
    uv run --extra gpu --locked python -u scripts/cert_atlas.py fs --procs 48 --out A.npz
    uv run --extra gpu --locked python -u scripts/cert_atlas.py transitions --procs 48 --out T.npz
    uv run --extra gpu --locked python -u scripts/cert_atlas.py combine --runs P0.npz P1.npz \\
        --fs A.npz --transitions T.npz
"""
import argparse
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
# The Taylor coefficients of a jet of degree 3 in (t_0, t_1), in this order.
MONS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
IDX = {m: k for k, m in enumerate(MONS)}
PAIRS = [(a, b, IDX[(ma[0] + mb[0], ma[1] + mb[1])]) for a, ma in enumerate(MONS)
         for b, mb in enumerate(MONS) if sum(ma) + sum(mb) <= 3]


class Jet:
    """A holomorphic function of (t_0, t_1) to degree 3 about a point: the Taylor coefficients
    of MONS as acb balls (or Duals)."""

    def __init__(self, c):
        self.c = c

    @staticmethod
    def const(z):
        return Jet([z if isinstance(z, Dual) else acb(z)] + [acb(0)] * 9)

    @staticmethod
    def var(k, z):
        c = [z if isinstance(z, Dual) else acb(z)] + [acb(0)] * 9
        c[1 + k] = acb(1)
        return Jet(c)

    def __add__(self, o):
        return Jet([a + b for a, b in zip(self.c, o.c)])

    def __sub__(self, o):
        return Jet([a - b for a, b in zip(self.c, o.c)])

    def __mul__(self, o):
        if not isinstance(o, Jet):
            return Jet([a * o for a in self.c])
        out = [acb(0)] * 10
        for a, b, k in PAIRS:
            out[k] += self.c[a] * o.c[b]
        return Jet(out)

    def rel(self):
        """y = x / x(0) - 1, the jet with no constant term."""
        x0 = self.c[0]
        return Jet([acb(0)] + [v / x0 for v in self.c[1:]])


def series(x, value, coeffs):
    """value times the sum of coeffs[k] y^k with y = x / x(0) - 1, to degree 3."""
    y = x.rel()
    y2 = y * y
    s = Jet.const(coeffs[0]) + y * coeffs[1] + y2 * coeffs[2] + y2 * y * coeffs[3]
    return s * value


def inv(x):
    return series(x, 1 / x.c[0], [acb(1), acb(-1), acb(1), acb(-1)])


class Dual:
    """A value and its partial derivatives in the real coordinates (Re t_0, Im t_0, Re t_1,
    Im t_1), as acb balls. With the balls of a box as the input, each partial encloses its
    values on the whole box. d is None once a partial has no finite enclosure."""
    __slots__ = ('v', 'd')

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def lift(x):
        return x if isinstance(x, Dual) else Dual(acb(x), [acb(0)] * 4)

    def _join(self, o, f):
        return None if self.d is None or o.d is None else [f(p, q) for p, q in zip(self.d, o.d)]

    def __add__(self, o):
        o = Dual.lift(o)
        return Dual(self.v + o.v, self._join(o, lambda p, q: p + q))

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, None if self.d is None else [-p for p in self.d])

    def __sub__(self, o):
        return self + (-Dual.lift(o))

    def __rsub__(self, o):
        return Dual.lift(o) + (-self)

    def __mul__(self, o):
        o = Dual.lift(o)
        return Dual(self.v * o.v, self._join(o, lambda p, q: self.v * q + o.v * p))

    __rmul__ = __mul__

    def inv(self):
        r = 1 / self.v
        return Dual(r, None if self.d is None else [-r * r * p for p in self.d])

    def __truediv__(self, o):
        return self * Dual.lift(o).inv()

    def __rtruediv__(self, o):
        return Dual.lift(o) * self.inv()

    def conjugate(self):
        return Dual(self.v.conjugate(), None if self.d is None else [p.conjugate() for p in self.d])

    @property
    def real(self):
        return Dual(acb(self.v.real), None if self.d is None else [acb(p.real) for p in self.d])


def root4(v):
    """A fourth root of the ball v, continuous on it: the principal root of v, or
    e^(i pi/4) times the principal root of -v when v is nearer the negative real axis."""
    if isinstance(v, Dual):
        r = root4(v.v)
        return Dual(r, None if v.d is None else [r / (4 * v.v) * p for p in v.d])
    if float(v.real.mid()) >= 0:
        return v.root(4)
    return acb(arb(1) / 4).exp_pi_i() * (-v).root(4)


def root4_jet(v):
    """A fourth root of the jet v, with the branch of root4 at its constant term."""
    return series(v, root4(v.c[0]), [acb(1), acb(1) / 4, acb(-3) / 32, acb(7) / 128])


# ---------------------------------------------------------------- the boxes


def sq(x):
    """x * x: python-flint gives NaN for x ** 2 of a ball that holds 0."""
    return x * x


def abs2(x):
    """abs(x)^2 as the real part of x conj(x), for an acb or a Dual."""
    return (x * x.conjugate()).real


def root_up(x):
    """sqrt of the upper end of the ball x, for a true value not below 0: its upper end is an
    upper bound of the true root (the lower end of x can be below 0 for a ball that holds 0)."""
    return arb(up(x)).sqrt()


def sqrt_up(x):
    """A root of x >= 0 whose upper end bounds the true root; for a Dual, with the partials of
    the root where x proves to be above 0 (else no partials)."""
    if not isinstance(x, Dual):
        return root_up(x)
    v = x.v.real
    if down(v) > 0:
        r = v.sqrt()
        return Dual(acb(r), None if x.d is None else [p / (2 * r) for p in x.d])
    return Dual(acb(root_up(v)), None)


def wof(t0, t1):
    """w with w^4 = -(1 + t_0^4 + t_1^4), for acb balls or Duals."""
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


def fs_eigs(t0, t1, w):
    """The two eigenvalues of the FS metric of volume 1 in the chart b, for acb balls or Duals:
    the lower end of the first and the upper end of the second are bounds (sqrt_up)."""
    y = [t0, t1, w]
    r0, r1 = t0 / w, t1 / w
    jac = [[acb(1), acb(0)], [acb(0), acb(1)], [-r0 * r0 * r0, -r1 * r1 * r1]]
    s = 1 + abs2(t0) + abs2(t1) + abs2(w)
    h = [[(1 if m == q else 0) / s - y[m].conjugate() * y[q] / (s * s) for q in range(3)]
         for m in range(3)]
    g = [[sum((jac[m][i] * h[m][q] * jac[q][j].conjugate() for m in range(3) for q in range(3)),
              acb(0)) for j in range(2)] for i in range(2)]
    a, d = g[0][0].real, g[1][1].real
    mid = (a + d) / 2
    rad = sqrt_up(sq((a - d) / 2) + abs2(g[0][1]))
    scale = arb.pi() * arb(2).sqrt()
    return (mid - rad) / scale, (mid + rad) / scale


def chart_jets(t0, t1, chart):
    """The two coordinates of chart a = (i, j) as jets about the point t of the chart b."""
    z = [Jet.var(0, t0), Jet.var(1, t1), None, Jet.const(1)]
    u = Jet.const(1) + z[0] * z[0] * z[0] * z[0] + z[1] * z[1] * z[1] * z[1]
    z[2] = root4_jet(Jet.const(0) - u)
    i, j = chart
    k, l = (x for x in range(4) if x not in chart)
    zi = inv(z[i])
    return [z[k] * zi, z[l] * zi]


def deriv_norms(comps):
    """The real norms of D^2 F and D^3 F (2 and 2 sqrt 2 times the complex norms), with upper
    ends that bound them (sqrt_up)."""
    a2 = lambda f, m: abs2(f.c[IDX[m]])
    d2 = sum((4 * a2(f, (2, 0)) + 2 * a2(f, (1, 1)) + 4 * a2(f, (0, 2)) for f in comps), arb(0))
    d3 = sum((36 * a2(f, (3, 0)) + 12 * a2(f, (2, 1)) + 12 * a2(f, (1, 2)) + 36 * a2(f, (0, 3))
              for f in comps), arb(0))
    return 2 * sqrt_up(d2), 2 * sqrt_up(2 * d3)


def value(kind, t0, t1):
    """The value of the quantity `kind` at t (acb balls, or Duals): ('fs', 0 or 1) or ('d2' or
    'd3', chart)."""
    if kind[0] == 'fs':
        return fs_eigs(t0, t1, wof(t0, t1))[kind[1]]
    d2, d3 = deriv_norms(chart_jets(t0, t1, kind[1]))
    return d2 if kind[0] == 'd2' else d3


def domains(kind):
    return [CHART_B] if kind[0] == 'fs' else [CHART_B, kind[1]]


def is_min(kind):
    return kind == ('fs', 0)


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


# The partials of t_0 and t_1 in (Re t_0, Im t_0, Re t_1, Im t_1).
D_T0 = [acb(1), acb(0, 1), acb(0), acb(0)]
D_T1 = [acb(0), acb(0), acb(1), acb(0, 1)]


def step(job):
    """None if the box cannot meet the domain, else (its bound, the value at its center if the
    center is in the domain, else None). The bound is the better of the ball value on the box
    and the centered form q(c) + sum of the partials on the box times [-s/2, s/2]; a NaN bound
    counts as no bound."""
    kind, lo, s = job
    t0, t1 = ball(lo, s)
    if not all(may_meet(moduli(t0, t1), ch) for ch in domains(kind)):
        return None
    small = is_min(kind)
    pick = down if small else up
    q = value(kind, Dual(t0, D_T0), Dual(t1, D_T1))
    b = pick(q.v.real)
    c = [lo[k] + s / 2 for k in range(4)]
    c0, c1 = acb(c[0], c[1]), acb(c[2], c[3])
    qc = value(kind, c0, c1)
    if q.d is not None:
        half = arb(0, s / 2)
        bc = pick(qc + sum((p.real * half for p in q.d), arb(0)))
        if not math.isnan(bc):
            b = bc if math.isnan(b) else (max(b, bc) if small else min(b, bc))
    if math.isnan(b):
        b = -math.inf if small else math.inf
    point = None
    if c[1] <= c[0] and all(inside(moduli(c0, c1), ch) for ch in domains(kind)):
        point = up(qc) if small else down(qc)
    return b, point


def run_kind(kind, pool, tol, min_side):
    """The branch-and-bound in rounds of one side, so that each round knows the best point
    value of all boxes so far."""
    small = is_min(kind)
    best = math.inf if small else 0.0
    top = math.inf if small else -math.inf
    level, n, at_min = first_boxes(), 0, 0
    while level:
        res = pool.map(step, [(kind, lo, s) for lo, s in level], chunksize=16)
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
        level = nxt
    return dict(bound=top, best=best, boxes=n, at_min=at_min)


def stage(args, kinds, name):
    t0 = time.perf_counter()
    out = {}
    with multiprocessing.Pool(args.procs) as pool:
        for kind in kinds:
            t1 = time.perf_counter()
            r = run_kind(kind, pool, args.tol, R / 2 ** args.min_depth)
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
    fs, tr = np.load(args.fs), np.load(args.transitions)
    x = lambda v: arb(float(v))
    c_up, mu_min = x(top('mu_up')), x(float(np.min(rec[:, col['mu_low']])))
    lam = mu_min * x(fs['lam0_bound'])
    big = c_up * x(fs['Lam0_bound'])
    d2 = max(float(tr[f'd2_{i}{j}_bound']) for i, j in CHARTS)
    d3 = max(float(tr[f'd3_{i}{j}_bound']) for i, j in CHARTS)
    c1 = big.sqrt() / lam * x(d2)
    c2 = big.sqrt() / (lam * lam.sqrt()) * x(d3)
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
        ('largest real norm of D^2 F_a', d2), ('largest real norm of D^3 F_a', d3),
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
        p.add_argument('--tol', type=float, default=0.02)
        p.add_argument('--min-depth', type=int, default=9,
                       help='the smallest side of a box is R / 2^min_depth')
        p.add_argument('--out', required=True)
    c = sub.add_parser('combine')
    c.add_argument('--runs', nargs='+', required=True)
    c.add_argument('--fs', required=True)
    c.add_argument('--transitions', required=True)
    c.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.stage == 'combine':
        return combine(args)
    if Path(args.out).exists():
        raise SystemExit(f'{args.out} exists; choose a new file')
    if args.stage == 'fs':
        return stage(args, [('fs', 0), ('fs', 1)], 'fs')
    return stage(args, [(kind, ch) for ch in CHARTS for kind in ('d2', 'd3')], 'transitions')


if __name__ == '__main__':
    sys.exit(main())
