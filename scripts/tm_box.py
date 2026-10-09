"""A Taylor model of f of a one-stage level fit on boxes of a chart, to measure the cost.

A box is the polydisc abs(t_i - c_i) <= delta in the free coordinates t of a chart,
with w = conj(t) taken as independent variables. Each function is then a series in
the four variables s = t - c and r = w - conj(c). A Taylor model is its polynomial of
total degree <= N and a bound of the rest on the polydisc. The bounds are carried for
several radii delta at once, because the polynomials do not depend on delta.

The potential is Phi = log p0 + sum_L N_L p0^(-L/k0). p0 and each N_L are Hermitian
forms of sections. In a chart with z_j = 1 a section is z_m^e times an exact
polynomial in t, with z_m^4 = -(1 + t_0^4 + t_1^4). So a form is the sum over
e = 0..3 of z_m(t)^e conj(z_m)(w)^e Q_e(s, r), with Q_e an exact polynomial, and
only the powers of z_m need a remainder. The form and its first derivatives in t and
w come from matrix products of polynomial coefficients. g = ddbar Phi, det g, v and f
then follow in Taylor-model arithmetic, the powers of p0 from the binomial series.

The output is, for each box center, order and radius, the bound of abs(f - f(c)) on
the box and the remainder in it, and the time and the multiply-accumulate count of
each stage. The center value is checked against the dd v of the export.

Inputs: the export of `level_oracle.py` (the base, the points) and the output of
`level_matrices.py` (the forms of the levels). Usage, on CPU cores:
    JAX_PLATFORMS=cpu uv run --extra gpu --with python-flint python -u \\
        scripts/tm_box.py --data runs_tm/r20_data.npz --forms runs_tm/r20_forms.npz \\
        --orders 12 16 20 --deltas 0.005 0.01 0.02 0.04 --procs 12
"""
import argparse
import itertools
import math
import multiprocessing
import time

import numpy as np
import flint
from flint import acb, acb_mat, arb

PERMS = list(itertools.permutations(range(4)))
COUNT = {'mac': 0.0}


def ub(x):
    """An upper bound of abs(x) as a float."""
    a = abs(x)
    return (float(a.mid()) + float(a.rad())) * (1 + 1e-13)


def compositions(total, k):
    """All k-tuples of non-negative integers with this sum."""
    if k == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest


class Space:
    """The monomials of total degree <= n in nvars variables, and the product table."""

    def __init__(self, nvars, n):
        self.nvars, self.n = nvars, n
        self.mons = [e for d in range(n + 1) for e in compositions(d, nvars)]
        self.index = {e: i for i, e in enumerate(self.mons)}
        self.deg = np.array([sum(e) for e in self.mons])
        self.size = len(self.mons)
        self._table = None

    def table(self):
        """The pairs (i, j) with deg i + deg j <= n, sorted by the index of the product."""
        if self._table is None:
            base = self.n + 1
            codes = np.array([sum(c * base ** v for v, c in enumerate(e)) for e in self.mons])
            lookup = np.full(base ** self.nvars, -1, dtype=np.int64)
            lookup[codes] = np.arange(self.size)
            count = np.searchsorted(self.deg, np.arange(self.n + 1), side='right')
            ii, jj, kk = [], [], []
            for i in range(self.size):
                js = np.arange(count[self.n - self.deg[i]])
                ii.append(np.full(js.size, i))
                jj.append(js)
                kk.append(lookup[codes[i] + codes[js]])
            ii, jj, kk = np.concatenate(ii), np.concatenate(jj), np.concatenate(kk)
            order = np.argsort(kk, kind='stable')
            ii, jj, kk = ii[order], jj[order], kk[order]
            starts = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
            self._table = (ii, jj, starts)
        return self._table


def segment_sum(prods, starts):
    """The sums of prods over the segments that start at `starts`."""
    try:
        return np.add.reduceat(prods, starts)
    except TypeError:
        ends = np.r_[starts[1:], prods.size]
        return np.array([sum(prods[a + 1:b], prods[a]) for a, b in zip(starts, ends)],
                        dtype=object)


def tail(dist, n, deltas):
    """sum over D > n of dist[D] delta^D, for each delta."""
    d = np.arange(len(dist))
    keep = d > n
    return np.array([float(np.sum(dist[keep] * dl ** d[keep])) for dl in deltas])


def poly_bound(g0, deltas):
    d = np.arange(len(g0))
    return np.array([float(np.sum(g0 * dl ** d)) for dl in deltas])


class TM:
    """A Taylor model: coefficients (object array of acb) and remainders, one per delta."""

    def __init__(self, sp, c, rem, deltas):
        self.sp, self.c, self.rem, self.deltas = sp, c, np.asarray(rem, float), deltas

    @classmethod
    def const(cls, sp, value, deltas):
        c = np.array([acb(0)] * sp.size, dtype=object)
        c[0] = acb(value)
        return cls(sp, c, np.zeros(len(deltas)), deltas)

    def graded(self):
        u = np.array([ub(x) for x in self.c])
        return np.bincount(self.sp.deg, weights=u, minlength=self.sp.n + 1)

    def poly_bound(self):
        return poly_bound(self.graded(), self.deltas)

    def bound(self):
        return self.poly_bound() + self.rem

    def at(self, point):
        """The polynomial at one point (a list of nvars acb), with no remainder."""
        total = acb(0)
        for coef, e in zip(self.c, self.sp.mons):
            term = coef
            for x, k in zip(point, e):
                if k:
                    term *= x ** k
            total += term
        return total

    def __add__(self, o):
        return TM(self.sp, self.c + o.c, self.rem + o.rem, self.deltas)

    def __sub__(self, o):
        return TM(self.sp, self.c - o.c, self.rem + o.rem, self.deltas)

    def scale(self, k):
        k = acb(k)
        return TM(self.sp, self.c * k, self.rem * ub(k), self.deltas)

    def add_const(self, k):
        c = self.c.copy()
        c[0] = c[0] + acb(k)
        return TM(self.sp, c, self.rem.copy(), self.deltas)

    def __mul__(self, o):
        ii, jj, starts = self.sp.table()
        c = segment_sum(self.c[ii] * o.c[jj], starts)
        COUNT['mac'] += ii.size
        ga, gb = self.graded(), o.graded()
        rem = (tail(np.convolve(ga, gb), self.sp.n, self.deltas)
               + poly_bound(ga, self.deltas) * o.rem + self.rem * poly_bound(gb, self.deltas)
               + self.rem * o.rem)
        return TM(self.sp, c, rem, self.deltas)

    def binomial(self, a, prefactor):
        """prefactor (1 + x)^a with x = self / c0 - 1, from the binomial series of order n."""
        c0 = self.c[0]
        x = self.add_const(-c0).scale(1 / c0)
        eta = x.bound()
        k = self.sp.n
        coef = [arb(1)]
        for i in range(k + 1):
            coef.append(coef[-1] * (a - i) / (i + 1))
        s = TM.const(self.sp, coef[k], self.deltas)
        for i in range(k - 1, -1, -1):
            s = (s * x).add_const(coef[i])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            lag = ub(coef[k + 1]) * eta ** (k + 1) * (1 - eta) ** (float(a) - k - 1)
        lag = np.where(eta < 1, lag, np.inf)
        out = s.scale(prefactor)
        out.rem = out.rem + lag * ub(prefactor)
        out.eta = eta
        return out


def tm_power(x, n):
    """x^n for an integer n >= 1, by repeated squares."""
    out, base = None, x
    while n:
        if n & 1:
            out = base if out is None else out * base
        n >>= 1
        if n:
            base = base * base
    return out


def radius_model(sp4, sp_n, chart, z1, w1, deltas):
    """abs(z)^2 = 1 + t_0 w_0 + t_1 w_1 + z_m w_m in the chart, as a Taylor model."""
    c = np.array([acb(0)] * sp4.size, dtype=object)
    c[0] = acb(1)
    for i in range(2):
        ci = chart.c[i]
        e_s = [0, 0, 0, 0]
        e_r = [0, 0, 0, 0]
        e_s[i], e_r[2 + i] = 1, 1
        both = [0, 0, 0, 0]
        both[i], both[2 + i] = 1, 1
        c[0] += ci * ci.conjugate()
        c[sp4.index[tuple(e_s)]] += ci.conjugate()
        c[sp4.index[tuple(e_r)]] += ci
        c[sp4.index[tuple(both)]] += acb(1)
    return TM(sp4, c, np.zeros(len(deltas)), deltas) + tensor(sp4, sp_n, z1, w1, deltas)


# ---------------------------------------------------------------------------------------
# Exact polynomials in the two free coordinates, and the forms.

def conj_mat(m):
    """The entrywise complex conjugate of an acb_mat."""
    return acb_mat(m.nrows(), m.ncols(),
                   [m[i, k].conjugate() for i in range(m.nrows()) for k in range(m.ncols())])


def poly_mul(sp2, a, b):
    """The exact product of two polynomials of sp2 whose degrees add up to <= sp2.n."""
    ii, jj, starts = sp2.table()
    COUNT['mac'] += ii.size
    return segment_sum(a[ii] * b[jj], starts)


def coordinate_polys(sp2, c):
    """t_0, t_1 as polynomials c_i + s_i of sp2."""
    out = []
    for i in range(2):
        p = np.array([acb(0)] * sp2.size, dtype=object)
        p[0] = c[i]
        e = [0, 0]
        e[i] = 1
        p[sp2.index[tuple(e)]] = acb(1)
        out.append(p)
    return out


class Chart:
    """The chart z_j = 1 at a point: the free coordinates, z_m, and the exact u polynomials."""

    def __init__(self, z, patch, sp2):
        m, f0, f1 = patch
        j = ({0, 1, 2, 3} - {m, f0, f1}).pop()
        self.j, self.m, self.free = j, m, (f0, f1)
        self.c = [z[f0] / z[j], z[f1] / z[j]]
        cu = -(1 + self.c[0] ** 4 + self.c[1] ** 4)
        approx = z[m] / z[j]
        root = cu.root(4)
        units = [acb(1), acb(0, 1), acb(-1), acb(0, -1)]
        self.zm = min((root * w for w in units), key=lambda v: ub(v - approx))
        self.cu = cu
        self.sp2 = sp2
        t = coordinate_polys(sp2, self.c)
        one = np.array([acb(0)] * sp2.size, dtype=object)
        one[0] = acb(1)
        t4 = []
        for p in t:
            q = poly_mul(sp2, p, p)
            t4.append(poly_mul(sp2, q, q))
        um = -(one + t4[0] + t4[1])
        self.t = t
        self.u = {j: one, f0: t4[0], f1: t4[1], m: um}
        self.powers = {}

    def space(self, size):
        """The 2-variable Space whose monomial count is `size`."""
        deg = 0
        while (deg + 1) * (deg + 2) // 2 < size:
            deg += 1
        key = ('space', deg)
        if key not in self.powers:
            self.powers[key] = Space(2, deg)
        return self.powers[key]

    def u_power(self, coord, n):
        key = (coord, n)
        if key not in self.powers:
            if n == 0:
                self.powers[key] = self.u[self.j]
            else:
                self.powers[key] = poly_mul(self.sp2, self.u_power(coord, n - 1), self.u[coord])
        return self.powers[key]

    def t_monomial(self, r):
        """t^(r_free) for a residue r, as an exact polynomial."""
        key = ('t', r[self.free[0]], r[self.free[1]])
        if key not in self.powers:
            p = self.u[self.j]
            for i, f in enumerate(self.free):
                for _ in range(r[f]):
                    p = poly_mul(self.sp2, p, self.t[i])
            self.powers[key] = p
        return self.powers[key]

    def monomial(self, exps, perm):
        """u(sigma z)^E = prod_a u_(sigma(a))^(E_a) as an exact polynomial."""
        key = ('m', tuple(exps), perm)
        if key not in self.powers:
            p = self.u[self.j]
            for a in range(4):
                if exps[a]:
                    p = poly_mul(self.sp2, p, self.u_power(perm[a], exps[a]))
            self.powers[key] = p
        return self.powers[key]


def sections_matrix(chart, exps, frame_t, perm, r, n_p):
    """The coefficients (d, n_p) of t^(r'_free) frame^T u(sigma z)^E, and e = r'_m.

    r' is the residue of the block that sigma maps the orbit's block to."""
    rp = [0] * 4
    for a in range(4):
        rp[perm[a]] = r[a]
    key = ('mono', tuple(map(tuple, exps)), perm, n_p)
    if key not in chart.powers:
        rows = [chart.monomial(e, perm) for e in exps]
        chart.powers[key] = acb_mat(len(rows), n_p, [x for row in rows for x in row[:n_p]])
    mono = chart.powers[key]
    tkey = ('tmat', rp[chart.free[0]], rp[chart.free[1]], n_p)
    if tkey not in chart.powers:
        sp_p = chart.space(n_p)
        chart.powers[tkey] = mult_matrix(sp_p, sp_p, chart.t_monomial(rp)[:n_p]).transpose()
    psi = frame_t * mono
    COUNT['mac'] += frame_t.nrows() * frame_t.ncols() * n_p + frame_t.nrows() * n_p * n_p
    return psi * chart.powers[tkey], rp[chart.m]


def form_polys(chart, terms, n_p):
    """Q_e = sum over terms of V^T C conj(V), the exact bidegree polynomial of each e."""
    q = [None] * 4
    for v, e, cmat in terms:
        vt = v.transpose()
        part = vt * cmat * conj_mat(v)
        d = cmat.nrows()
        COUNT['mac'] += n_p * d * d + n_p * n_p * d
        q[e] = part if q[e] is None else q[e] + part
    return [x if x is not None else acb_mat(n_p, n_p) for x in q]


# ---------------------------------------------------------------------------------------
# From the bidegree forms to Taylor models in (s, r).

def zm_power(chart, sp, e, deltas):
    """z_m(t)^e as a Taylor model in s: z_m(c)^e (u_m / u_m(c))^(e/4)."""
    um = TM(sp, np.array(list(chart.u[chart.m][:sp.size]), dtype=object),
            np.zeros(len(deltas)), deltas)
    if e == 0:
        return TM.const(sp, 1, deltas)
    return um.binomial(arb(e) / 4, chart.zm ** e)


def mult_matrix(sp_out, sp_in, coeffs, weight=None):
    """M with (M P)[g] = sum over a of coeffs[g - a] P[a] (times weight[a]), as acb_mat."""
    rows = []
    for g in sp_out.mons:
        row = []
        for a in sp_in.mons:
            diff = (g[0] - a[0], g[1] - a[1])
            if diff[0] < 0 or diff[1] < 0 or diff not in sp_out.index:
                row.append(acb(0))
                continue
            val = coeffs[sp_out.index[diff]]
            row.append(val if weight is None else val * weight(a))
        rows.append(row)
    return acb_mat(sp_out.size, sp_in.size, [x for row in rows for x in row])


def shift_deriv(sp_out, sp_in, coeffs, axis):
    """M D_axis: (M D P)[g] = sum over a of coeffs[g - a] (a_axis + 1) P[a + e_axis]."""
    rows = []
    for g in sp_out.mons:
        row = []
        for a in sp_in.mons:
            if a[axis] == 0:
                row.append(acb(0))
                continue
            b = list(a)
            b[axis] -= 1
            diff = (g[0] - b[0], g[1] - b[1])
            if diff[0] < 0 or diff[1] < 0 or diff not in sp_out.index:
                row.append(acb(0))
                continue
            row.append(coeffs[sp_out.index[diff]] * a[axis])
        rows.append(row)
    return acb_mat(sp_out.size, sp_in.size, [x for row in rows for x in row])


def bigraded(mat, sp_a, sp_b):
    """U[m, n] = sum of upper bounds of abs(mat[a, b]) over deg a = m, deg b = n."""
    u = np.zeros((sp_a.n + 1, sp_b.n + 1))
    for i in range(mat.nrows()):
        for k in range(mat.ncols()):
            u[sp_a.deg[i], sp_b.deg[k]] += ub(mat[i, k])
    return u


def deriv_bigraded(u, axis_side, sp):
    """A bound of the bigraded norms of a derivative of the form, from those of the form."""
    out = np.zeros_like(u)
    if axis_side == 0:
        for m in range(1, u.shape[0]):
            out[m - 1] += u[m] * m
    else:
        for n in range(1, u.shape[1]):
            out[:, n - 1] += u[:, n] * n
    return out


def form_models(chart, sp4, sp_n, sp_p, q, z, w, dz, dw, deltas):
    """The Taylor models of F, F_a, F_bbar and F_abbar of one form, a, b in {0, 1}.

    q: the four Q_e (n_p, n_p); z[e], w[e]: TM of z_m^e and its conjugate in sp_n;
    dz[e][a], dw[e][b]: their derivatives. Index 0 of each side is the plain factor."""
    n = sp4.n
    out = {}
    lmat = {}
    rmat = {}
    for e in range(4):
        key = ('ops', e, sp_p.n)
        if key not in chart.powers:
            left = [mult_matrix(sp_n, sp_p, z[e].c)]
            right = [mult_matrix(sp_n, sp_p, w[e].c)]
            for a in range(2):
                left.append(mult_matrix(sp_n, sp_p, dz[e][a].c)
                            + shift_deriv(sp_n, sp_p, z[e].c, a))
                right.append(mult_matrix(sp_n, sp_p, dw[e][a].c)
                             + shift_deriv(sp_n, sp_p, w[e].c, a))
            chart.powers[key] = (left, right)
        lmat[e], rmat[e] = chart.powers[key]
    mats = {}
    for e in range(4):
        xs = [q[e] * rmat[e][j].transpose() for j in range(3)]
        COUNT['mac'] += 3 * sp_p.size * sp_p.size * sp_n.size
        for i in range(3):
            for j in range(3):
                part = lmat[e][i] * xs[j]
                COUNT['mac'] += sp_n.size * sp_p.size * sp_n.size
                mats[i, j] = part if (i, j) not in mats else mats[i, j] + part
    # The remainders: for each piece X Q' Y, the tail beyond degree n and the remainders
    # of X and Y. A derivative acts on X or on Q' (the left index) and on Y or Q' (right).
    rems = {key: np.zeros(len(deltas)) for key in mats}
    for e in range(4):
        u = bigraded(q[e], sp_p, sp_p)
        for i in range(3):
            for j in range(3):
                lefts = [(z[e], 0)] if i == 0 else [(dz[e][i - 1], 0), (z[e], 1 + (i - 1))]
                rights = [(w[e], 0)] if j == 0 else [(dw[e][j - 1], 0), (w[e], 1 + (j - 1))]
                for xm, xq in lefts:
                    for ym, yq in rights:
                        uq = u
                        if xq:
                            uq = deriv_bigraded(uq, 0, sp_p)
                        if yq:
                            uq = deriv_bigraded(uq, 1, sp_p)
                        qd = np.zeros(uq.shape[0] + uq.shape[1] - 1)
                        for mm in range(uq.shape[0]):
                            qd[mm:mm + uq.shape[1]] += uq[mm]
                        gx, gy = xm.graded(), ym.graded()
                        dist = np.convolve(np.convolve(gx, qd), gy)
                        bq = poly_bound(qd, deltas)
                        bx, by = poly_bound(gx, deltas), poly_bound(gy, deltas)
                        rems[i, j] += (tail(dist, n, deltas) + xm.rem * by * bq
                                       + bx * ym.rem * bq + xm.rem * ym.rem * bq)
    for (i, j), mat in mats.items():
        c = np.array([acb(0)] * sp4.size, dtype=object)
        for k, (a0, a1, b0, b1) in enumerate(sp4.mons):
            c[k] = mat[sp_n.index[(a0, a1)], sp_n.index[(b0, b1)]]
        out[i, j] = TM(sp4, c, rems[i, j], deltas)
    return out


def tensor(sp4, sp_n, x, y, deltas):
    """x(s) y(r) as a Taylor model in the four variables."""
    c = np.array([x.c[sp_n.index[(a0, a1)]] * y.c[sp_n.index[(b0, b1)]]
                  for a0, a1, b0, b1 in sp4.mons], dtype=object)
    gx, gy = x.graded(), y.graded()
    rem = (tail(np.convolve(gx, gy), sp4.n, deltas) + x.rem * poly_bound(gy, deltas)
           + poly_bound(gx, deltas) * y.rem + x.rem * y.rem)
    return TM(sp4, c, rem, deltas)


# ---------------------------------------------------------------------------------------

D = {}
FACTOR = {'on': False}


def load(args):
    flint.ctx.prec = args.prec
    data = np.load(args.data)
    forms = np.load(args.forms)
    D['k0'] = int(data['k0'])
    D['z'] = [[acb(complex(a)) + acb(complex(b)) for a, b in zip(data['z_hi'][k], data['z_lo'][k])]
              for k in range(data['z_hi'].shape[0])]
    D['patch'] = data['patch'].tolist()
    D['ratio'] = data['ratio'].tolist()
    D['v1'] = [acb(float(a)) + acb(float(b))
               for a, b in zip(data['v1_hi'][:, args.col], data['v1_lo'][:, args.col])]
    base = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        d = frame.shape[1]
        frame_t = acb_mat(d, frame.shape[0], [acb(float(x)) for x in frame.T.ravel()])
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        for b, r in enumerate(data[f'g{i}_res'].tolist()):
            h = acb_mat(d, d, [acb(float(x)) + acb(float(y))
                               for x, y in zip(hhi[b].ravel(), hlo[b].ravel())])
            base.append((data[f'g{i}_exps'].tolist(), frame_t, r, h))
    D['base'] = base
    levels = []
    for k1 in forms['k1'].tolist():
        groups = []
        for gi in range(int(forms[f'L{k1}_n_groups'])):
            frame = forms[f'L{k1}_g{gi}_frame']
            groups.append((forms[f'L{k1}_g{gi}_exps'].tolist(),
                           acb_mat(frame.shape[1], frame.shape[0],
                                   [acb(float(x)) for x in frame.T.ravel()])))
        orbits = []
        for oi in range(int(forms[f'L{k1}_n_orbits'])):
            cs = forms[f'L{k1}_o{oi}_C']
            d = cs.shape[0]
            cmat = acb_mat(d, d, [acb(arb(str(x))) for x in cs.ravel()])
            orbits.append((int(forms[f'L{k1}_o{oi}_group']),
                           forms[f'L{k1}_o{oi}_r'].tolist(), cmat))
        levels.append((int(k1), groups, orbits))
    D['levels'] = levels


def _psi_direct(exps, frame_t, u):
    mono = []
    for row in exps:
        val = acb(1)
        for a in range(4):
            if row[a]:
                val *= u[a] ** row[a]
        mono.append(val)
    return frame_t * acb_mat(len(mono), 1, mono)


def _hermitian(v, h):
    d = v.nrows()
    vc = acb_mat(1, d, [v[i, 0].conjugate() for i in range(d)])
    return (vc * h * v)[0, 0]


def direct_p(z):
    """p0 at a point z, as the export holds it."""
    u = [c ** 4 for c in z]
    total = acb(0)
    for exps, frame_t, r, h in D['base']:
        w = acb(1)
        for a in range(4):
            if r[a]:
                w *= abs(z[a]) ** (2 * r[a])
        total += w * _hermitian(_psi_direct(exps, frame_t, u), h)
    return total


def direct_form(level, z):
    """N_L at a point z, the sum over S_4 of one block of each orbit."""
    k1, groups, orbits = level
    total = acb(0)
    for perm in PERMS:
        zp = [z[perm[a]] for a in range(4)]
        u = [c ** 4 for c in zp]
        for gi, r, cmat in orbits:
            exps, frame_t = groups[gi]
            w = acb(1)
            for a in range(4):
                if r[a]:
                    w *= abs(zp[a]) ** (2 * r[a])
            total += w * _hermitian(_psi_direct(exps, frame_t, u), cmat)
    return total


def moved_point(chart, s):
    """The point of X with z_j = 1 and t = c + s, on the branch of z_m of the center."""
    t = [chart.c[0] + s[0], chart.c[1] + s[1]]
    um = -(1 + t[0] ** 4 + t[1] ** 4)
    zm = chart.zm * (um / chart.cu) ** (arb(1) / 4)
    z = [None] * 4
    z[chart.j], z[chart.m] = acb(1), zm
    z[chart.free[0]], z[chart.free[1]] = t
    return z


def run_box(job):
    k, n, deltas = job
    deltas = np.asarray(deltas, float)
    COUNT['mac'] = 0.0
    times = {}
    t0 = time.perf_counter()
    k0 = D['k0']
    top = max(k0, max(lv[0] for lv in D['levels']))
    sp2 = Space(2, max(top, n))         # exact polynomials in s
    sp_n = Space(2, n)                  # one side of a Taylor model
    sp4 = Space(4, n)
    sp4.table()
    times['tables'] = time.perf_counter() - t0
    chart = Chart(D['z'][k], D['patch'][k], sp2)

    t1 = time.perf_counter()
    z, w, dz, dw = {}, {}, {}, {}
    for e in range(4):
        z[e] = zm_power(chart, sp_n, e, deltas)
        w[e] = TM(sp_n, np.array([x.conjugate() for x in z[e].c], dtype=object),
                  z[e].rem.copy(), deltas)
        dz[e], dw[e] = [], []
        for a in range(2):
            if e == 0:
                d0 = TM.const(sp_n, 0, deltas)
            else:
                # d_a z_m^e = -e t_a^3 z_m^(e-4)
                ta3 = np.array(list(poly_mul(sp2, poly_mul(sp2, chart.t[a], chart.t[a]),
                                             chart.t[a])[:sp_n.size]), dtype=object)
                d0 = TM(sp_n, ta3, np.zeros(len(deltas)), deltas) * zm_power(chart, sp_n, e - 4, deltas)
                d0 = d0.scale(-e)
            dz[e].append(d0)
            dw[e].append(TM(sp_n, np.array([x.conjugate() for x in d0.c], dtype=object),
                            d0.rem.copy(), deltas))
    times['z_m powers'] = time.perf_counter() - t1

    def models(deg, terms_fn):
        sp_p = Space(2, deg)
        terms = terms_fn(sp_p.size)
        q = form_polys(chart, terms, sp_p.size)
        return form_models(chart, sp4, sp_n, sp_p, q, z, w, dz, dw, deltas)

    t1 = time.perf_counter()

    def base_terms(n_p):
        return [(*sections_matrix(chart, exps, frame_t, (0, 1, 2, 3), r, n_p), h)
                for exps, frame_t, r, h in D['base']]
    fb = models(k0, base_terms)
    times['base form'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    fl = []
    for k1, groups, orbits in D['levels']:
        def level_terms(n_p, groups=groups, orbits=orbits):
            out = []
            for gi, r, cmat in orbits:
                exps, frame_t = groups[gi]
                for perm in PERMS:
                    v, e = sections_matrix(chart, exps, frame_t, perm, r, n_p)
                    out.append((v, e, cmat))
            return out
        fl.append((k1, models(k1, level_terms)))
    times['level forms'] = time.perf_counter() - t1
    mac_forms = COUNT['mac']

    t1 = time.perf_counter()
    COUNT['mac'] = 0.0
    p = fb[0, 0]
    alpha_u = arb(-1) / k0
    if FACTOR['on']:
        # p0 = abs(z)^(2 k0) p~, so that the binomial series act on functions that vary slowly.
        r2 = radius_model(sp4, sp_n, chart, z[1], w[1], deltas)
        ir2 = r2.binomial(arb(-1), r2.c[0] ** -1)
        ir2k = tm_power(ir2, k0)
        pt = p * ir2k
        ipt = pt.binomial(arb(-1), pt.c[0] ** -1)
        ip = ir2k * ipt
        u = ir2 * pt.binomial(alpha_u, pt.c[0] ** alpha_u)
        etas = {'abs(z)^2': ir2.eta, 'p~': ipt.eta}
    else:
        ip = p.binomial(arb(-1), p.c[0] ** -1)
        u = p.binomial(alpha_u, p.c[0] ** alpha_u)
        etas = {'p0': ip.eta}
    pa = [fb[1 + a, 0] * ip for a in range(2)]
    pb = [fb[0, 1 + b] * ip for b in range(2)]
    pab = {(a, b): fb[1 + a, 1 + b] * ip for a in range(2) for b in range(2)}
    pp = {(a, b): pa[a] * pb[b] for a in range(2) for b in range(2)}
    g = {key: pab[key] - pp[key] for key in pab}
    powers = {1: u}

    def upow(L):
        if L not in powers:
            half = upow(L // 2)
            sq = half * half
            powers[L] = sq if L % 2 == 0 else sq * u
        return powers[L]
    acc = None
    for k1, f in fl:
        al = arb(k1) / k0
        ul = upow(k1)
        parts = {'n': ul * f[0, 0]}
        for a in range(2):
            parts['a', a] = ul * f[1 + a, 0]
            parts['b', a] = ul * f[0, 1 + a]
        for a in range(2):
            for b in range(2):
                parts['ab', a, b] = ul * f[1 + a, 1 + b]
        scaled = {'A': {key: parts[('ab',) + key] for key in pab},
                  'Ba': [parts['a', a].scale(al) for a in range(2)],
                  'Bb': [parts['b', b].scale(al) for b in range(2)],
                  'C': parts['n'].scale(al), 'E': parts['n'].scale(al * (al + 1))}
        if acc is None:
            acc = scaled
        else:
            acc = {'A': {key: acc['A'][key] + scaled['A'][key] for key in pab},
                   'Ba': [acc['Ba'][a] + scaled['Ba'][a] for a in range(2)],
                   'Bb': [acc['Bb'][a] + scaled['Bb'][a] for a in range(2)],
                   'C': acc['C'] + scaled['C'], 'E': acc['E'] + scaled['E']}
    if acc is not None:
        for a in range(2):
            for b in range(2):
                g[a, b] = (g[a, b] + acc['A'][a, b] - acc['Ba'][a] * pb[b]
                           - acc['Bb'][b] * pa[a] - acc['C'] * pab[a, b]
                           + acc['E'] * pp[a, b])
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    zz = tensor(sp4, sp_n, z[3], w[3], deltas)
    v = (det * zz).scale(16)
    pi = arb.pi()
    vbar = 512 * pi ** 4 * k0 ** 2 / arb(0.25).gamma() ** 8
    f = v.scale(1 / vbar).add_const(-1)
    times['assembly'] = time.perf_counter() - t1
    # Checks inside the smallest box: the models of p0 and of the top form at a real point.
    small = float(deltas.min())
    s_pt = [acb(0.6 * small, 0.3 * small), acb(-0.2 * small, 0.5 * small)]
    point4 = s_pt + [x.conjugate() for x in s_pt]
    z_pt = moved_point(chart, s_pt)
    check_p = ub(p.at(point4) - direct_p(z_pt)) / ub(p.c[0])
    check_n = 0.0
    if fl:
        top_level = max(D['levels'], key=lambda lv: lv[0])
        f_top = dict(fl)[top_level[0]]
        check_n = ub(f_top[0, 0].at(point4) - direct_form(top_level, z_pt)) / ub(f_top[0, 0].c[0])
    f_center = f.c[0]
    f_dd = D['v1'][k] / vbar - 1
    var = f.add_const(-f_center)
    return {'point': k, 'ratio': D['ratio'][k], 'n': n, 'deltas': deltas.tolist(),
            'f_center': float(f_center.real), 'f_dd': float(f_dd.real),
            'center_gap': ub(f_center - f_dd), 'poly': var.poly_bound().tolist(),
            'check_p': check_p, 'check_n': check_n,
            'rem_p': (p.rem / ub(p.c[0])).tolist(),
            'rem': f.rem.tolist(), 'rem_ip': ip.rem.tolist(), 'rem_u': u.rem.tolist(),
            'rem_g': [float(x) for x in g[0, 0].rem], 'times': times,
            'etas': {key: [float(x) for x in val] for key, val in etas.items()},
            'mac_forms': mac_forms, 'mac_assembly': COUNT['mac'],
            'seconds': time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--forms', type=str, required=True)
    ap.add_argument('--col', type=int, default=3, help='the column of the export v')
    ap.add_argument('--points', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--orders', type=int, nargs='+', default=[12, 16, 20])
    ap.add_argument('--deltas', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.04])
    ap.add_argument('--prec', type=int, default=160)
    ap.add_argument('--procs', type=int, default=9)
    ap.add_argument('--factor', action='store_true',
                    help='take the binomial series of p0 / abs(z)^(2 k0) and of abs(z)^2')
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()
    t0 = time.perf_counter()
    FACTOR['on'] = args.factor
    load(args)
    print(f'loaded: k0 {D["k0"]}, levels {[lv[0] for lv in D["levels"]]}, '
          f'{len(D["z"])} points with min/max abs(z) {D["ratio"]}, '
          f'{time.perf_counter() - t0:.0f}s', flush=True)
    jobs = [(k, n, args.deltas) for n in args.orders for k in args.points]
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        results = []
        for res in pool.imap_unordered(run_box, jobs):
            results.append(res)
            print(f'point {res["point"]} (rho {res["ratio"]:.3f}), order {res["n"]}: '
                  f'f(c) {res["f_center"]:+.3e}, dd {res["f_dd"]:+.3e}, gap {res["center_gap"]:.1e}; '
                  f'{res["seconds"]:.0f}s, forms {res["mac_forms"]:.2e} MAC, assembly '
                  f'{res["mac_assembly"]:.2e} MAC; times '
                  + ', '.join(f'{key} {val:.0f}s' for key, val in res['times'].items()),
                  flush=True)
            print('    eta of the binomial series by delta: ' + '; '.join(
                f'{key} ' + ' '.join(f'{x:.3f}' for x in val)
                for key, val in res['etas'].items()), flush=True)
            print(f'    check at a point of the smallest box: p0 {res["check_p"]:.1e} '
                  f'(remainder {res["rem_p"][0]:.1e}), top form {res["check_n"]:.1e}, '
                  'both relative to the center value', flush=True)
            for i, dl in enumerate(res['deltas']):
                print(f'    delta {dl:.4f}: bound of abs(f - f(c)) {res["poly"][i]:.2e} '
                      f'+ remainder {res["rem"][i]:.2e}  (remainders: 1/p0 {res["rem_ip"][i]:.1e},'
                      f' p0^(-1/k0) {res["rem_u"][i]:.1e}, g_00 {res["rem_g"][i]:.1e})',
                      flush=True)
    if args.out:
        np.save(args.out, np.array(results, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()
