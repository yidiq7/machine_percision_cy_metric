"""The forms of a level fit as global polynomials in the chart z_3 = 1, m = 2, in arb.

In the chart, z = (t_0, t_1, z_2, 1) with z_2^4 = u_m = -(1 + t_0^4 + t_1^4). A block
term abs(z^r)^2 (u^E)^T C u^E is z_2^(r_2) w_2^(r_2) times sum over E, F of
C[E, F] h_E(t) h_F(w), with the integer polynomial
h_E(t) = t_0^(r_0 + 4 E_0) t_1^(r_1 + 4 E_1) u_m(t)^(E_2). So each form is

    sum over e = 0..3 of z_2(t)^e w_2(w)^e Q_e(t, w),   Q_e = sum over r with r_2 = e of V_r^T C_r V_r,

with V_r[E, alpha] the coefficient of t^alpha in h_E. The coefficients of Q_e are real
and symmetric, and they are formed in arb from the stored dd numbers of C_r, so each
one carries a strict radius.

A level form N_L of `level_matrices.py` is a sum over S_4 of one block of each orbit.
Each orbit block goes to the residues r' = sigma r with the monomial matrix
Pi_sigma (F C_o F^T) Pi_sigma^T, and the matrices of one residue are summed first.

The levels cancel strongly, so they are combined here in arb, at a fixed x0:
M~_m = sum over L of C(L, m) x0^L N_L. A box with x_c = x0 (1 + zeta) then takes
F_k = (1 + zeta)^k sum over m >= k of C(m, k) zeta^(m - k) M~_m, which is the exact
sum over L of C(L, k) x_c^L N_L, and its terms do not cancel.
"""
import itertools
import math

import numpy as np
from flint import acb, arb, arb_mat, ctx
from tmgpu.host import to_td, up

from .. import blocks

S4 = list(itertools.permutations(range(4)))
J, M, FREE = 3, 2, (0, 1)


def monomials2(d):
    """The monomials (a, b) of t_0^a t_1^b of degree <= d, in the order of tmgpu.tm.Space(2, d)."""
    return [(a, s - a) for s in range(d + 1) for a in range(s, -1, -1)]


class Poly2:
    """Polynomials in t_0, t_1 of degree <= d as integer coefficient vectors."""

    def __init__(self, d):
        self.d = d
        self.mons = monomials2(d)
        self.index = {e: i for i, e in enumerate(self.mons)}
        self.size = len(self.mons)

    def zero(self):
        return np.zeros(self.size, dtype=object)

    def mono(self, a, b):
        v = self.zero()
        v[self.index[(a, b)]] = 1
        return v

    def mul(self, x, y):
        out = self.zero()
        for i, ei in enumerate(self.mons):
            if x[i] == 0:
                continue
            for j, ej in enumerate(self.mons):
                if y[j] == 0:
                    continue
                e = (ei[0] + ej[0], ei[1] + ej[1])
                if e[0] + e[1] > self.d:
                    raise ValueError('a product beyond the degree of the space')
                out[self.index[e]] += x[i] * y[j]
        return out

    def power(self, x, n):
        out = self.mono(0, 0)
        for _ in range(n):
            out = self.mul(out, x)
        return out

    def u_m(self):
        """u_m = -(1 + t_0^4 + t_1^4)."""
        v = self.zero()
        v[self.index[(0, 0)]] = -1
        v[self.index[(4, 0)]] = -1
        v[self.index[(0, 4)]] = -1
        return v


def h_matrix(p2, r, exps):
    """V[E, alpha]: the integer coefficients of h_E(t) for the residue r, as an arb_mat."""
    um = p2.u_m()
    rows = []
    for e in exps:
        v = p2.mul(p2.mono(r[FREE[0]] + 4 * e[FREE[0]], r[FREE[1]] + 4 * e[FREE[1]]),
                   p2.power(um, e[M]))
        rows.append(v)
    return arb_mat(len(exps), p2.size, [arb(int(x)) for row in rows for x in row])


def form_q(p2, blocks_list):
    """Q_e for e = 0..3 from (r, exps, C) blocks, C an arb_mat in the monomial basis."""
    q = [arb_mat(p2.size, p2.size) for _ in range(4)]
    for r, exps, c in blocks_list:
        v = h_matrix(p2, r, exps)
        q[r[M]] += v.transpose() * c * v
    return q


def base_blocks(data):
    """(r, exps, C) of every block of the invariant base in an export of level_oracle."""
    out = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        if frame.shape[0] != frame.shape[1] or not np.array_equal(frame, np.eye(frame.shape[0])):
            raise SystemExit('the export does not hold the invariant base (--invariant-base)')
        exps = [tuple(int(x) for x in e) for e in data[f'g{i}_exps']]
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        n = len(exps)
        for b, r in enumerate(data[f'g{i}_res'].tolist()):
            c = arb_mat(n, n, [arb(float(x)) + arb(float(y))
                               for x, y in zip(hhi[b].ravel(), hlo[b].ravel())])
            out.append((tuple(int(x) for x in r), exps, c))
    return out


def level_blocks(forms, k1):
    """(r', exps, C') for every residue of the level form N_L: the S_4 sum by residue."""
    groups = {}
    for gi in range(int(forms[f'L{k1}_n_groups'])):
        exps = [tuple(int(x) for x in e) for e in forms[f'L{k1}_g{gi}_exps']]
        groups[gi] = (exps, forms[f'L{k1}_g{gi}_frame'])
    acc = {}
    for oi in range(int(forms[f'L{k1}_n_orbits'])):
        gi = int(forms[f'L{k1}_o{oi}_group'])
        exps, frame = groups[gi]
        r0 = tuple(int(x) for x in forms[f'L{k1}_o{oi}_r'])
        cs = forms[f'L{k1}_o{oi}_C']
        d = cs.shape[0]
        c_o = arb_mat(d, d, [arb(str(x)) for x in cs.ravel()])
        f = arb_mat(frame.shape[0], d, [arb(float(x)) for x in frame.ravel()])
        c_mono = f * c_o * f.transpose()
        n = len(exps)
        index = {e: i for i, e in enumerate(exps)}
        for s in S4:
            rp = blocks.act(s, r0)
            pi = [index[blocks.act(s, e)] for e in exps]
            perm = arb_mat(n, n)
            for i in range(n):
                perm[pi[i], i] = 1
            term = perm * c_mono * perm.transpose()
            key = (rp, gi)
            acc[key] = term if key not in acc else acc[key] + term
    return [(rp, groups[gi][0], c) for (rp, gi), c in acc.items()]


def chart_forms_arb(data, forms, prec=256):
    """Q_e of the base (degree k0) and of each N_L (degree top), as arb_mat."""
    ctx.prec = prec
    k0 = int(data['k0'])
    top = int(max(forms['k1']))
    out = {'k0': k0, 'top': top, 'levels': [int(x) for x in forms['k1']]}
    out['base'] = form_q(Poly2(k0), base_blocks(data))
    p2 = Poly2(top)
    for k1 in out['levels']:
        out[f'N{k1}'] = form_q(p2, level_blocks(forms, k1))
    return out


def shift_matrix(p2, c):
    """T with sum over a of q_a t^a = sum over a' of (T q)_a' (t - c)^a', for t^a in p2.

    c is a pair of exact acb numbers.
    """
    from flint import acb, acb_mat
    vals = []
    for ap in p2.mons:
        for a in p2.mons:
            if ap[0] <= a[0] and ap[1] <= a[1]:
                vals.append(math.comb(a[0], ap[0]) * math.comb(a[1], ap[1])
                            * c[0] ** (a[0] - ap[0]) * c[1] ** (a[1] - ap[1]))
            else:
                vals.append(acb(0))
    return acb_mat(p2.size, p2.size, vals)


def conj_transpose(m):
    from flint import acb_mat
    return acb_mat(m.ncols(), m.nrows(), [m[i, j].conjugate() for j in range(m.ncols())
                                           for i in range(m.nrows())])


def to_td_complex(q):
    """An acb_mat as (6, n, m) td parts (three limbs of the midpoint of re, then of im), the
    radii and the abs bounds of the entries.

    For the td entry e and any point v of the ball q[i, j], rad bounds
    abs(Re(e - v)) + abs(Im(e - v)), and abs bounds both abs(v) and abs(e)."""
    n, m = q.nrows(), q.ncols()
    parts = np.zeros((6, n, m))
    rad = np.zeros((n, m))
    absb = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            x = q[i, j]
            # tmgpu.host.to_td: the exact residual of the limbs plus the radius, up once.
            limbs, err = to_td(x)
            parts[:, i, j] = limbs
            rad[i, j] = err
            # abs(td entry) <= abs(midpoint) + err; the ball abs(x) contains abs(midpoint).
            absb[i, j] = up(abs(x) + arb(err)) if math.isfinite(err) else math.inf
    return parts, rad, absb


def delta_split(qs, s2):
    """R_j = sum over e >= j of C(e, j) s2^(e - j) Q_e, for j = 0..3.

    With Delta = abs(z_2)^2 - s2, sum over e of abs(z_2)^(2e) Q_e = sum over j of
    Delta^j R_j. Near the curves z_a = 0 the Q_e cancel strongly and R_0 does not.
    """
    out = []
    for j in range(4):
        acc = None
        for e in range(j, 4):
            term = qs[e] * (math.comb(e, j) * s2 ** (e - j))
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def local_forms(cf, c, x0, s2):
    """The base and the M~_m at x0, moved to t = c in arb, as td parts, radii, abs bounds.

    M~_m = sum over L of C(L, m) x0^L N_L, with N_L the chart polynomials of the levels.
    A box with x_c = x0 (1 + zeta) takes
    F_k = (1 + zeta)^k sum over m >= k of C(m, k) zeta^(m - k) M~_m, and these terms do
    not cancel, so the cancellation of the levels happens here in arb only.
    With the exact real s2, each form is written as the R_j of `delta_split` (keys
    `<name>_j<j>`), each entry in three limbs (6 parts).
    """
    from flint import acb_mat
    out = {}
    tb = shift_matrix(Poly2(cf['k0']), c)
    tbh = conj_transpose(tb)
    tl = shift_matrix(Poly2(cf['top']), c)
    tlh = conj_transpose(tl)
    moved = {k1: [tl * acb_mat(q) * tlh for q in cf[f'N{k1}']] for k1 in cf['levels']}
    named = [('base', [tb * acb_mat(q) * tbh for q in cf['base']])]
    for m in range(cf['top'] + 1):
        qs = []
        for e in range(4):
            acc = None
            for k1 in cf['levels']:
                if m > k1:
                    continue
                term = moved[k1][e] * acb(arb(math.comb(k1, m)) * x0 ** k1)
                acc = term if acc is None else acc + term
            qs.append(acc)
        named.append((f'mt{m}', qs))
    for name, qs in named:
        qs = delta_split(qs, acb(s2))
        for e, q in enumerate(qs):
            parts, rad, absb = to_td_complex(q)
            out[f'{name}_j{e}'] = parts
            out[f'{name}_j{e}_rad'] = rad
            out[f'{name}_j{e}_abs'] = absb
    return out
