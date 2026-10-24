"""Tests of scripts/cert_atlas.py (CPU).

1. At t = 0 the FS metric of volume 1 is I / (2 pi sqrt 2), and at random points of D(3, 2)
   fs_eigs agrees with the eigenvalues of ddbar log(1 + abs(y)^2) from mpmath differences in
   real coordinates, divided by pi sqrt 2; the closed form fs_closed agrees with fs_eigs.
2. At random points, the real norms of D^2 F_a and D^3 F_a of `value` agree with the
   Frobenius norms of the second and third partial derivatives of F_a as a map R^4 -> R^4,
   from mpmath differences, for 4 charts.
3. The branch-and-bound from 24 first boxes (with the corner of lam0) to the side R/512 gives
   bounds at or beyond the values at random points of those boxes in the domain, for lam0, Lam0,
   the norm of D^2 F of one chart and of D^3 F of another.
6. The FS normalization of the transitions: the norm of Q0_a D^n F [Q0_b^-1 .] from the FS
   matrices of both charts equals that from Q0_b DF^-1 (one metric), and value at the point
   gives it within 1 + 1e-6.
4. combine on a synthetic run gives the formulas of the design.
5. The jets: the constant terms of fs_jets equal fs_matrix, and the Taylor coefficients of
   chart_jets (degree 5) and of fs_jets (the terms of g to degree 2, phi to degree 4) equal the
   derivatives from mpmath.

Run on CPU cores: uv run --extra gpu --locked python -u -m tests.test_cert_atlas
"""
import itertools
import math
import multiprocessing
import os
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import mpmath as mp
import numpy as np
from flint import acb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import cert_atlas as ca                                    # noqa: E402
from quartic.cert import records as rs                     # noqa: E402

mp.mp.dps = 40
RNG = np.random.default_rng(20261025)
PROCS = int(os.environ.get('PROCS', '8'))


def random_points(count, domains):
    """Points t (two Python complex numbers) of the wedge that lie in every domain."""
    out = []
    while len(out) < count:
        x = RNG.uniform(0, ca.R, size=4)
        if x[1] > x[0]:
            continue
        t0, t1 = acb(x[0], x[1]), acb(x[2], x[3])
        m = ca.moduli(t0, t1)
        if all(ca.inside(m, ch) for ch in domains):
            out.append((complex(x[0], x[1]), complex(x[2], x[3])))
    return out


def w_of(t0, t1):
    return (-(1 + t0 ** 4 + t1 ** 4)) ** mp.mpf(0.25)


def real_map(chart):
    """F_a as a map of the four real coordinates to the four real coordinates."""
    i, j = chart
    k, l = (x for x in range(4) if x not in chart)

    def f(a, b, c, d):
        t0, t1 = mp.mpc(a, b), mp.mpc(c, d)
        z = [t0, t1, w_of(t0, t1), mp.mpc(1)]
        u, v = z[k] / z[i], z[l] / z[i]
        return [mp.re(u), mp.im(u), mp.re(v), mp.im(v)]
    return f


def test_fs():
    lo, hi = ca.fs_eigs(acb(0), acb(0), ca.root4(acb(-1)))
    want = 1 / (2 * mp.pi * mp.sqrt(2))
    assert abs(float(lo.mid()) - want) < 1e-15 and abs(float(hi.mid()) - want) < 1e-15
    worst, worst_closed = 0, 0
    for t0, t1 in random_points(6, [ca.CHART_B]):
        def phi(a, b, c, d):
            s0, s1 = mp.mpc(a, b), mp.mpc(c, d)
            return mp.log(1 + abs(s0) ** 2 + abs(s1) ** 2 + abs(w_of(s0, s1)) ** 2)
        x = [mp.mpf(t0.real), mp.mpf(t0.imag), mp.mpf(t1.real), mp.mpf(t1.imag)]
        hess = [[mp.diff(phi, x, tuple(int(q == a) + int(q == b) for q in range(4)))
                 for b in range(4)] for a in range(4)]
        g = mp.matrix(2, 2)
        for i in range(2):
            for j in range(2):
                xi, yi, xj, yj = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
                g[i, j] = (hess[xi][xj] + hess[yi][yj] + 1j * (hess[xi][yj] - hess[yi][xj])) / 4
        ev = sorted(mp.re(v) / (mp.pi * mp.sqrt(2)) for v in mp.eig(g)[0])
        a0, a1 = acb(t0), acb(t1)
        lo, hi = ca.fs_eigs(a0, a1, ca.wof(a0, a1))
        worst = max(worst, abs(float(lo.mid()) - ev[0]) / ev[0], abs(float(hi.mid()) - ev[1]) / ev[1])
        c_lo, c_hi = ca.fs_closed(a0, a1)
        e_lo, e_hi = (float(x.mid()) for x in ca.fs_eigs(a0, a1, ca.wof(a0, a1)))
        worst_closed = max(worst_closed, abs(c_lo - e_lo) / e_lo, abs(c_hi - e_hi) / e_hi)
    assert worst < 1e-12 and worst_closed < 1e-12, (worst, worst_closed)
    print(f'test_fs: PASS (largest relative difference {float(worst):.1e}; closed form '
          f'{float(worst_closed):.1e})', flush=True)


def test_norms():
    worst = 0
    for chart in [(0, 1), (2, 3), (3, 0), (1, 2)]:
        for t0, t1 in random_points(2, [ca.CHART_B]):
            f = real_map(chart)
            x = [mp.mpf(t0.real), mp.mpf(t0.imag), mp.mpf(t1.real), mp.mpf(t1.imag)]

            def part(orders):
                return [mp.diff(lambda *v: f(*v)[c], x, orders) for c in range(4)]
            idx = range(4)
            n2 = mp.sqrt(mp.fsum(v ** 2 for a in idx for b in idx
                                 for v in part(tuple(int(q == a) + int(q == b) for q in idx))))
            n3 = mp.sqrt(mp.fsum(v ** 2 for a in idx for b in idx for c in idx
                                 for v in part(tuple(int(q == a) + int(q == b) + int(q == c)
                                                     for q in idx))))
            d2, d3 = (ca.value((k, chart), acb(t0), acb(t1)) for k in ('d2', 'd3'))
            worst = max(worst, abs(float(d2.mid()) - n2) / n2, abs(float(d3.mid()) - n3) / n3)
    assert worst < 1e-10, worst
    print(f'test_norms: PASS (largest relative difference {float(worst):.1e})', flush=True)


def to_mp(z):
    """An acb ball as the mpmath complex number of its midpoint."""
    return mp.mpc(mp.mpf(z.real.mid().str(50, radius=False)), mp.mpf(z.imag.mid().str(50, radius=False)))


def branch_w(base):
    """w(t) with the branch of cert_atlas.root4 at the base point t."""
    b0, b1 = base
    v0 = -(1 + b0 ** 4 + b1 ** 4)

    def w(t0, t1):
        v = -(1 + t0 ** 4 + t1 ** 4)
        if mp.re(v0) >= 0:
            return v ** mp.mpf(0.25)
        return mp.exp(1j * mp.pi / 4) * (-v) ** mp.mpf(0.25)
    return w


def test_jets():
    worst_c, worst_t, worst_f = 0, 0, 0
    for t0, t1 in random_points(2, [ca.CHART_B]):
        a0, a1 = acb(t0), acb(t1)
        sp2, g = ca.fs_jets(a0, a1)
        fm = ca.fs_matrix(a0, a1, ca.wof(a0, a1))
        for i in range(2):
            for j in range(2):
                worst_c = max(worst_c, float(abs(g[i][j][0] - fm[i][j]).mid()) /
                              float(abs(fm[0][0]).mid()))
        base = (mp.mpc(t0), mp.mpc(t1))
        w = branch_w(base)
        for chart in [(0, 1), (2, 3)]:
            i, j = chart
            k, l = (x for x in range(4) if x not in chart)
            jets = ca.chart_jets(a0, a1, chart)
            for comp, jet in zip((k, l), jets):
                def f(x0, x1, comp=comp):
                    z = [x0, x1, w(x0, x1), mp.mpc(1)]
                    return z[comp] / z[i]
                for n, m in enumerate(ca.SP2.mons):
                    want = mp.diff(f, base, m) / (math.factorial(m[0]) * math.factorial(m[1]))
                    worst_t = max(worst_t, abs(to_mp(jet.c[n]) - want) / max(1, abs(want)))
        wt = lambda s0, s1: mp.conj(w(mp.conj(s0), mp.conj(s1)))
        phi = lambda x0, x1, y0, y1: mp.log(1 + x0 * y0 + x1 * y1 + w(x0, x1) * wt(y0, y1))
        x = [base[0], base[1], mp.conj(base[0]), mp.conj(base[1])]
        for i in range(2):
            for j in range(2):
                for n, m in enumerate(sp2.mons):
                    e = list(m)
                    e[i] += 1
                    e[2 + j] += 1
                    want = mp.diff(phi, x, tuple(e)) / math.prod(math.factorial(v) for v in m)
                    worst_f = max(worst_f, abs(to_mp(g[i][j][n]) - want) / max(1, abs(want)))
    assert worst_c < 1e-25 and worst_t < 1e-20 and worst_f < 1e-20, (worst_c, worst_t, worst_f)
    print(f'test_jets: PASS (constant terms {worst_c:.1e}, chart jets {float(worst_t):.1e}, '
          f'FS jets {float(worst_f):.1e})', flush=True)


def conj_mat(m):
    """The conjugate of a 2 x 2 acb matrix as an mpmath matrix."""
    return mp.matrix([[mp.conj(to_mp(m[i][j])) for j in range(2)] for i in range(2)])


def test_fs_normalized():
    """At points of D(3, 2) and D(a), the real norm of Q0_a D^n F [Q0_b^-1 .] from the FS
    matrices of both charts equals that from Q0_b DF^-1 in chart b, and value(c1 or c2), with
    the normalization at the point, bounds it within a factor 1 + 1e-6."""
    worst_id, worst_ratio, count = 0, 0, 0
    scale = mp.pi * mp.sqrt(2)
    for chart in [(2, 3), (3, 0), (0, 2)]:
        i, j = chart
        k, l = (x for x in range(4) if x not in chart)
        for t0, t1 in random_points(2, [ca.CHART_B, chart]):
            a0, a1 = acb(t0), acb(t1)
            w = ca.wof(a0, a1)
            z = [a0, a1, w, acb(1)]
            hb = conj_mat(ca.fs_matrix(a0, a1, w)) / scale
            ha = conj_mat(ca.fs_matrix(z[k] / z[i], z[l] / z[i], z[j] / z[i])) / scale
            jets = ca.chart_jets(a0, a1, chart)
            df = mp.matrix([[to_mp(f.c[1 + q]) for q in range(2)] for f in jets])
            qa, qb = mp.sqrtm(ha), mp.sqrtm(hb)
            qbi = qb ** -1
            for n, kind in ((2, 'c1'), (3, 'c2')):
                ks = list(itertools.product(range(2), repeat=n))

                def entry(f, idx):
                    e = tuple(idx.count(q) for q in range(2))
                    return to_mp(f.c[ca.SP2.idx[e]]) * math.prod(math.factorial(v) for v in e)
                tens = {(m,) + idx: entry(jets[m], idx) for m in range(2) for idx in ks}

                def norm(left):
                    tot = 0
                    for m in range(2):
                        for idx in ks:
                            v = 0
                            for m2 in range(2):
                                for src in ks:
                                    c = left[m, m2] * tens[(m2,) + src]
                                    for p, q in zip(idx, src):
                                        c *= qbi[q, p]
                                    v += c
                            tot += abs(v) ** 2
                    return mp.sqrt(tot) * (2 if n == 2 else 2 * mp.sqrt(2))
                direct, via = norm(qa), norm(qb * df ** -1)
                worst_id = max(worst_id, abs(direct - via) / direct)
                val = float(ca.value((kind, chart), a0, a1).mid())
                assert val >= direct, (chart, kind, val, direct)
                worst_ratio = max(worst_ratio, val / direct)
                count += 1
    assert worst_id < 1e-20 and worst_ratio < 1 + 1e-6, (worst_id, worst_ratio)
    print(f'test_fs_normalized: PASS ({count} values; the two forms agree to {float(worst_id):.1e}; '
          f'value over the norm at most {float(worst_ratio):.3f})', flush=True)


def test_search():
    """The branch-and-bound from 24 first boxes that meet each domain (the corner box of
    lam0 among them) down to the side R/512: each bound holds at random points of those boxes
    in the domain."""
    t0 = time.perf_counter()
    kinds = [('fs', 0), ('fs', 1), ('d2', (2, 3)), ('d3', (3, 0))]
    side = ca.R / 8
    with multiprocessing.Pool(PROCS) as pool:
        for kind in kinds:
            meet = [b for b in ca.first_boxes()
                    if all(ca.may_meet(ca.moduli(*ca.ball(*b)), ch) for ch in ca.domains(kind))]
            corner = [b for b in meet if all(abs(b[0][k] + side / 2 - x) < side / 2
                                             for k, x in enumerate((0.62, 0.50, 0.45, 0.70)))]
            boxes = corner + meet[::max(1, len(meet) // 23)][:24 - len(corner)]
            r = ca.run_kind(kind, pool, 0.3, ca.R / 2 ** 9, boxes=boxes)
            vals, tries = [], 0
            while len(vals) < 40 and tries < 100000:
                tries += 1
                lo, s = boxes[RNG.integers(len(boxes))]
                x = [lo[k] + s * RNG.uniform() for k in range(4)]
                if x[1] > x[0]:
                    continue
                a0, a1 = acb(x[0], x[1]), acb(x[2], x[3])
                if all(ca.inside(ca.moduli(a0, a1), ch) for ch in ca.domains(kind)):
                    vals.append(float(ca.value(kind, a0, a1).mid()))
            assert len(vals) >= 10, (kind, len(vals), tries)
            if ca.is_min(kind):
                assert 0 < r['bound'] <= min(vals), (kind, r, min(vals))
            else:
                assert r['bound'] >= max(vals), (kind, r, max(vals))
            print(f'  {kind}: bound {r["bound"]:.6g}, best {r["best"]:.6g}, {len(vals)} random points '
                  f'{min(vals) if ca.is_min(kind) else max(vals):.6g}, {r["boxes"]} boxes, '
                  f'{r["at_min"]} at the smallest side', flush=True)
    print(f'test_search: PASS ({time.perf_counter() - t0:.0f}s)', flush=True)


def test_combine():
    rec = np.zeros((3, len(rs.COLUMNS)))
    col = {c: k for k, c in enumerate(rs.COLUMNS)}
    rec[:, col['ref']] = [0, 0, 1]
    rec[:, col['bound']] = [1e-20, 4e-20, 2e-20]
    rec[:, col['grad_g']] = [1e-16, 3e-16, 2e-16]
    rec[:, col['mu_up']] = [1.5, 2.0, 1.8]
    rec[:, col['mu_low']] = [0.7, 0.5, 0.6]
    rec[:, col['ric']] = [3e-14, 6e-14, 1e-14]
    rec[:, col['k_t']] = [10.0, 30.0, 20.0]
    rec[:, col['k_atlas']] = [12.0, 30.0, 25.0]
    rec[:, col['lam_atlas']] = [0.2, 0.08, 0.3]
    rec[:, col['Lam_atlas']] = [1.5, 1.2, 3.0]
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / 'run.npz'
        np.savez(run, records=rec, target=5e-20, columns=np.array(rs.COLUMNS), commit='abc')
        fs = Path(d) / 'fs.npz'
        np.savez(fs, lam0_bound=0.1, Lam0_bound=1.0)
        tr = Path(d) / 'tr.npz'
        vals = {}
        for n, (i, j) in enumerate(ca.CHARTS):
            vals[f'c1_{i}{j}_bound'] = 3.0 if n == 4 else 1.0
            vals[f'c2_{i}{j}_bound'] = 5.0 if n == 7 else -math.inf
        np.savez(tr, **vals)
        out = Path(d) / 'out.npz'
        ca.combine(Namespace(runs=[str(run)], fs=str(fs), transitions=[str(tr)], out=str(out)))
        z = np.load(out)
        got = dict(zip([str(s) for s in z['names']], z['values']))
    # lambda: max(0.5 * 0.1, 0.08); Lambda: min(2.0 * 1.0, 3.0); K~: min(30, 30 + 2 c~1).
    lam, big = 0.08, 2.0
    c1 = math.sqrt(2.0) / 0.5 * 3.0
    want = {'C: omega <= C omega_0': 2.0, 'mu: omega >= mu omega_0': 0.5,
            'kappa: Ric >= -3 kappa g': 2e-14, 'lambda: lambda I <= G_a': lam,
            'Lambda: G_a <= Lambda I': big, 'c~1': c1,
            'c~2': math.sqrt(2.0) / 0.5 ** 1.5 * 5.0, 'K~': 30.0,
            'K~ from the chart of F + 2 c~1': 30 + 2 * c1,
            '[f o psi_a^-1]_{1,0;V_a} <= sqrt(Lambda) sup abs(grad f)_G': math.sqrt(big) * 3e-16}
    for k, v in want.items():
        assert abs(got[k] - v) <= 1e-12 * abs(v), (k, got[k], v)
    print('test_combine: PASS', flush=True)


if __name__ == '__main__':
    test_fs()
    test_jets()
    test_norms()
    test_fs_normalized()
    test_combine()
    test_search()
    print('RESULT: PASS', flush=True)
