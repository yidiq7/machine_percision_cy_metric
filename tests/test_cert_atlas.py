"""Tests of scripts/cert_atlas.py (CPU).

1. At t = 0 the FS metric of volume 1 is I / (2 pi sqrt 2), and at random points of D(3, 2)
   fs_eigs agrees with the eigenvalues of ddbar log(1 + abs(y)^2) from mpmath differences in
   real coordinates, divided by pi sqrt 2.
2. At random points, the real norms of D^2 F_a and D^3 F_a of deriv_norms agree with the
   Frobenius norms of the second and third partial derivatives of F_a as a map R^4 -> R^4,
   from mpmath differences, for 4 charts.
3. The branch-and-bound with coarse settings gives bounds at or beyond the values at random
   points of the domain, for lam0, Lam0 and two norms.
4. combine on a synthetic run gives the formulas of the design.

Run on CPU cores: uv run --extra gpu --locked python -u -m tests.test_cert_atlas
"""
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
        m, _ = ca.moduli(t0, t1)
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
    worst = 0
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
        lo, hi = ca.fs_eigs(a0, a1, ca.moduli(a0, a1)[1])
        worst = max(worst, abs(float(lo.mid()) - ev[0]) / ev[0], abs(float(hi.mid()) - ev[1]) / ev[1])
    assert worst < 1e-12, worst
    print(f'test_fs: PASS (largest relative difference {worst:.1e})', flush=True)


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
            d2, d3 = ca.deriv_norms(ca.chart_jets(acb(t0), acb(t1), chart))
            worst = max(worst, abs(float(d2.mid()) - n2) / n2, abs(float(d3.mid()) - n3) / n3)
    assert worst < 1e-10, worst
    print(f'test_norms: PASS (largest relative difference {worst:.1e})', flush=True)


def test_search():
    t0 = time.perf_counter()
    kinds = [('fs', 0), ('fs', 1), ('d2', (2, 3)), ('d3', (0, 1))]
    with multiprocessing.Pool(PROCS) as pool:
        for kind in kinds:
            r = ca.run_kind(kind, pool, 0.2, ca.R / 2 ** 5)
            vals = []
            for p0, p1 in random_points(40, ca.domains(kind)):
                a0, a1 = acb(p0), acb(p1)
                _, w = ca.moduli(a0, a1)
                vals.append(float(ca.value(kind, a0, a1, w).mid()))
            if ca.is_min(kind):
                assert r['bound'] <= min(vals) and r['bound'] > 0, (kind, r, min(vals))
            else:
                assert r['bound'] >= max(vals), (kind, r, max(vals))
            print(f'  {kind}: bound {r["bound"]:.6g}, best {r["best"]:.6g}, random points '
                  f'{min(vals) if ca.is_min(kind) else max(vals):.6g}, {r["boxes"]} boxes',
                  flush=True)
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
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / 'run.npz'
        np.savez(run, records=rec, target=5e-20, columns=np.array(rs.COLUMNS), commit='abc')
        fs = Path(d) / 'fs.npz'
        np.savez(fs, lam0_bound=0.1, Lam0_bound=1.0)
        tr = Path(d) / 'tr.npz'
        vals = {}
        for n, (i, j) in enumerate(ca.CHARTS):
            vals[f'd2_{i}{j}_bound'] = 3.0 if n == 4 else 1.0
            vals[f'd3_{i}{j}_bound'] = 5.0 if n == 7 else -math.inf
        np.savez(tr, **vals)
        out = Path(d) / 'out.npz'
        ca.combine(Namespace(runs=[str(run)], fs=str(fs), transitions=str(tr), out=str(out)))
        z = np.load(out)
        got = dict(zip([str(s) for s in z['names']], z['values']))
    lam, big = 0.5 * 0.1, 2.0 * 1.0
    c1 = math.sqrt(big) / lam * 3.0
    want = {'C: omega <= C omega_0': 2.0, 'mu: omega >= mu omega_0': 0.5,
            'kappa: Ric >= -3 kappa g': 2e-14, 'lambda: lambda I <= G_a': lam,
            'Lambda: G_a <= Lambda I': big, 'c~1': c1,
            'c~2': math.sqrt(big) / lam ** 1.5 * 5.0, 'K~ (K in the chart of F + 2 c~1)': 30 + 2 * c1,
            '[f o psi_a^-1]_{1,0;V_a} <= sqrt(Lambda) sup abs(grad f)_G': math.sqrt(big) * 3e-16}
    for k, v in want.items():
        assert abs(got[k] - v) <= 1e-12 * abs(v), (k, got[k], v)
    print('test_combine: PASS', flush=True)


if __name__ == '__main__':
    test_fs()
    test_norms()
    test_combine()
    test_search()
    print('RESULT: PASS', flush=True)
