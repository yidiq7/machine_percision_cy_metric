"""Tests of `quartic.cert.box.metric_bounds` against definitions in real coordinates.

1. Polynomial models: a potential Phi of degree 4 in (tau, sigma) with Hermitian
   coefficients, g = d_tau d_sigma Phi, f = kap det g - 1 (so Ric = -ddbar log(1 + f)), a
   second potential for g_FS and a shift h(tau). At random points of the slice sigma =
   conj(tau) of two boxes, mpmath at 40 digits gives from the real 4 x 4 matrices: the
   eigenvalues of G_0^-1 G; the eigenvalues of G^-1 Ric with Ric from the Christoffel
   symbols of G; the real norm of DG with each argument normalized by G^(-1/2); and the
   same norm in the chart t = tau + n_01 h(tau), from central differences of G_t. Each
   bound must hold at each point, and lie within a factor of the largest value. The test
   also checks the two identities of the design: the eigenvalues of G^-1 Ric are 2 / c
   times those of g^-1 R, R = -ddbar log det g, and the real norm of DG is sqrt(8) times
   the complex norm of c dg in the coordinates of c g = I.
2. A real batch (4 cells of the reference center 297, cover v3, local forms v3): g_FS at
   each center equals E_tau^T h_FS(p) conj(E_tau) from the FS formula, with
   E_tau = dy / dtau = e + n dh(0); mu_low <= mu(center) <= mu_up; each output is finite.

Run on a GPU node: uv run --extra gpu --locked python -u -m tests.test_metric_bounds
"""
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmgpu import tm                                         # noqa: E402
from tmgpu.tools import space                                # noqa: E402
from quartic.cert import box as tc                           # noqa: E402

COVER = ROOT / 'runs_tm' / 'cover_s0.03_c6_v3.npz'
LOCAL = Path('/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td_v3/center_297.npz')
RNG = np.random.default_rng(20261024)
K0 = 2
N_ORDER, NY = 12, 14
DELTAS = [0.02, 0.04]
A = 0.35
POINTS = 24
# The largest allowed bound over the largest value at the points.
FACTOR = {'mu_up': 2.0, 'mu_low': 2.0, 'ric': 20.0, 'k_tau': 20.0, 'k_t': 20.0}
mp.mp.dps = 40


# ---------------------------------------------------------------- polynomials as dicts


def deriv(poly, v):
    """d / dx_v of a polynomial {exponents: complex}."""
    out = {}
    for e, c in poly.items():
        if e[v]:
            f = list(e)
            f[v] -= 1
            out[tuple(f)] = out.get(tuple(f), 0) + c * e[v]
    return out


def lin(*terms):
    """The polynomial sum of k * p over the pairs (k, p)."""
    out = {}
    for k, p in terms:
        for e, c in p.items():
            out[e] = out.get(e, 0) + k * c
    return out


def at(poly, x):
    """The value of the polynomial at the point x (mpmath numbers)."""
    s = mp.mpc(0)
    for e, c in poly.items():
        term = mp.mpc(c.real, c.imag)
        for xi, k in zip(x, e):
            term *= xi ** k
        s += term
    return s


def hermitian(quad, scale, rng):
    """A potential with the Hermitian quadratic part quad and random Hermitian terms of
    degree 3 and 4 of size scale; the coefficient of tau^a sigma^b is the conjugate of that
    of tau^b sigma^a, so Phi is real on the slice."""
    p = {(1, 0, 1, 0): complex(quad[0][0]), (0, 1, 0, 1): complex(quad[1][1]),
         (1, 0, 0, 1): complex(quad[0][1]), (0, 1, 1, 0): complex(quad[0][1]).conjugate()}
    mons = [(a0, a1, b0, b1) for a0 in range(5) for a1 in range(5) for b0 in range(5)
            for b1 in range(5) if 3 <= a0 + a1 + b0 + b1 <= 4]
    for e in mons:
        f = (e[2], e[3], e[0], e[1])
        if e > f or e in p:
            continue
        z = complex(*(scale * rng.standard_normal(2)))
        if e == f:
            z = complex(z.real)
        p[e], p[f] = z, z.conjugate()
    return p


def to_tm(sp, poly, delta):
    """The model of the polynomial with exact FP64 coefficients and no remainder."""
    b = delta.shape[0]
    c = np.zeros((4, sp.size, b))
    for e, z in poly.items():
        c[0, sp.index[e]] = z.real
        c[2, sp.index[e]] = z.imag
    return tm.TM(cp.asarray(c), cp.zeros(b), delta)


# ---------------------------------------------------------------- real coordinates


def realify(m):
    """The real 4 x 4 matrix of the form Re(sum m_ij u^i conj(v^j)) in the basis e_0, i e_0,
    e_1, i e_1."""
    units = [mp.mpc(1), mp.mpc(0, 1)]
    r = mp.matrix(4, 4)
    for i in range(2):
        for j in range(2):
            for p, a in enumerate(units):
                for q, b in enumerate(units):
                    r[2 * i + p, 2 * j + q] = mp.re(a * mp.conj(b) * m[i][j])
    return r


def real_linear(m):
    """The real 4 x 4 matrix of the complex linear map m in the coordinates (Re, Im)."""
    r = mp.matrix(4, 4)
    for i in range(2):
        for j in range(2):
            x, y = mp.re(m[i][j]), mp.im(m[i][j])
            r[2 * i, 2 * j], r[2 * i, 2 * j + 1] = x, -y
            r[2 * i + 1, 2 * j], r[2 * i + 1, 2 * j + 1] = y, x
    return r


def wirtinger(v):
    """The real derivative d/dx_v (v = 0..3 for Re tau_0, Im tau_0, Re tau_1, Im tau_1) on the
    slice, as the list of (factor, variable) of the polarized model."""
    k = v // 2
    return [(1, k), (1, 2 + k)] if v % 2 == 0 else [(1j, k), (-1j, 2 + k)]


def dreal(poly, v):
    return lin(*[(f, deriv(poly, var)) for f, var in wirtinger(v)])


def sym_sqrt_inv(m):
    """G^(-1/2) of a real symmetric positive matrix."""
    w, q = mp.eigsy(m)
    return q * mp.diag([1 / mp.sqrt(x) for x in w]) * q.T


def sym_eigs(m):
    """The eigenvalues of a real symmetric matrix (mpmath eigsy; the QR of eig does not
    converge on the double eigenvalues of a realified Hermitian matrix)."""
    w = mp.eigsy((m + m.T) / 2)[0]
    return [w[k] for k in range(m.rows)]


def frob3(t):
    return mp.sqrt(mp.fsum(t[a][b][c] ** 2 for a in range(4) for b in range(4) for c in range(4)))


def normalized(dg, p):
    """T_abc = sum DG[a'](b', c') P_a'a P_b'b P_c'c, as nested lists."""
    pd = [[sum(p[e, a] * dg[e] for e in range(4)) for a in range(4)]]
    return [[[mp.fsum(p[x, b] * pd[0][a][x, y] * p[y, c] for x in range(4) for y in range(4))
              for c in range(4)] for b in range(4)] for a in range(4)]


def ricci(g, dg, ddg):
    """The Ricci tensor of the real metric g with the derivatives dg[e] and ddg[e][f]."""
    gi = g ** -1
    dgi = [-gi * dg[e] * gi for e in range(4)]

    def low(d, b, c):
        return (dg[b][d, c] + dg[c][d, b] - dg[d][b, c]) / 2

    def dlow(e, d, b, c):
        return (ddg[e][b][d, c] + ddg[e][c][d, b] - ddg[e][d][b, c]) / 2

    gam = [[[mp.fsum(gi[a, d] * low(d, b, c) for d in range(4)) for c in range(4)]
            for b in range(4)] for a in range(4)]

    def dgam(e, a, b, c):
        return mp.fsum(dgi[e][a, d] * low(d, b, c) + gi[a, d] * dlow(e, d, b, c) for d in range(4))

    r = mp.matrix(4, 4)
    for b in range(4):
        for c in range(4):
            r[b, c] = mp.fsum(
                [dgam(a, a, b, c) - dgam(c, a, b, a) for a in range(4)]
                + [gam[a][a][d] * gam[d][b][c] - gam[a][c][d] * gam[d][b][a]
                   for a in range(4) for d in range(4)])
    return r


# ---------------------------------------------------------------- test 1


def synthetic():
    rng = np.random.default_rng(20261024)
    phi = hermitian([[2.0, 0.5 + 0.3j], [0.5 - 0.3j, 1.2]], 0.15, rng)
    fsp = hermitian([[1.0, -0.2 + 0.1j], [-0.2 - 0.1j, 0.8]], 0.1, rng)
    h = {(1, 0): 0.01 + 0j, (2, 0): 0.7 + 0j, (1, 1): -0.4j, (0, 2): 0.5 + 0j, (3, 0): 0.3 + 0j,
         (1, 2): 0.2 - 0.1j}
    n01 = [0.6 + 0.2j, -0.3 + 0.4j]
    det0 = 2.0 * 1.2 - abs(0.5 + 0.3j) ** 2
    kap = 1 / det0
    return phi, fsp, h, n01, kap


def models(phi, fsp, h, n01, kap):
    d = cp.asarray(np.array(DELTAS))
    d_f = d * (1 + A)
    d_phi = d_f * (1 + A)
    sp4, sp2 = space(4, N_ORDER), space(2, NY)
    phi_m = to_tm(sp4, phi, d_phi)
    g = {(i, j): tm.cauchy_mixed(sp4, phi_m, i, 2 + j, d_f) for i in range(2) for j in range(2)}
    det = tm.sub(sp4, tm.mul(sp4, g[0, 0], g[1, 1]), tm.mul(sp4, g[0, 1], g[1, 0]))
    b = len(DELTAS)
    f = tm.add_const(sp4, tm.scale(sp4, det, tm.cconst(kap, b)), -tm.cconst(1, b))
    log_r2 = to_tm(sp4, fsp, d_phi)
    h_m = to_tm(sp2, h, d_phi)
    n = np.zeros((3, 4, b))
    for k, z in enumerate(n01):
        n[k, 0], n[k, 2] = z.real, z.imag
    grad = tc.gradient_bounds(sp4, f, g, det, d, K0)
    out = tc.metric_bounds(sp4, sp2, f, g, log_r2, h_m, cp.asarray(n), d, K0, grad)
    return {k: cp.asnumpy(v) for k, v in out.items()}


class Reference:
    """The values of the definitions at points of the slice, for the synthetic potential."""

    def __init__(self, phi, fsp, h, n01, kap):
        # Exact mpmath coefficients, so that the derivatives and products do not round in FP64.
        phi, fsp, h = ({e: mp.mpc(c) for e, c in p.items()} for p in (phi, fsp, h))
        self.c = 1 / (mp.sqrt(2) * mp.pi * K0)
        self.g = [[deriv(deriv(phi, i), 2 + j) for j in range(2)] for i in range(2)]
        self.gf = [[deriv(deriv(fsp, i), 2 + j) for j in range(2)] for i in range(2)]
        self.dg = [[[dreal(self.g[i][j], v) for j in range(2)] for i in range(2)] for v in range(4)]
        self.ddg = [[[[dreal(dreal(self.g[i][j], v), w) for j in range(2)] for i in range(2)]
                     for w in range(4)] for v in range(4)]
        self.h = h
        self.dh = [{(e[0], e[1]): c for e, c in deriv({(a, b, 0, 0): c for (a, b), c in h.items()},
                                                       k).items()} for k in range(2)]
        self.n01 = [mp.mpc(z.real, z.imag) for z in n01]
        self.kap = kap
        self.phi = phi

    def cmat(self, polys, x):
        return [[at(polys[i][j], x) for j in range(2)] for i in range(2)]

    def big_g(self, tau):
        x = list(tau) + [mp.conj(t) for t in tau]
        return self.c * realify(self.cmat(self.g, x))

    def values(self, tau):
        x = list(tau) + [mp.conj(t) for t in tau]
        G = self.c * realify(self.cmat(self.g, x))
        G0 = realify(self.cmat(self.gf, x)) / (mp.pi * mp.sqrt(2))
        s0 = sym_sqrt_inv(G0)
        mu = sorted(sym_eigs(s0 * G * s0))
        dG = [self.c * realify(self.cmat(self.dg[v], x)) for v in range(4)]
        ddG = [[self.c * realify(self.cmat(self.ddg[v][w], x)) for w in range(4)] for v in range(4)]
        ric = ricci(G, dG, ddG)
        sg = sym_sqrt_inv(G)
        eig_r = sym_eigs(sg * (ric + ric.T) / 2 * sg)
        k_tau = frob3(normalized(dG, sym_sqrt_inv(G)))
        return dict(mu=mu, ric=max(abs(v) for v in eig_r), eig_r=eig_r, k_tau=k_tau,
                    k_t=self.k_chart(tau), dG=dG, G=G, x=x)

    def t_of(self, tau):
        hv = at({(a, b, 0, 0): c for (a, b), c in self.h.items()}, list(tau) + [0, 0])
        return [tau[k] + self.n01[k] * hv for k in range(2)]

    def jac(self, tau):
        x = list(tau) + [0, 0]
        dh = [at({(e[0], e[1], 0, 0): c for e, c in self.dh[k].items()}, x) for k in range(2)]
        return [[(1 if i == j else 0) + self.n01[i] * dh[j] for j in range(2)] for i in range(2)]

    def tau_of(self, t, start):
        tau = list(start)
        for _ in range(60):
            res = [a - b for a, b in zip(self.t_of(tau), t)]
            j = mp.matrix(self.jac(tau))
            step = j ** -1 * mp.matrix(res)
            tau = [tau[k] - step[k] for k in range(2)]
            if max(abs(step[k]) for k in range(2)) < mp.mpf(10) ** (-mp.mp.dps + 5):
                break
        return tau

    def g_t(self, t, start):
        tau = self.tau_of(t, start)
        dtau = real_linear(self.jac(tau)) ** -1
        return dtau.T * self.big_g(tau) * dtau

    def k_chart(self, tau):
        t = self.t_of(tau)
        eps = mp.mpf(10) ** -15
        units = [mp.mpc(1), mp.mpc(0, 1)]
        dG = []
        for v in range(4):
            e = [0, 0]
            e[v // 2] = units[v % 2] * eps
            plus = self.g_t([t[k] + e[k] for k in range(2)], tau)
            minus = self.g_t([t[k] - e[k] for k in range(2)], tau)
            dG.append((plus - minus) / (2 * eps))
        return frob3(normalized(dG, sym_sqrt_inv(self.g_t(t, tau))))

    def identities(self, vals):
        """The two identities of the design at one point: the relative errors."""
        x = vals['x']
        g = mp.matrix(self.cmat(self.g, x))
        # R = -ddbar log det g from the polynomial of det g.
        det = lin((1, mul(self.g[0][0], self.g[1][1])), (-1, mul(self.g[0][1], self.g[1][0])))
        dv = at(det, x)
        dd = [[deriv(deriv(det, i), 2 + j) for j in range(2)] for i in range(2)]
        d1 = [at(deriv(det, i), x) for i in range(2)]
        d2 = [at(deriv(det, 2 + j), x) for j in range(2)]
        r = mp.matrix(2, 2)
        for i in range(2):
            for j in range(2):
                r[i, j] = -(at(dd[i][j], x) / dv - d1[i] * d2[j] / dv ** 2)
        eig_c = sorted(mp.re(v) * 2 / self.c for v in mp.eig(g ** -1 * r)[0])
        eig_r = sorted(vals['eig_r'])
        err_ric = max(abs(eig_r[2 * k] - eig_c[k]) + abs(eig_r[2 * k + 1] - eig_c[k])
                      for k in range(2)) / max(abs(v) for v in eig_c)
        # sqrt(8) abs(Gamma)_F with Gamma_k = c dg_k in the coordinates z = P w of c g = I:
        # P^T (c g) conj(P) = I for P = conj(q) W^(-1/2), with c g = q W q^H.
        w, q = mp.eighe(self.c * g)
        p = q.H.T * mp.diag([1 / mp.sqrt(mp.re(v)) for v in w])
        dk = [mp.matrix(self.cmat([[deriv(self.g[i][j], k) for j in range(2)] for i in range(2)],
                                  x)) * self.c for k in range(2)]
        gam = mp.fsum(abs(mp.fsum(p[k, c] * p[i, a] * mp.conj(p[j, b]) * dk[k][i, j]
                                  for k in range(2) for i in range(2) for j in range(2))) ** 2
                      for c in range(2) for a in range(2) for b in range(2))
        err_k = abs(mp.sqrt(8 * gam) - vals['k_tau']) / vals['k_tau']
        return err_ric, err_k


def mul(p, q):
    out = {}
    for e, a in p.items():
        for f, b in q.items():
            k = tuple(x + y for x, y in zip(e, f))
            out[k] = out.get(k, 0) + a * b
    return out


def points(delta, rng):
    """POINTS points of the slice: half on the distinguished boundary abs(tau_i) = delta."""
    out = []
    for k in range(POINTS):
        r = delta * (np.ones(2) if k % 2 == 0 else np.sqrt(rng.uniform(size=2)))
        a = rng.uniform(0, 2 * np.pi, size=2)
        out.append([mp.mpc(float(r[i] * np.cos(a[i])), float(r[i] * np.sin(a[i])))
                    for i in range(2)])
    return out


def test_synthetic():
    t0 = time.perf_counter()
    data = synthetic()
    out = models(*data)
    ref = Reference(*data)
    for bx, delta in enumerate(DELTAS):
        vals = [ref.values(tau) for tau in points(delta, RNG)]
        top = dict(mu_up=max(v['mu'][-1] for v in vals), mu_low=min(v['mu'][0] for v in vals),
                   ric=max(v['ric'] for v in vals), k_tau=max(v['k_tau'] for v in vals),
                   k_t=max(v['k_t'] for v in vals))
        bound = {k: out[k][bx] for k in top}
        assert bound['mu_up'] >= top['mu_up'] and bound['mu_low'] <= top['mu_low'], (bound, top)
        for k in ('ric', 'k_tau', 'k_t'):
            assert bound[k] >= top[k], (k, bound[k], top[k])
        assert bound['mu_up'] <= FACTOR['mu_up'] * top['mu_up']
        assert bound['mu_low'] * FACTOR['mu_low'] >= top['mu_low']
        for k in ('ric', 'k_tau', 'k_t'):
            assert bound[k] <= FACTOR[k] * top[k], (k, bound[k], top[k])
        errs = [ref.identities(v) for v in vals[:4]]
        assert max(e[0] for e in errs) < 1e-25, errs
        assert max(e[1] for e in errs) < 1e-25, errs
        print(f'  box {bx} (delta {delta}): ' + ', '.join(
            f'{k} {bound[k]:.6g} over the points {float(top[k]):.6g}' for k in top)
            + f'; identities {float(max(max(e) for e in errs)):.1e}', flush=True)
    print(f'test_synthetic: PASS ({time.perf_counter() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------- test 2


def test_real_box():
    t0 = time.perf_counter()
    cov = np.load(COVER)
    local = tc.local_on_gpu(str(LOCAL))
    k0 = int(local['k0'])
    idx = np.flatnonzero(cov['ref'] == 297)[:4]
    centers = [[list(x) for x in cov['centers'][i]] for i in idx]
    fr = tc.make_frames(centers, k0)
    keep = {}
    f, (d, _, _), info, parts = tc.box_f(fr, local, [float(cov['deltas'][i]) for i in idx], 28,
                                         28 + tc.NY_EXTRA, 0.35,
                                         cp.asarray(cov['hmaxs'][idx].astype(float)), grad=True,
                                         keep=keep)
    sp4, sp2 = parts['sp4'], parts['sp2']
    out = {k: cp.asnumpy(info[k]) for k in ('mu_up', 'mu_low', 'ric', 'k_tau', 'k_t')}
    for k, v in out.items():
        assert np.all(np.isfinite(v) & (v > 0)), (k, v)
    assert np.all(out['k_t'] >= out['k_tau'])
    gfs = {(i, j): cp.asnumpy(tm.cauchy_mixed(sp4, keep['log_r2'], i, 2 + j, d).c[:, 0, :])
           for i in range(2) for j in range(2)}
    gc = {k: cp.asnumpy(v.c[:, 0, :]) for k, v in keep['g'].items()}
    hlin = [cp.asnumpy(keep['h_dd'].c[:, sp2.index[e], :]) for e in ((1, 0), (0, 1))]
    cz = lambda a, b: mp.mpc(mp.mpf(a[0, b]) + mp.mpf(a[1, b]), mp.mpf(a[2, b]) + mp.mpf(a[3, b]))
    worst_fs, worst_mu = 0, 0
    for b in range(len(idx)):
        host = fr.host[b]
        p, e, n = host['p'], host['e'], host['n']
        dh = [cz(hlin[k], b) for k in range(2)]
        ey = [[e[i][m] + n[m] * dh[i] for m in range(3)] for i in range(2)]
        s = 1 + mp.fsum(abs(v) ** 2 for v in p)
        hfs = [[(1 if m == q else 0) / s - mp.conj(p[m]) * p[q] / s ** 2 for q in range(3)]
               for m in range(3)]
        for i in range(2):
            for j in range(2):
                want = mp.fsum(ey[i][m] * hfs[m][q] * mp.conj(ey[j][q]) for m in range(3)
                               for q in range(3))
                worst_fs = max(worst_fs, abs(cz(gfs[i, j], b) - want) / abs(want))
        gm = mp.matrix([[cz(gc[i, j], b) for j in range(2)] for i in range(2)])
        fm = mp.matrix([[cz(gfs[i, j], b) for j in range(2)] for i in range(2)])
        mu = [mp.re(v) / k0 for v in mp.eig(fm ** -1 * gm)[0]]
        assert out['mu_low'][b] <= min(mu) and max(mu) <= out['mu_up'][b], (mu, out)
        worst_mu = max(worst_mu, out['mu_up'][b] / max(mu), min(mu) / out['mu_low'][b])
    assert worst_fs < 1e-25, worst_fs
    # The ratio holds the change of mu over the box as well as the looseness of the bound.
    assert worst_mu < 1.5, worst_mu
    print(f'  g_FS against the FS formula: largest relative error {float(worst_fs):.2e}; '
          f'mu bounds over the center values at most {float(worst_mu):.4f}; '
          + ', '.join(f'{k} {v.max():.4g}' for k, v in out.items()), flush=True)
    print(f'test_real_box: PASS ({time.perf_counter() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    test_synthetic()
    test_real_box()
    print('RESULT: PASS', flush=True)
