"""Tests of the arb reference of the steps 1 to 9 of the box code (tests/box_reference.py).

1. exact and poly_at give the exact sums of the limbs and the value of a polynomial.
2. ppow and plog give the principal branches of mpmath at 300 digits, also near the cut.
3. Box.from_npz reads the layout of the export file.
4. solve_h proves the root h with a radius below 1e-80, and a start value off the root
   raises ValueError.
5. On the slice zb = conj(z), the polarized p0 and N_L agree with global_values of
   scripts/cert_local_values.py at 3 points of X.
6. Off the slice, the polarized forms are Hermitian: F(z, zb) = conj(F(conj(zb), conj(z))).
7. Off the slice, the sum over k of F_k xi^k agrees with psi.

The tests read runs_tm/f6t_data.npz, runs_tm/f6t_forms.npz and runs_tm/cover_s0.03_c6_v2.npz,
and the box takes its frame from quartic.cert.frame.

Run on a GPU node: uv run --extra gpu --locked python -u -m tests.test_box_reference
"""
import hashlib
import math
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from flint import acb, arb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts'))

from tests import box_reference as br                     # noqa: E402
from quartic.cert.frame import frame                      # noqa: E402
import cert_local_values as clv                           # noqa: E402

DATA = ROOT / 'runs_tm' / 'f6t_data.npz'
FORMS = ROOT / 'runs_tm' / 'f6t_forms.npz'
COVER = ROOT / 'runs_tm' / 'cover_s0.03_c6_v2.npz'
SHA = {DATA: '69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1',
       FORMS: '2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2'}
# The moduli of (t_0, t_1) of the test points; rho is near 0.9, 0.4 and 0.05.
MODULI = [(0.95, 0.9), (0.7, 0.4), (0.5, 0.05)]
# The test box is the first cell of this reference center.
REF = 297
RNG = np.random.default_rng(20261021)
STATE = {}


def check_inputs():
    for path, want in SHA.items():
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for block in iter(lambda: fh.read(1 << 24), b''):
                h.update(block)
        assert h.hexdigest() == want, f'{path}: SHA-256 {h.hexdigest()}, not {want}'


def potential():
    if 'pot' not in STATE:
        STATE['pot'] = br.load(DATA, FORMS)
    return STATE['pot']


def rad(x):
    """The larger of the radii of the real and the imaginary part of an acb."""
    return max(float(x.real.rad()), float(x.imag.rad()))


def rel_rad(x):
    """The radius of the ball x over abs(x)."""
    return rad(x) / max(float(abs(x).mid()), 1e-300)


def mp_ball(x):
    """The exact midpoint and radius of an arb as mpmath numbers."""
    m, e = x.mid().man_exp()
    rm, re = x.rad().man_exp()
    return mp.mpf(int(m)) * mp.mpf(2) ** int(e), mp.mpf(int(rm)) * mp.mpf(2) ** int(re)


def holds(ball, ref):
    """True if the acb ball holds the mpmath complex ref, part by part."""
    for part, r in ((ball.real, ref.real), (ball.imag, ref.imag)):
        mid, width = mp_ball(part)
        if not abs(mid - r) <= width:
            return False
    return True


def cis(r, phase):
    return complex(r * math.cos(phase), r * math.sin(phase))


def on_x(t0, t1):
    """z = (t_0, t_1, z_2, 1) on X, with z_2 the principal root of -(1 + t_0^4 + t_1^4)."""
    a, b = acb(t0.real, t0.imag), acb(t1.real, t1.imag)
    return [a, b, (-(1 + a ** 4 + b ** 4)).root(4), acb(1)]


def conj(z):
    return [v.conjugate() for v in z]


def dd_real(x):
    """A real dd list near the arb x."""
    hi = float(x.mid())
    return [hi, float(x.mid() - hi), 0.0, 0.0]


def dd_complex(x):
    """A complex dd list near the acb x."""
    re, im = dd_real(x.real), dd_real(x.imag)
    return [re[0], re[1], im[0], im[1]]


def make_box():
    """A box of the cover at the first cell of the reference REF, with its frame from
    quartic.cert.frame and free dd numbers near the values that box_f picks."""
    if 'box' in STATE:
        return STATE['box']
    pot = potential()
    cov = np.load(COVER)
    i = int(np.flatnonzero(np.asarray(cov['ref']) == REF)[0])
    c = np.asarray(cov['centers'][i], dtype=float)
    fr = frame([c[0].tolist(), c[1].tolist()], pot.k0)
    p = [br.exact(v) for v in fr['p']]
    zc = p + [acb(1)]
    p0c = br.forms_polar(pot, zc, conj(zc))[0]
    r20 = 1 + sum((v * v.conjugate() for v in p), acb(0))
    pt0 = p0c / r20 ** pot.k0
    dd = dict(s_r=dd_real((1 / r20).real), s_p=dd_real((1 / pt0).real),
              x0=dd_real(br.ppow(acb(p0c.real), Fraction(-1, pot.k0)).real),
              zeta=[1e-3, 0.0, 1e-3, 0.0], s2=dd_real(abs(p[2]) ** 2))
    kap = (br.ppow(br.exact(dd['s_p']), Fraction(1, pot.k0))
           / (br.exact(dd['x0']) * (1 + br.exact(dd['zeta']))))
    dd['kappa'] = dd_complex(kap)
    c_r = np.asarray(cov['ref_centers'][REF], dtype=float)
    box = br.Box(p=p, e=[[br.exact(v) for v in row] for row in fr['e']],
                 n=[br.exact(v) for v in fr['n']], c1inv=br.exact(fr['c1inv']),
                 vol=br.exact(fr['vol']), s_r=br.exact(dd['s_r']), s_p=br.exact(dd['s_p']),
                 zeta=br.exact(dd['zeta']), kappa=br.exact(dd['kappa']),
                 c_r=[br.exact(v) for v in c_r], x0=br.exact(dd['x0']),
                 s2=br.exact(dd['s2']), k0=pot.k0, top=pot.top)
    mods = [abs(complex(v)) for v in zc]
    print(f'  test box: cell {i} of the reference {REF}, rho {min(mods) / max(mods):.3f}, '
          f'p0 {complex(p0c).real:.6e}', flush=True)
    STATE.update(box=box, fr=fr, dd=dd, c_r=c_r)
    return box


def newton_h(box, tau):
    """An FP64 start value of h at tau: Newton steps on P(p + E tau + h n) = 0."""
    p, n = [complex(v) for v in box.p], [complex(v) for v in box.n]
    e = [[complex(v) for v in row] for row in box.e]
    base = [p[k] + e[0][k] * tau[0] + e[1][k] * tau[1] for k in range(3)]
    h = 0j
    for _ in range(40):
        y = [b + h * n[k] for k, b in enumerate(base)]
        h -= (sum(v ** 4 for v in y) + 1) / sum(4 * v ** 3 * n[k] for k, v in enumerate(y))
    return h


def random_tau(r, edge):
    """Two complex numbers with random phases, of modulus 0.999 r if edge is true, else of a
    random modulus below r."""
    return [cis(r * (0.999 if edge else float(RNG.uniform())), float(RNG.uniform(0, 2 * math.pi)))
            for _ in range(2)]


def test_exact_and_poly():
    v = br.exact([1.0, 2.0 ** -60, 0.5, -2.0 ** -70])
    assert v.is_exact() and v.real == 1 + arb(2.0 ** -60), v
    assert v.imag == arb(0.5) - arb(2.0 ** -70), v
    w = br.exact([1.0, 1e-300, 0.0, 0.0])
    # The limbs are 997 bits apart; the sum stays exact.
    assert w.is_exact() and w.real - 1 == arb(1e-300), w
    t = br.exact([1.0, 2.0 ** -55, 2.0 ** -110, -3.0, 2.0 ** -54, -2.0 ** -108])
    assert t.real == 1 + arb(2.0 ** -55) + arb(2.0 ** -110), t
    assert t.imag == -3 + arb(2.0 ** -54) - arb(2.0 ** -108), t
    # 1 + (2 + i) x0 + 3 x0 x1^2 at an exact point.
    coefs = np.array([[1.0, 2.0, 3.0], [0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    mons = np.array([[0, 0], [1, 0], [1, 2]])
    x = [acb(0.5, 0.25), acb(-1.5, 2.0)]
    want = 1 + acb(2, 1) * x[0] + 3 * x[0] * x[1] ** 2
    got = br.poly_at(coefs, mons, x)
    assert got.overlaps(want) and rad(got) < 1e-90, (got, want)
    for bad in (lambda: br.poly_at(coefs, mons[:2], x), lambda: br.exact([1.0, 2.0, 3.0])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError('a wrong shape passed')
    print('  exact: dd and td sums exact, also with limbs 997 bits apart; poly_at agrees',
          flush=True)


def test_principal_branch():
    mp.mp.dps = 300
    for phase in (0.4, -3.1):
        xc = cis(0.9, phase)
        x, ref = acb(xc.real, xc.imag), mp.mpc(xc.real, xc.imag)
        pw, lg = br.ppow(x, Fraction(-1, 16)), br.plog(x)
        want_pw, want_lg = mp.power(ref, mp.mpf(-1) / 16), mp.log(ref)
        assert holds(pw, want_pw), (phase, pw, want_pw)
        assert holds(lg, want_lg), (phase, lg, want_lg)
        assert rad(pw) < 1e-90 and rad(lg) < 1e-90, (rad(pw), rad(lg))
        # The value of the next branch, arg x + 2 pi, lies off the ball.
        other = want_pw * mp.exp(-2j * mp.pi / 16)
        assert not holds(pw, other), (phase, pw, other)
        print(f'  arg x = {phase}: ppow {complex(pw):.15f}, plog {complex(lg):.15f}; radii '
              f'{rad(pw):.1e}, {rad(lg):.1e}', flush=True)


def test_from_npz():
    make_box()
    fr, dd, c_r = STATE['fr'], STATE['dd'], STATE['c_r']
    arrays = dict(fr_p=np.array(fr['p'], float), fr_e=np.array(fr['e'], float),
                  fr_n=np.array(fr['n'], float), fr_c1inv=np.array(fr['c1inv'], float),
                  fr_vol=np.array(fr['vol'], float), c_r=c_r, k0=make_box().k0,
                  top=make_box().top,
                  **{k: np.array(v, float) for k, v in dd.items()})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'box.npz'
        np.savez(path, **arrays)
        got = br.Box.from_npz(path)
        want = make_box()
        for key in ('c1inv', 'vol', 's_r', 's_p', 'zeta', 'kappa', 'x0', 's2'):
            assert getattr(got, key) == getattr(want, key), key
        for key in ('p', 'n', 'c_r'):
            assert all(a == b for a, b in zip(getattr(got, key), getattr(want, key))), key
        assert all(got.e[i][k] == want.e[i][k] for i in range(2) for k in range(3))
        # e[i][k] = dy_k / dt_i: the identity in the first two columns.
        assert got.e[0][0] == 1 and got.e[0][1] == 0 and got.e[1][1] == 1
        assert (got.k0, got.top) == (want.k0, want.top)
        arrays['fr_e'] = arrays['fr_e'][:, :2]
        np.savez(path, **arrays)
        try:
            br.Box.from_npz(np.load(path))
        except ValueError:
            pass
        else:
            raise AssertionError('fr_e of shape (2, 2, 4) passed')
    print('  Box.from_npz: every field equal to the Box of the frame; a wrong shape fails',
          flush=True)


def test_solve_h():
    box = make_box()
    worst, r_h = 0.0, 1e-3
    for j in range(4):
        tau = random_tau(0.02, j % 2 == 0)
        h0 = newton_h(box, tau)
        h = br.solve_h(box, tau, h0, r_h)
        z, _ = br.polar_point(box, tau, [v.conjugate() for v in tau], h, h)
        assert br.quartic(z[:3]).contains(0), br.quartic(z[:3])
        assert rad(h) < 1e-80, rad(h)
        worst = max(worst, rad(h))
        try:
            br.solve_h(box, tau, h0 + 10 * r_h, r_h)
        except ValueError:
            pass
        else:
            raise AssertionError('a start value 10 rad off the root passed')
        print(f'  tau {abs(tau[0]):.4f}, {abs(tau[1]):.4f}: h {complex(h):.6e}', flush=True)
    print(f'  solve_h: P holds 0 at 4 points, largest radius of h {worst:.1e}; a start '
          f'value off by 10 rad raises ValueError', flush=True)


def test_slice_equals_global():
    pot = potential()
    clv.setup(str(DATA), str(FORMS))
    worst_rad, worst_gap = 0.0, 0.0
    for m0, m1 in MODULI:
        z = on_x(cis(m0, math.pi / 8), cis(m1, -math.pi / 8))
        mods = [abs(complex(v)) for v in z]
        p0, nl = br.forms_polar(pot, z, conj(z))
        g0, gnl = clv.global_values(z)
        assert sorted(nl) == sorted(gnl), (sorted(nl), sorted(gnl))
        for name, a, b in [('p0', p0, g0)] + [(f'N_{L}', nl[L], gnl[L]) for L in sorted(nl)]:
            assert a.overlaps(b), (name, a, b)
            assert rel_rad(a) < 1e-60 and rel_rad(b) < 1e-60, (name, rel_rad(a), rel_rad(b))
            worst_rad = max(worst_rad, rel_rad(a), rel_rad(b))
            worst_gap = max(worst_gap, float(abs(a - b).mid()) / float(abs(a).mid()))
        print(f'  rho {min(mods) / max(mods):.3f}: p0 {complex(p0).real:.10e}, N_24 '
              f'{complex(nl[24]).real:.10e}', flush=True)
    print(f'  slice: p0 and {len(nl)} N_L overlap global_values at {len(MODULI)} points; '
          f'largest relative radius {worst_rad:.1e}, gap of the midpoints {worst_gap:.1e}',
          flush=True)


def test_hermitian():
    pot = potential()
    worst = 0.0
    for m0, m1 in MODULI:
        t0, t1 = cis(m0, math.pi / 8), cis(m1, -math.pi / 8)
        d0, d1 = (cis(0.01, float(RNG.uniform(0, 2 * math.pi))) for _ in range(2))
        z, zb = on_x(t0, t1), conj(on_x(t0 + d0, t1 + d1))
        a0, an = br.forms_polar(pot, z, zb)
        b0, bn = br.forms_polar(pot, conj(zb), conj(z))
        # Off the slice the polarized value is not real.
        assert not a0.imag.contains(0), a0
        for name, a, b in [('p0', a0, b0)] + [(f'N_{L}', an[L], bn[L]) for L in sorted(an)]:
            assert a.overlaps(b.conjugate()), (name, a, b)
            assert rel_rad(a) < 1e-60, (name, rel_rad(a))
            worst = max(worst, rel_rad(a), rel_rad(b))
        print(f'  off the slice: p0 {complex(a0):.10e}', flush=True)
    print(f'  Hermitian: F(z, zb) = conj(F(conj zb, conj z)) for p0 and each N_L at '
          f'{len(MODULI)} points; largest relative radius {worst:.1e}', flush=True)


def test_identity_fk():
    pot, box = potential(), make_box()
    tau, sigma = random_tau(0.01, True), random_tau(0.01, False)
    sc = [v.conjugate() for v in sigma]
    h_tau = br.solve_h(box, tau, newton_h(box, tau))
    h_sc = br.solve_h(box, sc, newton_h(box, sc))
    z, zb = br.polar_point(box, tau, sigma, h_tau, h_sc)
    v = br.step_values(pot, box, z, zb)
    keys = (['delta', 'p0', 'r2', 'ir2', 'log_r2', 'pt', 'xh', 'log_pt', 'xw', 'xi']
            + [f'fk{k}' for k in range(pot.top + 1)] + ['psi', 'phi'])
    assert sorted(v) == sorted(keys), sorted(set(v) ^ set(keys))
    xi, psi = v['xi'], v['psi']
    total = sum((v[f'fk{k}'] * xi ** k for k in range(pot.top + 1)), acb(0))
    tail = total - v['fk0']
    assert total.overlaps(psi), (total, psi)
    assert rel_rad(total) < 1e-60 and rel_rad(psi) < 1e-60, (rel_rad(total), rel_rad(psi))
    # The terms of degree >= 1 in xi are far above the radius, so the test sees them.
    assert float(abs(tail).mid()) > 1e20 * rad(total), (tail, rad(total))
    assert not xi.imag.contains(0), xi
    gap = float(abs(total - psi).mid())
    print(f'  identity: xi {complex(xi):.6e}, psi {complex(psi):.15f}, sum over k >= 1 '
          f'{complex(tail):.3e}; gap {gap:.1e}, radius {rad(total):.1e}', flush=True)
    print(f'  phi {complex(v["phi"]):.15f}, log_pt {complex(v["log_pt"]):.6e}', flush=True)


def main():
    t_all = time.perf_counter()
    check_inputs()
    print('inputs: the SHA-256 of f6t_data.npz and f6t_forms.npz agree', flush=True)
    for fn in (test_exact_and_poly, test_principal_branch, test_from_npz, test_solve_h,
               test_slice_equals_global, test_hermitian, test_identity_fk):
        t0 = time.perf_counter()
        fn()
        print(f'{fn.__name__} passes ({time.perf_counter() - t0:.0f} s)', flush=True)
    print(f'all box reference tests pass ({time.perf_counter() - t_all:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
