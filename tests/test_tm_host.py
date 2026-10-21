"""Test of the host numbers of `quartic.cert.box` that enter the strict bound.

1. The offsets p_i - c_i and the constant abs(p_2)^2 - s2 of Delta do not depend on the
   precision of mpmath, and lie within their bounds of the exact values.
2. The weights of the re-expansion in zeta (reexpand_weights), in td, lie within their
   bounds of the values in mpmath at 150 digits, and the moduli bound the values.
3. A cell that quartic.cert.cover.may_meet_f drops holds no point of F.
4. quartic.cert.box._real_low is a lower bound of the real part of a model on its polydisc.
5. box_f stops on local forms that are not in Delta and td (_check_local).

The tests of frac_dd and lower are in tests/test_host.py of the repository yidiq7/tmgpu.

Run on a GPU node: uv run --extra gpu \\
    python -m tests.test_tm_host
"""
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quartic.cert import box as tc                # noqa: E402
from tmgpu import host, tm                                       # noqa: E402
from quartic.cert import cover as cc                                         # noqa: E402
from quartic.cert import frame as fr                                         # noqa: E402

RNG = np.random.default_rng(20261010)


def random_dd(scale=1.0):
    """A random complex dd list, its low parts about u times the high parts."""
    out = []
    for _ in range(2):
        h = float(RNG.normal() * scale)
        out += [h, float(h * 2.0 ** -53 * RNG.uniform(-1, 1))]
    return out


class _Frames:
    def __init__(self, p_dd):
        self.host = [dict(p_dd=p) for p in p_dd]


def check_offsets_and_delta():
    boxes = [[random_dd(0.5), random_dd(0.5), random_dd(0.8)] for _ in range(64)]
    fr = _Frames(boxes)
    ref = [random_dd(0.5), random_dd(0.5)]
    s2 = [float(RNG.uniform(0.3, 0.9)), float(RNG.uniform(-1, 1) * 2.0 ** -54), 0.0, 0.0]
    outs = []
    for dps in (15, 100):
        mp.mp.dps = dps
        outs.append((tc.offset_consts(fr, ref), tc.delta_const(fr, np.array(s2))))
    for a, b in zip(outs[0], outs[1]):
        for x, y in zip(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]):
            assert all(bool(cp.all(u == v)) for u, v in zip(x, y)), 'depends on mpmath precision'
    (off3, (d0, derr)) = outs[1]
    for i in range(2):
        rq = host.dd_frac(ref[i])
        for limbs, (vals, errs) in ((3, off3[i]),):
            vals, errs = cp.asnumpy(vals), cp.asnumpy(errs)
            for bi, box in enumerate(boxes):
                re, im = (a - c for a, c in zip(host.dd_frac(box[i]), rq))
                got = vals[:, bi]
                gap = abs(re - sum(Fraction(x) for x in got[:limbs])) + abs(im - sum(Fraction(x) for x in got[limbs:]))
                assert gap <= Fraction(errs[bi]), ('offset', i, limbs, float(gap), errs[bi])
    d0, derr = cp.asnumpy(d0), cp.asnumpy(derr)
    s2q = host.dd_frac(s2)[0]
    for bi, box in enumerate(boxes):
        re, im = host.dd_frac(box[2])
        exact = re * re + im * im - s2q
        gap = abs(exact - Fraction(d0[0, bi]) - Fraction(d0[1, bi])) + abs(Fraction(d0[2, bi]))
        assert gap <= Fraction(derr[bi]), ('delta', float(gap), derr[bi])
    print('  offsets and the constant of Delta: the same at 15 and at 100 digits, and within '
          'their bounds', flush=True)


def check_weights():
    top = 24
    zeta = [[float(RNG.normal() * 0.05), float(RNG.normal() * 2.0 ** -58),
             float(RNG.normal() * 0.05), float(RNG.normal() * 2.0 ** -58)] for _ in range(3)]
    mp.mp.dps = 150
    for limbs in (3,):
        w, werr, wabs = tc.reexpand_weights(zeta, top)
        worst, worst_rel = 0.0, 0.0
        for i, zd in enumerate(zeta):
            z = host.dd_mp(zd)
            for k in range(top + 1):
                for m in range(k, top + 1):
                    v = (1 + z) ** k * mp.binomial(m, k) * z ** (m - k)
                    got = w[i, :, k, m]
                    gr = sum(mp.mpf(x) for x in got[:limbs])
                    gi = sum(mp.mpf(x) for x in got[limbs:])
                    gap = abs(v.real - gr) + abs(v.imag - gi)
                    assert gap <= werr[i, k, m], (limbs, k, m, float(gap), werr[i, k, m])
                    assert abs(v) <= wabs[i, k, m], (limbs, k, m)
                    worst = max(worst, float(gap / werr[i, k, m]))
                    worst_rel = max(worst_rel, werr[i, k, m] / float(abs(v)))
        print(f'  weights (td): largest error over its bound {worst:.2f}; largest bound over '
              f'abs(weight) {worst_rel:.1e}', flush=True)
        assert worst_rel < 1e-46


def in_f(t0, t1):
    z2 = (-(1 + t0 ** 4 + t1 ** 4)) ** (mp.mpf(1) / 4)
    if not (abs(t1) <= abs(t0) <= abs(z2) <= 1):
        return False
    return 0 <= mp.arg(t0) <= mp.pi / 4 and 0 <= mp.arg(t1) < mp.pi / 2


def check_dropped_cells():
    mp.mp.dps = 30
    s = mp.mpf(0.03)
    grid = [34, 24, 34, 34]
    dropped, kept = 0, 0
    while dropped < 300:
        idx = tuple(int(RNG.integers(g)) for g in grid)
        lo, _ = cc.cell(idx, s)
        if cc.may_meet_f(lo, s):
            kept += 1
            continue
        dropped += 1
        for _ in range(40):
            x = [lo[k] + s * mp.mpf(RNG.uniform()) for k in range(4)]
            assert not in_f(mp.mpc(x[0], x[1]), mp.mpc(x[2], x[3])), ('a dropped cell meets F', idx)
    print(f'  may_meet_f: {dropped} dropped cells hold no point of F in 40 samples each '
          f'({kept} kept cells on the way)', flush=True)


def check_real_low():
    sp = tm.Space(4, 6)
    b = 4
    delta = cp.asarray([0.02, 0.05, 0.1, 0.2])
    c = cp.asarray(RNG.normal(size=(4, sp.size, b)) * 0.1)
    c[1] *= 2.0 ** -53
    c[3] *= 2.0 ** -53
    c[0, 0] = 2.0
    x = tm.TM(c, cp.full(b, 1e-6), delta)
    low = cp.asnumpy(tc._real_low(sp, x, delta))
    cn = cp.asnumpy(c)
    for bi in range(b):
        d = float(delta[bi])
        for _ in range(200):
            pt = [d * complex(*RNG.uniform(-0.7, 0.7, 2)) for _ in range(4)]
            val = 0j
            for i, e in enumerate(sp.mons):
                v = complex(cn[0, i, bi] + cn[1, i, bi], cn[2, i, bi] + cn[3, i, bi])
                for xv, kk in zip(pt, e):
                    v *= xv ** kk
                val += v
            assert val.real - 1e-6 >= low[bi], (bi, val.real, low[bi])
    print(f'  _real_low: below Re(model) - rem at 800 points; bounds {low.round(4)}', flush=True)


def check_old_local_forms():
    for bad in ({'parts': 4, 's2': np.zeros(4)}, {'parts': 6}):
        try:
            tc._check_local(bad)
        except ValueError:
            continue
        raise AssertionError(f'local forms with keys {sorted(bad)} passed the check')
    tc._check_local({'parts': 6, 's2': np.zeros(4)})
    print('  _check_local: local forms that are not in Delta and td fail', flush=True)


def _dd_mp(d):
    """The complex mpmath number of a dd list [re_hi, re_lo, im_hi, im_lo]."""
    return mp.mpc(mp.mpf(d[0]) + mp.mpf(d[1]), mp.mpf(d[2]) + mp.mpf(d[3]))


def _td_mp(t):
    """The complex mpmath number of a td list of 6 floats."""
    return mp.mpc(sum(mp.mpf(x) for x in t[:3]), sum(mp.mpf(x) for x in t[3:]))


def _rand_center():
    """A random dd center with abs(t1) <= abs(t0) <= 1 and abs(1 + t0^4 + t1^4) >= 1e-3."""
    while True:
        r0, r1 = RNG.uniform(0.05, 1.0), None
        r1 = RNG.uniform(0.0, r0)
        t = []
        for r in (r0, r1):
            a = float(RNG.uniform(0, 2 * np.pi))
            c = [r * np.cos(a), r * np.sin(a)]
            t.append([c[0], float(c[0] * 2.0 ** -53 * RNG.uniform(-1, 1)),
                      c[1], float(c[1] * 2.0 ** -53 * RNG.uniform(-1, 1))])
        with mp.workdps(60):
            if abs(1 + _dd_mp(t[0]) ** 4 + _dd_mp(t[1]) ** 4) >= 1e-3:
                return t


FRAME_NAMES = ['ha0', 'ha1', 'ha2', 'kappa', 'a6', 'a4', 'vol']


def _ratio(d, bound):
    """The error d over its bound; inf if the bound is 0 and d is not."""
    if bound > 0:
        return float(d / bound)
    return 0.0 if d == 0 else float('inf')


def frame_errors(centers, k0):
    """For each frame number, (largest error over its bound, count of centers with error > 0).

    The errors are against mpmath at 300 digits, with the dd numbers of the frame as exact.
    """
    worst = {n: 0.0 for n in FRAME_NAMES}
    count = {n: 0 for n in FRAME_NAMES}
    with mp.workdps(300):
        vbar = 512 * mp.pi ** 4 * k0 ** 2 / mp.gamma(mp.mpf(1) / 4) ** 8
        for c in centers:
            f = fr.frame(c, k0)
            p = [_dd_mp(x) for x in f['p']]
            e = [[mp.mpc(1), mp.mpc(0), _dd_mp(f['e'][0][2])],
                 [mp.mpc(0), mp.mpc(1), _dd_mp(f['e'][1][2])]]
            n = [_dd_mp(x) for x in f['n']]
            c1inv = _dd_mp(f['c1inv'])
            grad = [4 * x ** 3 for x in p]

            def dot(u, v):
                return sum(a * b for a, b in zip(u, v))

            ref = {'ha0': c1inv * (1 + sum(x ** 4 for x in p)),
                   'ha1': c1inv * dot(grad, e[0]), 'ha2': c1inv * dot(grad, e[1]),
                   'kappa': 1 - c1inv * dot(grad, n)}
            got = {'ha0': f['ha_td'][0], 'ha1': f['ha_td'][1], 'ha2': f['ha_td'][2],
                   'kappa': f['kappa_td']}
            bound = dict(zip(['ha0', 'ha1', 'ha2', 'kappa', 'a6', 'a4'], f['h_err_td']))
            pairs = [(k, _td_mp(got[k]), ref[k]) for k in got]
            pairs += [('a6', _td_mp(f['a6_td'][i]), 6 * p[i] ** 2) for i in range(3)]
            pairs += [('a4', _td_mp(f['a4_td'][i]), 4 * p[i]) for i in range(3)]
            nz = set()
            for k, v, r in pairs:
                d = abs(v - r)
                worst[k] = max(worst[k], _ratio(d, bound[k]))
                if d > 0:
                    nz.add(k)
            u = mp.matrix(3, 3)
            for i in range(3):
                for j, col in enumerate((e[0], e[1], n)):
                    u[i, j] = col[i]
            vref = 1 / (abs(mp.det(u)) ** 2 * vbar)
            d = abs(_dd_mp(f['vol']) - vref)
            worst['vol'] = max(worst['vol'], _ratio(d / vref, f['vol_err']))
            if d > 0:
                nz.add('vol')
            for k in nz:
                count[k] += 1
    return {k: (worst[k], count[k]) for k in FRAME_NAMES}


def check_frame():
    cov = np.load(ROOT / 'runs_tm' / 'cover_s0.03_c6_v2.npz')
    centers = [_rand_center() for _ in range(64)]
    centers += [[[float(x) for x in c[0]], [float(x) for x in c[1]]] for c in cov['ref_centers']]
    res = frame_errors(centers, 16)
    exempt = ['a4']
    bad = [k for k, (w, _) in res.items() if not w <= 1.0]
    assert not bad, f'frame: error above its bound for {bad}: {res}'
    zero = [k for k, (_, c) in res.items() if c == 0 and k not in exempt]
    assert not zero, f'frame: the numbers {zero} have error 0 at all {len(centers)} centers, ' \
                     f'so the check cannot see an error there'
    print(f'  frame: {len(centers)} centers; largest error over its bound '
          + ', '.join(f'{k} {w:.2f}' for k, (w, _) in res.items())
          + f'; exempt (exact in td): {exempt}; centers with nonzero error: '
          + ', '.join(f'{k} {c}' for k, (_, c) in res.items()), flush=True)


def _rdd(x):
    """A real dd pair: the float x and a low part of about 2^-53 of it."""
    return [float(x), float(x * 2.0 ** -53 * RNG.uniform(-1, 1))]


def kappa_errors(inputs, k0):
    """(largest error over its bound, count of inputs with error > 0) of fr.kappa.

    inputs is a list of (s_dd, x0_dd, zeta_dd); the reference is mpmath at 300 digits.
    """
    worst, count = 0.0, 0
    with mp.workdps(300):
        for s, x0, zeta in inputs:
            got, err = fr.kappa(s, k0, x0, zeta)
            ref = _dd_mp(s).real ** (mp.mpf(1) / k0) / (_dd_mp(x0).real * (1 + _dd_mp(zeta)))
            d = abs(_dd_mp(got) - ref)
            worst = max(worst, _ratio(d, err))
            count += d > 0
    return worst, count


def check_kappa():
    inputs = []
    for _ in range(64):
        r, a = RNG.uniform(0, 1e-3 * 0.999), RNG.uniform(0, 2 * np.pi)
        inputs.append((_rdd(10.0 ** RNG.uniform(-10, 10)) + [0.0, 0.0],
                       _rdd(RNG.uniform(0.1, 10)) + [0.0, 0.0],
                       _rdd(r * np.cos(a)) + _rdd(r * np.sin(a))))
    worst, count = kappa_errors(inputs, 16)
    assert worst <= 1.0, ('kappa', worst)
    assert count > 0, 'kappa: error 0 at all 64 inputs, so the check cannot see an error'
    print(f'  kappa: 64 inputs; largest error over its bound {worst:.2f}; {count} inputs have '
          f'a nonzero error', flush=True)


def main():
    check_offsets_and_delta()
    check_weights()
    check_dropped_cells()
    check_real_low()
    check_old_local_forms()
    check_frame()
    check_kappa()
    print('all host tests pass', flush=True)


if __name__ == '__main__':
    main()
