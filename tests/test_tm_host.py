"""Test of the host numbers of `quartic.tm_cert` that enter the strict bound.

1. Exact rationals rounded to dd and td (frac_dd) lie within their bounds.
2. The offsets p_i - c_i and the constant abs(p_2)^2 - s2 of Delta do not depend on the
   precision of mpmath, and lie within their bounds of the exact values.
3. The weights of the re-expansion in zeta (reexpand_weights), in dd and in td, lie within
   their bounds of the values in mpmath at 150 digits, and the moduli bound the values.
4. tm_frame.lower gives a lower bound, and 0 for a wide ball.
5. A cell that cert_cover.may_meet_f drops holds no point of F.
6. tm_cert._real_low is a lower bound of the real part of a model on its polydisc.

Run on a GPU node: uv run --extra gpu --with cupy-cuda12x --with python-flint \\
    python -m tests.test_tm_host
"""
import os
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quartic import tmc, tm_cert as tc, tm_frame                # noqa: E402
import cert_cover as cc                                         # noqa: E402

RNG = np.random.default_rng(20261010)


def random_dd(scale=1.0):
    """A random complex dd list, its low parts about u times the high parts."""
    out = []
    for _ in range(2):
        h = float(RNG.normal() * scale)
        out += [h, float(h * 2.0 ** -53 * RNG.uniform(-1, 1))]
    return out


def check_frac_dd():
    worst = [0.0, 0.0]
    for _ in range(2000):
        re = Fraction(float(RNG.normal())) + Fraction(float(RNG.normal())) * Fraction(1, 2 ** 70)
        im = Fraction(float(RNG.normal())) * Fraction(1, 3)
        for k, limbs in enumerate((2, 3)):
            v, err = tc.frac_dd(re, im, limbs)
            gap = abs(re - sum(Fraction(x) for x in v[:limbs])) + abs(im - sum(Fraction(x) for x in v[limbs:]))
            assert gap <= Fraction(err), (limbs, float(gap), err)
            worst[k] = max(worst[k], float(gap) / err)
    print(f'  frac_dd: largest error over its bound {worst[0]:.2f} (dd), {worst[1]:.2f} (td)',
          flush=True)


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
        outs.append((tc.offset_consts(fr, ref, 2), tc.offset_consts(fr, ref, 3),
                     tc.delta_const(fr, np.array(s2))))
    for a, b in zip(outs[0], outs[1]):
        for x, y in zip(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]):
            assert all(bool(cp.all(u == v)) for u, v in zip(x, y)), 'depends on mpmath precision'
    (off2, off3, (d0, derr)) = outs[1]
    for i in range(2):
        rq = tc.dd_frac(ref[i])
        for limbs, (vals, errs) in ((2, off2[i]), (3, off3[i])):
            vals, errs = cp.asnumpy(vals), cp.asnumpy(errs)
            for bi, box in enumerate(boxes):
                re, im = (a - c for a, c in zip(tc.dd_frac(box[i]), rq))
                got = vals[:, bi]
                gap = abs(re - sum(Fraction(x) for x in got[:limbs])) + abs(im - sum(Fraction(x) for x in got[limbs:]))
                assert gap <= Fraction(errs[bi]), ('offset', i, limbs, float(gap), errs[bi])
    d0, derr = cp.asnumpy(d0), cp.asnumpy(derr)
    s2q = tc.dd_frac(s2)[0]
    for bi, box in enumerate(boxes):
        re, im = tc.dd_frac(box[2])
        exact = re * re + im * im - s2q
        gap = abs(exact - Fraction(d0[0, bi]) - Fraction(d0[1, bi])) + abs(Fraction(d0[2, bi]))
        assert gap <= Fraction(derr[bi]), ('delta', float(gap), derr[bi])
    print('  offsets and the constant of Delta: the same at 15 and at 100 digits, and within '
          'their bounds', flush=True)


def check_weights():
    k_max, top = 24, 24
    zeta = [[float(RNG.normal() * 0.05), float(RNG.normal() * 2.0 ** -58),
             float(RNG.normal() * 0.05), float(RNG.normal() * 2.0 ** -58)] for _ in range(3)]
    mp.mp.dps = 150
    for limbs in (2, 3):
        w, werr, wabs = tc.reexpand_weights(zeta, k_max, top, limbs)
        worst, worst_rel = 0.0, 0.0
        for i, zd in enumerate(zeta):
            z = tc.dd_mp(zd)
            for k in range(k_max + 1):
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
        print(f'  weights ({"dd" if limbs == 2 else "td"}): largest error over its bound '
              f'{worst:.2f}; largest bound over abs(weight) {worst_rel:.1e}', flush=True)
        assert worst_rel < (1e-30 if limbs == 2 else 1e-46)


def check_lower():
    from flint import arb
    with tm_frame.precision():
        assert tm_frame.lower(arb(1, 0.6)) == 0.0
        assert tm_frame.lower(arb(-1, 0.6)) == 0.0
        x = arb(3, 1e-20)
        assert 0 < tm_frame.lower(x) <= 3 - 1e-20
        assert tm_frame.upper(x) >= 3 + 1e-20
    print('  tm_frame.lower: 0 for a wide ball, a lower bound for a narrow one', flush=True)


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
    sp = tmc.Space(4, 6)
    b = 4
    delta = cp.asarray([0.02, 0.05, 0.1, 0.2])
    c = cp.asarray(RNG.normal(size=(4, sp.size, b)) * 0.1)
    c[1] *= 2.0 ** -53
    c[3] *= 2.0 ** -53
    c[0, 0] = 2.0
    x = tmc.TM(c, cp.full(b, 1e-6), delta)
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


def main():
    check_frac_dd()
    check_offsets_and_delta()
    check_weights()
    check_lower()
    check_dropped_cells()
    check_real_low()
    check_old_local_forms()
    print('all host tests pass', flush=True)


if __name__ == '__main__':
    main()
