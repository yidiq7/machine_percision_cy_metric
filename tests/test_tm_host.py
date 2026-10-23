"""Test of the host numbers of `quartic.cert.box` that enter the strict bound.

1. The offsets p_i - c_i and the constant abs(p_2)^2 - s2 of Delta do not depend on the
   precision of mpmath, and lie within their bounds of the exact values.
2. The weights of the re-expansion in zeta (reexpand_weights), in td, lie within their
   bounds of the values in mpmath at 150 digits, and the moduli bound the values.
3. A cell that quartic.cert.cover.may_meet_f drops holds no point of F.
4. quartic.cert.box._real_low is a lower bound of the real part of a model on its polydisc,
   and its last step is at most the exact value. On planted constants where round to nearest
   exceeds the exact value, _real_low stays at or below it.
5. box_f stops on local forms that are not in Delta and td (_check_local).
6. The other directed steps of box.py lie on the correct side of their exact values:
   metric_c_low, _lam_low, _grad_g, _vol_factor and the checks _h_checks of implicit_h_td.
   Each test also has inputs where round to nearest gives the wrong side.

The tests of frac_dd and lower are in tests/test_host.py of the repository yidiq7/tmgpu.

Run on a GPU node: uv run --extra gpu \\
    python -m tests.test_tm_host
"""
import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp
from flint import arb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quartic.cert import box as tc                # noqa: E402
from tmgpu import host, tm                                       # noqa: E402
from quartic.cert import cover as cc                                         # noqa: E402
from tests.host_reference import _dd_mp, frame_errors, kappa_errors     # noqa: E402

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


def _real_low_exact(c, rem, pb, bi):
    """Re c0 - abs(Im c0) - pb - rem of box bi as an exact Fraction, for dd coefficients c
    (4, S, B) and the floats rem and pb."""
    re = Fraction(float(c[0, 0, bi])) + Fraction(float(c[1, 0, bi]))
    im = Fraction(float(c[2, 0, bi])) + Fraction(float(c[3, 0, bi]))
    return re - abs(im) - Fraction(float(pb)) - Fraction(float(rem))


def _poly_part_bound(sp, x, delta):
    """B(x - c0) on delta, the same call as in _real_low."""
    var = x.c.copy()
    var[:, 0, :] = 0
    return cp.asnumpy(tm.poly_bound(sp, tm.TM(var, x.rem, delta), delta))


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
    pb = _poly_part_bound(sp, x, delta)
    for bi in range(b):
        # The last step in exact arithmetic: low <= Re c0 - abs(Im c0) - B(x - c0) - rem.
        assert Fraction(float(low[bi])) <= _real_low_exact(cn, 1e-6, pb[bi], bi), (bi, low[bi])
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


# Constant models (a0, a1, b0, b1, rem) with Re c0 = a0 + a1 and Im c0 = b0 + b1, t = 2^-60.
# Each one fails a _real_low with round to nearest (RN) in place of the directed step named.
_T = 2.0 ** -60
REAL_LOW_PLANTED = (
    (1.0, -_T, 0.0, 0.0, 0.0),    # a0 + a1 must round down: RN(1 - t) = 1 > 1 - t
    (1.0, 0.0, 0.0, 0.0, _T),     # the last difference must round down: RN(1 - t) = 1
    (2.0, 0.0, -1.0, -_T, 0.0),   # abs(b0 + b1) needs the downward sum: abs(RU(-1 - t)) = 1
    (2.0, 0.0, 1.0, _T, 0.0),     # abs(b0 + b1) needs the upward sum: abs(RD(1 + t)) = 1
    (2.0, 0.0, 1.0, 0.0, _T),     # the subtracted sum must round up: RN(1 + 0 + t) = 1
)


def check_real_low_planted():
    """_real_low on the planted constant models is at most the exact value, and the same
    formula in round to nearest exceeds the exact value on each of them."""
    sp = tm.Space(4, 6)
    b = len(REAL_LOW_PLANTED)
    delta = cp.full(b, 0.1)
    c = np.zeros((4, sp.size, b))
    rem = np.zeros(b)
    for bi, (a0, a1, b0, b1, r) in enumerate(REAL_LOW_PLANTED):
        c[:, 0, bi] = a0, a1, b0, b1
        rem[bi] = r
    x = tm.TM(cp.asarray(c), cp.asarray(rem), delta)
    low = cp.asnumpy(tc._real_low(sp, x, delta))
    pb = _poly_part_bound(sp, x, delta)
    assert np.all(pb == 0.0), ('a constant model has B(x - c0) = 0', pb)
    for bi, (a0, a1, b0, b1, r) in enumerate(REAL_LOW_PLANTED):
        exact = _real_low_exact(c, r, pb[bi], bi)
        # Python floats round to nearest: this is _real_low with no directed step.
        nearest = (a0 + a1) - abs(b0 + b1) - float(pb[bi]) - r
        assert Fraction(nearest) > exact, ('the planted case does not see the direction', bi)
        assert Fraction(float(low[bi])) <= exact, ('_real_low is above the exact value', bi,
                                                   low[bi], float(exact))
    print(f'  _real_low: {b} planted constants where round to nearest exceeds the exact value; '
          f'_real_low stays at or below it', flush=True)


def check_old_local_forms():
    for bad in ({'parts': 4, 's2': np.zeros(4)}, {'parts': 6}):
        try:
            tc._check_local(bad)
        except ValueError:
            continue
        raise AssertionError(f'local forms with keys {sorted(bad)} passed the check')
    tc._check_local({'parts': 6, 's2': np.zeros(4)})
    print('  _check_local: local forms that are not in Delta and td fail', flush=True)


def _rand_center():
    """A random dd center with abs(t1) <= abs(t0) <= 1 and abs(1 + t0^4 + t1^4) >= 1e-3."""
    while True:
        r0 = RNG.uniform(0.05, 1.0)
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
          + ', '.join(f'{k} {w:.6f}' for k, (w, _) in res.items())
          + f'; exempt (exact in td): {exempt}; centers with nonzero error: '
          + ', '.join(f'{k} {c}' for k, (_, c) in res.items()), flush=True)


def _rdd(x):
    """A real dd pair: the float x and a low part of about 2^-53 of it."""
    return [float(x), float(x * 2.0 ** -53 * RNG.uniform(-1, 1))]


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
    print(f'  kappa: 64 inputs; largest error over its bound {worst:.6f}; {count} inputs have '
          f'a nonzero error', flush=True)


# The tests of the directed steps take their own generator, so the inputs of the other tests
# stay the same.
DRNG = np.random.default_rng(20261023)


def check_metric_c():
    """metric_c_low(k0) is the largest float below c = 1 / (sqrt(2) pi k0) for k0 = 1 to 64,
    with c from another arb formula, and for some k0 the float nearest to c lies above c."""
    above = []
    with host.precision():
        for k0 in range(1, 65):
            c = 1 / (2 * arb.pi() ** 2 * k0 ** 2).sqrt()
            low = tc.metric_c_low(k0)
            nxt = math.nextafter(low, math.inf)
            assert host.down(c - arb(low)) > 0, ('metric_c_low is not below c', k0, low)
            assert host.down(arb(nxt) - c) > 0, ('a float between metric_c_low and c', k0, low)
            # The midpoint of low and nxt is exact; c above it makes nxt the nearest float.
            if host.down(c - (arb(low) + arb(nxt)) / 2) > 0:
                above.append(k0)
    assert above, 'the float nearest to c is below c for each k0, so no k0 sees the direction'
    print(f'  metric_c_low: below c for k0 = 1 to 64; round to nearest lies above c for '
          f'{len(above)} of them, for example k0 = {above[0]}', flush=True)


def check_lam_low():
    """_lam_low is the largest float at or below det_low / tr_up, 0 without the proof, and the
    quotient in round to nearest lies above the exact one for 1 / 10 and other pairs."""
    pairs = [(1.0, 10.0)] + [(float(DRNG.uniform(0.1, 5.0)), float(DRNG.uniform(1.0, 20.0)))
                             for _ in range(63)]
    det = cp.asarray([p[0] for p in pairs])
    tr = cp.asarray([p[1] for p in pairs])
    low = cp.asnumpy(tc._lam_low(det, cp.ones(len(pairs)), tr))
    above = 0
    for (dl, tu), lo in zip(pairs, low):
        exact = Fraction(dl) / Fraction(tu)
        assert Fraction(float(lo)) <= exact, ('_lam_low above det_low / tr_up', dl, tu, lo)
        assert Fraction(math.nextafter(float(lo), math.inf)) > exact, ('not the largest', dl, tu)
        above += Fraction(dl / tu) > exact
    assert Fraction(1.0 / 10.0) > Fraction(1, 10), 'RN(1 / 10) is not above 1 / 10'
    assert above > 0, 'round to nearest is not above the exact quotient for any pair'
    none = cp.asnumpy(tc._lam_low(cp.asarray([1.0, -1.0, 1.0, math.nan]),
                                  cp.asarray([1.0, 1.0, -1.0, 1.0]), cp.full(4, 2.0)))
    assert none[0] > 0 and np.all(none[1:] == 0.0), none
    print(f'  _lam_low: at or below det_low / tr_up for 64 pairs; round to nearest lies above '
          f'for {above} of them (1 / 10 among them); 0 without the proof', flush=True)


def check_grad_g():
    """_grad_g is at least 2 grad_tau / sqrt(c lam_low) in exact arb and +inf where lam_low is
    not above 0, and the formula in round to nearest lies below the exact value on some rows."""
    c16 = tc.metric_c_low(16)
    rows = [(1.0, 2.0, 1.0)] + [(float(DRNG.uniform(1e-18, 1e-16)), c16,
                                 float(DRNG.uniform(0.5, 5.0))) for _ in range(63)]
    cols = [cp.asarray([x[i] for x in rows]) for i in range(3)]
    got = cp.asnumpy(tc._grad_g(cols[0], cols[1], cols[2]))
    below = 0
    with host.precision():
        for (gt, c, lam), val in zip(rows, got):
            exact = 2 * arb(gt) / (arb(c) * arb(lam)).sqrt()
            assert host.down(arb(float(val)) - exact) >= 0, ('_grad_g below the exact value',
                                                              gt, c, lam, val)
            below += host.up(arb(2 * gt / math.sqrt(c * lam)) - exact) < 0
    assert below > 0, 'round to nearest is not below the exact value for any row'
    edge = cp.asnumpy(tc._grad_g(cp.full(3, 1e-17), c16, cp.asarray([0.0, -1.0, math.nan])))
    assert np.all(np.isposinf(edge)), edge
    print(f'  _grad_g: at or above 2 grad_tau / sqrt(c lam_low) for 64 rows; round to nearest '
          f'lies below for {below} of them; +inf where lam_low is not above 0', flush=True)


def check_vol_factor():
    """_vol_factor is at least e / (1 - e), +inf for e >= 1, and round to nearest lies below
    e / (1 - e) for e = 2^-80, where 1 - e rounds to 1."""
    errs = [2.0 ** -80, 2.0 ** -53, 0.0] + [float(10.0 ** DRNG.uniform(-40, -1))
                                            for _ in range(61)]
    fac = cp.asnumpy(tc._vol_factor(cp.asarray(errs)))
    below = 0
    for e, f in zip(errs, fac):
        exact = Fraction(e) / (1 - Fraction(e))
        assert Fraction(float(f)) >= exact, ('_vol_factor below e / (1 - e)', e, f)
        below += Fraction(e / (1.0 - e)) < exact
    e = 2.0 ** -80
    assert Fraction(e / (1.0 - e)) < Fraction(e) / (1 - Fraction(e)), 'RN(1 - 2^-80) is not 1'
    assert fac[2] == 0.0, fac[2]
    edge = cp.asnumpy(tc._vol_factor(cp.asarray([1.0, 2.0])))
    assert np.all(np.isposinf(edge)), edge
    print(f'  _vol_factor: at or above e / (1 - e) for 64 values; round to nearest lies below '
          f'for {below} of them (2^-80 among them); +inf for e >= 1', flush=True)


def _second_exact(x, nabs, by):
    """The sum over k of 12 nabs_k^2 (by_k + x nabs_k)^2 as an exact Fraction."""
    return sum(12 * n * n * (b + x * n) ** 2 for n, b in zip(nabs, by))


def check_h_checks():
    """_h_checks forms lam and lam_big at or above their exact values, its mask good implies
    the exact checks, and good fails a planted box where bd + lam r <= r holds only in round to
    nearest. On the random boxes round to nearest puts lam below its exact value."""
    t = 2.0 ** -60
    # (r, bd, lam0, iabs, nabs, by, hmax, bh): box 0 must fail, box 1 must pass.
    boxes = [(1.0, 2.0 ** -53 + t, 1.0 - 2.0 ** -53, 1.0, (0.0,) * 3, (0.5,) * 3, 0.0, 0.0),
             (1.0, 0.25, 0.5, 1.0, (0.0,) * 3, (0.5,) * 3, 0.0, 0.0)]
    for _ in range(62):
        r = float(10.0 ** DRNG.uniform(-32, -28))
        boxes.append((r, r * float(DRNG.uniform(0.1, 0.4)), float(DRNG.uniform(0.01, 0.5)),
                      float(DRNG.uniform(0.05, 1.0)),
                      tuple(float(v) for v in DRNG.uniform(0.0, 1.0, 3)),
                      tuple(float(v) for v in DRNG.uniform(0.5, 1.5, 3)),
                      float(DRNG.uniform(0.0, 1e-2)), float(DRNG.uniform(0.0, 1e-2))))
    col = lambda i: cp.asarray([x[i] for x in boxes])
    nabs = [cp.asarray([x[4][k] for x in boxes]) for k in range(3)]
    by = [cp.asarray([x[5][k] for x in boxes]) for k in range(3)]
    lam, good, lam_big = (cp.asnumpy(v) for v in
                          tc._h_checks(col(0), col(1), col(2), col(3), nabs, by, col(6), col(7)))
    below = 0
    for i, (r, bd, lam0, iabs, na, bb, hmax, bh) in enumerate(boxes):
        q = [Fraction(v) for v in (r, bd, lam0, iabs, hmax, bh)]
        nq, bq = [Fraction(v) for v in na], [Fraction(v) for v in bb]
        lam_x = q[2] + q[0] * _second_exact(q[0], nq, bq) * q[3]
        big_x = q[4] + q[5] + q[0]
        lam_big_x = q[2] + big_x * _second_exact(big_x, nq, bq) * q[3]
        assert Fraction(float(lam[i])) >= lam_x, ('lam below its exact value', i)
        assert Fraction(float(lam_big[i])) >= lam_big_x, ('lam_big below its exact value', i)
        if good[i]:
            assert q[2] < 1 and lam_x < 1 and q[1] + lam_x * q[0] <= q[0], ('good, not exact', i)
        rn = lam0 + r * sum(12 * n ** 2 * (b + r * n) ** 2 for n, b in zip(na, bb)) * iabs
        below += Fraction(rn) < lam_x
    r, bd, lam0 = boxes[0][:3]
    assert bd + lam0 * r <= r and Fraction(bd) + Fraction(lam0) * Fraction(r) > Fraction(r), \
        'box 0 does not separate round to nearest from the exact check'
    assert not good[0] and good[1], ('the planted boxes', good[:2])
    assert below > 0, 'round to nearest is not below the exact lam for any box'
    print(f'  _h_checks: lam and lam_big at or above their exact values for 64 boxes; good '
          f'implies the exact checks ({int(good.sum())} good); the planted box fails, where '
          f'round to nearest passes; round to nearest lies below lam for {below} boxes',
          flush=True)


def main():
    check_offsets_and_delta()
    check_weights()
    check_dropped_cells()
    check_real_low()
    check_real_low_planted()
    check_old_local_forms()
    check_frame()
    check_kappa()
    check_metric_c()
    check_lam_low()
    check_grad_g()
    check_vol_factor()
    check_h_checks()
    print('all host tests pass', flush=True)


if __name__ == '__main__':
    main()
