"""Tests of `tmgpu.tm`: each model must enclose the exact function on its polydisc.

Each test builds models of functions that mpmath evaluates exactly, and checks
abs(f(x) - P(x)) <= R at random points x of the polydisc, also near its edge. The
cases include a sum that cancels by 1e8 and a product after it.

Run on a GPU node: uv run --extra gpu --with cupy-cuda12x python -m tmgpu.tests.test_tm
"""
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmgpu import tm                                     # noqa: E402

mp.mp.dps = 60
RNG = np.random.default_rng(3)


def to_dd(vals):
    """Complex mpmath coefficients [box][coefficient] as a CuPy array (4, S, B)."""
    b, s = len(vals), len(vals[0])
    out = np.zeros((4, s, b))
    for k in range(b):
        for i, z in enumerate(vals[k]):
            re, im = mp.re(z), mp.im(z)
            rh, ih = float(re), float(im)
            out[:, i, k] = [rh, float(re - rh), ih, float(im - ih)]
    return cp.asarray(out)


def to_mp(c, k):
    c = cp.asnumpy(c)
    return [mp.mpc(mp.mpf(float(c[0, i, k])) + mp.mpf(float(c[1, i, k])),
                   mp.mpf(float(c[2, i, k])) + mp.mpf(float(c[3, i, k])))
            for i in range(c.shape[1])]


def evaluate(sp, coef, x):
    total = mp.mpc(0)
    for cv, e in zip(coef, sp.mons):
        if cv == 0:
            continue
        t = cv
        for xv, k in zip(x, e):
            if k:
                t *= xv ** k
        total += t
    return total


def random_points(nvars, delta, count, edge=0.999):
    out = []
    for i in range(count):
        r = delta * (edge if i % 2 == 0 else RNG.random())
        out.append([mp.mpc(r * np.cos(a), r * np.sin(a)) for a in RNG.uniform(0, 2 * np.pi, nvars)])
    return out


def random_poly(sp, b, scale, decay):
    return [[mp.mpc(*(RNG.normal(size=2) * scale * decay ** sum(e))) for e in sp.mons]
            for _ in range(b)]


def model(sp, vals, delta):
    return tm.TM(to_dd(vals), cp.zeros(len(delta)), cp.asarray(delta))


def check(sp, m, exact_fn, delta, name, count=12):
    worst = 0.0
    rem = cp.asnumpy(m.rem)
    for k in range(rem.shape[0]):
        coef = to_mp(m.c, k)
        for x in random_points(sp.nvars, delta[k], count):
            gap = abs(exact_fn(k, x) - evaluate(sp, coef, x))
            assert gap <= rem[k], (name, k, float(gap), rem[k])
            worst = max(worst, float(gap / rem[k]) if rem[k] > 0 else 0.0)
    print(f"  {name}: largest error over remainder {worst:.2e}, remainders "
          f"{rem.min():.2e} to {rem.max():.2e}", flush=True)


def test_product_and_cancellation():
    sp = tm.Space(4, 6)
    delta = np.array([0.05, 0.1, 0.2])
    va, vb = random_poly(sp, 3, 1.0, 1.5), random_poly(sp, 3, 1.0, 1.5)
    a, bb = model(sp, va, delta), model(sp, vb, delta)
    check(sp, tm.mul(sp, a, bb), lambda k, x: evaluate(sp, va[k], x) * evaluate(sp, vb[k], x),
          delta, 'product')
    vc = random_poly(sp, 3, 1e8, 1.5)
    c = model(sp, vc, delta)
    canc = tm.sub(sp, tm.add(sp, a, c), c)
    check(sp, canc, lambda k, x: evaluate(sp, va[k], x), delta, 'cancellation by 1e8')
    big = tm.sub(sp, tm.mul(sp, tm.add(sp, a, c), bb), tm.mul(sp, c, bb))
    check(sp, big, lambda k, x: evaluate(sp, va[k], x) * evaluate(sp, vb[k], x), delta,
          'product after a cancellation by 1e8')
    k = tm.cconst(complex(0.3, -1.7), 3)
    check(sp, tm.add_const(sp, tm.scale(sp, a, k), tm.cconst(complex(2, 1), 3)),
          lambda kk, x: evaluate(sp, va[kk], x) * mp.mpc(0.3, -1.7) + mp.mpc(2, 1), delta,
          'scale and constant')


def test_series():
    sp = tm.Space(4, 10)
    delta = np.array([0.02, 0.05])
    vy = random_poly(sp, 2, 0.05, 2.0)
    vy = [[mp.mpc(0) if i == 0 else v for i, v in enumerate(row)] for row in vy]
    y = model(sp, vy, delta)
    pw = tm.powers(sp, y, sp.n)
    eta = tm.bound(sp, y)
    yv = lambda k, x: evaluate(sp, vy[k], x)
    check(sp, tm.power_series(sp, pw, Fraction(-1, 14), eta),
          lambda k, x: (1 + yv(k, x)) ** (mp.mpf(-1) / 14), delta, '(1 + y)^(-1/14)')
    check(sp, tm.power_series(sp, pw, -20, eta), lambda k, x: (1 + yv(k, x)) ** -20, delta,
          '(1 + y)^-20')
    check(sp, tm.log1p_series(sp, pw, eta), lambda k, x: mp.log(1 + yv(k, x)), delta,
          'log(1 + y)')


def test_cauchy_derivatives():
    sp = tm.Space(4, 10)
    big = np.array([0.03, 0.06])
    small = big / (1 + 1.0 / sp.n) ** 2
    vy = random_poly(sp, 2, 0.05, 2.0)
    vy = [[mp.mpc(0) if i == 0 else v for i, v in enumerate(row)] for row in vy]
    y = model(sp, vy, big)
    f = tm.power_series(sp, tm.powers(sp, y, sp.n), Fraction(-1, 14), tm.bound(sp, y))
    mixed = tm.cauchy_mixed(sp, f, 0, 2, cp.asarray(small))
    first = tm.deriv1(sp, f, 1, cp.asarray(small))
    a = mp.mpf(-1) / 14

    def d(k, x, i):
        h = mp.mpf(10) ** -25
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        return (evaluate(sp, vy[k], xp) - evaluate(sp, vy[k], xm)) / (2 * h)

    def dd2(k, x, i, j):
        h = mp.mpf(10) ** -20
        out = mp.mpc(0)
        for si, sj, w in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            xp = list(x)
            xp[i] += si * h
            xp[j] += sj * h
            out += w * evaluate(sp, vy[k], xp)
        return out / (4 * h * h)

    def exact_mixed(k, x):
        yv = evaluate(sp, vy[k], x)
        return (a * (a - 1) * (1 + yv) ** (a - 2) * d(k, x, 0) * d(k, x, 2)
                + a * (1 + yv) ** (a - 1) * dd2(k, x, 0, 2))

    check(sp, mixed, exact_mixed, small, 'd^2 / ds_0 dr_0 of (1 + y)^(-1/14)', count=6)
    check(sp, first, lambda k, x: a * (1 + evaluate(sp, vy[k], x)) ** (a - 1) * d(k, x, 1),
          small, 'd / ds_1 of (1 + y)^(-1/14)', count=6)


def test_tensor():
    sp4, sp2 = tm.Space(4, 8), tm.Space(2, 10)
    delta = np.array([0.05, 0.1])
    vz, vw = random_poly(sp2, 2, 1.0, 1.5), random_poly(sp2, 2, 1.0, 1.5)
    t = tm.tensor(sp4, sp2, model(sp2, vz, delta), model(sp2, vw, delta))
    check(sp4, t, lambda k, x: evaluate(sp2, vz[k], x[:2]) * evaluate(sp2, vw[k], x[2:]),
          delta, 'tensor product z(s) w(r)', count=6)


def test_kernel_speed():
    for n in (20, 24):
        sp = tm.Space(4, n)
        b = 64
        x = tm.TM(cp.asarray(RNG.normal(size=(4, sp.size, b))), cp.zeros(b), cp.full(b, 0.02))
        tm.mul(sp, x, x)
        cp.cuda.Device().synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(5):
            tm.mul(sp, x, x)
        cp.cuda.Device().synchronize()
        each = (time.perf_counter() - t0) / 5
        print(f"  order {n}: {each / b * 1e3:.3f} ms for each product and box, with the bounds",
              flush=True)


if __name__ == "__main__":
    for fn in (test_product_and_cancellation, test_series, test_cauchy_derivatives, test_tensor,
               test_kernel_speed):
        print(fn.__name__, flush=True)
        fn()
    print("\nall tm tests pass")
