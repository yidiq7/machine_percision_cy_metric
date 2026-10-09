"""Tests of `quartic.tm_gpu`: each model must enclose the exact function on its polydisc.

Each test builds models of functions that mpmath evaluates exactly, and checks
abs(f(x) - P(x)) <= R at random points x of the polydisc, also at points near its
edge. The cases include a sum that cancels by 1e8.

Run on a GPU node: uv run --extra gpu python -m tests.test_tm_gpu
"""
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax                                                  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                     # noqa: E402
from quartic import dd, tm_gpu as tg                        # noqa: E402

mp.mp.dps = 60
RNG = np.random.default_rng(3)


def coeffs_dd(vals):
    """Complex mpmath coefficients (B, size) as a complex dd array, exactly rounded."""
    vals = np.asarray(vals, dtype=object)
    re_hi = np.vectorize(lambda z: float(mp.re(z)))(vals)
    re_lo = np.vectorize(lambda z, h: float(mp.re(z) - h))(vals, re_hi)
    im_hi = np.vectorize(lambda z: float(mp.im(z)))(vals)
    im_lo = np.vectorize(lambda z, h: float(mp.im(z) - h))(vals, im_hi)
    return ((jnp.asarray(re_hi), jnp.asarray(re_lo)), (jnp.asarray(im_hi), jnp.asarray(im_lo)))


def to_mp(c, b):
    (rh, rl), (ih, il) = jax.tree_util.tree_map(np.asarray, c)
    return [mp.mpc(mp.mpf(float(rh[b, i])) + mp.mpf(float(rl[b, i])),
                   mp.mpf(float(ih[b, i])) + mp.mpf(float(il[b, i]))) for i in range(rh.shape[1])]


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
        out.append([mp.mpc(r * np.cos(a), r * np.sin(a))
                    for a in RNG.uniform(0, 2 * np.pi, nvars)])
    return out


def random_poly(sp, b, scale, decay):
    """Random complex coefficients, of size scale * decay^degree."""
    vals = []
    for _ in range(b):
        row = []
        for e in sp.mons:
            mag = scale * decay ** sum(e)
            row.append(mp.mpc(*(RNG.normal(size=2) * mag)))
        vals.append(row)
    return vals


def check(sp, model, exact_fn, delta, name, count=12):
    """abs(exact - P) <= R at random points, for every box of the batch."""
    worst = 0.0
    rem = np.asarray(model.rem)
    for b in range(rem.shape[0]):
        coef = to_mp(model.c, b)
        for x in random_points(sp.nvars, delta[b], count):
            gap = abs(exact_fn(b, x) - evaluate(sp, coef, x))
            assert gap <= rem[b], (name, b, float(gap), rem[b])
            worst = max(worst, float(gap / rem[b]) if rem[b] > 0 else 0.0)
    print(f"  {name}: largest error over remainder {worst:.2e}, remainders "
          f"{rem.min():.2e} to {rem.max():.2e}")


def test_product_and_cancellation():
    sp = tg.Space(4, 6)
    b = 3
    delta = np.array([0.05, 0.1, 0.2])
    va = random_poly(sp, b, 1.0, 1.5)
    vb = random_poly(sp, b, 1.0, 1.5)
    a = tg.from_coeffs(coeffs_dd(va), jnp.asarray(delta))
    bb = tg.from_coeffs(coeffs_dd(vb), jnp.asarray(delta))
    prod = jax.jit(lambda x, y: tg.mul(sp, x, y))(a, bb)
    check(sp, prod, lambda k, x: evaluate(sp, va[k], x) * evaluate(sp, vb[k], x),
          delta, 'product')
    # A sum that cancels by 1e8: (A + 1e8 C) - 1e8 C, with C of the size of A.
    vc = random_poly(sp, b, 1e8, 1.5)
    c = tg.from_coeffs(coeffs_dd(vc), jnp.asarray(delta))
    canc = jax.jit(lambda x, y: tg.sub(sp, tg.add(sp, x, y), y))(a, c)
    check(sp, canc, lambda k, x: evaluate(sp, va[k], x), delta, 'cancellation by 1e8')
    big = jax.jit(lambda x, y, z: tg.sub(sp, tg.mul(sp, tg.add(sp, x, y), z),
                                         tg.mul(sp, y, z)))(a, c, bb)
    check(sp, big, lambda k, x: evaluate(sp, va[k], x) * evaluate(sp, vb[k], x),
          delta, 'product after a cancellation by 1e8')


def test_series():
    sp = tg.Space(4, 10)
    b = 2
    delta = np.array([0.02, 0.05])
    vy = random_poly(sp, b, 0.05, 2.0)
    vy = [[mp.mpc(0) if i == 0 else v for i, v in enumerate(row)] for row in vy]
    y = tg.from_coeffs(coeffs_dd(vy), jnp.asarray(delta))

    def build(y):
        pw = tg.powers(sp, y, sp.n)
        eta = tg.bound(sp, y)
        return (tg.power_series(sp, pw, Fraction(-1, 14), eta),
                tg.power_series(sp, pw, -20, eta), tg.log1p_series(sp, pw, eta))
    p14, p20, lg = jax.jit(build)(y)
    yv = lambda k, x: evaluate(sp, vy[k], x)
    check(sp, p14, lambda k, x: (1 + yv(k, x)) ** (mp.mpf(-1) / 14), delta, '(1 + y)^(-1/14)')
    check(sp, p20, lambda k, x: (1 + yv(k, x)) ** -20, delta, '(1 + y)^-20')
    check(sp, lg, lambda k, x: mp.log(1 + yv(k, x)), delta, 'log(1 + y)')


def test_cauchy_derivatives():
    sp = tg.Space(4, 10)
    b = 2
    big = np.array([0.03, 0.06])
    small = big / (1 + 1.0 / sp.n) ** 2
    vy = random_poly(sp, b, 0.05, 2.0)
    vy = [[mp.mpc(0) if i == 0 else v for i, v in enumerate(row)] for row in vy]
    y = tg.from_coeffs(coeffs_dd(vy), jnp.asarray(big))

    def build(y):
        pw = tg.powers(sp, y, sp.n)
        f = tg.power_series(sp, pw, Fraction(-1, 14), tg.bound(sp, y))
        return (tg.cauchy_mixed(sp, f, 0, 2, jnp.asarray(small)),
                tg.deriv1(sp, f, 1, jnp.asarray(small)))
    mixed, first = jax.jit(build)(y)
    a = mp.mpf(-1) / 14

    def d(k, x, i):
        h = mp.mpf(10) ** -25
        xp = list(x)
        xp[i] += h
        xm = list(x)
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

    def exact_first(k, x):
        return a * (1 + evaluate(sp, vy[k], x)) ** (a - 1) * d(k, x, 1)
    check(sp, mixed, exact_mixed, small, 'd^2 / ds_0 dr_0 of (1 + y)^(-1/14)', count=6)
    check(sp, first, exact_first, small, 'd / ds_1 of (1 + y)^(-1/14)', count=6)


def test_bidegree_and_tensor():
    sp4, sp2 = tg.Space(4, 8), tg.Space(2, 12)
    b = 2
    delta = np.array([0.05, 0.1])
    q = [[[mp.mpc(*(RNG.normal(size=2) * 1.3 ** -(sum(ea) + sum(eb))))
           for eb in sp2.mons] for ea in sp2.mons] for _ in range(b)]
    qd = coeffs_dd(q)
    model = jax.jit(lambda x, dl: tg.from_bidegree(sp4, sp2, x, dl, jnp.zeros(b)))(
        qd, jnp.asarray(delta))

    def exact_q(k, x):
        total = mp.mpc(0)
        for i, ea in enumerate(sp2.mons):
            for j, eb in enumerate(sp2.mons):
                total += q[k][i][j] * x[0] ** ea[0] * x[1] ** ea[1] * x[2] ** eb[0] * x[3] ** eb[1]
        return total
    check(sp4, model, exact_q, delta, 'bidegree polynomial of degree (12, 12) at order 8',
          count=6)
    sp2n = tg.Space(2, 8)
    vz = random_poly(sp2n, b, 1.0, 1.5)
    vw = random_poly(sp2n, b, 1.0, 1.5)
    z = tg.from_coeffs(coeffs_dd(vz), jnp.asarray(delta))
    w = tg.from_coeffs(coeffs_dd(vw), jnp.asarray(delta))
    t = jax.jit(lambda x, y: tg.tensor(sp4, sp2n, x, y))(z, w)
    check(sp4, t, lambda k, x: evaluate(sp2n, vz[k], x[:2]) * evaluate(sp2n, vw[k], x[2:]),
          delta, 'tensor product z(s) w(r)', count=6)


if __name__ == "__main__":
    for fn in (test_product_and_cancellation, test_series, test_cauchy_derivatives,
               test_bidegree_and_tensor):
        print(fn.__name__)
        fn()
    print("\nall tm_gpu tests pass")
