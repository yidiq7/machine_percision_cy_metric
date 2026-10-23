"""The cells of the cover of the region F, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). A cell is a cube of side s
in (Re t_0, Im t_0, Re t_1, Im t_1) with the exact corner idx s. `may_meet_f` is False
only if interval arithmetic (mpmath.iv) proves that the cell misses F. `box_radius`
proves, in ball arithmetic, the radius delta and the bound hmax of the box at the center
of a cell in the frame of `quartic.cert.frame`.
"""
import math
from fractions import Fraction

import mpmath as mp
from mpmath import iv

from tmgpu.host import lower, mp_dd, precision, up, upper

from .frame import frame

# The corners and centers are short binary numbers, exact at 30 digits; a caller with a
# higher precision keeps it.
mp.mp.dps = max(mp.mp.dps, 30)
iv.dps = 30


def _exact_corners():
    """Raise ValueError if mpmath has too few bits for exact corners k s and their sums."""
    if mp.mp.prec < 100:
        raise ValueError(f'the exact corners need mp.prec >= 100, not {mp.mp.prec}')


def cpow4(x, y):
    """Re and Im of (x + i y)^4 for real intervals x, y."""
    x2, y2 = x * x, y * y
    return (x2 - y2) * (x2 - y2) - 4 * x2 * y2, 4 * x * y * (x2 - y2)


def may_meet_f(lo, s):
    """False only if the cell [lo, lo + s]^4 proves to miss F."""
    _exact_corners()
    a0, b0, a1, b1 = (iv.mpf([lo[k], lo[k] + s]) for k in range(4))
    if b0.a > a0.b:                        # arg t_0 > pi/4 on the whole cell: Im t_0 > Re t_0
        return False
    if a0.b < 0 or b0.b < 0 or a1.b < 0 or b1.b < 0:
        return False
    r0, r1 = a0 * a0 + b0 * b0, a1 * a1 + b1 * b1
    if r1.a > r0.b:                        # abs(t_1) > abs(t_0)
        return False
    p0, q0 = cpow4(a0, b0)
    p1, q1 = cpow4(a1, b1)
    w2 = (1 + p0 + p1) ** 2 + (q0 + q1) ** 2   # abs(z_2)^8
    if w2.a > 1:                           # abs(z_2) > 1
        return False
    if (r0 ** 4).a > w2.b:                 # abs(t_0)^8 > abs(z_2)^8, in interval arithmetic
        return False
    return True


def _to_arb(x):
    """An mpmath mpf as an exact arb."""
    from flint import arb
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = arb(int(man)) * arb(2) ** int(exp)
    return -v if sign else v


def _ball(lo, s):
    """The arb ball of the interval [lo, lo + s] for exact mpmath numbers lo and s."""
    from flint import arb
    return arb(_to_arb(lo + s / 2), _to_arb(s / 2))


def box_radius(lo, s, c_dd):
    """(delta, hmax) for the cell [lo, lo + s]^4 (exact mpmath numbers) and the box at c_dd.

    In the frame of `quartic.cert.frame.frame`, a point of X over the cell is p + E tau + h n.
    Here tau_i = (t_i - c_i) - h n_i and h = (z2(t) - p_2 - d . (t - c)) / den, with d = E_2
    and den = n_2 - d_0 n_0 - d_1 n_1. The numerator is at most hs r^2 / 2 + abs(z2(c) - p_2) +
    sum over i of abs(d(c)_i - d_i) r. Here hs bounds the second derivatives of z2 over the
    cell, and d(c) is the exact derivative. `_radius_inputs` gives the directed inputs, and
    `_radius_from` forms the two results.
    """
    return _radius_from(*_radius_inputs(lo, s, c_dd))


def _radius_inputs(lo, s, c_dd):
    """The inputs (den, r, zmin, tmax, e_z, e_d, nmax) of `_radius_from` as floats.

    Each one is the directed float of an endpoint of an arb ball. den and zmin are lower
    bounds, and the others are upper bounds. zmin is 0 when the interval lower bound of abs(z2)^8
    on the cell is not above 0.
    """
    from flint import acb, arb
    _exact_corners()
    f = frame(c_dd, None)
    A = f['acb']
    with precision():
        p, n, y2 = A['p'], A['n'], A['y2']
        d = [A['e'][0][2], A['e'][1][2]]
        den = lower(n[2] - d[0] * n[0] - d[1] * n[1])
        t = [acb(_ball(lo[0], s), _ball(lo[1], s)), acb(_ball(lo[2], s), _ball(lo[3], s))]
        # abs(t_i - c_i) <= r on the cell; abs(z2) >= zmin and abs(t_i) <= tmax_i there.
        r = upper(_to_arb(s) * arb(2).sqrt() / 2)
        # abs(z2)^8 = abs(1 + t_0^4 + t_1^4)^2 from real intervals (mpmath.iv), which is
        # sharper than the complex ball of the cell.
        a0, b0, a1, b1 = (iv.mpf([lo[k], lo[k] + s]) for k in range(4))
        p0, q0 = cpow4(a0, b0)
        p1, q1 = cpow4(a1, b1)
        w2 = (1 + p0 + p1) ** 2 + (q0 + q1) ** 2
        zmin = lower((_to_arb(w2.a).log() / 8).exp()) if w2.a > 0 else 0.0
        tmax = [upper(t[i]) for i in range(2)]
        exact_d = [-p[i] ** 3 / y2 ** 3 for i in range(2)]
        e_z = upper(y2 - p[2])
        e_d = [upper(exact_d[i] - d[i]) for i in range(2)]
        nmax = [upper(n[0]), upper(n[1])]
    return den, r, zmin, tmax, e_z, e_d, nmax


def _radius_from(den, r, zmin, tmax, e_z, e_d, nmax):
    """(delta, hmax) from the directed inputs of `_radius_inputs`, in exact rationals.

    hmax and delta each take one upward step to a float. The result is (inf, inf) if den or
    zmin is not above 0 (NaN included), or if an input or a result is not finite.
    """
    if not (den > 0 and zmin > 0):
        return float('inf'), float('inf')
    if not all(math.isfinite(x) for x in [den, r, zmin, *tmax, e_z, *e_d, *nmax]):
        return float('inf'), float('inf')
    # The bounds increase with r, tmax, e_z, e_d and nmax, and decrease with zmin and den.
    rq, zq, dq = Fraction(r), Fraction(zmin), Fraction(den)
    tq = [Fraction(x) for x in tmax]
    # Second derivatives of z2: -3 t_i^2 delta_ij / z2^3 - 3 t_i^3 t_j^3 / z2^7.
    hs = Fraction(0)
    for i in range(2):
        for j in range(2):
            v = 3 * tq[i] ** 3 * tq[j] ** 3 / zq ** 7
            if i == j:
                v += 3 * tq[i] ** 2 / zq ** 3
            hs += v
    num = hs * rq * rq / 2 + Fraction(e_z) + sum(Fraction(x) for x in e_d) * rq
    hq = num / dq
    hmax = up(hq)
    delta = up(max(rq + hq * Fraction(nmax[0]), rq + hq * Fraction(nmax[1])))
    # up gives inf for a value above the largest float.
    if not (math.isfinite(delta) and math.isfinite(hmax)):
        return float('inf'), float('inf')
    return delta, hmax


def grid(s):
    """The number of cells of side s along each axis of the box around F: Re t_0 in [0, 1],
    Im t_0 in [0, 1 / sqrt 2], Re t_1 and Im t_1 in [0, 1]."""
    return [math.ceil(1 / s), math.ceil(0.7072 / s), math.ceil(1 / s), math.ceil(1 / s)]


def center(lo, s):
    """The exact dd center of the cell [lo, lo + s]^4 (exact mpmath numbers)."""
    _exact_corners()
    return [mp_dd(mp.mpc(lo[0] + s / 2, lo[1] + s / 2)), mp_dd(mp.mpc(lo[2] + s / 2, lo[3] + s / 2))]


def cell(idx, s):
    """The exact lower corner idx s and the exact dd center of a cell."""
    _exact_corners()
    lo = [mp.mpf(int(i)) * s for i in idx]
    return lo, center(lo, s)
