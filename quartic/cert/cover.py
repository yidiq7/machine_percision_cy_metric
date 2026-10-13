"""The cells of the cover of the region F, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). A cell is a cube of side s
in (Re t_0, Im t_0, Re t_1, Im t_1) with the exact corner idx s. `may_meet_f` is False
only if interval arithmetic (mpmath.iv) proves that the cell misses F. `box_radius`
proves, in ball arithmetic, the radius delta and the bound hmax of the box at the center
of a cell in the frame of `quartic.cert.frame`.
"""
import math

import mpmath as mp
from mpmath import iv

from tmgpu.host import lower, precision, upper

from .frame import frame

# The corners and centers are short binary numbers, exact at 30 digits; a caller with a
# higher precision keeps it.
mp.mp.dps = max(mp.mp.dps, 30)
iv.dps = 30


def cpow4(x, y):
    """Re and Im of (x + i y)^4 for real intervals x, y."""
    x2, y2 = x * x, y * y
    return (x2 - y2) * (x2 - y2) - 4 * x2 * y2, 4 * x * y * (x2 - y2)


def may_meet_f(lo, s):
    """False only if the cell [lo, lo + s]^4 proves to miss F."""
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

    In the frame of `quartic.cert.frame.frame` a point of X over the cell is p + E tau + h n
    with tau_i = (t_i - c_i) - h n_i and h = (z2(t) - p_2 - d . (t - c)) / den, d = E_2 and
    den = n_2 - d_0 n_0 - d_1 n_1. The numerator is at most half the bound of the second
    derivatives of z2 over the cell times r^2, plus abs(z2(c) - p_2) and
    abs(d(c) - d_i) r, with d(c) the exact derivative. All bounds come from arb balls.
    """
    from flint import acb, arb
    f = frame(c_dd, None)
    A = f['acb']
    with precision():
        p, n, y2 = A['p'], A['n'], A['y2']
        d = [A['e'][0][2], A['e'][1][2]]
        den = lower(n[2] - d[0] * n[0] - d[1] * n[1])
        if den <= 0:
            return float('inf'), float('inf')
        t = [acb(_ball(lo[0], s), _ball(lo[1], s)), acb(_ball(lo[2], s), _ball(lo[3], s))]
        # abs(t_i - c_i) <= r on the cell; abs(z2) >= zmin and abs(t_i) <= tmax_i there.
        r = upper(_to_arb(s) * arb(2).sqrt() / 2)
        # abs(z2)^8 = abs(1 + t_0^4 + t_1^4)^2 from real intervals (mpmath.iv), which is
        # sharper than the complex ball of the cell.
        a0, b0, a1, b1 = (iv.mpf([lo[k], lo[k] + s]) for k in range(4))
        p0, q0 = cpow4(a0, b0)
        p1, q1 = cpow4(a1, b1)
        w2 = (1 + p0 + p1) ** 2 + (q0 + q1) ** 2
        if not w2.a > 0:
            return float('inf'), float('inf')
        zmin = lower((_to_arb(w2.a).log() / 8).exp())
        tmax = [upper(t[i]) for i in range(2)]
        # Second derivatives of z2: -3 t_i^2 delta_ij / z2^3 - 3 t_i^3 t_j^3 / z2^7.
        hs = 0.0
        for i in range(2):
            for j in range(2):
                v = 3 * tmax[i] ** 3 * tmax[j] ** 3 / zmin ** 7
                if i == j:
                    v += 3 * tmax[i] ** 2 / zmin ** 3
                hs += v
        exact_d = [-p[i] ** 3 / y2 ** 3 for i in range(2)]
        num = (hs * (1 + 1e-12) * r * r / 2 + upper(y2 - p[2])
               + sum(upper(exact_d[i] - d[i]) for i in range(2)) * r) * (1 + 1e-12)
        hmax = num / den * (1 + 1e-12)
        delta = max(r + hmax * upper(n[0]), r + hmax * upper(n[1])) * (1 + 1e-12)
    # Python max drops a NaN in its second argument, so test both numbers here.
    if not (math.isfinite(delta) and math.isfinite(hmax)):
        return float('inf'), float('inf')
    return delta, hmax


def dd(x):
    x = mp.mpc(x)
    rh, ih = float(x.real), float(x.imag)
    return [rh, float(x.real - rh), ih, float(x.imag - ih)]


def cell(idx, s):
    """The exact lower corner idx s and the exact dd center of a cell."""
    lo = [mp.mpf(int(i)) * s for i in idx]
    c = (mp.mpc(lo[0] + s / 2, lo[1] + s / 2), mp.mpc(lo[2] + s / 2, lo[3] + s / 2))
    return lo, [dd(c[0]), dd(c[1])]
