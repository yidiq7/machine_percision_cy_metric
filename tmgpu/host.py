"""Host numbers for the GPU models, on the CPU: Arb balls and exact rationals to dd or td
with a bound of each error, and mpmath numbers to and from dd.

A dd list is [re_hi, re_lo, im_hi, im_lo] and a td list is [re0, re1, re2, im0, im1, im2].
After the rounding, the dd or td number is the exact value that the GPU uses, and the
bound of its distance to the true number goes into a remainder.
"""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp
from flint import acb, arb, ctx

# The default precision of `precision`, in bits.
PREC = 512


class precision:
    """`with precision():` runs ball arithmetic at PREC bits and restores the old setting."""

    def __init__(self, prec=PREC):
        self.prec = prec

    def __enter__(self):
        self.old = ctx.prec
        ctx.prec = self.prec

    def __exit__(self, *exc):
        ctx.prec = self.old


def exact(v):
    """A dd list [re_hi, re_lo, im_hi, im_lo] as an exact acb (inside `precision`)."""
    return acb(arb(v[0]) + arb(v[1]), arb(v[2]) + arb(v[3]))


def upper(x):
    """An upper bound of abs(x) for an arb or acb ball, as a float."""
    a = abs(x)
    return (abs(float(a.mid())) + float(a.rad())) * (1 + 1e-12) + 1e-300


def lower(x):
    """A lower bound of abs(x) for an arb or acb ball, as a float (0 if it may vanish).

    With mid > 2 rad the FP64 round-off of mid - rad is below 3 u of it, which the
    factor 1 - 1e-12 covers; a wider ball gives 0.
    """
    a = abs(x)
    m, r = float(a.mid()), float(a.rad())
    if not m > 2 * r:
        return 0.0
    return (m - r) * (1 - 1e-12)


def to_dd(x):
    """The dd rounding of the midpoint of an acb ball, and a bound of its distance to x."""
    out, err = [], 0.0
    for part in (x.real, x.imag):
        mid = part.mid()
        h = float(mid)
        lo = float(mid - arb(h))
        out += [h, lo]
        err += upper(part - arb(h) - arb(lo))
    return out, err


def to_td(x):
    """The td rounding of the midpoint of an acb ball (6 floats), and a bound of its
    distance to x."""
    out, err = [], 0.0
    for part in (x.real, x.imag):
        mid = part.mid()
        h = float(mid)
        m = float(mid - arb(h))
        lo = float(mid - arb(h) - arb(m))
        out += [h, m, lo]
        err += upper(part - arb(h) - arb(m) - arb(lo))
    return out, err


def mp_dd(z):
    """An mpmath complex as dd parts [re_hi, re_lo, im_hi, im_lo]."""
    z = mp.mpc(z)
    rh, ih = float(z.real), float(z.imag)
    return [rh, float(z.real - rh), ih, float(z.imag - ih)]


def dd_mp(v):
    return mp.mpc(mp.mpf(v[0]) + mp.mpf(v[1]), mp.mpf(v[2]) + mp.mpf(v[3]))


def dd_frac(v):
    """A dd list [re_hi, re_lo, im_hi, im_lo] as the exact rationals (re, im)."""
    return Fraction(float(v[0])) + Fraction(float(v[1])), Fraction(float(v[2])) + Fraction(float(v[3]))


def frac_dd(re, im, limbs=2):
    """The exact complex rational re + i im rounded to dd (limbs 2) or td (limbs 3), as
    [re limbs, im limbs], and an upper bound of the distance (abs re + abs im)."""
    out, err = [], Fraction(0)
    for x in (re, im):
        rest = x
        for _ in range(limbs):
            v = float(rest)
            out.append(v)
            rest -= Fraction(v)
        err += abs(rest)
    return out, float(err) * (1 + 1e-12) + 1e-300
