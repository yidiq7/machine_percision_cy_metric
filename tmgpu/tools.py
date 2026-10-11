"""Helpers on batches of Taylor models of `tmgpu.tm`: one Space for each size, the bound
of a model on its box, the order of a series, cached powers, the conjugate model, and CuPy
dd arrays to and from mpmath."""
from __future__ import annotations

import math

import numpy as np
import cupy as cp

from . import tm
from .host import dd_mp, mp_dd

_SPACES = {}


def space(nvars, n):
    """One Space for each (nvars, n), so that its tables are built once."""
    if (nvars, n) not in _SPACES:
        _SPACES[nvars, n] = tm.Space(nvars, n)
    return _SPACES[nvars, n]


def sup_bound(sp4, f, d):
    """abs(f) <= abs(f(c)) + B(f - f(c)) on the box delta d + the remainder of f."""
    c0 = tm.abs_ub(f.c[:, :1, :])[0]
    var = f.c.copy()
    var[:, 0, :] = 0
    pb = tm.poly_bound(sp4, tm.TM(var, f.rem, d), d)
    return (c0 + pb + f.rem) * tm.SAFE + tm.TINY, c0, pb


def series_order(eta, n, eps):
    """The number of terms with eta^(k+1) below eps for the largest eta of the batch: at
    least 4 and at most n, and n if the largest eta is not in (0, 0.5)."""
    e = float(cp.max(eta))
    if not 0 < e < 0.5:
        return n
    return int(min(n, max(4, math.ceil(math.log(eps) / math.log(e)) - 1)))


def power(sp, x, m, cache):
    """x^m by squares; cache is a dict that holds {1: x} and keeps every power."""
    if m not in cache:
        half = power(sp, x, m // 2, cache)
        cache[m] = (tm.mul(sp, half, half) if m % 2 == 0
                    else tm.mul(sp, power(sp, x, m - 1, cache), x))
    return cache[m]


def conj(x):
    """The model with conjugate coefficients: for a model of g(tau), the model of conj(g)
    as a function of sigma = conj(tau), on the same monomials."""
    c = x.c.copy()
    c[2:] = -c[2:]
    return tm.TM(c, x.rem.copy(), x.delta)


def host_dd(x):
    """A CuPy dd array (4, B) as a list of mpmath complex numbers."""
    h = cp.asnumpy(x)
    return [dd_mp(h[:, i]) for i in range(h.shape[1])]


def gpu_dd(vals):
    """A list of mpmath complex numbers as a CuPy dd array (4, B)."""
    return cp.asarray(np.array([mp_dd(v) for v in vals]).T)
