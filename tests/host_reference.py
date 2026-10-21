"""Reference values in mpmath for the host numbers of quartic.cert.frame.

The functions compare the frame numbers and kappa with mpmath at 300 digits. They need no
GPU: the module imports only the standard library, numpy, mpmath and quartic.cert.frame.
An error, a bound or a ratio that is not finite raises ValueError.
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quartic.cert import frame as fr                                         # noqa: E402


def _dd_mp(d):
    """The complex mpmath number of a dd list [re_hi, re_lo, im_hi, im_lo]."""
    return mp.mpc(mp.mpf(d[0]) + mp.mpf(d[1]), mp.mpf(d[2]) + mp.mpf(d[3]))


def _td_mp(t):
    """The complex mpmath number of a td list of 6 floats."""
    return mp.mpc(sum(mp.mpf(x) for x in t[:3]), sum(mp.mpf(x) for x in t[3:]))


FRAME_NAMES = ['ha0', 'ha1', 'ha2', 'kappa', 'a6', 'a4', 'vol']


def _ratio(d, bound, where):
    """The error d over its bound; inf if the bound is 0 and d is not."""
    if not (mp.isfinite(d) and mp.isfinite(bound)):
        raise ValueError(f'{where}: error {d} or bound {bound} is not finite')
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
        for ic, c in enumerate(centers):
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
                worst[k] = max(worst[k], _ratio(d, bound[k], f'{k}, center {ic}'))
                if d > 0:
                    nz.add(k)
            u = mp.matrix(3, 3)
            for i in range(3):
                for j, col in enumerate((e[0], e[1], n)):
                    u[i, j] = col[i]
            vref = 1 / (abs(mp.det(u)) ** 2 * vbar)
            d = abs(_dd_mp(f['vol']) - vref)
            worst['vol'] = max(worst['vol'], _ratio(d / vref, f['vol_err'], f'vol, center {ic}'))
            if d > 0:
                nz.add('vol')
            for k in nz:
                count[k] += 1
    return {k: (worst[k], count[k]) for k in FRAME_NAMES}


def kappa_errors(inputs, k0):
    """(largest error over its bound, count of inputs with error > 0) of fr.kappa.

    inputs is a list of (s_dd, x0_dd, zeta_dd); the reference is mpmath at 300 digits.
    """
    worst, count = 0.0, 0
    with mp.workdps(300):
        for i, (s, x0, zeta) in enumerate(inputs):
            got, err = fr.kappa(s, k0, x0, zeta)
            ref = _dd_mp(s).real ** (mp.mpf(1) / k0) / (_dd_mp(x0).real * (1 + _dd_mp(zeta)))
            d = abs(_dd_mp(got) - ref)
            worst = max(worst, _ratio(d, err, f'kappa, input {i}'))
            count += d > 0
    return worst, count


