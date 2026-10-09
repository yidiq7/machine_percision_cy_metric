"""Test of `quartic.ddgemm`: the products against mpmath, inside the a priori bound.

Run on a GPU node: uv run --extra gpu --with cupy-cuda12x python -m tests.test_ddgemm
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quartic import ddgemm                                  # noqa: E402

mp.mp.dps = 50
RNG = np.random.default_rng(5)


def random_dd(shape, scale):
    """Random complex dd entries (..., 4, r, c), with entries that cancel strongly."""
    hi = RNG.normal(size=shape[:-3] + (2,) + shape[-2:]) * scale
    lo = hi * 1e-17 * RNG.normal(size=hi.shape)
    out = np.zeros(shape)
    out[..., 0, :, :], out[..., 2, :, :] = hi[..., 0, :, :], hi[..., 1, :, :]
    out[..., 1, :, :], out[..., 3, :, :] = lo[..., 0, :, :], lo[..., 1, :, :]
    return out


def to_mp(x):
    return mp.mpc(mp.mpf(x[0]) + mp.mpf(x[1]), mp.mpf(x[2]) + mp.mpf(x[3]))


def run(nb, m, k, n, mode, shared):
    a = random_dd((nb, 4, m, k), 1.0)
    bshape = (1 if shared else nb, 4) + ((k, n) if mode == 0 else (n, k))
    b = random_dd(bshape, 1e3)
    c = cp.asnumpy(ddgemm.zgemm(cp.asarray(a), cp.asarray(b), mode))
    g = ddgemm.gamma(k)
    worst = 0.0
    for bi in range(nb):
        bb = b[0 if shared else bi]
        for i in range(0, m, max(1, m // 5)):
            for j in range(0, n, max(1, n // 5)):
                exact = mp.mpc(0)
                absum = 0.0
                for t in range(k):
                    x = to_mp(a[bi, :, i, t])
                    y = to_mp(bb[:, t, j]) if mode == 0 else mp.conj(to_mp(bb[:, j, t]))
                    exact += x * y
                    absum += float((abs(x.real) + abs(x.imag)) * (abs(y.real) + abs(y.imag)))
                got = to_mp(c[bi, :, i, j])
                err = abs(got.real - exact.real) + abs(got.imag - exact.imag)
                assert err <= g * absum, (mode, bi, i, j, float(err), g * absum)
                worst = max(worst, float(err / (g * absum)))
    print(f'  mode {mode}, shared B {shared}, ({m} x {k}) ({k} x {n}): largest error over the '
          f'bound {worst:.2e}', flush=True)


if __name__ == '__main__':
    for mode in (0, 1):
        for shared in (True, False):
            run(3, 37, 53, 29, mode, shared)
    print('\nall ddgemm tests pass')
