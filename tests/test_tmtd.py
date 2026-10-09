"""Test of `quartic.tmtd` and `ddgemm.zgemm_tt` against mpmath: every coefficient of a
product, a sum, a scaled model and every GEMM entry lies within its running error bound.

Run on a GPU node: uv run --extra gpu --with cupy-cuda12x python -m tests.test_tmtd
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quartic import tmc, tmtd, ddgemm                       # noqa: E402

mp.mp.dps = 80
RNG = np.random.default_rng(11)
U = 2.0 ** -53


def random_td(shape, scale=1.0):
    """Random complex td limbs (6, ...): each lower limb about u times the one above."""
    out = np.zeros((6,) + shape)
    for part in (0, 3):
        hi = RNG.normal(size=shape) * scale
        mid = hi * U * RNG.uniform(-1, 1, size=shape)
        lo = mid * U * RNG.uniform(-1, 1, size=shape)
        out[part], out[part + 1], out[part + 2] = hi, mid, lo
    return out


def to_mp(v):
    """A td number (6,) as an mpmath complex number."""
    return mp.mpc(mp.mpf(v[0]) + mp.mpf(v[1]) + mp.mpf(v[2]), mp.mpf(v[3]) + mp.mpf(v[4]) + mp.mpf(v[5]))


def absdiff(a, b):
    return float(abs(a.real - b.real) + abs(a.imag - b.imag))


def check_tm_ops():
    sp = tmc.Space(2, 8)
    b = 3
    delta = cp.full(b, 1.0)          # with delta = 1 each coefficient error is at most rem
    x = tmtd.TMT(cp.asarray(random_td((sp.size, b))), cp.zeros(b), delta)
    y = tmtd.TMT(cp.asarray(random_td((sp.size, b))), cp.zeros(b), delta)
    z, err = tmtd.mul(sp, x, y, return_err=True)
    zc, er, xc, yc = (cp.asnumpy(v) for v in (z.c, err, x.c, y.c))
    start, cnt, ii, jj, _ = (cp.asnumpy(v) if hasattr(v, 'get') else v for v in sp.table())
    worst, worst_rel = 0.0, 0.0
    for k in range(sp.size):
        for bi in range(b):
            exact = mp.mpc(0)
            for t in range(start[k], start[k] + cnt[k]):
                exact += to_mp(xc[:, ii[t], bi]) * to_mp(yc[:, jj[t], bi])
            got = to_mp(zc[:, k, bi])
            e = absdiff(got, exact)
            assert e <= er[k, bi], ('mul', k, bi, e, er[k, bi])
            worst = max(worst, e / er[k, bi] if er[k, bi] > 0 else 0.0)
            worst_rel = max(worst_rel, er[k, bi] / max(float(abs(exact)), 1e-300))
    print(f'  td product: largest error over its bound {worst:.2e}; largest bound over abs(value) '
          f'{worst_rel:.2e} (u^3 = {U ** 3:.1e})', flush=True)
    # A sum that cancels to about 1e-20 of its terms.
    yneg = tmtd.TMT(cp.asarray(-xc + random_td((sp.size, b), 1e-20)), cp.zeros(b), delta)
    w = tmtd.add(sp, x, yneg)
    wc, ync = cp.asnumpy(w.c), cp.asnumpy(yneg.c)
    worst = 0.0
    for k in range(sp.size):
        for bi in range(b):
            exact = to_mp(xc[:, k, bi]) + to_mp(ync[:, k, bi])
            worst = max(worst, absdiff(to_mp(wc[:, k, bi]), exact))
    print(f'  td sum with a cancellation to 1e-20: largest error {worst:.1e}; remainder '
          f'{float(cp.max(w.rem)):.1e}', flush=True)
    assert worst <= float(cp.min(w.rem)) or worst == 0.0
    # Scale by a td number for each box.
    k6 = cp.asarray(random_td((b,)))
    v = tmtd.scale(sp, x, k6)
    vc, kc = cp.asnumpy(v.c), cp.asnumpy(k6)
    worst = 0.0
    for k in range(sp.size):
        for bi in range(b):
            exact = to_mp(xc[:, k, bi]) * to_mp(kc[:, bi])
            worst = max(worst, absdiff(to_mp(vc[:, k, bi]), exact))
    print(f'  td scale: largest error {worst:.1e}; remainder {float(cp.max(v.rem)):.1e}', flush=True)
    assert worst <= float(cp.min(v.rem)) or worst == 0.0
    # The conversion to dd.
    d = tmtd.to_dd(sp, x)
    dc = cp.asnumpy(d.c)
    worst = 0.0
    for k in range(sp.size):
        for bi in range(b):
            got = mp.mpc(mp.mpf(dc[0, k, bi]) + mp.mpf(dc[1, k, bi]), mp.mpf(dc[2, k, bi]) + mp.mpf(dc[3, k, bi]))
            worst = max(worst, absdiff(got, to_mp(xc[:, k, bi])))
    print(f'  td to dd: largest error {worst:.1e}; remainder {float(cp.max(d.rem)):.1e}', flush=True)
    assert worst <= float(cp.min(d.rem))


def check_gemm(mode, out_td, cancel):
    nb, m, k, n = 2, 9, 40, 7
    a = random_td((nb, m, k))
    a = np.moveaxis(a, 0, 1)                                    # (nb, 6, m, k)
    bshape = (nb, k, n) if mode == 0 else (nb, n, k)
    bm = np.moveaxis(random_td(bshape, 1e3), 0, 1)
    if cancel:
        a = np.concatenate([a, a], axis=-1)
        noise = np.moveaxis(random_td(bshape, 1e-17), 0, 1)
        bm = np.concatenate([bm, -bm + noise], axis=-2 if mode == 0 else -1)
        k = 2 * k
    c, err = ddgemm.zgemm_tt(cp.asarray(a), cp.asarray(bm), mode, out_td)
    c, err = cp.asnumpy(c), cp.asnumpy(err)
    worst, rel = 0.0, 0.0
    for bi in range(nb):
        for i in range(m):
            for j in range(n):
                exact = mp.mpc(0)
                absum = 0.0
                for t in range(k):
                    x = to_mp(a[bi, :, i, t])
                    y = to_mp(bm[bi, :, t, j]) if mode == 0 else mp.conj(to_mp(bm[bi, :, j, t]))
                    exact += x * y
                    absum += float(abs(x) * abs(y))
                if out_td:
                    got = to_mp(c[bi, :, i, j])
                else:
                    got = mp.mpc(mp.mpf(c[bi, 0, i, j]) + mp.mpf(c[bi, 1, i, j]),
                                 mp.mpf(c[bi, 2, i, j]) + mp.mpf(c[bi, 3, i, j]))
                e = absdiff(got, exact)
                assert e <= err[bi, i, j], (mode, out_td, cancel, bi, i, j, e, err[bi, i, j])
                worst = max(worst, e / err[bi, i, j] if err[bi, i, j] > 0 else 0.0)
                rel = max(rel, err[bi, i, j] / absum)
    print(f'  zgemm_tt mode {mode}, out {"td" if out_td else "dd"}, cancel {cancel}: largest '
          f'error over its bound {worst:.2e}; largest bound over abs(A) abs(B) {rel:.1e}', flush=True)


if __name__ == '__main__':
    check_tm_ops()
    for mode in (0, 1):
        for out_td in (True, False):
            for cancel in (False, True):
                check_gemm(mode, out_td, cancel)
    print('\nall tmtd tests pass')
