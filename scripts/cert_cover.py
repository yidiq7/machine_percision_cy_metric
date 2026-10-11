"""The cover of the fundamental region F by cells, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). S_4, the phases and the
conjugation map every point of X into F.

The cells are cubes of side s in (Re t_0, Im t_0, Re t_1, Im t_1) with corners i s, for
integers i and the exact binary value s of the side, so neighbor cells meet exactly. A
cell is dropped only when interval arithmetic (mpmath.iv) proves that it does not meet F.
Each kept cell gets the box at its center c (exact in dd) with the radius delta and the
bound hmax that ball arithmetic proves in the frame of `quartic.tm_frame` (the frame of
the box): every point of X over the cell is p + E tau + h n with abs(tau_i) <= delta and
abs(h) <= hmax.

The reference centers are the centers of the coarse cells of side S = m s; each kept
cell belongs to the coarse cell that holds its center.

Output: an npz with the cells (lower corners, side), the box centers as dd, the radii,
the index of the reference center of each box, and the reference centers as dd.

Usage, on CPU cores:
    uv run --extra gpu python -u scripts/cert_cover.py --side 0.03 --coarse 5 \\
        --out runs_tm/cover_s030.npz --procs 16
"""
import argparse
import itertools
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import mpmath as mp
from mpmath import iv

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    In the frame of `quartic.tm_frame.frame` a point of X over the cell is p + E tau + h n
    with tau_i = (t_i - c_i) - h n_i and h = (z2(t) - p_2 - d . (t - c)) / den, d = E_2 and
    den = n_2 - d_0 n_0 - d_1 n_1. The numerator is at most half the bound of the second
    derivatives of z2 over the cell times r^2, plus abs(z2(c) - p_2) and
    abs(d(c) - d_i) r, with d(c) the exact derivative. All bounds come from arb balls.
    """
    from flint import acb, arb
    from quartic.tm_frame import frame
    from tmgpu.host import lower, precision, upper
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


def one_slab(job):
    """All cells of one value of the first coordinate."""
    i0, s, grid = job
    s = mp.mpf(s)
    out = []
    for i1, i2, i3 in itertools.product(*(range(g) for g in grid[1:])):
        idx = (i0, i1, i2, i3)
        lo, c = cell(idx, s)
        if not may_meet_f(lo, s):
            continue
        delta, hmax = box_radius(lo, s, c)
        out.append((idx, [float(x) for x in lo], c, delta, hmax))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--side', type=float, required=True)
    ap.add_argument('--coarse', type=int, default=5)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--procs', type=int, default=8)
    args = ap.parse_args()
    s = args.side
    # The bounding box: Re t_0 in [0, 1], Im t_0 in [0, 1/sqrt 2], Re t_1, Im t_1 in [0, 1].
    grid = [int(np.ceil(1 / s)), int(np.ceil(0.7072 / s)), int(np.ceil(1 / s)), int(np.ceil(1 / s))]
    t0 = time.perf_counter()
    jobs = [(i0, s, grid) for i0 in range(grid[0])]
    cells = []
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        for part in pool.imap_unordered(one_slab, jobs):
            cells += part
    cells.sort(key=lambda x: x[0])
    idx = np.array([c[0] for c in cells], dtype=np.int64)
    lo = np.array([c[1] for c in cells])
    centers = np.array([c[2] for c in cells])          # (n, 2, 4)
    deltas = np.array([c[3] for c in cells])
    hmaxs = np.array([c[4] for c in cells])
    big = args.coarse * s
    coarse = np.floor((lo + s / 2) / big).astype(np.int64)
    keys, ref = np.unique(coarse, axis=0, return_inverse=True)
    ref_centers = np.array([[dd(mp.mpc((k[0] + 0.5) * big, (k[1] + 0.5) * big)),
                             dd(mp.mpc((k[2] + 0.5) * big, (k[3] + 0.5) * big))] for k in keys])
    np.savez(args.out, idx=idx, lo=lo, side=s, centers=centers, deltas=deltas, hmaxs=hmaxs,
             ref=ref.ravel(), ref_centers=ref_centers, coarse=big)
    vol = len(cells) * s ** 4
    print(f'side {s}: {len(cells)} cells of {np.prod(grid)} kept, volume {vol:.4f}; delta '
          f'{deltas.min():.4f} to {deltas.max():.4f} (r = {np.sqrt(2) * s / 2:.4f}), hmax up to '
          f'{hmaxs.max():.2e}; '
          f'{len(keys)} reference centers of side {big:.3f}; {time.perf_counter() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
