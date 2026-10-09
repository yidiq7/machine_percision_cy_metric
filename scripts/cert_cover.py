"""The cover of the fundamental region F by cells, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). S_4, the phases and the
conjugation map every point of X into F.

The cells are cubes of side s in (Re t_0, Im t_0, Re t_1, Im t_1). A cell is dropped only
when interval arithmetic (mpmath.iv) proves that it does not meet F. Each kept cell gets
the box at its center c with the radius delta that interval arithmetic proves:
abs(tau_i) <= delta for every point of X over the cell, with tau = U^-1 (y - p) the
tangent coordinates of the box (quartic.tm_cert, E = dy/dt at p).

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
import time

import numpy as np
import mpmath as mp
from mpmath import iv

mp.mp.dps = 30
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
    if r0.a ** 4 > w2.b:                   # abs(t_0)^8 > abs(z_2)^8
        return False
    return True


def box_radius(lo, s, c):
    """A delta with abs(tau_i) <= delta on the cell, for the box at the center c.

    tau_i = (t_i - c_i) - h n_i and h = (z2(t) - z2(c) - d . (t - c)) / den, with d = dz2/dt
    and den = n_2 - d_0 n_0 - d_1 n_1 at c. The numerator is at most half the bound of the
    second derivatives of z2 = u^(1/4) over the cell times abs(t - c)^2.
    """
    t0, t1 = c
    u = -(1 + t0 ** 4 + t1 ** 4)
    z2 = u ** (mp.mpf(1) / 4)
    grad = [4 * t0 ** 3, 4 * t1 ** 3, 4 * z2 ** 3]
    norm = mp.sqrt(sum(abs(g) ** 2 for g in grad))
    n = [mp.conj(g) / norm for g in grad]
    d = [-t0 ** 3 / z2 ** 3, -t1 ** 3 / z2 ** 3]
    den = abs(n[2] - d[0] * n[0] - d[1] * n[1])
    r = mp.sqrt(2) * s / 2                 # abs(t_i - c_i) on the cell
    # Over the cell: abs(t_i) <= tmax and abs(z2) >= zmin.
    tmax = [abs(t0) + r, abs(t1) + r]
    a0, b0, a1, b1 = (iv.mpf([lo[k], lo[k] + s]) for k in range(4))
    p0, q0 = cpow4(a0, b0)
    p1, q1 = cpow4(a1, b1)
    w2 = (1 + p0 + p1) ** 2 + (q0 + q1) ** 2
    zmin = mp.mpf(w2.a) ** (mp.mpf(1) / 8)
    if zmin <= 0:
        return mp.inf
    # Second derivatives of z2: -3 t_i^2 delta_ij / z2^3 - 3 t_i^3 t_j^3 / z2^7.
    hs = 0
    for i in range(2):
        for j in range(2):
            v = 3 * tmax[i] ** 3 * tmax[j] ** 3 / zmin ** 7
            if i == j:
                v += 3 * tmax[i] ** 2 / zmin ** 3
            hs += v
    hmax = hs * r * r / 2 / den
    delta = max(r + hmax * abs(n[0]), r + hmax * abs(n[1]))
    return float(delta) * (1 + 1e-9)


def dd(x):
    x = mp.mpc(x)
    rh, ih = float(x.real), float(x.imag)
    return [rh, float(x.real - rh), ih, float(x.imag - ih)]


def one_slab(job):
    """All cells of one value of the first coordinate."""
    i0, s, grid = job
    out = []
    for i1, i2, i3 in itertools.product(*(range(g) for g in grid[1:])):
        lo = [i0 * s, i1 * s, i2 * s, i3 * s]
        if not may_meet_f(lo, s):
            continue
        c = (mp.mpc(lo[0] + s / 2, lo[1] + s / 2), mp.mpc(lo[2] + s / 2, lo[3] + s / 2))
        delta = box_radius(lo, s, c)
        out.append((lo, [dd(c[0]), dd(c[1])], delta))
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
    lo = np.array([c[0] for c in cells])
    centers = np.array([c[1] for c in cells])          # (n, 2, 4)
    deltas = np.array([c[2] for c in cells])
    big = args.coarse * s
    coarse = np.floor((lo + s / 2) / big).astype(np.int64)
    keys, ref = np.unique(coarse, axis=0, return_inverse=True)
    ref_centers = np.array([[dd(mp.mpc((k[0] + 0.5) * big, (k[1] + 0.5) * big)),
                             dd(mp.mpc((k[2] + 0.5) * big, (k[3] + 0.5) * big))] for k in keys])
    np.savez(args.out, lo=lo, side=s, centers=centers, deltas=deltas, ref=ref.ravel(),
             ref_centers=ref_centers, coarse=big)
    vol = len(cells) * s ** 4
    print(f'side {s}: {len(cells)} cells of {np.prod(grid)} kept, volume {vol:.4f}; delta '
          f'{deltas.min():.4f} to {deltas.max():.4f} (r = {np.sqrt(2) * s / 2:.4f}); '
          f'{len(keys)} reference centers of side {big:.3f}; {time.perf_counter() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
