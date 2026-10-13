"""The cover of the fundamental region F by cells, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). S_4, the phases and the
conjugation map every point of X into F.

The cells are cubes of side s in (Re t_0, Im t_0, Re t_1, Im t_1) with corners i s, for
integers i and the exact binary value s of the side, so neighbor cells meet exactly. A
cell is dropped only when interval arithmetic (mpmath.iv) proves that it does not meet F.
Each kept cell gets the box at its center c (exact in dd) with the radius delta and the
bound hmax that ball arithmetic proves in the frame of `quartic.cert.frame` (the frame of
the box): every point of X over the cell is p + E tau + h n with abs(tau_i) <= delta and
abs(h) <= hmax.

The reference centers are the centers of the coarse cells of side S = m s; each kept
cell belongs to the coarse cell that holds its center.

Output: an npz with the cells (lower corners, side), the box centers as dd, the radii,
the index of the reference center of each box, and the reference centers as dd.

Usage, on CPU cores:
    uv run --extra gpu python -u scripts/cert_cover.py --side 0.03 --coarse 6 \\
        --out runs_tm/cover_s0.03_c6_v2.npz --procs 16
"""
import argparse
import itertools
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmgpu.host import mp_dd                                          # noqa: E402
from quartic.cert import cover as cc                                  # noqa: E402


def one_slab(job):
    """All cells of one value of the first coordinate."""
    i0, s, grid = job
    s = mp.mpf(s)
    out = []
    for i1, i2, i3 in itertools.product(*(range(g) for g in grid[1:])):
        idx = (i0, i1, i2, i3)
        lo, c = cc.cell(idx, s)
        if not cc.may_meet_f(lo, s):
            continue
        delta, hmax = cc.box_radius(lo, s, c)
        out.append((idx, [float(x) for x in lo], c, delta, hmax))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--side', type=float, required=True)
    ap.add_argument('--coarse', type=int, default=6)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--procs', type=int, default=8)
    args = ap.parse_args()
    s = args.side
    grid = cc.grid(s)
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
    ref_centers = np.array([[mp_dd(mp.mpc((k[0] + 0.5) * big, (k[1] + 0.5) * big)),
                             mp_dd(mp.mpc((k[2] + 0.5) * big, (k[3] + 0.5) * big))] for k in keys])
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
