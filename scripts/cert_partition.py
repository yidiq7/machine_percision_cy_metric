"""The exact check that the final boxes of a run of scripts/cert_run.py fill its cover.

The strict bound of sup abs(f) over F needs every point of F in a final box. This script
proves it for the cells of the grid of scripts/cert_cover.py, with the exact corners of
the cells:

1. Every cell of the grid that is not in the cover fails `cert_cover.may_meet_f`.
2. Every cell of the cover is a final box of the record, or each of its 16 cells of half
   the side is a final box, fails `may_meet_f`, or divides in the same way, down to the
   largest depth of the run. So the final boxes and the cells that miss F make a
   partition of each cell of the cover.
3. Each record is one final box of that partition, and each record has a finite bound at
   or below the target and lam_big < 1.

The corners of the records are multiples of s / 2^depth, so they are compared as
integers. Usage, on CPU cores:
    uv run --extra gpu python -u scripts/cert_partition.py --run runs_tm/run_f6t.npz \\
        --cover runs_tm/cover_s0.03_c6_v2.npz --procs 32
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

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import cert_cover as cc                                     # noqa: E402

G = {}


def _grid(s):
    return [int(np.ceil(1 / s)), int(np.ceil(0.7072 / s)), int(np.ceil(1 / s)), int(np.ceil(1 / s))]


def dropped_slab(i0):
    """The cells of the grid with first index i0 that are not in the cover and may meet F."""
    s = mp.mpf(G['side'])
    bad = []
    for rest in itertools.product(*(range(g) for g in G['grid'][1:])):
        idx = (i0,) + rest
        if idx in G['kept']:
            continue
        if cc.may_meet_f(cc.cell(idx, s)[0], s):
            bad.append(idx)
    return bad


def _covered(corner, size, depth, recs, used, missing):
    """True if records and cells that miss F make a partition of the cell (integer corner,
    size in units of s / 2^D)."""
    key = (corner, size)
    if key in recs:
        used.add(recs[key])
        return True
    if depth > 0:
        unit = mp.mpf(G['side']) / 2 ** G['depth']
        lo = [mp.mpf(int(c)) * unit for c in corner]
        if not cc.may_meet_f(lo, size * unit):
            return True
    if size == 1:
        missing.append(key)
        return False
    half = size // 2
    ok = True
    for k in range(16):
        child = tuple(corner[i] + half * ((k >> i) & 1) for i in range(4))
        ok = _covered(child, half, depth + 1, recs, used, missing) and ok
    return ok


def cover_chunk(cells):
    """The partition of a list of cover cells (their grid indices)."""
    used, missing = set(), []
    size = 2 ** G['depth']
    for idx in cells:
        _covered(tuple(int(i) * size for i in idx), size, 0, G['recs'], used, missing)
    return used, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=str, required=True)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--depth', type=int, default=2, help='the --depth of the run')
    ap.add_argument('--procs', type=int, default=16)
    args = ap.parse_args()
    t0 = time.perf_counter()
    z = np.load(args.run)
    rec, target = z['records'], float(z['target'])
    cov = np.load(args.cover)
    side = float(cov['side'])
    depth = max(int(rec[:, 1].max()), args.depth)
    unit = side / 2 ** depth
    corners = rec[:, 2:6] / unit
    sizes = rec[:, 6] / unit
    ci, si = np.rint(corners).astype(np.int64), np.rint(sizes).astype(np.int64)
    if np.abs(corners - ci).max() > 1e-6 or np.abs(sizes - si).max() > 1e-6:
        raise SystemExit('a record corner is not a multiple of s / 2^depth')
    recs = {}
    for j, (c, n) in enumerate(zip(map(tuple, ci.tolist()), si.tolist())):
        if (c, n) in recs:
            raise SystemExit(f'record {j} repeats the cell of record {recs[(c, n)]}')
        recs[(c, n)] = j
    G.update(side=side, depth=depth, grid=_grid(side), recs=recs,
             kept={tuple(int(v) for v in i) for i in cov['idx']})
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        bad = [x for part in pool.map(dropped_slab, range(G['grid'][0])) for x in part]
        chunks = np.array_split(cov['idx'], 4 * args.procs)
        res = pool.map(cover_chunk, chunks)
    used = set().union(*(u for u, _ in res))
    missing = [m for _, ms in res for m in ms]
    unused = sorted(set(range(len(rec))) - used)
    bounds, lam = rec[:, 8], rec[:, 13]
    bad_bound = int(np.sum(~(bounds <= target)))
    bad_lam = int(np.sum(~(lam < 1)))
    print(f'grid {G["grid"]}: {len(bad)} dropped cells that may meet F', flush=True)
    print(f'cover: {len(cov["idx"])} cells, depth {depth}; {len(missing)} cells of the '
          f'partition without a record; {len(unused)} records outside the partition', flush=True)
    print(f'records: {len(rec)}; largest bound {np.max(bounds):.4e} (target {target:.1e}), '
          f'{bad_bound} not at or below it; largest lam_big {np.max(lam):.3f}, {bad_lam} not '
          f'below 1; {time.perf_counter() - t0:.0f}s', flush=True)
    ok = not bad and not missing and not unused and bad_bound == 0 and bad_lam == 0
    print('the partition check passes' if ok else 'THE PARTITION CHECK FAILS', flush=True)
    if not ok:
        print('first dropped cells:', bad[:5], 'first missing cells:', missing[:5],
              'first unused records:', unused[:5], flush=True)


if __name__ == '__main__':
    main()
