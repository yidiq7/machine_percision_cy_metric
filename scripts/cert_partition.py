"""The exact check that the final boxes of a run of scripts/cert_run.py fill its cover.

The strict bound of sup abs(f) over F needs every point of F in a final box. This script
proves it for the cells of the grid of scripts/cert_cover.py, with the exact corners of
the cells:

1. Every cell of the grid that is not in the cover fails `cert_cover.may_meet_f`.
2. Every cell of the cover is a final box of the record, or each of its 16 cells of half
   the side is a final box, fails `may_meet_f`, or divides in the same way, down to the
   largest depth of the run. So the final boxes and the cells that miss F make a
   partition of each cell of the cover.
3. Each record is one final box of that partition, and each record has a bound in
   (0, target] and 0 <= lam_big < 1.
4. With --radii, `cert_cover.box_radius` with the present code gives, for the cell and the
   center of each record, a radius and a bound of abs(h) at or below the delta and the
   h_max that the record used (columns 7 and 12); a NaN fails. The center of a box is
   the center of its cell: for the cells of the cover the cover file holds it, and the
   script checks that file; cert_run forms the center of each smaller cell in the same
   way.
5. With --grad (for a run of cert_run.py --grad), every record must have a finite bound
   of abs(grad f) in G and in tau above 0, and a finite lam_low > 0; a NaN fails. With no
   --grad, the script only reports these columns if one of them is finite.

The corners of the records are multiples of s / 2^depth, so they are compared as
integers. Usage, on CPU cores:
    uv run --extra gpu python -u scripts/cert_partition.py --run runs_tm/run_f6t.npz \\
        --cover runs_tm/cover_s0.03_c6_v2.npz --radii --grad --procs 32
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

from quartic.cert import cover as cc                                     # noqa: E402

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


def radii_chunk(rows):
    """For records j: delta and h_max of box_radius for the cell of the record and its
    center, against the values of the record. Returns the largest ratios and the records
    that fail; a NaN fails."""
    unit = mp.mpf(G['side']) / 2 ** G['depth']
    worst, bad = [0.0, 0.0], []
    for j in rows:
        corner, size = G['cells'][j]
        lo = [mp.mpf(int(v)) * unit for v in corner]
        h = mp.mpf(int(size)) * unit
        cen = [cc.dd(mp.mpc(lo[0] + h / 2, lo[1] + h / 2)),
               cc.dd(mp.mpc(lo[2] + h / 2, lo[3] + h / 2))]
        delta, hmax = cc.box_radius(lo, h, cen)
        d_rec, h_rec = float(G['rec'][j, 7]), float(G['rec'][j, 12])
        if not (delta <= d_rec and hmax <= h_rec):
            bad.append(int(j))
            continue
        worst[0] = max(worst[0], delta / d_rec)
        worst[1] = max(worst[1], hmax / h_rec)
    return worst, bad


def centers_chunk(rows):
    """The cover cells whose stored center is not the exact center of the cell."""
    s = mp.mpf(G['side'])
    bad = []
    for i in rows:
        _, c = cc.cell(tuple(int(v) for v in G['idx'][i]), s)
        if not np.array_equal(np.array(c), np.asarray(G['centers'][i])):
            bad.append(int(i))
    return bad


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
    ap.add_argument('--radii', action='store_true',
                    help='also compute delta and h_max of the cell of every record again')
    ap.add_argument('--grad', action='store_true',
                    help='require the gradient bound in every record (a run with --grad)')
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
    # The FP64 corner of a record is within about 1e-14 units of its exact corner.
    if not (np.abs(corners - ci).max() <= 1e-9 and np.abs(sizes - si).max() <= 1e-9):
        raise SystemExit('a record corner is not a multiple of s / 2^depth')
    recs = {}
    for j, (c, n) in enumerate(zip(map(tuple, ci.tolist()), si.tolist())):
        if (c, n) in recs:
            raise SystemExit(f'record {j} repeats the cell of record {recs[(c, n)]}')
        recs[(c, n)] = j
    G.update(side=side, depth=depth, grid=_grid(side), recs=recs, rec=rec,
             cells=list(zip(map(tuple, ci.tolist()), si.tolist())),
             kept={tuple(int(v) for v in i) for i in cov['idx']}, idx=cov['idx'],
             centers=cov['centers'])
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        bad = [x for part in pool.map(dropped_slab, range(G['grid'][0])) for x in part]
        chunks = np.array_split(cov['idx'], 4 * args.procs)
        res = pool.map(cover_chunk, chunks)
        if args.radii:
            radii = pool.map(radii_chunk, np.array_split(np.arange(len(rec)), 4 * args.procs))
            bad_cen = [i for part in pool.map(centers_chunk, np.array_split(
                np.arange(len(cov['idx'])), 4 * args.procs)) for i in part]
    used = set().union(*(u for u, _ in res))
    missing = [m for _, ms in res for m in ms]
    unused = sorted(set(range(len(rec))) - used)
    bounds, lam = rec[:, 8], rec[:, 13]
    # Each comparison is False for NaN, so a NaN counts as bad.
    bad_bound = int(np.sum(~((bounds > 0) & (bounds <= target))))
    bad_lam = int(np.sum(~((lam >= 0) & (lam < 1))))
    print(f'grid {G["grid"]}: {len(bad)} dropped cells that may meet F', flush=True)
    print(f'cover: {len(cov["idx"])} cells, depth {depth}; {len(missing)} cells of the '
          f'partition without a record; {len(unused)} records outside the partition', flush=True)
    print(f'records: {len(rec)}; largest bound {np.max(bounds):.4e} (target {target:.1e}), '
          f'{bad_bound} not in (0, target]; largest lam_big {np.max(lam):.3f}, {bad_lam} not '
          f'in [0, 1); {time.perf_counter() - t0:.0f}s', flush=True)
    ok = not bad and not missing and not unused and bad_bound == 0 and bad_lam == 0
    if args.radii:
        rd = max(r[0][0] for r in radii)
        rh = max(r[0][1] for r in radii)
        bad_rad = [j for r in radii for j in r[1]]
        print(f'radii of the {len(rec)} records: delta from box_radius over the delta of the '
              f'record at most {rd:.15f}, h_max over the h_max of the record at most {rh:.15f}; '
              f'{len(bad_rad)} records above or NaN {bad_rad[:5]}; {len(bad_cen)} cover cells '
              f'whose center is not the center of the cell', flush=True)
        ok = ok and not bad_rad and not bad_cen
    print('the partition check passes' if ok else 'THE PARTITION CHECK FAILS', flush=True)
    has_grad = rec.shape[1] > 18
    if args.grad and not has_grad:
        print('THE GRADIENT CHECK FAILS: the records have no gradient columns', flush=True)
    elif args.grad or (has_grad and np.any(np.isfinite(rec[:, 17]))):
        gt, gg, ll = rec[:, 16], rec[:, 17], rec[:, 18]
        good = (np.isfinite(gg) & (gg > 0) & np.isfinite(gt) & (gt > 0) & np.isfinite(ll)
                & (ll > 0))
        bad_grad = int(np.sum(~good))
        print(f'gradient: largest bound in G {np.max(gg):.4e}, in tau {np.max(gt):.4e}; '
              f'smallest lam_low {np.min(ll):.4e}; {bad_grad} records without a finite bound '
              f'above 0 or with lam_low not finite and above 0', flush=True)
        print('the gradient bound holds on every record' if bad_grad == 0 and ok
              else 'THE GRADIENT CHECK FAILS', flush=True)
    if not ok:
        print('first dropped cells:', bad[:5], 'first missing cells:', missing[:5],
              'first unused records:', unused[:5], flush=True)


if __name__ == '__main__':
    main()
