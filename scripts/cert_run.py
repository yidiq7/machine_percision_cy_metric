"""The strict bound of sup abs(f) over F: the run of the boxes of a cover on the GPU.

Input: a cover of scripts/cert_cover.py and the directory of the local forms of its
reference centers (scripts/tm_local_forms.py --centers <cover>), with center_<i>.npz
for reference center i.

For each reference center that has a file, its boxes run in batches. A box whose
remainder exceeds --target splits into the cells of half the side that may meet F,
each with the radius and the bound of abs(h) of cert_cover.box_radius, at most --depth
times; cell corners stay exact mpmath numbers. Each final box gives one record: the
lower corner and side of its cell, its radius, the bound of abs(f), the parts of that
bound, and lam_big, the contraction factor of implicit_h on the ball that holds every
point of X over the cell. A box that still misses the target at the last depth stays in
the record with its bound, which then enters the maximum.

Usage, on one GPU node:
    uv run --extra gpu --with cupy-cuda12x python -u scripts/cert_run.py \\
        --cover runs_tm/cover_s0.03.npz --local /n/netscratch/.../local --order 28 \\
        --target 2e-17 --out runs_tm/run_s0.03.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import mpmath as mp

# The quartic package imports JAX, which would take most of the GPU memory from CuPy.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quartic import tm_cert as tc                           # noqa: E402
import cert_cover as cc                                     # noqa: E402


def split(lo, s):
    """The cells of side s/2 inside the cell [lo, lo + s]^4 that may meet F, each with its
    exact dd center, radius and bound of abs(h); lo and s are exact mpmath numbers."""
    out = []
    h = s / 2
    for k in range(16):
        sub = [lo[i] + h * ((k >> i) & 1) for i in range(4)]
        if not cc.may_meet_f(sub, h):
            continue
        c = (mp.mpc(sub[0] + h / 2, sub[1] + h / 2), mp.mpc(sub[2] + h / 2, sub[3] + h / 2))
        cdd = [cc.dd(c[0]), cc.dd(c[1])]
        rad, hmax = cc.box_radius(sub, h, cdd)
        out.append((sub, h, cdd, rad, hmax))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--order', type=int, default=28)
    ap.add_argument('--ny-extra', type=int, default=6)
    ap.add_argument('--target', type=float, default=2e-17)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--a', type=float, default=0.0, help='the gap fraction (0: 2 / order)')
    ap.add_argument('--td', action='store_true', help='triple-double compositions')
    ap.add_argument('--refs', type=int, nargs='*', default=None,
                    help='only these reference centers (default: every one with a file)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    cov = np.load(args.cover)
    refs = args.refs
    if refs is None:
        refs = sorted(int(p.stem.split('_')[1]) for p in Path(args.local).glob('center_*.npz'))
    t0 = time.perf_counter()
    records = []
    n_boxes = 0
    for r in refs:
        local = tc.local_on_gpu(Path(args.local) / f'center_{r}.npz')
        k0 = int(local['k0'])
        idx = np.flatnonzero(cov['ref'] == r)
        side = mp.mpf(float(cov['side']))
        queue = [([mp.mpf(int(v)) * side for v in cov['idx'][i]], side,
                  [list(x) for x in cov['centers'][i]], float(cov['deltas'][i]),
                  float(cov['hmaxs'][i]), 0) for i in idx]
        t1 = time.perf_counter()
        while queue:
            batch, queue = queue[:args.batch], queue[args.batch:]
            fr = tc.make_frames([b[2] for b in batch], k0)
            deltas = [b[3] for b in batch]
            hmaxs = cp.asarray(np.array([b[4] for b in batch]))
            a = args.a if args.a > 0 else None
            f, (d, _, _), info, parts = tc.box_f(fr, local, deltas, args.order,
                                                 args.order + args.ny_extra, a=a,
                                                 td=args.td, hmax=hmaxs)
            sup, c0, pb = tc.sup_bound(parts['sp4'], f, d)
            sup, c0, pb, rem, lam_big = (cp.asnumpy(x) for x in (sup, c0, pb, f.rem,
                                                                  info['lam_big']))
            n_boxes += len(batch)
            n_split = 0
            for j, (lo, s, cen, dl, hm, depth) in enumerate(batch):
                if rem[j] > args.target and depth < args.depth:
                    queue += [(sub, h, c, rad, hmx, depth + 1) for sub, h, c, rad, hmx in split(lo, s)]
                    n_split += 1
                    continue
                records.append((r, depth, *[float(v) for v in lo], float(s), dl, sup[j], c0[j],
                                pb[j], rem[j], hm, lam_big[j]))
            if args.verbose:
                print(f'    batch of {len(batch)} (depth {batch[0][5]} to {batch[-1][5]}): {n_split} '
                      f'split, largest remainder {rem.max():.2e}, largest bound {sup.max():.2e}, '
                      f'radius {min(deltas):.4f} to {max(deltas):.4f}; queue {len(queue)}; '
                      f'{time.perf_counter() - t1:.1f}s', flush=True)
        done = [x for x in records if x[0] == r]
        worst = max(x[8] for x in done) if done else float("nan")
        print(f'reference {r}: {len(idx)} cells, {len(done)} final boxes, largest bound '
              f'{worst:.3e}; {time.perf_counter() - t1:.1f}s; total {n_boxes} boxes in '
              f'{time.perf_counter() - t0:.0f}s', flush=True)
        np.savez(args.out, records=np.array(records), order=args.order, target=args.target,
                 columns=np.array(['ref', 'depth', 'lo0', 'lo1', 'lo2', 'lo3', 'side', 'delta',
                                   'bound', 'center', 'polynomial', 'remainder', 'hmax',
                                   'lam_big']))
    rec = np.array(records)
    print(f'done: {len(rec)} boxes, {n_boxes} computed, largest bound {rec[:, 8].max():.4e}, '
          f'{np.sum(rec[:, 11] > args.target)} above the target; {time.perf_counter() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
