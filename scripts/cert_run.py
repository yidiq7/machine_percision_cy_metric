"""The strict bound of sup abs(f) over F: the run of the boxes of a cover on the GPU.

Input: a cover of scripts/cert_cover.py and the directory of the local forms of its
reference centers (scripts/tm_local_forms.py --centers <cover>), with center_<i>.npz
for reference center i.

For each reference center that has a file, its boxes run in batches of one order: a
cell with rho >= --rho-high at --order-high (with --a-high), the others at --order
(with --a). A box whose bound of abs(f) exceeds --target tries --order-high first, and
then splits into the cells of half the side that may meet F, each with the radius and
the bound of abs(h) of cert_cover.box_radius, at most --depth times; cell corners stay
exact mpmath numbers. Each final box gives one record: the
lower corner and side of its cell, its radius, the bound of abs(f), the parts of that
bound, and lam_big, the contraction factor of implicit_h on the ball that holds every
point of X over the cell. A box that still misses the target at the last depth stays in
the record with its bound, which then enters the maximum.

Usage, on one GPU node (local forms in Delta and td, tm_local_forms.py --delta --td):
    uv run --extra gpu --with cupy-cuda12x --with python-flint python -u \\
        scripts/cert_run.py --cover runs_tm/cover_s0.03_c6_v2.npz \\
        --local /n/netscratch/.../tm_local_f6t_td --target 5e-20 --out runs_tm/run_f6t.npz
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


def cell_rho(c_dd):
    """rho = min / max of abs(z_a) at a dd center t, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4)."""
    t = [complex(v[0] + v[1], v[2] + v[3]) for v in c_dd]
    z = [abs(t[0]), abs(t[1]), abs(1 + t[0] ** 4 + t[1] ** 4) ** 0.25, 1.0]
    return min(z) / max(z)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--order', type=int, default=28, help='the order of the cells below --rho-high')
    ap.add_argument('--order-high', type=int, default=32,
                    help='the order of the cells at or above --rho-high, and of each second try')
    ap.add_argument('--rho-high', type=float, default=0.3)
    ap.add_argument('--a', type=float, default=0.35, help='the gap fraction at --order (0: 2 / order)')
    ap.add_argument('--a-high', type=float, default=0.12, help='the gap fraction at --order-high')
    ap.add_argument('--kmax', type=int, default=0, help='the largest k with a full model (0: top)')
    ap.add_argument('--ny-extra', type=int, default=6)
    ap.add_argument('--target', type=float, default=5e-20,
                    help='a box whose bound of abs(f) exceeds this tries --order-high, then splits')
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--td', action='store_true', help='td GEMMs for dd local forms')
    ap.add_argument('--refs', type=int, nargs='*', default=None,
                    help='only these reference centers (default: every one with a file)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    cov = np.load(args.cover)
    refs = args.refs
    if refs is None:
        refs = sorted(int(p.stem.split('_')[1]) for p in Path(args.local).glob('center_*.npz'))
    settings = {args.order: args.a or None, args.order_high: args.a_high or None}
    t0 = time.perf_counter()
    records = []
    n_boxes = 0
    columns = ['ref', 'depth', 'lo0', 'lo1', 'lo2', 'lo3', 'side', 'delta', 'bound', 'center',
               'polynomial', 'remainder', 'hmax', 'lam_big', 'order', 'a']
    for r in refs:
        local = tc.local_on_gpu(Path(args.local) / f'center_{r}.npz')
        k0, top = int(local['k0']), int(local['top'])
        kmax = args.kmax or top
        idx = np.flatnonzero(cov['ref'] == r)
        side = mp.mpf(float(cov['side']))
        # Queue items: (lo, side, center, delta, hmax, depth, order).
        queue = []
        for i in idx:
            cen = [list(x) for x in cov['centers'][i]]
            order = args.order_high if cell_rho(cen) >= args.rho_high else args.order
            queue.append(([mp.mpf(int(v)) * side for v in cov['idx'][i]], side, cen,
                          float(cov['deltas'][i]), float(cov['hmaxs'][i]), 0, order))
        t1 = time.perf_counter()
        n_ref = 0
        while queue:
            # One order for each batch.
            order = queue[0][6]
            batch = [q for q in queue if q[6] == order][:args.batch]
            chosen = {id(q) for q in batch}
            queue = [q for q in queue if id(q) not in chosen]
            fr = tc.make_frames([b[2] for b in batch], k0)
            deltas = [b[3] for b in batch]
            hmaxs = cp.asarray(np.array([b[4] for b in batch]))
            a = settings[order]
            f, (d, _, _), info, parts = tc.box_f(fr, local, deltas, order, order + args.ny_extra,
                                                 k_max=kmax, a=a, td=args.td, hmax=hmaxs)
            sup, c0, pb = tc.sup_bound(parts['sp4'], f, d)
            sup, c0, pb, rem, lam_big = (cp.asnumpy(x) for x in (sup, c0, pb, f.rem,
                                                                  info['lam_big']))
            n_boxes += len(batch)
            n_ref += len(batch)
            n_split, n_retry = 0, 0
            for j, (lo, s, cen, dl, hm, depth, od) in enumerate(batch):
                if sup[j] > args.target and od < args.order_high:
                    queue.append((lo, s, cen, dl, hm, depth, args.order_high))
                    n_retry += 1
                    continue
                if sup[j] > args.target and depth < args.depth:
                    for sub, h, c, rad, hmx in split(lo, s):
                        sub_order = args.order_high if cell_rho(c) >= args.rho_high else args.order
                        queue.append((sub, h, c, rad, hmx, depth + 1, sub_order))
                    n_split += 1
                    continue
                records.append((r, depth, *[float(v) for v in lo], float(s), dl, sup[j], c0[j],
                                pb[j], rem[j], hm, lam_big[j], od, a or 2.0 / od))
            if args.verbose:
                print(f'    batch of {len(batch)} at order {order} (depth {min(b[5] for b in batch)} '
                      f'to {max(b[5] for b in batch)}): {n_retry} to order {args.order_high}, '
                      f'{n_split} split; largest bound {sup.max():.2e}, largest remainder '
                      f'{rem.max():.2e}, lam_big up to {np.nanmax(lam_big):.2f}; radius '
                      f'{min(deltas):.4f} to {max(deltas):.4f}; queue {len(queue)}; '
                      f'{time.perf_counter() - t1:.1f}s', flush=True)
        done = [x for x in records if x[0] == r]
        worst = max(x[8] for x in done) if done else float('nan')
        print(f'reference {r}: {len(idx)} cells, {n_ref} boxes computed, {len(done)} final '
              f'boxes, largest bound {worst:.3e}; {time.perf_counter() - t1:.1f}s; total '
              f'{n_boxes} boxes in {time.perf_counter() - t0:.0f}s', flush=True)
        np.savez(args.out, records=np.array(records), target=args.target,
                 columns=np.array(columns))
    rec = np.array(records)
    print(f'done: {len(rec)} boxes, {n_boxes} computed, largest bound {rec[:, 8].max():.4e}, '
          f'{np.sum(rec[:, 8] > args.target)} above the target, largest lam_big '
          f'{np.nanmax(rec[:, 13]):.3f}; {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
