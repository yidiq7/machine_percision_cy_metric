"""The strict bound of sup abs(f) over F: the run of the boxes of a cover on the GPU.

Input: a cover of scripts/cert_cover.py and the directory of the local forms of its
reference centers (scripts/tm_local_forms.py --centers <cover>), with center_<i>.npz
for reference center i.

For each reference center of the cover (or of --refs), its boxes run in batches of one order: a
cell with rho >= --rho-high at --order-high (with --a-high), the others at --order
(with --a); the models in tau alone take quartic.cert.box.NY_EXTRA more. Each box also
gets the bound of abs(grad f). A box whose bound of abs(f) exceeds --target tries
--order-high first, and
then splits into the cells of half the side that may meet F, each with the radius and
the bound of abs(h) of quartic.cert.cover.box_radius, at most --depth times; cell corners stay
exact mpmath numbers. Each final box gives one record: the
lower corner and side of its cell, its radius, the bound of abs(f), the parts of that
bound, and lam_big, the contraction factor of implicit_h_td on the ball that holds every
point of X over the cell. A box that still misses the target at the last depth stays in
the record with its bound, which then enters the maximum. The file also holds the git
commit of the code and the SHA-256 of each file of local forms (quartic.cert.records).

Usage, on one GPU node:
    uv run --extra gpu python -u \\
        scripts/cert_run.py --cover runs_tm/cover_s0.03_c6_v2.npz \\
        --local /n/netscratch/.../tm_local_f6t_td --target 5e-20 --out runs_tm/run_f6t.npz
"""
import argparse
import hashlib
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmgpu.tools import sup_bound                           # noqa: E402
from quartic.cert import box as tc                           # noqa: E402
from quartic.cert import cover as cc                                     # noqa: E402
from quartic.cert import records as rs                                   # noqa: E402


def split(lo, s):
    """The cells of side s/2 inside the cell [lo, lo + s]^4 that may meet F, each with its
    exact dd center, radius and bound of abs(h); lo and s are exact mpmath numbers."""
    out = []
    h = s / 2
    for k in range(16):
        sub = [lo[i] + h * ((k >> i) & 1) for i in range(4)]
        if not cc.may_meet_f(sub, h):
            continue
        cdd = cc.center(sub, h)
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
    ap.add_argument('--a', type=float, default=0.35, help='the gap fraction at --order')
    ap.add_argument('--a-high', type=float, default=0.12, help='the gap fraction at --order-high')
    ap.add_argument('--target', type=float, default=5e-20,
                    help='a box whose bound of abs(f) exceeds this tries --order-high, then splits')
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--refs', type=int, nargs='*', default=None,
                    help='only these reference centers (default: every one of the cover)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    cov = np.load(args.cover)
    refs = args.refs if args.refs is not None else list(range(len(cov['ref_centers'])))
    # The commit of the code, with -dirty if a tracked file differs from it.
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT).returncode != 0:
        commit += '-dirty'
    shas = []
    if args.order == args.order_high and args.a != args.a_high:
        raise SystemExit('--order equals --order-high, so --a must equal --a-high')
    settings = {args.order: args.a, args.order_high: args.a_high}
    t0 = time.perf_counter()
    records = []
    n_boxes = 0
    for r in refs:
        # One read of the file gives both the hash and the forms that the boxes use.
        raw = (Path(args.local) / f'center_{r}.npz').read_bytes()
        shas.append(hashlib.sha256(raw).hexdigest())
        local = tc.local_on_gpu(io.BytesIO(raw))
        k0 = int(local['k0'])
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
            f, (d, _, _), info, parts = tc.box_f(fr, local, deltas, order, order + tc.NY_EXTRA,
                                                 a, hmaxs, grad=True)
            sup, c0, pb = sup_bound(parts['sp4'], f, d)
            sup, c0, pb, rem, lam_big = (cp.asnumpy(x) for x in (sup, c0, pb, f.rem,
                                                                  info['lam_big']))
            gt, gg, ll = (cp.asnumpy(info[k]) for k in ('grad_tau', 'grad_g', 'lam_low'))
            n_boxes += len(batch)
            n_ref += len(batch)
            n_split, n_retry = 0, 0
            for j, (lo, s, cen, dl, hm, depth, od) in enumerate(batch):
                if not sup[j] <= args.target and od < args.order_high:
                    queue.append((lo, s, cen, dl, hm, depth, args.order_high))
                    n_retry += 1
                    continue
                if not sup[j] <= args.target and depth < args.depth:
                    for sub, h, c, rad, hmx in split(lo, s):
                        sub_order = args.order_high if cell_rho(c) >= args.rho_high else args.order
                        queue.append((sub, h, c, rad, hmx, depth + 1, sub_order))
                    n_split += 1
                    continue
                records.append((r, depth, *[float(v) for v in lo], float(s), dl, sup[j], c0[j],
                                pb[j], rem[j], hm, lam_big[j], od, a, gt[j], gg[j], ll[j]))
            if args.verbose:
                print(f'    batch of {len(batch)} at order {order} (depth {min(b[5] for b in batch)} '
                      f'to {max(b[5] for b in batch)}): {n_retry} to order {args.order_high}, '
                      f'{n_split} split; largest bound {sup.max():.2e}, largest gradient '
                      f'{np.nanmax(gg):.2e}, largest remainder '
                      f'{rem.max():.2e}, lam_big up to {np.nanmax(lam_big):.2f}; radius '
                      f'{min(deltas):.4f} to {max(deltas):.4f}; queue {len(queue)}; '
                      f'{time.perf_counter() - t1:.1f}s', flush=True)
        done = [x for x in records if x[0] == r]
        worst = float(np.max([x[8] for x in done])) if done else float('nan')
        print(f'reference {r}: {len(idx)} cells, {n_ref} boxes computed, {len(done)} final '
              f'boxes, largest bound {worst:.3e}; {time.perf_counter() - t1:.1f}s; total '
              f'{n_boxes} boxes in {time.perf_counter() - t0:.0f}s', flush=True)
        np.savez(args.out, records=np.array(records).reshape(-1, len(rs.COLUMNS)), target=args.target,
                 columns=np.array(rs.COLUMNS), commit=commit, local_refs=np.array(refs[:len(shas)]),
                 local_sha=np.array(shas))
    rec = np.array(records)
    # np.max and np.min give NaN if one record has NaN, so no NaN can hide here.
    grad_text = (f', largest gradient bound {np.max(rec[:, 17]):.4e} (in tau '
                 f'{np.max(rec[:, 16]):.4e}), smallest lam_low {np.min(rec[:, 18]):.3e}')
    # A bound that is NaN fails both tests below: max gives NaN, and it counts as above.
    print(f'done: {len(rec)} boxes, {n_boxes} computed, largest bound {rec[:, 8].max():.4e}, '
          f'{np.sum(~(rec[:, 8] <= args.target))} above the target, largest lam_big '
          f'{np.max(rec[:, 13]):.3f}{grad_text}; {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
