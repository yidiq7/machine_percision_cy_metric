"""The bounds of quartic.cert.box.metric_bounds on the cells of a cover, in every chart of the
atlas that can meet a cell, for the constants of notes/draft-main.tex (scripts/cert_atlas.py
combine --metric).

Input: a cover of scripts/cert_cover.py and the directory of the local forms of its reference
centers, as for scripts/cert_run.py. For each reference center of the cover (or of --refs), the
cells run in batches at one --order, each box with the charts of quartic.cert.cover.chart_mask.
A box whose bound k_atlas exceeds --k-target, or with a bound that is not finite, below 0, or
with mu_low or lam_atlas not above 0, splits into the cells of half the side that may meet F
(quartic.cert.cover.halves), at most --depth times. Each final box gives one record of
quartic.cert.records.METRIC_COLUMNS; a box that misses the target at the last depth keeps its
bounds, which then enter the maxima. These boxes do not bound abs(f): the metric bounds take g
at quartic.cert.box.METRIC_ORDER, so a low order is enough. The file also holds the git commit
and the SHA-256 of each file of local forms (quartic.cert.records), and
scripts/cert_partition.py --metric checks it.

Usage, on one GPU node:
    uv run --extra gpu --locked python -u scripts/cert_metric.py \\
        --cover runs_tm/cover_s0.03_c6_v3.npz --local /n/netscratch/.../tm_local_f6t_td_v3 \\
        --out runs_tm/metric_f6t.npz
"""
import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quartic.cert import box as tc                           # noqa: E402
from quartic.cert import cover as cc                         # noqa: E402
from quartic.cert import records as rs                       # noqa: E402

BOUNDS = rs.METRIC_COLUMNS[12:]


def final(vals, k_target):
    """True if the bounds of one box (a dict of floats) need no split."""
    good = all(np.isfinite(vals[k]) and vals[k] >= 0 for k in BOUNDS)
    return good and vals['mu_low'] > 0 and vals['lam_atlas'] > 0 and vals['k_atlas'] <= k_target


def run_reference(cov, r, local, args):
    """The records of the reference center r, the number of its cells and of computed boxes."""
    k0 = int(local['k0'])
    idx = np.flatnonzero(cov['ref'] == r)
    side = mp.mpf(float(cov['side']))
    # Queue items: (lo, side, center, delta, hmax, depth).
    queue = [([mp.mpf(int(v)) * side for v in cov['idx'][i]], side,
              [list(x) for x in cov['centers'][i]], float(cov['deltas'][i]),
              float(cov['hmaxs'][i]), 0) for i in idx]
    records, n_computed = [], 0
    while queue:
        batch, queue = queue[:args.batch], queue[args.batch:]
        fr = tc.make_frames([b[2] for b in batch], k0)
        charts = np.array([cc.chart_mask(b[0], b[1]) for b in batch])
        _, _, info, _ = tc.box_f(fr, local, [b[3] for b in batch], args.order,
                                 args.order + tc.NY_EXTRA, args.a,
                                 cp.asarray(np.array([b[4] for b in batch])), grad=True,
                                 charts=charts)
        info = {k: cp.asnumpy(info[k]) for k in ('lam_big',) + BOUNDS}
        n_computed += len(batch)
        for j, (lo, s, cen, dl, hm, depth) in enumerate(batch):
            vals = {k: float(info[k][j]) for k in BOUNDS}
            if not final(vals, args.k_target) and depth < args.depth:
                queue.extend((sub, h, c, rad, hmx, depth + 1)
                             for sub, h, c, rad, hmx in cc.halves(lo, s))
                continue
            records.append((r, depth, *[float(v) for v in lo], float(s), dl, hm,
                            float(info['lam_big'][j]), args.order, args.a,
                            *[vals[k] for k in BOUNDS]))
    return records, len(idx), n_computed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--order', type=int, default=12)
    ap.add_argument('--a', type=float, default=0.12, help='the gap fraction of the Cauchy estimate')
    ap.add_argument('--k-target', type=float, default=80.0,
                    help='a box whose bound k_atlas exceeds this splits')
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--refs', type=int, nargs='*', default=None,
                    help='only these reference centers (default: every one of the cover)')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f'{args.out} exists; choose a new file')
    cov = np.load(args.cover)
    refs = args.refs if args.refs is not None else list(range(len(cov['ref_centers'])))
    commit = rs.commit_of(ROOT)
    t0 = time.perf_counter()
    records, shas, n_boxes = [], [], 0
    col = {c: k for k, c in enumerate(rs.METRIC_COLUMNS)}
    for r in refs:
        # One read of the file gives both the hash and the forms that the boxes use.
        raw = (Path(args.local) / f'center_{r}.npz').read_bytes()
        shas.append(hashlib.sha256(raw).hexdigest())
        local = tc.local_on_gpu(io.BytesIO(raw))
        t1 = time.perf_counter()
        recs, n_cells, n_ref = run_reference(cov, r, local, args)
        records.extend(recs)
        n_boxes += n_ref
        worst = max((x[col['k_atlas']] for x in recs), default=float('nan'))
        print(f'reference {r}: {n_cells} cells, {n_ref} boxes computed, {len(recs)} final boxes, '
              f'largest k_atlas {worst:.3f}; {time.perf_counter() - t1:.1f}s; total {n_boxes} '
              f'boxes in {time.perf_counter() - t0:.0f}s', flush=True)
        np.savez(args.out, records=np.array(records).reshape(-1, len(rs.METRIC_COLUMNS)),
                 target=args.k_target, columns=np.array(rs.METRIC_COLUMNS), commit=commit,
                 local_refs=np.array(refs[:len(shas)]), local_sha=np.array(shas))
    rec = np.array(records).reshape(-1, len(rs.METRIC_COLUMNS))
    m = lambda k: rec[:, col[k]]
    # np.max and np.min give NaN if one record has NaN, so no NaN can hide here.
    print(f'done: {len(rec)} boxes, {n_boxes} computed, depth up to {int(np.max(m("depth")))}; '
          f'k_atlas {np.max(m("k_atlas")):.4f} ({int(np.sum(~(m("k_atlas") <= args.k_target)))} '
          f'above {args.k_target}), k_t {np.max(m("k_t")):.4f}, k_tau {np.max(m("k_tau")):.4f}; '
          f'eigenvalues of G in the atlas in [{np.min(m("lam_atlas")):.6f}, '
          f'{np.max(m("Lam_atlas")):.6f}]; mu in [{np.min(m("mu_low")):.6f}, '
          f'{np.max(m("mu_up")):.6f}], largest ric {np.max(m("ric")):.4e}, largest lam_big '
          f'{np.max(m("lam_big")):.3f}; {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
