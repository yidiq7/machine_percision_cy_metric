"""Independent checks of a run of scripts/cert_run.py.

1. The cover. Random points of F, from uniform t in the box around F and a test of
   the conditions of F in mpmath; each point must lie in a final box of the record:
   abs(tau_i) <= delta and abs(h) <= h_max, with (tau, h) = U^-1 (y - p) from the frame
   of that box.
2. The dropped cells. Random cells of the grid that the cover dropped must fail
   quartic.cert.cover.may_meet_f again, and no random point of F may lie in one.
3. The boxes. For random final boxes of the record, the GPU model of f at random
   points of the box (on the real slice sigma = conj(tau)) must agree with f from
   tm_box.py in arb (the chart z_3 = 1, m = 2, at order 2) within the remainder, the
   bound of the box must exceed abs(f) there, and the bound of the box computed again
   must agree with the record within --tol.
4. With --recompute R, all final boxes of R random reference centers again, in new
   batches: each bound of abs(f), and of abs(grad f) for a run with --grad, must agree
   with its record within --tol. The series order of a batch depends on its largest eta,
   so the agreement is not exact.
5. With --local, the file of each reference center and its center; the content of the
   files is the check of scripts/cert_local_values.py.

The centers of the boxes are exact: the corner of a record is a multiple of s / 2^depth.
The checks 3 and 4 take the defaults of cert_run.py for kmax (top) and --ny-extra.

Usage, on one GPU node:
    uv run --extra gpu --with cupy-cuda12x --with python-flint python -u \\
        scripts/cert_check.py --run runs_tm/run_r20.npz --cover runs_tm/cover_s0.03_c6.npz \\
        --local /n/netscratch/.../tm_local_c6 --data runs_tm/r20i_data.npz \\
        --forms runs_tm/r20_forms.npz --points 2000 --boxes 8
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from tmgpu.tools import sup_bound                           # noqa: E402
from quartic.cert import cover as cc                                     # noqa: E402
from quartic.cert import records as rs                                   # noqa: E402

mp.mp.dps = 40
RNG = np.random.default_rng(20261009)


def _nan_inf(x):
    """x, with NaN as infinity, so that max and <= count a NaN as a failure."""
    return x if x == x else math.inf


def in_f(t0, t1):
    """The conditions of F at a point t, in mpmath."""
    z2 = (-(1 + t0 ** 4 + t1 ** 4)) ** (mp.mpf(1) / 4)
    if not (abs(t1) <= abs(t0) <= abs(z2) <= 1):
        return False
    a0, a1 = mp.arg(t0), mp.arg(t1)
    return 0 <= a0 <= mp.pi / 4 and 0 <= a1 < mp.pi / 2


def exact_cell(row, side):
    """The exact lower corner and side (mpmath) and the dd center of the cell of a record;
    its corner is a multiple of s / 2^depth, with the depth of column 1."""
    unit = mp.mpf(side) / 2 ** int(row[1])
    k = [int(round(float(v) / float(unit))) for v in row[2:7]]
    if not max(abs(float(v) / float(unit) - kk) for v, kk in zip(row[2:7], k)) <= 1e-9:
        raise ValueError(f'the corner {row[2:7]} is not a multiple of s / 2^depth')
    lo = [mp.mpf(kk) * unit for kk in k[:4]]
    h = mp.mpf(k[4]) * unit
    c = [cc.dd(mp.mpc(lo[0] + h / 2, lo[1] + h / 2)), cc.dd(mp.mpc(lo[2] + h / 2, lo[3] + h / 2))]
    return lo, h, c


def tau_of(t, center):
    """The coordinates (tau_0, tau_1, h) of the point over t in the frame of the box at the
    center (dd lists)."""
    from tmgpu.host import dd_mp
    from quartic.cert.frame import frame
    f = frame(center, None)
    p, n = [dd_mp(v) for v in f['p']], [dd_mp(v) for v in f['n']]
    e = [[dd_mp(v) for v in row] for row in f['e']]
    z2 = (-(1 + t[0] ** 4 + t[1] ** 4)) ** (mp.mpf(1) / 4)
    # The branch of z2 nearest to that of the center.
    units = [1, 1j, -1, -1j]
    z2 = min((z2 * u for u in units), key=lambda v: abs(v - p[2]))
    y = [t[0], t[1], z2]
    u = mp.matrix([[e[0][i], e[1][i], n[i]] for i in range(3)])
    x = mp.lu_solve(u, mp.matrix([y[i] - p[i] for i in range(3)]))
    return [x[0], x[1], x[2]]


def check_cover(rec, cov, count):
    side = float(cov['side'])
    lo_all = rec[:, 2:6]
    sides = rec[:, 6]
    worst, worst_h, missed, done = 0.0, 0.0, 0, 0
    while done < count:
        t = [mp.mpc(RNG.uniform(0, 1), RNG.uniform(0, 0.7072)),
             mp.mpc(RNG.uniform(0, 1), RNG.uniform(0, 1))]
        if not in_f(*t):
            continue
        done += 1
        x = np.array([float(t[0].real), float(t[0].imag), float(t[1].real), float(t[1].imag)])
        # The FP64 search keeps a margin; the test of tau decides.
        hit = np.flatnonzero(np.all((lo_all - 1e-12 <= x) & (x <= lo_all + sides[:, None] + 1e-12),
                                    axis=1))
        if hit.size == 0:
            missed += 1
            continue
        ok = False
        for j in hit:
            x = tau_of(t, exact_cell(rec[j], side)[2])
            ratio = max(float(abs(v)) for v in x[:2]) / rec[j, 7]
            ratio_h = float(abs(x[2])) / rec[j, 12]
            worst, worst_h = max(worst, ratio), max(worst_h, ratio_h)
            ok = ok or (ratio <= 1 and ratio_h <= 1)
        missed += 0 if ok else 1
    print(f'cover: {count} random points of F, {missed} outside every box; largest '
          f'max abs(tau) / delta {worst:.4f}, largest abs(h) / h_max {worst_h:.4f} (cell side '
          f'{side})', flush=True)
    return missed == 0


def check_dropped(cov, count):
    s = float(cov['side'])
    side = mp.mpf(s)
    kept = {tuple(int(v) for v in i) for i in cov['idx']}
    grid = [int(np.ceil(1 / s)), int(np.ceil(0.7072 / s)), int(np.ceil(1 / s)), int(np.ceil(1 / s))]
    bad, tested = 0, 0
    while tested < count:
        idx = tuple(int(RNG.integers(g)) for g in grid)
        if idx in kept:
            continue
        tested += 1
        if cc.may_meet_f(cc.cell(idx, side)[0], side):
            bad += 1
    print(f'dropped cells: {tested} tested again, {bad} that may meet F', flush=True)
    return bad == 0


def _span(v):
    """The number of bits from the top bit of v[0] to the last bit of v[1] (dd parts)."""
    import math
    if v[1] == 0:
        return 53
    _, e0 = math.frexp(v[0])
    _, e1 = math.frexp(v[1])
    return e0 - e1 + 53


def check_local(cov, local_dir, prec=256):
    """Each reference center has a file with its exact center c, td forms (parts 6) in
    Delta, and dd numbers x0 and s2 that are exact balls at the precision of the forms.
    scripts/cert_local_values.py checks the content of the forms."""
    refs = len(cov['ref_centers'])
    bad = []
    for i in range(refs):
        path = Path(local_dir) / f'center_{i}.npz'
        if not path.exists():
            bad.append((i, 'no file'))
            continue
        z = np.load(path)
        if not np.array_equal(np.asarray(z['c']), np.asarray(cov['ref_centers'][i])):
            bad.append((i, 'c differs from the cover'))
        if int(z['parts']) != 6 or 's2' not in z.files:
            bad.append((i, 'not td forms in Delta'))
        x0, s2 = np.asarray(z['x0']), np.asarray(z['s2'])
        if max(_span(x0[:2]), _span(s2[:2])) > prec or x0[2] != 0 or s2[2] != 0:
            bad.append((i, 'x0 or s2 is not an exact ball'))
    print(f'local forms: {refs} reference centers, {len(bad)} with a defect {bad[:5]}',
          flush=True)
    return not bad


def check_boxes(rec, args):
    import cupy as cp
    from quartic.cert import box as tc
    sys.argv = ['tm_box']
    import tm_box as tb

    class A:
        data, forms, prec, col = args.data, args.forms, 192, 3
    tb.load(A)
    side = float(np.load(args.cover)['side'])
    picks = RNG.choice(len(rec), size=min(args.boxes, len(rec)), replace=False)
    worst_gap, worst_bound, worst_dev = 0.0, 0.0, 0.0
    for j in picks:
        r = int(rec[j, 0])
        lo_mp, h_mp, c = exact_cell(rec[j], side)
        lo, s, delta = [float(v) for v in lo_mp], float(h_mp), rec[j, 7]
        local = tc.local_on_gpu(Path(args.local) / f'center_{r}.npz')
        fr = tc.make_frames([c], int(local['k0']))
        # The order and gap fraction of the record (columns 14, 15), all k as models.
        order = int(rec[j, 14]) if rec.shape[1] > 14 else args.order
        a = float(rec[j, 15]) if rec.shape[1] > 15 else None
        grad = rec.shape[1] > 18 and bool(np.isfinite(rec[j, 17]))
        f, (d, _, _), info, parts = tc.box_f(fr, local, [delta], order, order + args.ny_extra,
                                             k_max=int(local['top']), a=a,
                                             hmax=cp.asarray([float(rec[j, 12])]), grad=grad)
        sup = float(sup_bound(parts['sp4'], f, d)[0][0])
        dev = abs(sup / rec[j, 8] - 1)
        if grad:
            dev = max(dev, abs(float(info['grad_g'][0]) / rec[j, 17] - 1))
        # A NaN counts as an infinite deviation.
        worst_dev = max(worst_dev, float(np.nan_to_num(dev, nan=np.inf)))
        sp4 = parts['sp4']
        coef = cp.asnumpy(f.c[:, :, 0])
        rem = float(f.rem[0])
        for _ in range(args.per_box):
            tpt = [mp.mpc(*RNG.uniform(lo[0:2], np.array(lo[0:2]) + s)),
                   mp.mpc(*RNG.uniform(lo[2:4], np.array(lo[2:4]) + s))]
            tau = tau_of(tpt, c)[:2]
            x = tau + [mp.conj(v) for v in tau]
            val = mp.mpc(0)
            for i, e in enumerate(sp4.mons):
                v = mp.mpc(mp.mpf(coef[0, i]) + mp.mpf(coef[1, i]), mp.mpf(coef[2, i]) + mp.mpf(coef[3, i]))
                for xv, kk in zip(x, e):
                    if kk:
                        v *= xv ** kk
                val += v
            z = [tpt[0], tpt[1], (-(1 + tpt[0] ** 4 + tpt[1] ** 4)) ** (mp.mpf(1) / 4), 1]
            from flint import acb
            tb.D['z'].append([acb(complex(v)) for v in z])
            tb.D['patch'].append([2, 0, 1])
            tb.D['ratio'].append(0.0)
            tb.D['v1'].append(acb(0))
            f_arb = tb.run_box((len(tb.D['z']) - 1, 2, [1e-4]))['f_center']
            gap = abs(float(val.real) - f_arb)
            worst_gap = max(worst_gap, _nan_inf(gap / rem))
            worst_bound = max(worst_bound, _nan_inf(abs(f_arb) / rec[j, 8]))
        print(f'box {j} (reference {r}, delta {delta:.4f}): largest gap over remainder '
              f'{worst_gap:.2e}, largest abs(f) over the bound {worst_bound:.2e}; bound again '
              f'{sup:.4e} against {rec[j, 8]:.4e} in the record', flush=True)
    print(f'boxes: largest deviation of a bound computed again from its record {worst_dev:.2e} '
          f'(at most {args.tol})', flush=True)
    return worst_gap <= 1 and worst_bound <= 1 and worst_dev <= args.tol


def check_recompute(rec, args):
    """All final boxes of --recompute random reference centers again, in new batches of one
    order and gap fraction, with the delta and h_max of their records."""
    import cupy as cp
    from quartic.cert import box as tc
    side = float(np.load(args.cover)['side'])
    grad = rec.shape[1] > 18 and bool(np.all(np.isfinite(rec[:, 17])))
    refs = np.unique(rec[:, 0].astype(int))
    pick = sorted(int(x) for x in RNG.choice(refs, size=min(args.recompute, len(refs)),
                                             replace=False))
    worst_f, worst_g, n_bad, n_box = 0.0, 0.0, 0, 0
    for r in pick:
        t0 = time.perf_counter()
        local = tc.local_on_gpu(Path(args.local) / f'center_{r}.npz')
        k0, top = int(local['k0']), int(local['top'])
        rows = np.flatnonzero(rec[:, 0] == r)
        keys = sorted({(int(rec[j, 14]), float(rec[j, 15])) for j in rows})
        dev_f, dev_g = [], []
        for order, a in keys:
            js = [int(j) for j in rows if int(rec[j, 14]) == order and float(rec[j, 15]) == a]
            for b in range(0, len(js), args.batch):
                part = js[b:b + args.batch]
                fr = tc.make_frames([exact_cell(rec[j], side)[2] for j in part], k0)
                f, (d, _, _), info, parts = tc.box_f(
                    fr, local, [float(rec[j, 7]) for j in part], order, order + args.ny_extra,
                    k_max=top, a=a, hmax=cp.asarray(rec[part, 12]), grad=grad)
                sup = cp.asnumpy(sup_bound(parts['sp4'], f, d)[0])
                dev_f.append(np.abs(sup / rec[part, 8] - 1))
                if grad:
                    dev_g.append(np.abs(cp.asnumpy(info['grad_g']) / rec[part, 17] - 1))
        dev_f = np.nan_to_num(np.concatenate(dev_f), nan=np.inf)
        dev_g = np.nan_to_num(np.concatenate(dev_g), nan=np.inf) if grad else np.zeros(1)
        n_bad += int(np.sum(~(dev_f <= args.tol))) + (int(np.sum(~(dev_g <= args.tol))) if grad else 0)
        n_box += len(rows)
        worst_f, worst_g = max(worst_f, float(dev_f.max())), max(worst_g, float(dev_g.max()))
        print(f'reference {r}: {len(rows)} final boxes again; largest deviation of the bound of '
              f'abs(f) {dev_f.max():.2e}' + (f', of abs(grad f) {dev_g.max():.2e}' if grad else '')
              + f'; {time.perf_counter() - t0:.0f}s', flush=True)
    print(f'recompute: {n_box} final boxes of {len(pick)} reference centers; largest deviation '
          f'of the bound of abs(f) {worst_f:.2e}' + (f', of abs(grad f) {worst_g:.2e}' if grad
                                                      else ' (no gradient in the run)')
          + f'; {n_bad} above {args.tol}', flush=True)
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=str, nargs='+', required=True, help='the files of all parts')
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, default='')
    ap.add_argument('--data', type=str, default='')
    ap.add_argument('--forms', type=str, default='')
    ap.add_argument('--order', type=int, default=28)
    ap.add_argument('--points', type=int, default=2000)
    ap.add_argument('--dropped', type=int, default=2000)
    ap.add_argument('--boxes', type=int, default=0)
    ap.add_argument('--per-box', type=int, default=3)
    ap.add_argument('--recompute', type=int, default=0,
                    help='compute all final boxes of this many random reference centers again')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ny-extra', type=int, default=6, help='the --ny-extra of the run')
    ap.add_argument('--tol', type=float, default=0.02,
                    help='the largest relative deviation of a bound computed again')
    args = ap.parse_args()
    for path in args.run + [args.cover]:
        print(f'{path}: SHA-256 {rs.sha256(path)}', flush=True)
    rec = rs.load(args.run)[0]
    cov = np.load(args.cover)
    ok = check_cover(rec, cov, args.points)
    ok = check_dropped(cov, args.dropped) and ok
    if args.local:
        ok = check_local(cov, args.local) and ok
    if args.boxes:
        ok = check_boxes(rec, args) and ok
    if args.recompute:
        ok = check_recompute(rec, args) and ok
    print('all checks pass' if ok else 'A CHECK FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
