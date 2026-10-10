"""Independent checks of a run of scripts/cert_run.py.

1. The cover. Random points of F, from uniform t in the box around F and a test of
   the conditions of F in mpmath; each point must lie in a final box of the record:
   abs(tau_i) <= delta, with tau = U^-1 (y - p) from the frame of that box.
2. The dropped cells. Random cells of the grid that the cover dropped must fail
   cert_cover.may_meet_f again, and no random point of F may lie in one.
3. The boxes. For random final boxes of the record, the GPU model of f at random
   points of the box (on the real slice sigma = conj(tau)) must agree with f from
   tm_box.py in arb (the chart z_3 = 1, m = 2, at order 2) within the remainder, and
   the bound of the box must exceed abs(f) there.

Usage, on one GPU node:
    uv run --extra gpu --with cupy-cuda12x --with python-flint python -u \\
        scripts/cert_check.py --run runs_tm/run_r20.npz --cover runs_tm/cover_s0.03_c6.npz \\
        --local /n/netscratch/.../tm_local_c6 --data runs_tm/r20i_data.npz \\
        --forms runs_tm/r20_forms.npz --points 2000 --boxes 8
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mpmath as mp

# The quartic package imports JAX, which would take most of the GPU memory from CuPy.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import cert_cover as cc                                     # noqa: E402

mp.mp.dps = 40
RNG = np.random.default_rng(20261009)


def in_f(t0, t1):
    """The conditions of F at a point t, in mpmath."""
    z2 = (-(1 + t0 ** 4 + t1 ** 4)) ** (mp.mpf(1) / 4)
    if not (abs(t1) <= abs(t0) <= abs(z2) <= 1):
        return False
    a0, a1 = mp.arg(t0), mp.arg(t1)
    return 0 <= a0 <= mp.pi / 4 and 0 <= a1 < mp.pi / 2


def tau_of(t, center):
    """The tangent coordinates of the point over t in the box at the center (dd lists)."""
    from quartic import tm_cert as tc
    fr = tc.make_frames([center], 14)
    h = fr.host[0]
    p, e, n = h['p'], h['e'], h['n']
    z2 = (-(1 + t[0] ** 4 + t[1] ** 4)) ** (mp.mpf(1) / 4)
    # The branch of z2 nearest to that of the center.
    units = [1, 1j, -1, -1j]
    z2 = min((z2 * u for u in units), key=lambda v: abs(v - p[2]))
    y = [t[0], t[1], z2]
    u = mp.matrix([[e[0][i], e[1][i], n[i]] for i in range(3)])
    x = mp.lu_solve(u, mp.matrix([y[i] - p[i] for i in range(3)]))
    return [x[0], x[1]]


def check_cover(rec, cov, count):
    side = float(cov['side'])
    lo_all = rec[:, 2:6]
    sides = rec[:, 6]
    worst, missed, done = 0.0, 0, 0
    while done < count:
        t = [mp.mpc(RNG.uniform(0, 1), RNG.uniform(0, 0.7072)),
             mp.mpc(RNG.uniform(0, 1), RNG.uniform(0, 1))]
        if not in_f(*t):
            continue
        done += 1
        x = np.array([float(t[0].real), float(t[0].imag), float(t[1].real), float(t[1].imag)])
        hit = np.flatnonzero(np.all((lo_all <= x) & (x <= lo_all + sides[:, None]), axis=1))
        if hit.size == 0:
            missed += 1
            continue
        ok = False
        for j in hit:
            lo, s = list(rec[j, 2:6]), rec[j, 6]
            c = [cc.dd(mp.mpc(lo[0] + s / 2, lo[1] + s / 2)), cc.dd(mp.mpc(lo[2] + s / 2, lo[3] + s / 2))]
            tau = tau_of(t, c)
            ratio = max(float(abs(v)) for v in tau) / rec[j, 7]
            worst = max(worst, ratio)
            ok = ok or ratio <= 1
        missed += 0 if ok else 1
    print(f'cover: {count} random points of F, {missed} outside every box; largest '
          f'max abs(tau) / delta {worst:.4f} (cell side {side})', flush=True)
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
    Delta, and dd numbers x0 and s2 that are exact balls at the precision of the forms."""
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
    from quartic import tm_cert as tc
    sys.argv = ['tm_box']
    import tm_box as tb

    class A:
        data, forms, prec, col = args.data, args.forms, 192, 3
    tb.load(A)
    picks = RNG.choice(len(rec), size=min(args.boxes, len(rec)), replace=False)
    worst_gap, worst_bound = 0.0, 0.0
    for j in picks:
        r = int(rec[j, 0])
        lo, s, delta = list(rec[j, 2:6]), rec[j, 6], rec[j, 7]
        local = tc.local_on_gpu(Path(args.local) / f'center_{r}.npz')
        c = [cc.dd(mp.mpc(lo[0] + s / 2, lo[1] + s / 2)), cc.dd(mp.mpc(lo[2] + s / 2, lo[3] + s / 2))]
        fr = tc.make_frames([c], int(local['k0']))
        # The order and gap fraction of the record (columns 14, 15), all k as models.
        order = int(rec[j, 14]) if rec.shape[1] > 14 else args.order
        a = float(rec[j, 15]) if rec.shape[1] > 15 else None
        f, (d, _, _), info, parts = tc.box_f(fr, local, [delta], order, order + 6,
                                             k_max=int(local['top']), a=a)
        sp4 = parts['sp4']
        coef = cp.asnumpy(f.c[:, :, 0])
        rem = float(f.rem[0])
        for _ in range(args.per_box):
            tpt = [mp.mpc(*RNG.uniform(lo[0:2], np.array(lo[0:2]) + s)),
                   mp.mpc(*RNG.uniform(lo[2:4], np.array(lo[2:4]) + s))]
            tau = tau_of(tpt, c)
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
            worst_gap = max(worst_gap, gap / rem)
            worst_bound = max(worst_bound, abs(f_arb) / rec[j, 8])
        print(f'box {j} (reference {r}, delta {delta:.4f}): largest gap over remainder '
              f'{worst_gap:.2e}, largest abs(f) over the bound {worst_bound:.2e}', flush=True)
    return worst_gap <= 1 and worst_bound <= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=str, required=True)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, default='')
    ap.add_argument('--data', type=str, default='')
    ap.add_argument('--forms', type=str, default='')
    ap.add_argument('--order', type=int, default=28)
    ap.add_argument('--points', type=int, default=2000)
    ap.add_argument('--dropped', type=int, default=2000)
    ap.add_argument('--boxes', type=int, default=0)
    ap.add_argument('--per-box', type=int, default=3)
    args = ap.parse_args()
    rec = np.load(args.run)['records']
    cov = np.load(args.cover)
    ok = check_cover(rec, cov, args.points)
    ok = check_dropped(cov, args.dropped) and ok
    if args.local:
        ok = check_local(cov, args.local) and ok
    if args.boxes:
        ok = check_boxes(rec, args) and ok
    print('all checks pass' if ok else 'A CHECK FAILED', flush=True)


if __name__ == '__main__':
    main()
