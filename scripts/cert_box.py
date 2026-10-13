"""The GPU Taylor models of f (quartic.cert.box) on boxes at given centers, with checks.

Each box has its center at a reference center of `scripts/tm_local_forms.py`, so the
local forms are those of the box, or (with --cover) at the cells of that reference center.
For each order and radius the script prints the bound of abs(f) on the box and the parts
of the remainder. With no cover a box has no cell, so h_max = 0 there, and the check of
implicit_h_td covers only the ball around the model of h.

The checks: the model of p0 at a point of the box against p0 in mpmath from the same
local forms; and f(c) against --fref (values from tm_box.py).

Usage, on one GPU node:
    uv run --extra gpu python -u scripts/cert_box.py \\
        --local <directory of the local forms> --centers 0 --orders 28 --deltas 0.01 0.02
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmgpu.host import dd_mp                                # noqa: E402
from tmgpu.tools import sup_bound                           # noqa: E402
from quartic.cert import box as tc                      # noqa: E402
from quartic.cert.forms import Poly2                          # noqa: E402

mp.mp.dps = 50


def tm_at(sp, c, k, x):
    """The polynomial of a model at a point x (list of mpmath), for box k."""
    h = cp.asnumpy(c[:, :, k])
    total = mp.mpc(0)
    for i, e in enumerate(sp.mons):
        v = mp.mpc(mp.mpf(h[0, i]) + mp.mpf(h[1, i]), mp.mpf(h[2, i]) + mp.mpf(h[3, i]))
        if v == 0:
            continue
        for xv, kk in zip(x, e):
            if kk:
                v *= xv ** kk
        total += v
    return total


def point_on_x(host, tau):
    """y(tau) on X: Newton's method in h for P(p + E tau + h n) = 0."""
    p, e, n = host['p'], host['e'], host['n']
    h = mp.mpc(0)
    for _ in range(30):
        y = [p[k] + e[0][k] * tau[0] + e[1][k] * tau[1] + h * n[k] for k in range(3)]
        val = sum(c ** 4 for c in y) + 1
        der = sum(4 * c ** 3 * m for c, m in zip(y, n))
        h = h - val / der
    return [p[k] + e[0][k] * tau[0] + e[1][k] * tau[1] + h * n[k] for k in range(3)]


def local_p0(local, y, k0):
    """p0 at a point y of X from the local base forms, in mpmath."""
    c = [dd_mp(local['c'][i]) for i in range(2)]
    s = [y[0] - c[0], y[1] - c[1]]
    p2 = Poly2(k0)
    mono = [s[0] ** a * s[1] ** b for a, b in p2.mons]
    monoc = [mp.conj(m) for m in mono]
    total = mp.mpc(0)
    # Block j has the factor Delta^j, with Delta = abs(y_2)^2 - s2.
    factor = abs(y[2]) ** 2 - dd_mp(local['s2']).real
    for e in range(4):
        q = local[f'base_j{e}']
        part = mp.mpc(0)
        for i in range(p2.size):
            row = mp.mpc(0)
            for j in range(p2.size):
                v = mp.mpc(mp.mpf(q[0, i, j]) + mp.mpf(q[1, i, j]) + mp.mpf(q[2, i, j]),
                           mp.mpf(q[3, i, j]) + mp.mpf(q[4, i, j]) + mp.mpf(q[5, i, j]))
                row += v * monoc[j]
            part += mono[i] * row
        total += part * factor ** e
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--centers', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--orders', type=int, nargs='+', default=[16, 20])
    ap.add_argument('--ny-extra', type=int, default=tc.NY_EXTRA)
    ap.add_argument('--repeat', type=int, default=1, help='copies of each box, for the timing')
    ap.add_argument('--a', type=float, nargs='+', default=[0.0],
                    help='the gap fractions of the Cauchy estimate (0: 2 / order)')
    ap.add_argument('--deltas', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.03])
    ap.add_argument('--cover', type=str, default='',
                    help='take the boxes from the cells of a cover (with --ref and --ncell)')
    ap.add_argument('--ref', type=int, default=0)
    ap.add_argument('--ncell', type=int, default=4)
    ap.add_argument('--fref', type=str, default='',
                    help='reference values of f(c), separated by commas')
    args = ap.parse_args()
    args.fref = [float(x) for x in args.fref.split(',') if x]
    print(cp.cuda.runtime.getDeviceProperties(0)['name'], flush=True)
    if args.cover:
        cov = np.load(args.cover)
        idx = np.flatnonzero(cov['ref'] == args.ref)
        pick = idx[np.linspace(0, len(idx) - 1, min(args.ncell, len(idx))).astype(int)]
        args.centers = [args.ref]
    for ci in args.centers:
        local = tc.local_on_gpu(Path(args.local) / f'center_{ci}.npz')
        k0 = int(local['k0'])
        local_host = np.load(Path(args.local) / f'center_{ci}.npz')
        cdd = [list(local_host['c'][i]) for i in range(2)]
        if args.cover:
            centers = [[list(x) for x in cov['centers'][i]] for i in pick]
            args.deltas = [float(cov['deltas'][i]) for i in pick]
            hmax = cp.asarray(np.array([float(cov['hmaxs'][i]) for i in pick] * args.repeat))
            off = [max(abs(dd_mp(c[k]) - dd_mp(cdd[k])) for k in range(2)) for c in centers]
            print(f'reference {ci}: cells {list(pick)}, offsets {[f"{float(o):.3f}" for o in off]}',
                  flush=True)
        deltas = list(args.deltas) * args.repeat
        nb = len(deltas)
        if not args.cover:
            hmax = cp.zeros(nb)
        fr = tc.make_frames(centers * args.repeat if args.cover else [cdd] * nb, k0)
        for n, a in [(n, a) for n in args.orders for a in args.a]:
            a = a or 2.0 / n
            if args.repeat > 1:
                tc.box_f(fr, local, deltas, n, n + args.ny_extra, a, hmax)
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            f, (d, d_f, d_phi), info, parts = tc.box_f(fr, local, deltas, n, n + args.ny_extra,
                                                      a, hmax)
            sup, c0, pb = sup_bound(parts['sp4'], f, d)
            cp.cuda.Device().synchronize()
            sec = time.perf_counter() - t0
            fc = dd_mp(cp.asnumpy(f.c[:, 0, 0]))
            ref = f', tm_box {args.fref[ci]:+.10e}' if ci < len(args.fref) else ''
            extra = f', bound of Delta {float(cp.max(info["bound_delta"])):.2e}'
            print(f'center {ci}, order {n} (tau alone {n + args.ny_extra}), a '
                  f'{a:.3f}{extra}: f(c) '
                  f'{mp.nstr(fc.real, 11)}{ref}; {sec:.1f}s for {nb} boxes ('
                  + ', '.join(f'{k} {v:.2f}s' for k, v in info['times'].items()) + ')', flush=True)
            # The model of p0 at a point of the smallest box against mpmath.
            k = int(np.argmin(args.deltas))
            small = args.deltas[k]
            tau = [mp.mpc(0.6 * small, 0.3 * small), mp.mpc(-0.2 * small, 0.5 * small)]
            y = point_on_x(fr.host[k], tau)
            p_tm = tm_at(parts['sp4'], parts['p0'].c, k, tau + [mp.conj(t) for t in tau])
            p_mp = local_p0(local_host, y, k0)
            print(f'    check p0 at a point: gap {float(abs(p_tm - p_mp)):.2e}, remainder '
                  f'{float(parts["p0"].rem[k]):.2e}, value {float(abs(p_mp)):.3e}', flush=True)
            h = {key: cp.asnumpy(val) for key, val in info.items() if isinstance(val, cp.ndarray)}
            pf = {k: cp.asnumpy(v) for k, v in info['parts_f0'].items()}
            for i in range(len(deltas[:len(args.deltas)])):
                print(f'    F_0 at box {i}: value {h["value_f0"][i]:.3e}, abs form over value '
                      f'{pf["abs"][i] / h["value_f0"][i]:.2e}; remainder parts: rounding '
                      f'{pf["round"][i]:.1e}, above the order {pf["high"][i]:.1e}, models '
                      f'{pf["models"][i]:.1e}, radii {pf["radius"][i]:.1e}, sums {pf["sums"][i]:.1e}',
                      flush=True)
            pp = {k: cp.asnumpy(v) for k, v in info['psi_parts'].items()}
            xp = {k: cp.asnumpy(v) for k, v in info['xi_parts'].items()}
            fr_ = cp.asnumpy(info['fk_rem'])
            for i, dl in enumerate(args.deltas):
                kk = int(np.argmax(fr_[:, i]))
                print(f'    psi at delta {dl:.4f}: ' + ', '.join(f'{k} {v[i]:.1e}' for k, v in pp.items())
                      + f'; largest F_k rem xi^k at k = {kk}: {fr_[kk, i]:.1e} | xi parts: '
                      + ', '.join(f'{k} {v[i]:.1e}' for k, v in xp.items()), flush=True)
            for i, dl in enumerate(args.deltas):
                print(f'    delta {dl:.4f}: abs(f) <= {float(sup[i]):.2e} (center {float(c0[i]):.2e},'
                      f' polynomial {float(pb[i]):.2e}, remainder {h["rem_f"][i]:.2e}) | h '
                      f'{h["rho_h"][i]:.1e} (lam {h["lam_h"][i]:.2f}, on the cell {h["lam_big"][i]:.2f}), '
                      f'p0 {h["rem_p0"][i]:.1e} rel, '
                      f'F_0 {h["rem_f0"][i]:.1e}, psi {h["rem_psi"][i]:.1e}, '
                      f'log p~ {h["rem_logpt"][i]:.1e}, log r2 {h["rem_logr2"][i]:.1e}, Phi '
                      f'{h["rem_phi"][i]:.1e}, g_00 {h["rem_g"][i]:.1e}; eta r2 {h["eta_r"][i]:.3f}, '
                      f'p~ {h["eta_p"][i]:.3f}, xi {h["bxi"][i]:.1e}', flush=True)


if __name__ == '__main__':
    main()
