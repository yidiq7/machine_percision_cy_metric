"""The GPU Taylor models of f (quartic.tm_cert) on boxes at given centers, with checks.

Each box has its center at a reference center of `scripts/tm_local_forms.py`, so the
local forms are those of the box. For each order and radius the script prints the bound
of abs(f) on the box and the parts of the remainder, as `scripts/tm_ref.py` does.

The checks: the model of p0 at a point of the box against p0 in mpmath from the same
local forms; and f(c) against --fref (values from tm_box.py or tm_ref.py).

Usage, on one GPU node:
    uv run --extra gpu --with cupy-cuda12x python -u scripts/cert_box.py \\
        --local runs_tm/local_r20i --centers 0 1 2 --orders 16 20 24 --ny-extra 6 \\
        --deltas 0.005 0.01 0.02 0.03 --fref=-3.03e-19,-1.33e-18,-2.23e-19
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

from quartic import tmc, tm_cert as tc                      # noqa: E402
from quartic.tm_forms import Poly2                          # noqa: E402

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
    c = [tc.dd_mp(local['c'][i]) for i in range(2)]
    s = [y[0] - c[0], y[1] - c[1]]
    p2 = Poly2(k0)
    mono = [s[0] ** a * s[1] ** b for a, b in p2.mons]
    monoc = [mp.conj(m) for m in mono]
    total = mp.mpc(0)
    for e in range(4):
        q = local[f'base_e{e}']
        part = mp.mpc(0)
        for i in range(p2.size):
            row = mp.mpc(0)
            for j in range(p2.size):
                v = mp.mpc(mp.mpf(q[0, i, j]) + mp.mpf(q[1, i, j]),
                           mp.mpf(q[2, i, j]) + mp.mpf(q[3, i, j]))
                row += v * monoc[j]
            part += mono[i] * row
        total += part * abs(y[2]) ** (2 * e)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--centers', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--orders', type=int, nargs='+', default=[16, 20])
    ap.add_argument('--ny-extra', type=int, default=6)
    ap.add_argument('--kmax', type=int, default=12)
    ap.add_argument('--deltas', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.03])
    ap.add_argument('--fref', type=str, default='',
                    help='reference values of f(c), separated by commas')
    args = ap.parse_args()
    args.fref = [float(x) for x in args.fref.split(',') if x]
    print(cp.cuda.runtime.getDeviceProperties(0)['name'], flush=True)
    for ci in args.centers:
        local = tc.local_on_gpu(Path(args.local) / f'center_{ci}.npz')
        k0 = int(local['k0'])
        local_host = np.load(Path(args.local) / f'center_{ci}.npz')
        cdd = [list(local_host['c'][i]) for i in range(2)]
        nb = len(args.deltas)
        fr = tc.make_frames([cdd] * nb, k0)
        for n in args.orders:
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            f, (d, d_f, d_phi), info, parts = tc.box_f(fr, local, args.deltas, n,
                                                      n + args.ny_extra, args.kmax)
            sup, c0, pb = tc.sup_bound(parts['sp4'], f, d)
            cp.cuda.Device().synchronize()
            sec = time.perf_counter() - t0
            fc = tc.dd_mp(cp.asnumpy(f.c[:, 0, 0]))
            ref = f', tm_box {args.fref[ci]:+.10e}' if ci < len(args.fref) else ''
            print(f'center {ci}, order {n} (tau alone {n + args.ny_extra}): f(c) '
                  f'{mp.nstr(fc.real, 11)}{ref}; {sec:.1f}s for {nb} boxes', flush=True)
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
            for i, dl in enumerate(args.deltas):
                print(f'    delta {dl:.4f}: abs(f) <= {float(sup[i]):.2e} (center {float(c0[i]):.2e},'
                      f' polynomial {float(pb[i]):.2e}, remainder {h["rem_f"][i]:.2e}) | h '
                      f'{h["rho_h"][i]:.1e} (lam {h["lam_h"][i]:.2f}), p0 {h["rem_p0"][i]:.1e} rel, '
                      f'F_0 {h["rem_f0"][i]:.1e}, psi {h["rem_psi"][i]:.1e} (tail {h["tail"][i]:.1e}), '
                      f'log p~ {h["rem_logpt"][i]:.1e}, log r2 {h["rem_logr2"][i]:.1e}, Phi '
                      f'{h["rem_phi"][i]:.1e}, g_00 {h["rem_g"][i]:.1e}; eta r2 {h["eta_r"][i]:.3f}, '
                      f'p~ {h["eta_p"][i]:.3f}, xi {h["bxi"][i]:.1e}', flush=True)


if __name__ == '__main__':
    main()
