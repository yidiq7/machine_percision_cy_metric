"""The local forms at reference centers, in arb, for the Taylor models of f on the GPU.

For each reference center c (the coordinates t = (z_0, z_1) of a point of X in the
chart z_3 = 1), the script moves the chart polynomials of the base and of the levels
to t = c, and combines the levels at x0 = p0(c)^(-1/k0) into M~_m (see
`quartic.cert.forms.local_forms`). All of it is in arb. Each center gives one file
with the td parts, the radii and the absolute-value bounds of every matrix, and the
exact dd numbers c, x0 and s2. Each form is the four R_j of
`quartic.cert.forms.delta_split`.

The centers are the ref_centers of a cover of scripts/cert_cover.py (--centers); with
--only, the script writes the files of those reference centers alone.

Usage, on CPU cores:
    uv run --extra gpu python -u scripts/tm_local_forms.py --data runs_tm/f6t_data.npz \\
        --forms runs_tm/f6t_forms.npz --centers runs_tm/cover_s0.03_c6_v2.npz \\
        --out <directory> --procs 48
"""
import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import flint
from flint import acb, arb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmgpu.host import exact, to_dd                                       # noqa: E402
from quartic.cert import forms as tm_forms                                # noqa: E402

G = {}


def chart_value(cf, name_q, k_deg, t, y2):
    """sum over e of abs(y2)^(2e) h(t)^T Q_e conj(h(t)) for chart polynomials Q_e of degree k_deg."""
    p2 = tm_forms.Poly2(k_deg)
    mono = [t[0] ** a * t[1] ** b for a, b in p2.mons]
    total = acb(0)
    for e, q in enumerate(name_q):
        s = acb(0)
        for i in range(p2.size):
            if mono[i] == 0:
                continue
            row = acb(0)
            for j in range(p2.size):
                row += q[i, j] * mono[j].conjugate()
            s += mono[i] * row
        total += s * abs(y2) ** (2 * e)
    return total


def center_point(t):
    """y2 with y2^4 = -1 - t0^4 - t1^4 on the branch nearest to the positive real axis."""
    u = -(1 + t[0] ** 4 + t[1] ** 4)
    return u.root(4)


def one(job):
    i, t = job
    flint.ctx.prec = G['prec']
    t0 = time.perf_counter()
    c = [exact(t[0]), exact(t[1])]
    y2 = center_point(c)
    p0 = chart_value(G['cf'], G['cf']['base'], G['cf']['k0'], c, y2)
    # x0 is the dd rounding of the midpoint of p0(c)^(-1/k0), so that the GPU holds exactly
    # the x0 of the M~_m; the file keeps the same dd list that defines the ball of the forms.
    x0_dd = to_dd((p0.real.log() * (-arb(1) / G['cf']['k0'])).exp())[0]
    x0 = exact(x0_dd)
    # s2 is a dd number near abs(y2)^2, taken as exact; the GPU forms Delta = abs(y2)^2 - s2.
    s2_dd = to_dd(acb(abs(y2) ** 2))[0]
    s2 = exact(s2_dd).real
    out = tm_forms.local_forms(G['cf'], c, x0, s2)
    out['c'] = np.array([t[0], t[1]])
    out['x0'] = np.array(x0_dd)
    out['s2'] = np.array(s2_dd)
    out['parts'] = 6
    out['k0'] = G['cf']['k0']
    out['top'] = G['cf']['top']
    np.savez(Path(G['out']) / f'center_{i}.npz', **out)
    return i, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--forms', type=str, required=True)
    ap.add_argument('--centers', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--prec', type=int, default=256)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--only', type=int, nargs='*', default=None,
                    help='only these indices of the centers; the files keep the indices')
    args = ap.parse_args()
    G['prec'], G['out'] = args.prec, args.out
    Path(args.out).mkdir(parents=True, exist_ok=True)
    flint.ctx.prec = args.prec
    data, forms = np.load(args.data), np.load(args.forms)
    t0 = time.perf_counter()
    G['cf'] = tm_forms.chart_forms_arb(data, forms, args.prec)
    print(f'chart forms in {time.perf_counter() - t0:.0f}s', flush=True)
    # The reference centers of a cover of scripts/cert_cover.py, (n, 2, 4) dd parts.
    centers = [[list(c[0]), list(c[1])] for c in np.load(args.centers)['ref_centers']]
    jobs = list(enumerate(centers))
    if args.only is not None:
        jobs = [jobs[i] for i in args.only]
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        for i, sec in pool.imap_unordered(one, jobs):
            print(f'center {i} in {sec:.0f}s', flush=True)
    print(f'{len(jobs)} centers in {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
