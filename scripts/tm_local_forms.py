"""The local forms at reference centers, in arb, for the Taylor models of f on the GPU.

For each reference center c (the coordinates t = (z_0, z_1) of a point of X in the
chart z_3 = 1), the script moves the chart polynomials of the base and of the levels
to t = c, and combines the levels at x0 = p0(c)^(-1/k0) into M~_m (see
`quartic.cert.forms.local_forms`). All of it is in arb. Each center gives one file
with the dd parts, the radii and the absolute-value bounds of every matrix, and the
exact dd numbers c, x0 and s2. Each form is the four R_j of
`quartic.cert.forms.delta_split`, with each entry in td.

The centers come from --points (the export points of the data, moved into F) or from
a cover of scripts/cert_cover.py (--centers, its ref_centers).

Usage, on CPU cores:
    uv run --extra gpu --with python-flint python -u \\
        scripts/tm_local_forms.py --data runs_tm/r20i_data.npz \\
        --forms runs_tm/r20_forms.npz --points 0 1 2 --out runs_tm/local_r20i --procs 3
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

from quartic.cert import forms as tm_forms                                # noqa: E402

G = {}


def exact(x):
    return acb(x.real.mid(), x.imag.mid())


def dd_of(x):
    """An acb as (re_hi, re_lo, im_hi, im_lo); its value is taken as exact afterwards."""
    out = []
    for part in (x.real, x.imag):
        mid = part.mid()
        h = float(mid)
        out += [h, float(mid - arb(h))]
    return out


def from_dd(v):
    return acb(arb(v[0]) + arb(v[1]), arb(v[2]) + arb(v[3]))


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
    c = [from_dd(t[0]), from_dd(t[1])]
    y2 = center_point(c)
    p0 = chart_value(G['cf'], G['cf']['base'], G['cf']['k0'], c, y2)
    # x0 is a dd number, so that the GPU holds exactly the x0 of the M~_m; the file keeps
    # the same dd list that defines the ball of the forms.
    x0_dd = dd_of(exact((p0.real.log() * (-arb(1) / G['cf']['k0'])).exp()))
    x0 = from_dd(x0_dd)
    # s2 is a dd number near abs(y2)^2, taken as exact; the GPU forms Delta = abs(y2)^2 - s2.
    s2_dd = dd_of(exact(acb(abs(y2) ** 2)))
    s2 = from_dd(s2_dd).real
    out = tm_forms.local_forms(G['cf'], c, x0, s2)
    out['c'] = np.array([t[0], t[1]])
    out['x0'] = np.array(x0_dd)
    out['s2'] = np.array(s2_dd)
    out['parts'] = 6
    out['k0'] = G['cf']['k0']
    out['top'] = G['cf']['top']
    np.savez(Path(G['out']) / f'center_{i}.npz', **out)
    return i, time.perf_counter() - t0


def to_region(z):
    """The S_4 image with abs(z_1) <= abs(z_0) <= abs(z_2) <= abs(z_3)."""
    order = sorted(range(4), key=lambda k: abs(complex(z[k])))
    return [z[order[1]], z[order[0]], z[order[2]], z[order[3]]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--forms', type=str, required=True)
    ap.add_argument('--points', type=int, nargs='*', default=[])
    ap.add_argument('--centers', type=str, default='')
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
    centers = []
    for k in args.points:
        z = [acb(complex(a)) + acb(complex(b)) for a, b in zip(data['z_hi'][k], data['z_lo'][k])]
        z = to_region(z)
        t = [z[0] / z[3], z[1] / z[3]]
        centers.append([dd_of(exact(x)) for x in t])
    if args.centers:
        # The reference centers of a cover of scripts/cert_cover.py, (n, 2, 4) dd parts.
        for c in np.load(args.centers)['ref_centers']:
            centers.append([list(c[0]), list(c[1])])
    jobs = list(enumerate(centers))
    if args.only is not None:
        jobs = [jobs[i] for i in args.only]
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        for i, sec in pool.imap_unordered(one, jobs):
            print(f'center {i} in {sec:.0f}s', flush=True)
    print(f'{len(jobs)} centers in {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
