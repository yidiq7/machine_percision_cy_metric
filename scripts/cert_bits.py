"""Compare the box code of two versions: the midpoints bit for bit, and the bounds box by box.

Stages:
* dump: for each reference center of --refs, the first batch of its cover cells runs through
  `cert_run.run_reference` with the hook of `box_f`. The file --out holds the batch (the lower
  corners, the side, the order), the coefficients (the midpoints) and the remainder of each
  kept model, the free numbers, and the bound and the gradient bound of each box.
* compare A B: two dumps. For each reference center, the batches must be the same. Then each
  model in the order of the steps: its coefficients bit for bit, and its remainder in B over A.
  The first model that differs shows where the two versions part; a move of the order of a
  series (through eta) shows first at ir2 or xh.
* records A B: two record files of cert_run. The boxes are joined on (ref, depth, lo0 to lo3,
  side, order), and the bound and the gradient bound of B over A are given for each box.

The stage dump uses the code of the repository in the present directory, so that one copy of
this file serves the old and the new code. Run it in the root of the repository:
    uv run --extra gpu --locked python -u scripts/cert_bits.py dump --cover C --local L \\
        --refs 0 243 297 231 --out D.npz
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

MODELS = (['h_td', 'h_dd', 'ys0', 'ys1', 'ys2', 'delta', 'p0', 'r2', 'ir2', 'log_r2', 'pt', 'xh',
           'log_pt', 'xw', 'xi'], ['psi', 'phi', 'g00', 'g01', 'g10', 'g11', 'det', 'q4', 'v', 'f',
                                   'df0', 'df1'])
FREE = ('s_r', 's_p', 'kappa')


def step_names(top):
    """The names of the kept models in the order of the steps, for F_0 to F_top."""
    return MODELS[0] + [f'fk{k}' for k in range(top + 1)] + MODELS[1]


def _models(keep):
    """{name: model} of the hook dict of box_f and gradient_bounds."""
    out = {'h_td': keep['h_td'], 'h_dd': keep['h_dd'], 'df0': keep['dfs'][0],
           'df1': keep['dfs'][1]}
    for k in range(3):
        out[f'ys{k}'] = keep['ys'][k]
    for name in MODELS[0][5:] + ['psi', 'phi', 'det', 'q4', 'v', 'f']:
        out[name] = keep[name]
    for k, m in enumerate(keep['fk']):
        out[f'fk{k}'] = m
    for (i, j), m in keep['g'].items():
        out[f'g{i}{j}'] = m
    return out


def dump(args):
    root = Path(os.getcwd())
    sys.path[:0] = [str(root), str(root / 'scripts')]
    import cupy as cp
    import cert_run
    from quartic.cert import box as tc

    cov = np.load(args.cover)
    run_args = Namespace(order=28, order_high=32, rho_high=0.3, target=5e-20, depth=2,
                         batch=64, verbose=False)
    settings = {28: 0.35, 32: 0.12}
    out = {}
    for r in args.refs:
        local = tc.local_on_gpu(str(Path(args.local) / f'center_{r}.npz'))
        held = {}

        def on_batch(batch, order, held=held):
            held['order'] = order
            held['keep'] = {'stop': True}
            return held['keep']

        cert_run.run_reference(cov, r, local, settings, run_args, on_batch=on_batch)
        keep = held['keep']
        top = int(local['top'])
        out[f'{r}/top'] = top
        out[f'{r}/order'] = held['order']
        out[f'{r}/lo'] = np.array([[float(v) for v in q[0]] for q in keep['batch']])
        out[f'{r}/side'] = np.array([float(q[1]) for q in keep['batch']])
        for name, m in _models(keep).items():
            out[f'{r}/{name}/c'] = cp.asnumpy(m.c)
            out[f'{r}/{name}/rem'] = cp.asnumpy(m.rem)
        for name in FREE:
            out[f'{r}/{name}'] = cp.asnumpy(cp.asarray(keep[name]))
        out[f'{r}/zeta'] = np.array(keep['zeta'], dtype=float)
        out[f'{r}/sup'] = cp.asnumpy(keep['sup'])
        out[f'{r}/grad_g'] = cp.asnumpy(keep['info']['grad_g'])
        print(f'reference {r}: order {held["order"]}, {len(keep["batch"])} boxes, '
              f'largest bound {float(cp.asnumpy(keep["sup"]).max()):.6e}', flush=True)
    out['refs'] = np.array(args.refs)
    np.savez(args.out, **out)


def same_bits(x, y):
    """True if the arrays have the same shape, dtype and bits (-0.0 differs from +0.0)."""
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(
        x.view(np.uint8), y.view(np.uint8))


def ratios(x, y):
    """(smallest, median, largest) of y / x over the entries where both are finite and x > 0."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not ok.any():
        return float('nan'), float('nan'), float('nan')
    q = y[ok] / x[ok]
    return float(q.min()), float(np.median(q)), float(q.max())


def compare(args):
    a, b = np.load(args.a, allow_pickle=False), np.load(args.b, allow_pickle=False)
    fail = False
    for r in a['refs'].tolist():
        if not all(same_bits(a[f'{r}/{k}'], b[f'{r}/{k}']) for k in ('lo', 'side', 'order', 'top')):
            print(f'reference {r}: the batches differ; no compare of the models', flush=True)
            fail = True
            continue
        names = step_names(int(a[f'{r}/top']))
        equal = [n for n in names if same_bits(a[f'{r}/{n}/c'], b[f'{r}/{n}/c'])]
        first = next((n for n in names if n not in equal), None)
        free = [n for n in FREE + ('zeta',) if not same_bits(a[f'{r}/{n}'], b[f'{r}/{n}'])]
        print(f'reference {r}: order {int(a[f"{r}/order"])}, {len(a[f"{r}/side"])} boxes; '
              f'{len(equal)} of {len(names)} models with the same midpoint bits; first that '
              f'differs: {first}; free numbers that differ: {free or "none"}', flush=True)
        fail = fail or first is not None or bool(free)
        for n in names:
            lo, med, hi = ratios(a[f'{r}/{n}/rem'], b[f'{r}/{n}/rem'])
            print(f'    {n:8s} remainder B / A: {lo:.6f} {med:.6f} {hi:.6f}', flush=True)
        for key in ('sup', 'grad_g'):
            lo, med, hi = ratios(a[f'{r}/{key}'], b[f'{r}/{key}'])
            print(f'    {key:8s} B / A: {lo:.6f} {med:.6f} {hi:.6f}', flush=True)
    print(f'RESULT: {"FAIL" if fail else "PASS"} (midpoints and free numbers bit for bit)',
          flush=True)
    return 1 if fail else 0


def records(args):
    keys = ('ref', 'depth', 'lo0', 'lo1', 'lo2', 'lo3', 'side', 'order')

    def table(path):
        z = np.load(path)
        cols = [str(c) for c in z['columns']]
        rec = np.asarray(z['records'])
        idx = [cols.index(k) for k in keys]
        bi, gi = cols.index('bound'), cols.index('grad_g')
        return {tuple(row[idx].tolist()): (row[bi], row[gi]) for row in rec}

    ta, tb = table(args.a), table(args.b)
    both = sorted(set(ta) & set(tb))
    print(f'{len(ta)} boxes in A, {len(tb)} in B, {len(both)} in both, {len(set(ta) - set(tb))} '
          f'only in A, {len(set(tb) - set(ta))} only in B', flush=True)
    for j, name in enumerate(('bound', 'grad_g')):
        x = np.array([ta[k][j] for k in both])
        y = np.array([tb[k][j] for k in both])
        lo, med, hi = ratios(x, y)
        above = int(np.sum(y > x * (1 + 1e-6)))
        print(f'{name}: B / A smallest {lo:.6f}, median {med:.6f}, largest {hi:.6f}; {above} boxes '
              f'above 1 + 1e-6; largest in A {np.nanmax(x):.6e}, in B {np.nanmax(y):.6e}',
              flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--cover', required=True)
    d.add_argument('--local', required=True)
    d.add_argument('--refs', type=int, nargs='+', required=True)
    d.add_argument('--out', required=True)
    for name in ('compare', 'records'):
        c = sub.add_parser(name)
        c.add_argument('a')
        c.add_argument('b')
    args = ap.parse_args()
    if args.stage == 'dump':
        if Path(args.out).exists():
            raise SystemExit(f'{args.out} exists; choose a new file')
        dump(args)
        return 0
    return compare(args) if args.stage == 'compare' else records(args)


if __name__ == '__main__':
    sys.exit(main())
