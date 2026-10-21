"""The test of the steps of the box code: the stage `export`.

`export` replays the batch of the run that made the record of a test box, keeps the
intermediate models of quartic.cert.box.box_f for that box, and writes them to one npz file
for each box and each radius factor (`<name>_x<factor>.npz`). The comparison with an
independent arb reference reads these files (tests/box_reference.py).

The test boxes are rows of the records: B1 the largest bound, B2 the largest gradient bound,
B3 and B4 the largest bound of the reference 0 and 243, B5 the largest bound of depth 1, B6
the first row of the reference 297, and B7 and B8 two other rows chosen at random.

At the factor 1 the export checks that the replay gives the center of the record cell, and
that sup and grad_g equal the record bit for bit. At another factor the whole batch runs
again with the radii times the factor and the same h_max; only the test box is written. At a
factor above 1, a box with a sup that is not finite gives no file.

Usage, on one GPU node:
    uv run --extra gpu --locked python -u scripts/cert_steps.py export \\
        --run runs_tm/run_f6t_v2.npz --cover runs_tm/cover_s0.03_c6_v2.npz \\
        --local <directory of the local forms> --data runs_tm/f6t_data.npz \\
        --forms runs_tm/f6t_forms.npz --manifest runs_tm/f6t_local_manifest.txt \\
        --boxes B1 --factors 1 --out runs_tm/steps_b1
"""
import argparse
import io
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import cupy as cp
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from tmgpu.tools import space, sup_bound                    # noqa: E402
from quartic.cert import box as tc                          # noqa: E402
from quartic.cert import records as rs                      # noqa: E402
import cert_run                                             # noqa: E402
from cert_check import exact_cell                           # noqa: E402

SHA_DATA = '69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1'
SHA_FORMS = '2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2'
COL = {name: i for i, name in enumerate(rs.COLUMNS)}
ARGS = Namespace(order=28, order_high=32, rho_high=0.3, target=5e-20, depth=2, batch=64,
                 verbose=False)
SETTINGS = {28: 0.35, 32: 0.12}


def select_boxes(rec):
    """{name: row index} of the test boxes B1 to B8 in the records `rec`."""
    ref, depth, bound, grad = (rec[:, COL[k]] for k in ('ref', 'depth', 'bound', 'grad_g'))

    def largest(mask, x):
        rows = np.flatnonzero(mask)
        return int(rows[np.argmax(x[rows])])

    every = np.ones(len(rec), bool)
    rows = {'B1': largest(every, bound), 'B2': largest(every, grad),
            'B3': largest(ref == 0, bound), 'B4': largest(ref == 243, bound),
            'B5': largest(depth == 1, bound), 'B6': int(np.flatnonzero(ref == 297)[0])}
    others = np.setdiff1d(np.arange(len(rec)), list(rows.values()))
    pick = np.random.default_rng(20261019).choice(others, 2, replace=False)
    rows['B7'], rows['B8'] = int(pick[0]), int(pick[1])
    return rows


def git_commit():
    """The git commit of the code, with -dirty if a tracked file differs from it."""
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT).returncode != 0:
        commit += '-dirty'
    return commit


def host(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def replay(cov, r, local, row, side):
    """Run the batches of the reference r up to the batch with the record row.

    Returns (keep, j): the keep dict of box_f for that batch, and the index of the test
    box in it. The record must match exactly one queue item.
    """
    found = []

    def matches(q):
        lo, s, _, _, _, depth, order = q
        return (all(float(lo[i]) == row[COL[f'lo{i}']] for i in range(4))
                and float(s) == row[COL['side']] and depth == row[COL['depth']]
                and order == row[COL['order']])

    def on_batch(batch, order):
        hits = [j for j, q in enumerate(batch) if matches(q)]
        if len(hits) > 1:
            raise SystemExit(f'{len(hits)} queue items of one batch match the record')
        if not hits:
            return None
        found.append(hits[0])
        return {'stop': True}

    holder = []

    def hook(batch, order):
        keep = on_batch(batch, order)
        if keep is not None:
            holder.append((keep, batch))
        return keep

    cert_run.run_reference(cov, r, local, SETTINGS, ARGS, on_batch=hook)
    if len(holder) != 1:
        raise SystemExit(f'{len(holder)} batches match the record, not 1')
    keep, batch = holder[0]
    j = found[0]
    lo, s, cdd = exact_cell(row, side)
    center = np.array(cdd, dtype=float)
    if not np.array_equal(center, np.array(batch[j][2], dtype=float)):
        raise SystemExit('the center of the queue item differs from the center of the record')
    p = host(keep['fr'].p)[:2, :, j]
    if not np.array_equal(p, center):
        raise SystemExit('fr.p[0:2] of the test box differs from the center of the record')
    return keep, j


def model_arrays(out, name, m, j, nvars, n):
    """The keys m_<name>_c, _rem, _delta, _mons and _part_<key> of the model m, box j."""
    sp = space(nvars, n)
    c = host(m.c)[:, :, j]
    if c.shape[1] != sp.size:
        raise SystemExit(f'model {name}: {c.shape[1]} coefficients, the space has {sp.size}')
    out[f'm_{name}_c'] = c
    out[f'm_{name}_rem'] = float(host(m.rem)[j])
    out[f'm_{name}_delta'] = float(host(m.delta)[j])
    out[f'm_{name}_mons'] = np.array(sp.mons)
    for key, val in (getattr(m, 'parts', None) or {}).items():
        if isinstance(val, cp.ndarray) and val.shape == (len(host(m.rem)),):
            out[f'm_{name}_part_{key}'] = float(host(val)[j])


def box_arrays(keep, info, sup, j, local, row, order, ny, a, hmax, factor, shas, commit):
    """The dict of the arrays of the export file for the test box j."""
    out = {}
    top = int(local['top'])
    model_arrays(out, 'h', keep['h_td'], j, 2, ny)
    for k in range(3):
        model_arrays(out, f'y{k}', keep['ys'][k], j, 2, ny)
    for name in ('delta', 'p0', 'r2', 'ir2', 'log_r2', 'pt', 'xh', 'log_pt', 'xw', 'xi',
                 'psi', 'phi', 'det', 'q4', 'v', 'f'):
        model_arrays(out, name, keep[name], j, 4, order)
    for k in range(top + 1):
        model_arrays(out, f'fk{k}', keep['fk'][k], j, 4, keep['orders'][k])
    for (i, k), m in keep['g'].items():
        model_arrays(out, f'g{i}{k}', m, j, 4, order)
    for i in range(2):
        model_arrays(out, f'df{i}', keep['dfs'][i], j, 4, order)
    for key, val in info['psi_parts'].items():
        out[f"psi_part_{key.replace(' ', '_')}"] = float(host(val)[j])
    hp = {k: host(info['h_parts'][k])[j] for k in ('residual', 'host', 'lam0', 'rho')}
    for key, val in hp.items():
        out[f'h_parts_{key}'] = float(val)
    safe = tc.SAFE

    def radius(res, hst):
        return 2 * ((res + hst) * safe) / (1 - hp['lam0']) * safe

    if np.isfinite(hp['rho']):
        out['h_parts_rho_host'] = float(hp['rho'] - radius(hp['residual'], 0.0))
        out['h_parts_rho_residual'] = float(hp['rho'] - radius(0.0, hp['host']))
    else:
        out['h_parts_rho_host'] = out['h_parts_rho_residual'] = float('inf')
    out['xi_kap'] = float(host(keep['xi_kap'])[j])
    out['v_vol'] = float(host(keep['v_vol'])[j])
    fr = keep['fr']
    out['fr_p'] = host(fr.p)[:, :, j]
    out['fr_e'] = host(fr.e)[:, :, :, j]
    out['fr_n'] = host(fr.n)[:, :, j]
    out['fr_c1inv'] = host(fr.c1inv)[:, j]
    out['fr_vol'] = host(fr.vol)[:, j]
    out['fr_vol_err'] = float(host(fr.vol_err)[j])
    out['s_r'] = host(keep['s_r'])[:, j]
    out['s_p'] = host(keep['s_p'])[:, j]
    out['zeta'] = np.array(keep['zeta'][j], dtype=float)
    out['kappa'] = host(keep['kappa'])[:, j]
    out['kap_err'] = float(host(keep['kap_err'])[j])
    out['c_r'] = host(local['c'])
    out['x0'] = host(local['x0'])
    out['s2'] = host(local['s2'])
    out['k0'], out['top'] = int(local['k0']), top
    out['d'], out['d_f'], out['d_phi'] = (float(host(keep[k])[j]) for k in ('d', 'd_f', 'd_phi'))
    out.update(order=order, ny=ny, a=a, hmax=float(host(hmax)[j]), factor=factor, record=row,
               sup=float(host(sup)[j]))
    for key in ('grad_tau', 'grad_g', 'lam_low'):
        out[key] = float(host(info[key])[j])
    out['det_low'] = float(host(keep['det_low'])[j])
    out['g00_low'] = float(host(keep['g00_low'])[j])
    out['tr_up'] = float(host(keep['tr_up'])[j])
    out['df_bound0'] = float(host(keep['df_bounds'][0])[j])
    out['df_bound1'] = float(host(keep['df_bounds'][1])[j])
    out.update(shas)
    out['commit'] = commit
    return out


def export_box(name, row, cov, side, local, sha_local, factors, out_dir, shas, commit):
    """Write the files of the box `name` with the record `row`, one for each factor."""
    keep, j = replay(cov, int(row[COL['ref']]), local, row, side)
    order, a = int(row[COL['order']]), keep['a']
    ny = order + tc.NY_EXTRA
    sup = keep['sup']
    if not (float(host(sup)[j]) == row[COL['bound']]
            and float(host(keep['info']['grad_g'])[j]) == row[COL['grad_g']]):
        raise SystemExit(f'{name}: sup or grad_g differs from the record')
    for factor in factors:
        if factor == 1:
            kp, info, sp = keep, keep['info'], sup
            deltas = keep['deltas']
            print(f'{name}: sup and grad_g equal the record bit for bit '
                  f'(sup {row[COL["bound"]]:.6e}, grad_g {row[COL["grad_g"]]:.6e})', flush=True)
        else:
            deltas = [d * factor for d in keep['deltas']]
            kp = {}
            f, (d, _, _), info, parts = tc.box_f(keep['fr'], local, deltas, order, ny, a,
                                                 keep['hmaxs'], grad=True, keep=kp)
            sp, _, _ = sup_bound(parts['sp4'], f, d)
            kp['fr'] = keep['fr']
        s = float(host(sp)[j])
        if factor > 1 and not np.isfinite(s):
            print(f'{name}: factor {factor:g}: sup {s} is not finite, no file', flush=True)
            continue
        arrays = box_arrays(kp, info, sp, j, local, row, order, ny, a, keep['hmaxs'], factor,
                            {**shas, 'sha_local': sha_local}, commit)
        path = Path(out_dir) / f'{name}_x{factor:g}.npz'
        np.savez(path, **arrays)
        print(f'{name}: factor {factor:g}: sup {s:.6e}, {len(arrays)} keys, wrote {path}',
              flush=True)


def export(args):
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f'{out_dir} exists')
    sha_data, sha_forms = rs.sha256(args.data), rs.sha256(args.forms)
    if sha_data != SHA_DATA or sha_forms != SHA_FORMS:
        raise SystemExit(f'the SHA-256 of --data or --forms differs: {sha_data} {sha_forms}')
    manifest = rs.read_manifest(args.manifest)
    rec, _, run_commit, run_shas = rs.load([args.run])
    rows = select_boxes(rec)
    for name in args.boxes:
        if name not in rows:
            raise SystemExit(f'unknown box {name}')
    cover_file = np.load(args.cover)
    cov = {k: cover_file[k] for k in cover_file.files}
    side = float(cov['side'])
    locals_ = {}
    for name in args.boxes:
        r = int(rec[rows[name], COL['ref']])
        if r in locals_:
            continue
        path = Path(args.local) / f'center_{r}.npz'
        raw = path.read_bytes()
        digest = rs.sha256(path)
        errors = rs.manifest_errors({r: digest}, [r], manifest, sha_data, sha_forms)
        if errors or run_shas.get(r, digest) != digest:
            raise SystemExit(f'the local forms of reference {r} fail the manifest: {errors}')
        locals_[r] = (tc.local_on_gpu(io.BytesIO(raw)), digest)
    out_dir.mkdir(parents=True)
    commit = git_commit()
    shas = dict(sha_data=sha_data, sha_forms=sha_forms, sha_run=rs.sha256(args.run),
                sha_cover=rs.sha256(args.cover))
    for name in args.boxes:
        row = rec[rows[name]]
        local, digest = locals_[int(row[COL['ref']])]
        print(f'{name}: row {rows[name]} of the records: '
              f'{dict(zip(rs.COLUMNS, [float(v) for v in row]))}', flush=True)
        export_box(name, row, cov, side, local, digest, args.factors, out_dir, shas, commit)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('export', help='write the models of the test boxes')
    for opt in ('run', 'cover', 'local', 'data', 'forms', 'manifest', 'out'):
        ex.add_argument(f'--{opt}', type=str, required=True)
    ex.add_argument('--boxes', nargs='+', required=True, help='names B1 to B8')
    ex.add_argument('--factors', nargs='+', type=float, default=[1.0])
    args = ap.parse_args()
    if args.stage == 'export':
        export(args)


if __name__ == '__main__':
    main()
