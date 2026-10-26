"""The test of the steps of the box code: the stages `export` and `compare`.

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

`compare` reads every export file of a directory. At random points of the polydisc of
each model of the steps 1 to 9, it evaluates the polynomial of the model exactly in arb and
compares it with the independent reference of tests/box_reference.py at the same exact
point. The steps 10 to 12 (the metric, f and the gradient) take points of the slice
sigma = conj(tau) and the reference of tests/box_reference.metric_values, which uses
scripts/tm_box.py: the models g<i><j>, det, q4, v and f at points of D(d_f), and df0, df1
at points of D(d).

* R1: the gap abs(P(x) - ref(x)), an upper bound in arb, is at most the remainder R;
* R2: for each part pi of R, gap / (R - pi); the test sees pi if this exceeds 1. A model
  with parts also gets the part rest = max(R - the sum of the parts, 0);
* R3: at the points of D(d), the arb ball of the reference proves each scalar bound of
  gradient_bounds (abs(df_i) <= df_bound_i, det >= det_low, Re g00 >= g00_low,
  tr <= tr_up, lam_min >= lam_low, grad_G <= grad_g);
* R4: frame(center) and kappa of quartic.cert.frame give the frame numbers and kappa of the
  file bit for bit, and each of them lies within its error bound of mpmath at 300 digits
  (tests/host_reference.py).

The models h and y_k take tau; the others take x = (tau_0, tau_1, sigma_0, sigma_1). The
stage writes report.txt and compare.npz into the directory, and stops if one of them
exists. The exit status is 1 if R1, R3 or R4 fails. It runs on a CPU node: only the stage
export imports CuPy and quartic.cert.box; R4 imports quartic.cert.frame and tmgpu.host.

Usage, on one GPU node:
    uv run --extra gpu --locked python -u scripts/cert_steps.py export \\
        --run runs_tm/full_f6t_v3/run_0.npz ... runs_tm/full_f6t_v3/run_3.npz \\
        --cover runs_tm/cover_s0.03_c6_v3.npz --local <directory of the local forms> \\
        --data runs_tm/f6t_data.npz --forms runs_tm/f6t_forms.npz \\
        --manifest runs_tm/full_f6t_v3/local_manifest.txt --boxes B1 --factors 1 \\
        --out runs_tm/steps_b1
Usage, on a CPU node with 16 cores:
    uv run --extra gpu --locked python -u scripts/cert_steps.py compare \\
        --steps runs_tm/steps_b1 --data runs_tm/f6t_data.npz --forms runs_tm/f6t_forms.npz \\
        --points 16 --slice-points 16 --procs 16
"""
import argparse
import hashlib
import io
import math
import multiprocessing
import re
import subprocess
import sys
import time
import zlib
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flint import acb, arb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quartic.cert import records as rs                      # noqa: E402

# The GPU modules of the stage export; _gpu_imports sets them.
cp = mag = space = sup_bound = tc = cert_run = exact_cell = None

SHA_DATA = '69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1'
SHA_FORMS = '2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2'
COL = {name: i for i, name in enumerate(rs.COLUMNS)}
ARGS = Namespace(order=28, order_high=32, rho_high=0.3, target=5e-20, depth=2, batch=64,
                 verbose=False)
SETTINGS = {28: 0.35, 32: 0.12}


def _gpu_imports():
    """Import CuPy, tmgpu, quartic.cert.box, cert_run and cert_check for the stage export."""
    global cp, mag, space, sup_bound, tc, cert_run, exact_cell
    import cupy as cp
    from tmgpu import mag
    from tmgpu.tools import space, sup_bound
    from quartic.cert import box as tc
    import cert_run
    from cert_check import exact_cell


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

    def radius(res, hst):
        # The same operations as implicit_h_td: bd = mag.add(residual, eps), r = 2 bd / (1 - lam0).
        return 2 * float(mag.add(res, hst)) / (1 - hp['lam0'])

    if np.isfinite(hp['rho']):
        # rho of implicit_h_td is this radius of the residual and the host errors, bit for bit.
        if not radius(hp['residual'], hp['host']) == hp['rho']:
            raise SystemExit(f"radius(residual, host) {radius(hp['residual'], hp['host'])!r} "
                             f"differs from rho {hp['rho']!r}")
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
    _gpu_imports()
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f'{out_dir} exists')
    sha_data, sha_forms = rs.sha256(args.data), rs.sha256(args.forms)
    if sha_data != SHA_DATA or sha_forms != SHA_FORMS:
        raise SystemExit(f'the SHA-256 of --data or --forms differs: {sha_data} {sha_forms}')
    manifest = rs.read_manifest(args.manifest)
    rec, _, run_commit, run_shas = rs.load(args.run)
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
        # One read: the hash and the GPU take the same bytes.
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        errors = rs.manifest_errors({r: digest}, [r], manifest, sha_data, sha_forms)
        if errors or run_shas.get(r, digest) != digest:
            raise SystemExit(f'the local forms of reference {r} fail the manifest: {errors}')
        locals_[r] = (tc.local_on_gpu(io.BytesIO(raw)), digest)
    out_dir.mkdir(parents=True)
    commit = git_commit()
    # The SHA-256 of each part of the run, in the order of --run.
    shas = dict(sha_data=sha_data, sha_forms=sha_forms,
                sha_run=' '.join(rs.sha256(p) for p in args.run), sha_cover=rs.sha256(args.cover))
    for name in args.boxes:
        row = rec[rows[name]]
        local, digest = locals_[int(row[COL['ref']])]
        print(f'{name}: row {rows[name]} of the records: '
              f'{dict(zip(rs.COLUMNS, [float(v) for v in row]))}', flush=True)
        export_box(name, row, cov, side, local, digest, args.factors, out_dir, shas, commit)


# ---- The stage compare: the models of the steps 1 to 9 against the arb reference.

# The models in tau alone; the other models of the steps 1 to 9 are in (tau, sigma).
TAU_MODELS = ('h', 'y0', 'y1', 'y2')
POLAR_MODELS = ('delta', 'p0', 'r2', 'ir2', 'log_r2', 'pt', 'xh', 'log_pt', 'xw', 'xi')
# The parts of R2: of a composition in Delta (p0, fk<k>), of a product (pt, xw), of psi.
COMPOSE_PARTS = ('round', 'high', 'models', 'radius')
PRODUCT_PARTS = ('tail', 'round')
PSI_PARTS = ('F_k', 'xi', 'xi_cut', 'tails', 'rounding')
# A reference ball with a radius above this fraction of R makes the gap meaningless.
REF_RAD_MAX = 1e-3
POINT_SEED = 20261019
TIME_KEYS = ('solve_h', 'polar_point', 'step_values', 'models')
# The steps 10 and 11 on the slice of D(d_f), and the step 12 on the slice of D(d).
SLICE_MODELS = ('g00', 'g01', 'g10', 'g11', 'det', 'q4', 'v', 'f')
GRAD_MODELS = ('df0', 'df1')
SLICE_SEED = 12
SLICE_TIME_KEYS = ('solve_h', 'metric_values', 'models')
# R3: (the value of metric_values, kind, the bound of the export file, the text). An upper
# relation is value <= bound, a lower relation is value >= bound.
R3_RELATIONS = (('df0', 'upper', 'df_bound0', 'abs(df0) <= df_bound0'),
                ('df1', 'upper', 'df_bound1', 'abs(df1) <= df_bound1'),
                ('det', 'lower', 'det_low', 'Re det >= det_low'),
                ('g00', 'lower', 'g00_low', 'Re g00 >= g00_low'),
                ('tr', 'upper', 'tr_up', 'Re tr <= tr_up'),
                ('lam_min', 'lower', 'lam_low', 'lam_min >= lam_low'),
                ('grad_G', 'upper', 'grad_g', 'grad_G <= grad_g'))
# R4: the keys of quartic.cert.frame.frame and the keys of the export file.
R4_FRAME_KEYS = (('p', 'fr_p'), ('e', 'fr_e'), ('n', 'fr_n'), ('c1inv', 'fr_c1inv'),
                 ('vol', 'fr_vol'), ('vol_err', 'fr_vol_err'))


def _reference():
    """tests/box_reference.py. Its import sets the arb precision to 320 bits, so only the
    stage compare and its tests load it."""
    from tests import box_reference
    return box_reference


def step_models(top):
    """The names of the models of the steps 1 to 9, in the order of the steps."""
    return (list(TAU_MODELS) + list(POLAR_MODELS) + [f'fk{k}' for k in range(top + 1)]
            + ['psi', 'phi'])


@dataclass
class Model:
    """A model of an export file: the limbs c (4 or 6, S) of its coefficients, the
    remainder rem on the polydisc of radius delta, the exponents mons (S, nvars), and the
    parts of rem for R2."""
    c: np.ndarray
    rem: float
    delta: float
    mons: np.ndarray
    parts: dict = field(default_factory=dict)


def _parts(z, name):
    """The parts of the remainder of the model `name` for R2, from the export file z."""
    if name == 'p0' or re.fullmatch(r'fk\d+', name):
        return {k: float(z[f'm_{name}_part_{k}']) for k in COMPOSE_PARTS}
    if name in ('pt', 'xw'):
        return {k: float(z[f'm_{name}_part_{k}']) for k in PRODUCT_PARTS}
    if name == 'psi':
        return {k: float(z[f'psi_part_{k}']) for k in PSI_PARTS}
    if name == 'h':
        return {'host': float(z['h_parts_rho_host']),
                'residual': float(z['h_parts_rho_residual'])}
    if name == 'xi':
        return {'kap': float(z['xi_kap'])}
    if name == 'v':
        return {'vol': float(z['v_vol'])}
    return {}


def with_rest(parts, rem):
    """The parts with the part `rest` = max(R - the sum of the parts, 0) added, so that R2
    of the rest is gap / (R - rest), gap over the sum of the parts. No parts give no rest."""
    if not parts:
        return {}
    return {**parts, 'rest': max(rem - math.fsum(parts.values()), 0.0)}


def load_models(z):
    """{name: Model} from the keys m_<name>_c, _rem, _delta and _mons of an export file. A
    model with parts gets the part `rest` of with_rest too."""
    names = sorted(k[2:-2] for k in z.files if k.startswith('m_') and k.endswith('_c'))
    out = {}
    for name in names:
        rem = float(z[f'm_{name}_rem'])
        out[name] = Model(c=np.asarray(z[f'm_{name}_c'], dtype=float), rem=rem,
                          delta=float(z[f'm_{name}_delta']),
                          mons=np.asarray(z[f'm_{name}_mons'], dtype=np.int64),
                          parts=with_rest(_parts(z, name), rem))
    return out


def _up(a):
    """The smallest FP64 number at or above the upper bound of the arb a; NaN or inf if the
    bound is not finite."""
    u = a.upper()
    g = float(u)
    if math.isfinite(g) and arb(g) < u:
        g = math.nextafter(g, math.inf)
    return g


def check_point(model, x, ref, factor):
    """R1 and R2 of `model` at the point x (one acb for each variable) against the acb ref.

    gap: an FP64 upper bound of abs(P(x) - ref), so the radius of ref is in it. Status: a
    NaN R fails; an infinite R fails at the factor 1 and is "no claim" at another factor;
    otherwise gap <= R passes. R2 of a part pi is gap / (R - pi), and inf if R - pi <= 0.
    ValueError: a point off the polydisc of the model, or a reference radius above
    REF_RAD_MAX R for a finite R.
    """
    br = _reference()
    rem = float(model.rem)
    if not all(abs(v) <= arb(model.delta) for v in x):
        raise ValueError(f'the point lies off the polydisc of radius {model.delta}')
    ref_rad = _up(ref.real.rad() + ref.imag.rad())
    if math.isfinite(rem) and not ref_rad <= REF_RAD_MAX * rem:
        raise ValueError(f'the reference radius {ref_rad:.3e} exceeds {REF_RAD_MAX:g} R, '
                         f'R = {rem:.3e}')
    gap = _up(abs(br.poly_at(model.c, model.mons, x) - ref))
    if math.isnan(rem):
        status = 'fail'
    elif math.isinf(rem):
        status = 'fail' if factor == 1 else 'no claim'
    else:
        status = 'pass' if gap <= rem else 'fail'
    r2 = {}
    for key, part in model.parts.items():
        den = rem - part
        r2[key] = math.inf if den <= 0 else gap / den
    return dict(gap=gap, rem=rem, ratio=gap / rem if rem != 0 else math.inf,
                ref_rad=ref_rad, r2=r2, status=status)


def _down(a):
    """The largest FP64 number at or below the lower bound of the arb a; NaN or -inf if the
    bound is not finite."""
    low = a.lower()
    g = float(low)
    if math.isfinite(g) and arb(g) > low:
        g = math.nextafter(g, -math.inf)
    return g


def check_relation(value, bound, kind, factor):
    """R3 of one relation at one point: value <= bound (kind 'upper') or value >= bound
    (kind 'lower'), for the arb ball `value` of the reference and the FP64 `bound`.

    The relation passes only when the ball proves it: its upper bound is at most the bound
    (upper), or its lower bound is at least the bound (lower). A NaN bound, or a ball that
    is not finite, fails. A bound that makes no claim (+inf for upper, -inf for lower)
    fails at the factor 1 and is "no claim" at another factor, as in R1; the other infinite
    bound fails. ratio: value over bound (upper) or bound over value (lower), so that it is
    at most 1 exactly when the value satisfies the relation; NaN if a number is NaN.
    """
    if kind not in ('upper', 'lower'):
        raise ValueError(f'the kind {kind!r} is not upper or lower')
    bound = float(bound)
    upper = kind == 'upper'
    val = _up(value) if upper else _down(value)
    if math.isnan(val) or math.isnan(bound):
        ratio = math.nan
    elif upper:
        ratio = val / bound if bound > 0 else (0.0 if val <= bound else math.inf)
    else:
        ratio = bound / val if val > 0 else (0.0 if bound <= val else math.inf)
    empty = math.inf if upper else -math.inf
    if math.isnan(bound) or not math.isfinite(val):
        status = 'fail'
    elif bound == empty:
        status = 'fail' if factor == 1 else 'no claim'
    elif math.isinf(bound):
        status = 'fail'
    else:
        status = 'pass' if (val <= bound if upper else val >= bound) else 'fail'
    return dict(value=val, bound=bound, ratio=ratio, status=status)


def r3_value(name, vals):
    """The arb of the relation `name` of R3 from the values of metric_values: abs for
    df<i>, the real part for the others (their true values are real on the slice)."""
    v = vals[name]
    return abs(v) if name in GRAD_MODELS else v.real


def make_points(name, r, count):
    """`count` points of the polydisc D_4(r) as FP64 pairs (count, 4, 2): (re, im) of tau_0,
    tau_1, sigma_0 and sigma_1. Each coordinate has a random phase. The even points have the
    modulus 0.999 r, the odd points r u with u uniform in (0, 1). The points 0, 4, 8, ...
    lie on the slice sigma = conj(tau). The seed comes from the file name."""
    rng = np.random.default_rng([POINT_SEED, zlib.crc32(name.encode())])
    out = np.zeros((count, 4, 2))
    for i in range(count):
        for v in range(2 if i % 4 == 0 else 4):
            mod = 0.999 * r if i % 2 == 0 else r * float(rng.uniform())
            phase = float(rng.uniform(0.0, 2 * math.pi))
            out[i, v] = mod * math.cos(phase), mod * math.sin(phase)
        if i % 4 == 0:
            out[i, 2:, 0], out[i, 2:, 1] = out[i, :2, 0], -out[i, :2, 1]
    return out


def make_slice_points(name, r, count):
    """`count` points (tau, conj(tau)) of the slice of D_4(r) as FP64 pairs (count, 4, 2).
    Each tau_i has a random phase. The even points have the modulus 0.999 r, the odd points
    r u with u uniform in (0, 1). The seed comes from the name."""
    rng = np.random.default_rng([POINT_SEED, zlib.crc32(name.encode()), SLICE_SEED])
    out = np.zeros((count, 4, 2))
    for i in range(count):
        for v in range(2):
            mod = 0.999 * r if i % 2 == 0 else r * float(rng.uniform())
            phase = float(rng.uniform(0.0, 2 * math.pi))
            out[i, v] = mod * math.cos(phase), mod * math.sin(phase)
        out[i, 2:, 0], out[i, 2:, 1] = out[i, :2, 0], -out[i, :2, 1]
    return out


_WORKER = {}


def _init_worker(data, forms):
    """Load the potential one time in each worker."""
    _WORKER['pot'] = _reference().load(data, forms)
    _WORKER['files'] = {}
    _WORKER['bounds'] = {}
    _WORKER['paths'] = (data, forms)


def _tm_box():
    """scripts/tm_box.py after tm_box.load at 320 bits, one time in each worker."""
    if 'tb' not in _WORKER:
        import tm_box
        data, forms = _WORKER['paths']
        tm_box.load(Namespace(data=data, forms=forms, prec=320, col=3))
        _WORKER['tb'] = tm_box
    return _WORKER['tb']


def _file_bounds(path):
    """The bounds of R3 of an export file, read one time."""
    if path not in _WORKER['bounds']:
        z = np.load(path)
        _WORKER['bounds'][path] = {key: float(z[key]) for _, _, key, _ in R3_RELATIONS}
    return _WORKER['bounds'][path]


def _file_state(path):
    """The box, the models, the factor and the top level of an export file, read one time."""
    if path not in _WORKER['files']:
        z = np.load(path)
        _WORKER['files'][path] = (_reference().Box.from_npz(z), load_models(z),
                                  float(z['factor']), int(z['top']))
    return _WORKER['files'][path]


def _check_file_point(task):
    """Compute the reference at one point of one file, and check each model of the steps
    1 to 9 there. Returns (path, i, {model: check_point result}, times)."""
    from flint import ctx
    path, i, pt = task
    if ctx.prec != 320:
        raise RuntimeError(f'the arb precision is {ctx.prec}, not 320')
    br = _reference()
    box, models, factor, top = _file_state(path)
    # The exact point, built one time: the models and the reference get the same objects.
    x = [acb(float(a), float(b)) for a, b in pt]
    tau, sigma = x[:2], x[2:]
    sc = [s.conjugate() for s in sigma]
    hm = models['h']
    where = f'{Path(path).name}, point {i}'
    t0 = time.perf_counter()
    try:
        h_tau = br.solve_h(box, tau, br.poly_at(hm.c, hm.mons, tau))
        h_sc = br.solve_h(box, sc, br.poly_at(hm.c, hm.mons, sc))
        t1 = time.perf_counter()
        z, zb = br.polar_point(box, tau, sigma, h_tau, h_sc)
        t2 = time.perf_counter()
        vals = br.step_values(_WORKER['pot'], box, z, zb)
        t3 = time.perf_counter()
    except ValueError as e:
        raise ValueError(f'{where}: the reference fails: {e}') from None
    refs = {'h': h_tau, 'y0': z[0], 'y1': z[1], 'y2': z[2]}
    out = {}
    for name in step_models(top):
        tau_only = name in TAU_MODELS
        try:
            out[name] = check_point(models[name], tau if tau_only else x,
                                    refs[name] if tau_only else vals[name], factor)
        except ValueError as e:
            raise ValueError(f'{where}, model {name}: {e}') from None
    t4 = time.perf_counter()
    return path, i, out, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)


def _check_slice_point(task):
    """The reference of the steps 10 to 12 at one point (tau, conj(tau)) of one file. The
    kind 'f' checks the models SLICE_MODELS (R1); the kind 'd' checks GRAD_MODELS (R1) and
    the relations of R3. Returns (kind, path, i, {model: check_point result},
    {relation: check_relation result}, times)."""
    from flint import ctx
    kind, path, i, pt = task
    if ctx.prec != 320:
        raise RuntimeError(f'the arb precision is {ctx.prec}, not 320')
    br = _reference()
    tb = _tm_box()
    box, models, factor, _ = _file_state(path)
    # The exact point, built one time: the models and the reference get the same objects.
    x = [acb(float(a), float(b)) for a, b in pt]
    tau = x[:2]
    where = f'{Path(path).name}, slice point {kind} {i}'
    if not all(s == t.conjugate() for s, t in zip(x[2:], tau)):
        raise ValueError(f'{where}: sigma is not conj(tau)')
    hm = models['h']
    t0 = time.perf_counter()
    try:
        h = br.solve_h(box, tau, br.poly_at(hm.c, hm.mons, tau))
        z, _ = br.polar_point(box, tau, x[2:], h, h)
        t1 = time.perf_counter()
        vals = br.metric_values(box, z, tb)
        t2 = time.perf_counter()
    except ValueError as e:
        raise ValueError(f'{where}: the reference fails: {e}') from None
    if ctx.prec != 320:
        raise RuntimeError(f'{where}: the arb precision is {ctx.prec} after tm_box, not 320')
    out = {}
    for name in (SLICE_MODELS if kind == 'f' else GRAD_MODELS):
        try:
            out[name] = check_point(models[name], x, vals[name], factor)
        except ValueError as e:
            raise ValueError(f'{where}, model {name}: {e}') from None
    r3 = {}
    if kind == 'd':
        bounds = _file_bounds(path)
        for name, rel_kind, key, _ in R3_RELATIONS:
            r3[name] = check_relation(r3_value(name, vals), bounds[key], rel_kind, factor)
    t3 = time.perf_counter()
    return kind, path, i, out, r3, (t1 - t0, t2 - t1, t3 - t2)


def _host_modules():
    """quartic.cert.frame (the host numbers of the box code) and tests/host_reference.py
    (their mpmath reference). Only R4 loads them; they need no GPU."""
    from quartic.cert import frame
    from tests import host_reference
    return frame, host_reference


def _same_bits(a, b):
    """True if the FP64 arrays a and b have the same shape and the same bytes (so -0.0 and
    +0.0 differ)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _check_host(path):
    """R4 at the box of one export file: frame(center, k0) and kappa(s_p, k0, x0, zeta) of
    quartic.cert.frame against the numbers of the file, bit for bit, and frame_errors and
    kappa_errors of tests/host_reference.py. Returns (path, result dict)."""
    from flint import ctx
    frame, href = _host_modules()
    z = np.load(path)
    k0 = int(z['k0'])
    center = [[float(v) for v in z['fr_p'][i]] for i in range(2)]
    args = [[float(v) for v in z[key]] for key in ('s_p', 'x0', 'zeta')]
    equal, errors, error = {}, {}, ''
    try:
        f = frame.frame(center, k0)
        for key, fkey in R4_FRAME_KEYS:
            equal[fkey] = _same_bits(f[key], z[fkey])
        kap, kerr = frame.kappa(args[0], k0, args[1], args[2])
        equal['kappa'] = _same_bits(kap, z['kappa'])
        # box_f stores the error bound of kappa as frame.kappa gives it, a float with no
        # conversion; an error of another type is a failure of R4.
        if not isinstance(kerr, float):
            raise ValueError(f'the error bound of kappa is a {type(kerr).__name__}, not a float')
        equal['kap_err'] = _same_bits(np.array([kerr]), [float(z['kap_err'])])
        errors = {f'frame {k}': v for k, v in href.frame_errors([center], k0).items()}
        errors['kappa'] = href.kappa_errors([tuple(args)], k0)
    except ValueError as e:
        error = str(e)
    prec = ctx.prec
    if prec != 320:
        ctx.prec = 320
    ok = (not error and len(equal) == len(R4_FRAME_KEYS) + 2 and all(equal.values())
          and all(w <= 1 for w, _ in errors.values()))
    return path, dict(equal=equal, errors=errors, error=error, prec=prec,
                      status='pass' if ok else 'fail')


def _run_task(task):
    """One task of the pool: ('steps', ...) for the steps 1 to 9, ('slice', ...) for the
    steps 10 to 12, ('host', path) for R4."""
    kind, arg = task
    if kind == 'steps':
        return kind, _check_file_point(arg)
    if kind == 'slice':
        return kind, _check_slice_point(arg)
    if kind == 'host':
        return kind, _check_host(arg)
    raise ValueError(f'unknown task {kind}')


def _largest(values):
    """The largest value; NaN if one value is NaN."""
    v = np.asarray(values, dtype=float)
    return math.nan if np.isnan(v).any() else float(v.max())


def _plan_file(path, count, slice_count):
    """The checks of the inputs of one export file, and its points."""
    z = np.load(path)
    for key, want in (('sha_data', SHA_DATA), ('sha_forms', SHA_FORMS)):
        if str(z[key]) != want:
            raise SystemExit(f'{path.name}: {key} {z[key]} is not {want}')
    models = load_models(z)
    top = int(z['top'])
    names = step_models(top)
    slice_names = list(SLICE_MODELS) + list(GRAD_MODELS)
    missing = [n for n in names + slice_names if n not in models]
    missing += [key for _, _, key, _ in R3_RELATIONS if key not in z.files]
    if missing:
        raise SystemExit(f'{path.name}: no model or key {" ".join(missing)}')
    d_phi, d_f, d = (float(z[key]) for key in ('d_phi', 'd_f', 'd'))
    if not 0 < d < d_f < d_phi:
        raise SystemExit(f'{path.name}: the radii d {d!r}, d_f {d_f!r}, d_phi {d_phi!r} do '
                         f'not increase')
    # The steps 1 to 9 are on D(d_phi), the steps 10 and 11 on D(d_f), the step 12 on D(d).
    radius = {**{n: ('d_phi', d_phi) for n in names},
              **{n: ('d_f', d_f) for n in SLICE_MODELS}, **{n: ('d', d) for n in GRAD_MODELS}}
    for n in names + slice_names:
        m, nvars = models[n], 2 if n in TAU_MODELS else 4
        if m.delta != radius[n][1]:
            raise SystemExit(f'{path.name}: model {n} has the radius {m.delta!r}, '
                             f'{radius[n][0]} is {radius[n][1]!r}')
        if m.mons.shape[1] != nvars:
            raise SystemExit(f'{path.name}: model {n} has {m.mons.shape[1]} variables')
    return dict(factor=float(z['factor']), d_phi=d_phi, d_f=d_f, d=d, order=int(z['order']),
                names=names, others=sorted(set(models) - set(names) - set(slice_names)),
                rems={n: models[n].rem for n in names + slice_names},
                parts={n: models[n].parts for n in names + slice_names},
                bounds={key: float(z[key]) for _, _, key, _ in R3_RELATIONS},
                points=make_points(path.name, d_phi, count),
                slice_f=make_slice_points(f'{path.name}:d_f', d_f, slice_count),
                slice_d=make_slice_points(f'{path.name}:d', d, slice_count))


R1_HEADER = (f'  {"model":<8} {"R":>10} {"largest gap":>12} {"gap / R":>10} {"pass":>5} '
             f'{"fail":>5} {"no claim":>9}')
R2_HEADER = ['  R2 = gap / (R - part); the test sees a part if R2 exceeds 1 at a point',
             f'  {"model":<8} {"part":<9} {"value":>10} {"over R":>10} {"largest R2":>11}']


def r2_label(big):
    """The label of the largest R2 of a part: seen above 1, NaN for a NaN, else not seen."""
    if math.isnan(big):
        return 'NaN'
    return 'seen' if big > 1 else 'not seen'


def no_split(plans):
    """The models of the plans with no part of R in any file, in the order of the report."""
    names = []
    for plan in plans.values():
        for n in plan['names'] + list(SLICE_MODELS) + list(GRAD_MODELS):
            if n not in names:
                names.append(n)
    return [n for n in names if not any(plan['parts'].get(n) for plan in plans.values())]


def _summary(plans, results, slices, hosts, count, slice_count, header):
    """The text of the report, the arrays of compare.npz, and the counts of the status of
    R1, R3 and R4."""
    lines = list(header)
    arrays = {}
    total = {'pass': 0, 'fail': 0, 'no claim': 0}
    total3 = {'pass': 0, 'fail': 0, 'no claim': 0}
    total4 = {'pass': 0, 'fail': 0}
    worst = [-math.inf, '', '']

    def model_rows(fname, stem, plan, names, res):
        """The rows of R1 and of R2 of the models `names` over the point results res."""
        rows, r2_rows = [], []
        for n in names:
            gaps = [r[n]['gap'] for r in res]
            ratios = [r[n]['ratio'] for r in res]
            status = [r[n]['status'] for r in res]
            counts = {k: status.count(k) for k in total}
            for k in total:
                total[k] += counts[k]
            big = _largest(ratios)
            # A NaN ratio ranks above every number.
            if (math.inf if math.isnan(big) else big) > worst[0]:
                worst[:] = [big, fname, n]
            rows.append(f'  {n:<8} {plan["rems"][n]:10.3e} {_largest(gaps):12.3e} '
                        f'{big:10.3e} {counts["pass"]:5d} {counts["fail"]:5d} '
                        f'{counts["no claim"]:9d}')
            arrays[f'{stem}:{n}:gap'] = np.array(gaps)
            arrays[f'{stem}:{n}:rem'] = np.array(plan['rems'][n])
            arrays[f'{stem}:{n}:ref_rad'] = np.array([r[n]['ref_rad'] for r in res])
            arrays[f'{stem}:{n}:status'] = np.array(status)
            for key, part in plan['parts'][n].items():
                vals = [r[n]['r2'][key] for r in res]
                arrays[f'{stem}:{n}:part:{key}'] = np.array(part)
                arrays[f'{stem}:{n}:r2:{key}'] = np.array(vals)
                big_r2 = _largest(vals)
                seen = r2_label(big_r2)
                arrays[f'{stem}:{n}:r2_label:{key}'] = np.array(seen)
                rem = plan['rems'][n]
                over = part / rem if rem != 0 else math.inf
                r2_rows.append(f'  {n:<8} {key:<9} {part:10.3e} {over:10.3e} '
                               f'{big_r2:11.3e}  {seen}')
        return rows, r2_rows

    for fname, plan in plans.items():
        stem = fname[:-len('.npz')]
        res = [results[(fname, i)][0] for i in range(count)]
        times = np.array([results[(fname, i)][1] for i in range(count)])
        arrays[f'{stem}:points'] = plan['points']
        arrays[f'{stem}:times'] = times
        lines.append(f'{fname}: factor {plan["factor"]:g}, d_phi {plan["d_phi"]:.6e}, order '
                     f'{plan["order"]}, {count} points (0, 4, 8, ... on the slice)')
        rows, r2_rows = model_rows(fname, stem, plan, plan['names'], res)
        lines += [R1_HEADER] + rows + R2_HEADER + r2_rows
        lines.append('  the time of the reference for each point (s): solve_h at tau and at '
                     'conj(sigma), polar_point, step_values; and the models in arb')
        for i in range(count):
            t = times[i]
            lines.append(f'  point {i:2d}: {t[0]:8.3f} {t[1]:8.4f} {t[2]:8.3f} | {t[3]:8.2f}')

        # The steps 10 to 12 on the slice, and R3.
        res_f = [slices[(fname, 'f', i)] for i in range(slice_count)]
        res_d = [slices[(fname, 'd', i)] for i in range(slice_count)]
        for kind, rs_ in (('f', res_f), ('d', res_d)):
            arrays[f'{stem}:slice_{kind}:points'] = plan[f'slice_{kind}']
            arrays[f'{stem}:slice_{kind}:times'] = np.array([r[2] for r in rs_])
        lines.append(f'  the steps 10 to 12 on the slice sigma = conj(tau): {slice_count} points '
                     f'in D(d_f), d_f {plan["d_f"]:.6e}, for {" ".join(SLICE_MODELS)}; '
                     f'{slice_count} points in D(d), d {plan["d"]:.6e}, for '
                     f'{" ".join(GRAD_MODELS)} and R3')
        rows_f, r2_f = model_rows(fname, stem, plan, SLICE_MODELS, [r[0] for r in res_f])
        rows_d, r2_d = model_rows(fname, stem, plan, GRAD_MODELS, [r[0] for r in res_d])
        lines += [R1_HEADER] + rows_f + rows_d + R2_HEADER + r2_f + r2_d
        lines.append(f'  R3 at the {slice_count} points in D(d): a relation passes when the arb '
                     f'ball of the reference proves it; ratio = value / bound (upper), bound / '
                     f'value (lower)')
        lines.append(f'  {"relation":<22} {"bound":>11} {"largest ratio":>14} {"pass":>5} '
                     f'{"fail":>5} {"no claim":>9}')
        r3 = [r[1] for r in res_d]
        for name, _, key, text in R3_RELATIONS:
            status = [r[name]['status'] for r in r3]
            ratios = [r[name]['ratio'] for r in r3]
            counts = {k: status.count(k) for k in total3}
            for k in total3:
                total3[k] += counts[k]
            lines.append(f'  {text:<22} {plan["bounds"][key]:11.4e} {_largest(ratios):14.4e} '
                         f'{counts["pass"]:5d} {counts["fail"]:5d} {counts["no claim"]:9d}')
            arrays[f'{stem}:r3:{name}:value'] = np.array([r[name]['value'] for r in r3])
            arrays[f'{stem}:r3:{name}:bound'] = np.array(plan['bounds'][key])
            arrays[f'{stem}:r3:{name}:ratio'] = np.array(ratios)
            arrays[f'{stem}:r3:{name}:status'] = np.array(status)
        lines.append('  the time of the reference for each slice point (s): solve_h and '
                     'polar_point, metric_values (tm_box); and the models in arb')
        for kind, rs_ in (('f', res_f), ('d', res_d)):
            for i, r in enumerate(rs_):
                t = r[2]
                lines.append(f'  point {kind}{i:<2d}: {t[0]:8.3f} {t[1]:8.1f} | {t[2]:8.2f}')

        # R4: the host numbers of the box.
        h = hosts[fname]
        total4[h['status']] += 1
        eq = ', '.join(f'{k} {"yes" if v else "NO"}' for k, v in h['equal'].items())
        lines.append(f'  R4: frame(center, k0) and kappa(s_p, k0, x0, zeta) of quartic.cert.frame '
                     f'equal the numbers of the file bit for bit: {eq or "not computed"}')
        lines.append('  R4: error over bound against mpmath at 300 digits (count of nonzero '
                     'errors): ' + ', '.join(f'{k} {w:.6f} ({c})'
                                             for k, (w, c) in h['errors'].items()))
        if h['error']:
            lines.append(f'  R4: ValueError: {h["error"]}')
        restored = ' (restored to 320)' if h['prec'] != 320 else ''
        lines.append(f'  R4: the arb precision after the calls {h["prec"]}{restored}; R4 '
                     f'{h["status"]}')
        arrays[f'{stem}:r4:status'] = np.array(h['status'])
        for k, v in h['equal'].items():
            arrays[f'{stem}:r4:equal:{k}'] = np.array(v)
        for k, (w, c) in h['errors'].items():
            arrays[f'{stem}:r4:error:{k}'] = np.array([w, c], dtype=float)
        lines.append(f'  models of the file that this stage does not compare: '
                     f'{" ".join(plan["others"]) or "none"}')
    lines.append(f'models with no split of R (no rows of R2): '
                 f'{" ".join(no_split(plans)) or "none"}')
    lines.append(f'largest gap / R: {worst[0]:.3e} ({worst[2]} of {worst[1]})')
    lines.append(f'checks R1 (the steps 1 to 12): {total["pass"]} pass, {total["fail"]} fail, '
                 f'{total["no claim"]} no claim')
    lines.append(f'checks R3: {total3["pass"]} pass, {total3["fail"]} fail, '
                 f'{total3["no claim"]} no claim')
    lines.append(f'checks R4: {total4["pass"]} files pass, {total4["fail"]} fail')
    failed = total['fail'] or total3['fail'] or total4['fail']
    lines.append(f'RESULT: {"FAIL" if failed else "PASS"} (no claim: {total["no claim"]} checks '
                 f'of R1, {total3["no claim"]} checks of R3)')
    return '\n'.join(lines) + '\n', arrays, (total, total3, total4)


def compare(args):
    steps = Path(args.steps)
    report_path, npz_path = steps / 'report.txt', steps / 'compare.npz'
    for p in (report_path, npz_path):
        if p.exists():
            raise SystemExit(f'{p} exists')
    sha_data, sha_forms = rs.sha256(args.data), rs.sha256(args.forms)
    if sha_data != SHA_DATA or sha_forms != SHA_FORMS:
        raise SystemExit(f'the SHA-256 of --data or --forms differs: {sha_data} {sha_forms}')
    files = sorted(p for p in steps.glob('*.npz') if p.name != npz_path.name)
    if not files:
        raise SystemExit(f'{steps} has no export file')
    if args.points < 1 or args.slice_points < 1:
        raise SystemExit('--points and --slice-points must be at least 1')
    plans = {p.name: _plan_file(p, args.points, args.slice_points) for p in files}
    # The long tasks (tm_box) go first, so that the short ones fill the pool at the end.
    tasks = ([('host', str(p)) for p in files]
             + [('slice', (kind, str(p), i, plans[p.name][f'slice_{kind}'][i]))
                for p in files for kind in ('d', 'f') for i in range(args.slice_points)]
             + [('steps', (str(p), i, plans[p.name]['points'][i]))
                for p in files for i in range(args.points)])
    commit = git_commit()
    header = [f'compare: {steps}, {len(files)} files, {args.points} points each for the steps '
              f'1 to 9, {args.slice_points} + {args.slice_points} slice points each for the '
              f'steps 10 to 12, commit {commit}', f'  data {sha_data}', f'  forms {sha_forms}']
    print('\n'.join(header), flush=True)
    results, slices, hosts = {}, {}, {}
    t_all = time.perf_counter()
    pool_ctx = multiprocessing.get_context('fork')
    with pool_ctx.Pool(args.procs, initializer=_init_worker,
                       initargs=(args.data, args.forms)) as pool:
        for kind, res in pool.imap_unordered(_run_task, tasks):
            if kind == 'host':
                path, r4 = res
                hosts[Path(path).name] = r4
                print(f'{Path(path).name} R4: {r4["status"]}', flush=True)
                continue
            if kind == 'steps':
                path, i, out, times = res
                results[(Path(path).name, i)] = (out, times)
                where, keys, r3 = f'point {i}', TIME_KEYS, {}
            else:
                sk, path, i, out, r3, times = res
                slices[(Path(path).name, sk, i)] = (out, r3, times)
                where, keys = f'slice point {sk} {i}', SLICE_TIME_KEYS
            worst = max(out, key=lambda n: np.nan_to_num(out[n]['ratio'], nan=np.inf))
            r3_text = ''
            if r3:
                big = max(r3, key=lambda n: np.nan_to_num(r3[n]['ratio'], nan=np.inf))
                bad = [n for n in r3 if r3[n]['status'] == 'fail']
                r3_text = (f'; R3 largest ratio {r3[big]["ratio"]:.3e} ({big}), fail: '
                           f'{" ".join(bad) or "none"}')
            print(f'{Path(path).name} {where}: ' +
                  ', '.join(f'{k} {t:.2f} s' for k, t in zip(keys, times)) +
                  f'; largest gap / R {out[worst]["ratio"]:.3e} ({worst}){r3_text}',
                  flush=True)
    header.append(f'  wall time {time.perf_counter() - t_all:.0f} s on {args.procs} processes')
    text, arrays, totals = _summary(plans, results, slices, hosts, args.points,
                                    args.slice_points, header)
    arrays.update(sha_data=sha_data, sha_forms=sha_forms, commit=commit)
    with open(npz_path, 'xb') as fh:
        np.savez(fh, **arrays)
    with open(report_path, 'x') as fh:
        fh.write(text)
    print(text, end='', flush=True)
    print(f'wrote {report_path} and {npz_path}', flush=True)
    return 1 if any(t['fail'] for t in totals) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('export', help='write the models of the test boxes')
    ex.add_argument('--run', type=str, nargs='+', required=True,
                    help='the record files of all parts of one run (quartic.cert.records.load)')
    for opt in ('cover', 'local', 'data', 'forms', 'manifest', 'out'):
        ex.add_argument(f'--{opt}', type=str, required=True)
    ex.add_argument('--boxes', nargs='+', required=True, help='names B1 to B8')
    ex.add_argument('--factors', nargs='+', type=float, default=[1.0])
    co = sub.add_parser('compare', help='compare the models of the steps 1 to 12 with arb')
    for opt in ('steps', 'data', 'forms'):
        co.add_argument(f'--{opt}', type=str, required=True)
    co.add_argument('--points', type=int, default=16,
                    help='points for each file for the steps 1 to 9')
    co.add_argument('--slice-points', type=int, default=16,
                    help='slice points in D(d_f) and again in D(d) for each file')
    co.add_argument('--procs', type=int, default=16, help='worker processes')
    args = ap.parse_args()
    if args.stage == 'export':
        export(args)
    elif args.stage == 'compare':
        sys.exit(compare(args))


if __name__ == '__main__':
    main()
