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
point:

* R1: the gap abs(P(x) - ref(x)), an upper bound in arb, is at most the remainder R;
* R2: for each part pi of R, gap / (R - pi); the test sees pi if this exceeds 1.

The models h and y_k take tau; the others take x = (tau_0, tau_1, sigma_0, sigma_1). The
stage writes report.txt and compare.npz into the directory, and stops if one of them
exists. It runs on a CPU node: only the stage export imports CuPy, tmgpu and quartic.cert.box.

Usage, on one GPU node:
    uv run --extra gpu --locked python -u scripts/cert_steps.py export \\
        --run runs_tm/run_f6t_v2.npz --cover runs_tm/cover_s0.03_c6_v2.npz \\
        --local <directory of the local forms> --data runs_tm/f6t_data.npz \\
        --forms runs_tm/f6t_forms.npz --manifest runs_tm/f6t_local_manifest.txt \\
        --boxes B1 --factors 1 --out runs_tm/steps_b1
Usage, on a CPU node with 16 cores:
    uv run --extra gpu --locked python -u scripts/cert_steps.py compare \\
        --steps runs_tm/steps_b1 --data runs_tm/f6t_data.npz --forms runs_tm/f6t_forms.npz \\
        --points 16 --procs 16
"""
import argparse
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
cp = space = sup_bound = tc = cert_run = exact_cell = None

SHA_DATA = '69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1'
SHA_FORMS = '2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2'
COL = {name: i for i, name in enumerate(rs.COLUMNS)}
ARGS = Namespace(order=28, order_high=32, rho_high=0.3, target=5e-20, depth=2, batch=64,
                 verbose=False)
SETTINGS = {28: 0.35, 32: 0.12}


def _gpu_imports():
    """Import CuPy, tmgpu, quartic.cert.box, cert_run and cert_check for the stage export."""
    global cp, space, sup_bound, tc, cert_run, exact_cell
    import cupy as cp
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
    _gpu_imports()
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


def load_models(z):
    """{name: Model} from the keys m_<name>_c, _rem, _delta and _mons of an export file."""
    names = sorted(k[2:-2] for k in z.files if k.startswith('m_') and k.endswith('_c'))
    return {name: Model(c=np.asarray(z[f'm_{name}_c'], dtype=float),
                        rem=float(z[f'm_{name}_rem']), delta=float(z[f'm_{name}_delta']),
                        mons=np.asarray(z[f'm_{name}_mons'], dtype=np.int64),
                        parts=_parts(z, name))
            for name in names}


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


_WORKER = {}


def _init_worker(data, forms):
    """Load the potential one time in each worker."""
    _WORKER['pot'] = _reference().load(data, forms)
    _WORKER['files'] = {}


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


def _largest(values):
    """The largest value; NaN if one value is NaN."""
    v = np.asarray(values, dtype=float)
    return math.nan if np.isnan(v).any() else float(v.max())


def _plan_file(path, count):
    """The checks of the inputs of one export file, and its points."""
    z = np.load(path)
    for key, want in (('sha_data', SHA_DATA), ('sha_forms', SHA_FORMS)):
        if str(z[key]) != want:
            raise SystemExit(f'{path.name}: {key} {z[key]} is not {want}')
    models = load_models(z)
    top = int(z['top'])
    names = step_models(top)
    missing = [n for n in names if n not in models]
    if missing:
        raise SystemExit(f'{path.name}: no model {" ".join(missing)}')
    d_phi = float(z['d_phi'])
    for n in names:
        m, nvars = models[n], 2 if n in TAU_MODELS else 4
        if m.delta != d_phi:
            raise SystemExit(f'{path.name}: model {n} has the radius {m.delta!r}, d_phi is '
                             f'{d_phi!r}')
        if m.mons.shape[1] != nvars:
            raise SystemExit(f'{path.name}: model {n} has {m.mons.shape[1]} variables')
    return dict(factor=float(z['factor']), d_phi=d_phi, order=int(z['order']), names=names,
                others=sorted(set(models) - set(names)),
                rems={n: models[n].rem for n in names},
                parts={n: models[n].parts for n in names},
                points=make_points(path.name, d_phi, count))


def _summary(plans, results, count, header):
    """The text of the report, the arrays of compare.npz, and the counts of the status."""
    lines = list(header)
    arrays = {}
    total = {'pass': 0, 'fail': 0, 'no claim': 0}
    worst = (-math.inf, '', '')
    for fname, plan in plans.items():
        stem = fname[:-len('.npz')]
        names = plan['names']
        res = [results[(fname, i)][0] for i in range(count)]
        times = np.array([results[(fname, i)][1] for i in range(count)])
        arrays[f'{stem}:points'] = plan['points']
        arrays[f'{stem}:times'] = times
        lines.append(f'{fname}: factor {plan["factor"]:g}, d_phi {plan["d_phi"]:.6e}, order '
                     f'{plan["order"]}, {count} points (0, 4, 8, ... on the slice)')
        lines.append(f'  {"model":<8} {"R":>10} {"largest gap":>12} {"gap / R":>10} '
                     f'{"pass":>5} {"fail":>5} {"no claim":>9}')
        r2_lines = []
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
                worst = (big, fname, n)
            lines.append(f'  {n:<8} {plan["rems"][n]:10.3e} {_largest(gaps):12.3e} '
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
                seen = 'seen' if big_r2 > 1 else 'not seen'
                rem = plan['rems'][n]
                over = part / rem if rem != 0 else math.inf
                r2_lines.append(f'  {n:<8} {key:<9} {part:10.3e} {over:10.3e} '
                                f'{big_r2:11.3e}  {seen}')
        lines.append('  R2 = gap / (R - part); the test sees a part if R2 exceeds 1 at a '
                     'point')
        lines.append(f'  {"model":<8} {"part":<9} {"value":>10} {"over R":>10} '
                     f'{"largest R2":>11}')
        lines.extend(r2_lines)
        lines.append('  the time of the reference for each point (s): solve_h at tau and at '
                     'conj(sigma), polar_point, step_values; and the models in arb')
        for i in range(count):
            t = times[i]
            lines.append(f'  point {i:2d}: {t[0]:8.3f} {t[1]:8.4f} {t[2]:8.3f} | {t[3]:8.2f}')
        lines.append(f'  not compared by this stage (the steps 10 to 12): '
                     f'{" ".join(plan["others"])}')
    lines.append(f'largest gap / R: {worst[0]:.3e} ({worst[2]} of {worst[1]})')
    lines.append(f'checks: {total["pass"]} pass, {total["fail"]} fail, {total["no claim"]} no '
                 f'claim')
    lines.append('RESULT: PASS' if total['fail'] == 0 else 'RESULT: FAIL')
    return '\n'.join(lines) + '\n', arrays, total


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
    plans = {p.name: _plan_file(p, args.points) for p in files}
    tasks = [(str(p), i, plans[p.name]['points'][i]) for p in files for i in range(args.points)]
    commit = git_commit()
    header = [f'compare: {steps}, {len(files)} files, {args.points} points each, commit '
              f'{commit}', f'  data {sha_data}', f'  forms {sha_forms}']
    print('\n'.join(header), flush=True)
    results = {}
    t_all = time.perf_counter()
    pool_ctx = multiprocessing.get_context('fork')
    with pool_ctx.Pool(args.procs, initializer=_init_worker,
                       initargs=(args.data, args.forms)) as pool:
        for path, i, out, times in pool.imap_unordered(_check_file_point, tasks):
            results[(Path(path).name, i)] = (out, times)
            worst = max(out, key=lambda n: np.nan_to_num(out[n]['ratio'], nan=np.inf))
            print(f'{Path(path).name} point {i}: ' +
                  ', '.join(f'{k} {t:.2f} s' for k, t in zip(TIME_KEYS, times)) +
                  f'; largest gap / R {out[worst]["ratio"]:.3e} ({worst})', flush=True)
    header.append(f'  wall time {time.perf_counter() - t_all:.0f} s on {args.procs} processes')
    text, arrays, total = _summary(plans, results, args.points, header)
    arrays.update(sha_data=sha_data, sha_forms=sha_forms, commit=commit)
    with open(npz_path, 'xb') as fh:
        np.savez(fh, **arrays)
    with open(report_path, 'x') as fh:
        fh.write(text)
    print(text, end='', flush=True)
    print(f'wrote {report_path} and {npz_path}', flush=True)
    return 1 if total['fail'] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('export', help='write the models of the test boxes')
    for opt in ('run', 'cover', 'local', 'data', 'forms', 'manifest', 'out'):
        ex.add_argument(f'--{opt}', type=str, required=True)
    ex.add_argument('--boxes', nargs='+', required=True, help='names B1 to B8')
    ex.add_argument('--factors', nargs='+', type=float, default=[1.0])
    co = sub.add_parser('compare', help='compare the models of the steps 1 to 9 with arb')
    for opt in ('steps', 'data', 'forms'):
        co.add_argument(f'--{opt}', type=str, required=True)
    co.add_argument('--points', type=int, default=16, help='points for each file')
    co.add_argument('--procs', type=int, default=16, help='worker processes')
    args = ap.parse_args()
    if args.stage == 'export':
        export(args)
    elif args.stage == 'compare':
        sys.exit(compare(args))


if __name__ == '__main__':
    main()
