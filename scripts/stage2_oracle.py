"""An mpmath reference for v of a two-stage fit of `vpa_dd_stage2.py` at a few points.

The model is Phi = log p0 + psi_1 + psi_2. psi_1 is the sum over the joint functions
of the first stage, whose level functions take the weight |z|^(2 k1) p0^(-k1/k0).
psi_2 is the sum over the joint functions of the second stage, whose level functions
take the weight |z|^(2 k1) p0^(-k1/k0) exp(-(k1/k0)(psi_1 - C)). The reference takes
the same steps in mpmath at 100 digits, with the functions of `level_oracle.py`, and
the Hessian of psi_1 + psi_2 along X from second differences of its values.

The stage `export` runs on the GPU and saves every number of the model and the dd v
at the chosen points in two charts. The stage `reference` needs numpy and mpmath only;
it is CPU work and evaluates psi at the points of the differences in parallel.

Usage, on one GPU node, then on CPU cores:
    uv run --extra gpu python -u scripts/stage2_oracle.py export --k0 12 \\
        --theta runs_explore/f2r/theta_k0_12.npy --stage1 runs_explore/s1b \\
        --stage1-k1 24 23 22 21 20 19 18 17 --k1 36 35 34 33 32 31 30 29 \\
        --load runs_explore/f2r --run runs_explore/s2d --exact-mul \\
        --data runs_stage2_oracle/data.npz
    uv run --extra gpu python -u scripts/stage2_oracle.py reference \\
        --data runs_stage2_oracle/data.npz --procs 16
"""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import level_oracle as lo                                   # noqa: E402


def _save_part(out, key, levels, jm, prefix, coeffs):
    """The level bases, the joint map and the coefficients of one stage, under `key`."""
    out[f'{key}_k1'] = np.array([lev.k1 for lev in levels])
    out[f'{key}_coeffs'] = coeffs
    out[f'{key}_j_sel'], out[f'{key}_j_level'] = jm.sel, jm.level
    out[f'{key}_j_h'], out[f'{key}_j_hlo'] = jm.h, jm.h_lo
    out[f'{key}_j_gamma_hi'], out[f'{key}_j_gamma_lo'] = jm.gamma
    out[f'{key}_j_prefix'] = prefix
    for i, lev in enumerate(levels):
        b = lev.basis
        if b.seed_slot is not None and np.any(np.asarray(b.seed_slot) >= 0):
            raise SystemExit(f'level {lev.k1} has seeds, which the reference does not hold')
        out[f'{key}_l{i}_h'] = np.asarray(b.h)
        out[f'{key}_l{i}_hlo'] = np.asarray(lev.h_lo)
        out[f'{key}_l{i}_mults'] = np.asarray(b.mults)
        out[f'{key}_l{i}_parents'] = np.asarray(b.parents)
        out[f'{key}_l{i}_deg'] = np.array([int(d) for d, _, _ in b.entries])
        out[f'{key}_l{i}_a'] = np.array([np.asarray(a) for _, a, _ in b.entries])
        out[f'{key}_l{i}_b'] = np.array([np.asarray(c) for _, _, c in b.entries])
        out[f'{key}_l{i}_shift'] = np.zeros(0) if b.shift is None else np.asarray(b.shift)
        out[f'{key}_l{i}_scale'] = np.zeros(0) if b.scale is None else np.asarray(b.scale)


def export(args):
    """The dd v at the chosen points in two charts, and the numbers of both stages."""
    sys.path.insert(0, str(ROOT))
    import MLGeometry as mlg
    mlg.set_precision(64)
    import jax
    import jax.numpy as jnp
    import correction_fit as cf
    import seam_levels as sl
    import vpa_dd_levels as vl
    import vpa_dd_stage2 as st
    import vpa_fit as vf
    from quartic import analytic, arnoldi, joint_dd, sections

    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    analytic.EXACT_MUL = args.exact_mul
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    counts = vf.hn_counts(max(args.k1 + args.stage1_k1))

    def load_levels(directory, degrees):
        class LoadArgs:
            load = directory
        return [vl.build_level(args.k0, k1, lay0, mats0, None, None, counts, LoadArgs,
                               Path('.'))[0] for k1 in degrees]
    s1_tag = f'k0_{args.k0}_' + '_'.join(str(k) for k in args.stage1_k1)
    s1_levels = load_levels(args.stage1, args.stage1_k1)
    jm1 = joint_dd.load(Path(args.stage1) / f'joint_map_{s1_tag}.npz')
    prefix1 = np.cumsum([int(np.sum(jm1.level == i)) for i in range(len(s1_levels))])
    c1 = np.load(Path(args.stage1) / f'vpa_levels_coeffs_{s1_tag}.npy')[:, args.stage1_col]
    s1 = st.Stage1(s1_levels, jm1, st.stage1_weights(c1, jm1, prefix1))
    tag = f"k0_{args.k0}_s1_{'_'.join(str(k) for k in args.stage1_k1)}_s2_" + \
        '_'.join(str(k) for k in args.k1)
    run = Path(args.run)
    s1.shift = float(json.loads((run / f'vpa_stage2_{tag}.json').read_text())['stage1_shift'])
    levels = load_levels(args.load, args.k1)
    jm = joint_dd.load(run / f'joint_map_{tag}.npz')
    prefix = np.cumsum([int(np.sum(jm.level == i)) for i in range(len(levels))])
    joint = (jm, None, tuple(int(x) for x in prefix))
    coeffs = np.load(run / f'vpa_stage2_coeffs_{tag}.npy')

    batch = cf.load_points(args.n, args.seed)
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    picks = np.array([int(np.argmin(np.abs(ratio - r))) for r in args.ratios])
    sub = {key: val[picks] for key, val in batch.items()}
    program = st.residual_program(lay0, parts0, s1, levels, args.chunk, joint)
    t1 = time.perf_counter()
    v1 = vl.volumes(program, coeffs, sub, args.chunk)
    v2 = vl.volumes(program, coeffs, sl.other_chart(sub), args.chunk)
    print(f'dd v at {len(picks)} points in two charts in {time.perf_counter() - t1:.0f}s',
          flush=True)

    out = {'k0': args.k0, 'perms': np.asarray(arnoldi.PERMS), 'shift': s1.shift,
           'z_hi': np.asarray(sub['points_hi']), 'z_lo': np.asarray(sub['points_lo']),
           'patch': np.asarray(sub['patch']), 'ratio': ratio[picks], 'v1_hi': v1[0],
           'v1_lo': v1[1], 'v2_hi': v2[0], 'v2_lo': v2[1], 'n_groups': len(lay0.groups)}
    for i, (g, (hi, lo_part)) in enumerate(zip(lay0.groups, parts0)):
        out[f'g{i}_exps'] = np.asarray(g.exps)
        out[f'g{i}_frame'] = np.asarray(g.frame)
        out[f'g{i}_res'] = np.asarray(g.residues)
        out[f'g{i}_hhi'] = np.asarray(hi)
        out[f'g{i}_hlo'] = np.asarray(lo_part)
    _save_part(out, 'a', s1_levels, jm1, prefix1, c1[:, None])
    _save_part(out, 'b', levels, jm, prefix, coeffs)
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.data, **out)
    print(f'saved to {args.data}', flush=True)


# The two stages in mpmath, set by `prepare` before the pool forks.
S = {}


def _prepare_part(data, key, columns):
    """The levels, the picks and the coefficients on the level functions of one stage."""
    levels, offset = [], 0
    for i, k1 in enumerate(data[f'{key}_k1'].tolist()):
        shift, scale = data[f'{key}_l{i}_shift'], data[f'{key}_l{i}_scale']
        lev = {'k1': int(k1), 'h': data[f'{key}_l{i}_h'], 'hlo': data[f'{key}_l{i}_hlo'],
               'mults': data[f'{key}_l{i}_mults'].tolist(),
               'parents': data[f'{key}_l{i}_parents'].tolist(),
               'entries': list(zip(data[f'{key}_l{i}_deg'].tolist(),
                                   data[f'{key}_l{i}_a'].tolist(),
                                   data[f'{key}_l{i}_b'].tolist())),
               'shift': [mp.mpf(x) for x in shift.tolist()] if shift.size else None,
               'scale': [mp.mpf(x) for x in scale.tolist()] if scale.size else None,
               'offset': offset}
        offset += lev['h'].shape[0]
        levels.append(lev)
    sel, level = data[f'{key}_j_sel'].tolist(), data[f'{key}_j_level'].tolist()
    picks = [(lv, s - levels[lv]['offset']) for s, lv in zip(sel, level)]
    n = len(sel)
    gamma = [lo._mpf(a, b) for a, b in zip(data[f'{key}_j_gamma_hi'].tolist(),
                                           data[f'{key}_j_gamma_lo'].tolist())]
    prefix = data[f'{key}_j_prefix'].tolist()
    coeffs = data[f'{key}_coeffs']
    d = {}
    for col in columns:
        c = coeffs[:, col].tolist()
        e = [mp.mpf(x) for x in c[:n]]
        for i, size in enumerate(prefix):
            if c[n + i] != 0.0:
                cn = mp.mpf(c[n + i])
                for t in range(size):
                    e[t] += cn * gamma[t]
        t1 = time.perf_counter()
        d[col] = lo._solve_upper(data[f'{key}_j_h'], data[f'{key}_j_hlo'], e)
        print(f'stage {key}, column {col}: coefficients on the level functions in '
              f'{time.perf_counter() - t1:.0f}s', flush=True)
    return {'levels': levels, 'picks': picks, 'd': d}


def prepare(data, columns):
    """The base in `level_oracle.M`, and both stages in S."""
    lo.M['k0'] = int(data['k0'])
    lo.M['perms'] = data['perms'].tolist()
    lo.M['groups'] = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        lo.M['groups'].append({
            'exps': data[f'g{i}_exps'].tolist(), 'res': data[f'g{i}_res'].tolist(),
            'frame_t': [[mp.mpf(x) for x in frame[:, al].tolist()]
                        for al in range(frame.shape[1])],
            'h': [[[lo._mpf(hhi[b, a, c], hlo[b, a, c]) for c in range(hhi.shape[2])]
                   for a in range(hhi.shape[1])] for b in range(hhi.shape[0])]})
    S['shift'] = mp.mpf(float(data['shift']))
    S['a'] = _prepare_part(data, 'a', [0])
    S['b'] = _prepare_part(data, 'b', columns)


def _stage_values(part, z, r2, p0, factor):
    """The selected level functions of one stage at a point, with an extra factor per level."""
    vals = []
    for lev in part['levels']:
        q = lo._recurrence(lev, lo._multipliers(lev, z, r2))
        w = r2 ** lev['k1'] * p0 ** (-mp.mpf(lev['k1']) / lo.M['k0']) * factor(lev['k1'])
        vals.append([w * x for x in q])
    return [vals[lv][i] for lv, i in part['picks']]


def psi_values(z):
    """psi_1, and psi_1 + psi_2 of every second-stage column, and log p0, at one point."""
    p0 = lo.potential(z)
    r2 = mp.fsum(abs(c) ** 2 for c in z)
    psi1 = mp.fdot(S['a']['d'][0], _stage_values(S['a'], z, r2, p0, lambda k1: 1))
    k0 = lo.M['k0']
    gap = psi1 - S['shift']
    flat = _stage_values(S['b'], z, r2, p0, lambda k1: mp.exp(-mp.mpf(k1) / k0 * gap))
    out = {col: psi1 + mp.fdot(d, flat) for col, d in S['b']['d'].items()}
    out['psi1'] = psi1
    out['logp'] = mp.log(p0)
    return out


def _task(job):
    k, off = job
    center, chart, h = S['centers'][k], S['charts'][k], S['h']
    dt = [mp.mpc(0), mp.mpc(0)]
    for (index, unit), n in zip(lo.DIRS, off):
        dt[index] += n * h * unit
    vals = psi_values(lo.on_x(center, chart, dt))
    return k, off, {col: mp.nstr(v, mp.mp.dps + 5) for col, v in vals.items()}


def reference(args):
    data = np.load(args.data)
    mp.mp.dps = args.dps
    coeffs = data['b_coeffs']
    columns = list(range(coeffs.shape[1]))
    t0 = time.perf_counter()
    prepare(data, columns)
    z_hi, z_lo, patch = data['z_hi'], data['z_lo'], data['patch']
    n_pts = z_hi.shape[0]
    S['h'] = mp.mpf(2) ** -args.log2_step
    S['charts'], S['centers'] = [], []
    for k in range(n_pts):
        m, f0, f1 = (int(x) for x in patch[k])
        j = ({0, 1, 2, 3} - {m, f0, f1}).pop()
        chart = (j, m, (f0, f1))
        center = [lo._mpc(a, b) for a, b in zip(z_hi[k], z_lo[k])]
        S['charts'].append(chart)
        S['centers'].append(lo.on_x(center, chart, (mp.mpc(0), mp.mpc(0))))
    jobs = [(k, off) for k in range(n_pts)
            for off in [(0, 0, 0, 0)] + lo._offsets(1) + lo._offsets(2)]
    print(f'{len(jobs)} values of psi for {n_pts} points with {args.procs} processes, '
          f'{args.dps} digits, h = 2^-{args.log2_step}', flush=True)
    t1 = time.perf_counter()
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        results = pool.map(_task, jobs, chunksize=1)
    print(f'values in {time.perf_counter() - t1:.0f}s', flush=True)
    table = {}
    for k, off, vals in results:
        table.setdefault(k, {})[off] = {col: mp.mpf(v) for col, v in vals.items()}

    ratios = {}
    for k in range(n_pts):
        center, chart = S['centers'][k], S['charts'][k]
        phi = lo.phi0(center)
        _, g0 = lo._volume(center, chart, phi, [[0, 0], [0, 0]])
        scale = max(abs(g0[i][j]) for i in range(2) for j in range(2))
        herm = [[(g0[i][j] + mp.conj(g0[j][i])) / 2 for j in range(2)] for i in range(2)]
        logp = {off: v['logp'] for off, v in table[k].items()}
        l1, l2 = lo._hessian(logp, 1, S['h']), lo._hessian(logp, 2, S['h'])
        control = max(abs((4 * l1[i][j] - l2[i][j]) / 3 - herm[i][j])
                      for i in range(2) for j in range(2))
        print(f'point {k}, min/max abs(z) {float(data["ratio"][k]):.3f}, chart (j, m, free) '
              f'{chart}; control: the differences of log p0 against R phi0 R^dag '
              f'{float(control / scale):.1e}', flush=True)
        for col in columns:
            vals = {off: v[col] for off, v in table[k].items()}
            g1 = lo._hessian(vals, 1, S['h'])
            g2 = lo._hessian(vals, 2, S['h'])
            g_add = [[(4 * g1[i][j] - g2[i][j]) / 3 for j in range(2)] for i in range(2)]
            trunc = max(abs(g1[i][j] - g2[i][j]) for i in range(2) for j in range(2))
            size = max(abs(g1[i][j]) for i in range(2) for j in range(2))
            v_ref, _ = lo._volume(center, chart, phi, g_add)
            v1 = lo._mpf(data['v1_hi'][k, col], data['v1_lo'][k, col])
            v2 = lo._mpf(data['v2_hi'][k, col], data['v2_lo'][k, col])
            ratios.setdefault(col, []).extend([v1 / v_ref - 1, v2 / v_ref - 1])
            print(f'    column {col}: max abs(dg) {float(size / scale):.3e} of max abs(g0), '
                  f'h against 2h {float(trunc / scale):.1e}; v_dd / v_ref - 1: '
                  f'{mp.nstr(v1 / v_ref - 1, 20)} in the first chart, '
                  f'{mp.nstr(v2 / v_ref - 1, 20)} in the second', flush=True)
    for col, vals in ratios.items():
        print(f'column {col}: v_dd / v_ref - 1 spans {mp.nstr(max(vals) - min(vals), 5)} over '
              f'the points and the charts, around {mp.nstr(sum(vals) / len(vals), 12)}')
    print(f'all in {time.perf_counter() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('export')
    ex.add_argument('--k0', type=int, required=True)
    ex.add_argument('--theta', type=str, required=True)
    ex.add_argument('--stage1', type=str, required=True)
    ex.add_argument('--stage1-k1', type=int, nargs='+', required=True)
    ex.add_argument('--stage1-col', type=int, default=-1)
    ex.add_argument('--k1', type=int, nargs='+', required=True)
    ex.add_argument('--load', type=str, required=True)
    ex.add_argument('--run', type=str, required=True,
                    help='the output directory of the second-stage fit')
    ex.add_argument('--exact-mul', action='store_true')
    ex.add_argument('--n', type=int, default=4000)
    ex.add_argument('--seed', type=int, default=20264813)
    ex.add_argument('--ratios', type=float, nargs='+', default=[0.9, 0.4, 0.1])
    ex.add_argument('--chunk', type=int, default=256)
    ex.add_argument('--data', type=str, required=True)
    ref = sub.add_parser('reference')
    ref.add_argument('--data', type=str, required=True)
    ref.add_argument('--dps', type=int, default=100)
    ref.add_argument('--log2-step', type=int, default=36)
    ref.add_argument('--procs', type=int, default=16)
    args = ap.parse_args()
    export(args) if args.stage == 'export' else reference(args)


if __name__ == '__main__':
    main()
