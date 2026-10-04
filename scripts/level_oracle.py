"""An mpmath reference for v of a level fit at a few points of X.

v from the production dd path is held against v from mpmath at the same points.
The reference takes phi0 = ddbar log p0 in closed form, and the Hessian of
psi = sum of c over the joint functions along X from second differences of psi
values. psi changes g by about 1e-12 only, so 17 correct digits of its Hessian
are enough, and 100 digits leave a large margin for the differences.

The stage `export` runs the dd path on the GPU and saves every number that the
reference needs. The stage `reference` needs only numpy and mpmath, so it can
fork, and it evaluates psi at the points of the differences in parallel.

Usage, on one GPU node:
    uv run --extra gpu python -u scripts/level_oracle.py export --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 44 40 36 \\
        --load runs_levels_load --load-joint MAP --coeffs COEFFS --exact-mul \\
        --data runs_level_oracle/data.npz
    uv run --extra gpu python -u scripts/level_oracle.py reference \\
        --data runs_level_oracle/data.npz --procs 32
"""
import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The four real directions x_0, y_0, x_1, y_1 of the free coordinates t, as (index, unit).
DIRS = ((0, 1), (0, 1j), (1, 1), (1, 1j))
MIXED = ((0, 2), (1, 3), (0, 3), (1, 2))       # x0 x1, y0 y1, x0 y1, y0 x1


def export(args):
    """The dd v at the chosen points in two charts, and the numbers of the model."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import MLGeometry as mlg
    mlg.set_precision(64)
    import jax
    import jax.numpy as jnp
    import correction_fit as cf
    import seam_levels as sl
    import vpa_dd_levels as vl
    import vpa_fit as vf
    from quartic import analytic, arnoldi, joint_dd, sections

    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    analytic.EXACT_MUL = args.exact_mul
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    counts = vf.hn_counts(max(args.k1))

    class LoadArgs:
        load = args.load
    levels = [vl.build_level(args.k0, k1, lay0, mats0, None, None, counts, LoadArgs,
                             Path('.'))[0] for k1 in args.k1]
    jm = joint_dd.load(args.load_joint)
    prefix = np.cumsum([int(np.sum(jm.level == i)) for i in range(len(levels))])
    joint = (jm, None, tuple(int(x) for x in prefix))
    coeffs = np.load(args.coeffs)

    batch = cf.load_points(args.n, args.seed)
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    picks = np.array([int(np.argmin(np.abs(ratio - r))) for r in args.ratios])
    sub = {key: val[picks] for key, val in batch.items()}
    program = vl.residual_program(lay0, parts0, levels, args.chunk, joint)
    t1 = time.perf_counter()
    v1 = vl.volumes(program, coeffs, sub, args.chunk)
    v2 = vl.volumes(program, coeffs, sl.other_chart(sub), args.chunk)
    print(f'dd v at {len(picks)} points in two charts in {time.perf_counter() - t1:.0f}s',
          flush=True)

    out = {'k0': args.k0, 'k1': np.array(args.k1), 'coeffs': coeffs,
           'perms': np.asarray(arnoldi.PERMS), 'z_hi': np.asarray(sub['points_hi']),
           'z_lo': np.asarray(sub['points_lo']), 'patch': np.asarray(sub['patch']),
           'ratio': ratio[picks], 'v1_hi': v1[0], 'v1_lo': v1[1], 'v2_hi': v2[0],
           'v2_lo': v2[1], 'j_sel': jm.sel, 'j_level': jm.level, 'j_h': jm.h,
           'j_hlo': jm.h_lo, 'j_gamma_hi': jm.gamma[0], 'j_gamma_lo': jm.gamma[1],
           'j_prefix': prefix, 'n_groups': len(lay0.groups)}
    for i, (g, (hi, lo)) in enumerate(zip(lay0.groups, parts0)):
        out[f'g{i}_exps'] = np.asarray(g.exps)
        out[f'g{i}_frame'] = np.asarray(g.frame)
        out[f'g{i}_res'] = np.asarray(g.residues)
        out[f'g{i}_hhi'] = np.asarray(hi)
        out[f'g{i}_hlo'] = np.asarray(lo)
    for i, lev in enumerate(levels):
        b = lev.basis
        if b.seed_slot is not None and np.any(np.asarray(b.seed_slot) >= 0):
            raise SystemExit(f'level {lev.k1} has seeds, which the reference does not hold')
        out[f'l{i}_h'] = np.asarray(b.h)
        out[f'l{i}_hlo'] = np.asarray(lev.h_lo)
        out[f'l{i}_mults'] = np.asarray(b.mults)
        out[f'l{i}_parents'] = np.asarray(b.parents)
        out[f'l{i}_deg'] = np.array([int(d) for d, _, _ in b.entries])
        out[f'l{i}_a'] = np.array([np.asarray(a) for _, a, _ in b.entries])
        out[f'l{i}_b'] = np.array([np.asarray(c) for _, _, c in b.entries])
        out[f'l{i}_shift'] = np.zeros(0) if b.shift is None else np.asarray(b.shift)
        out[f'l{i}_scale'] = np.zeros(0) if b.scale is None else np.asarray(b.scale)
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.data, **out)
    print(f'saved to {args.data}', flush=True)


# The model in mpmath, set by `prepare` before the pool forks.
M = {}


def _mpf(hi, lo=0.0):
    return mp.mpf(float(hi)) + mp.mpf(float(lo))


def _mpc(hi, lo):
    return (mp.mpc(float(np.real(hi)), float(np.imag(hi)))
            + mp.mpc(float(np.real(lo)), float(np.imag(lo))))


def _solve_upper(h, h_lo, e):
    """d with H d = e, H = h + h_lo upper triangular, by back substitution."""
    n = len(e)
    d = [mp.mpf(0)] * n
    for t in range(n - 1, -1, -1):
        row = [mp.mpf(a) + mp.mpf(b) for a, b in zip(h[t, t + 1:].tolist(),
                                                     h_lo[t, t + 1:].tolist())]
        d[t] = (e[t] - mp.fdot(row, d[t + 1:])) / _mpf(h[t, t], h_lo[t, t])
    return d


def prepare(data, columns):
    """The model in mpmath, and the coefficients of psi on the level functions."""
    M['k0'] = int(data['k0'])
    M['perms'] = data['perms'].tolist()
    M['groups'] = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        M['groups'].append({
            'exps': data[f'g{i}_exps'].tolist(), 'res': data[f'g{i}_res'].tolist(),
            'frame_t': [[mp.mpf(x) for x in frame[:, al].tolist()]
                        for al in range(frame.shape[1])],
            'h': [[[_mpf(hhi[b, a, c], hlo[b, a, c]) for c in range(hhi.shape[2])]
                   for a in range(hhi.shape[1])] for b in range(hhi.shape[0])]})
    M['levels'] = []
    offset = 0
    for i, k1 in enumerate(data['k1'].tolist()):
        shift, scale = data[f'l{i}_shift'], data[f'l{i}_scale']
        lev = {'k1': int(k1), 'h': data[f'l{i}_h'], 'hlo': data[f'l{i}_hlo'],
               'mults': data[f'l{i}_mults'].tolist(),
               'parents': data[f'l{i}_parents'].tolist(),
               'entries': list(zip(data[f'l{i}_deg'].tolist(), data[f'l{i}_a'].tolist(),
                                   data[f'l{i}_b'].tolist())),
               'shift': [mp.mpf(x) for x in shift.tolist()] if shift.size else None,
               'scale': [mp.mpf(x) for x in scale.tolist()] if scale.size else None,
               'offset': offset}
        offset += lev['h'].shape[0]
        M['levels'].append(lev)
    sel, level = data['j_sel'].tolist(), data['j_level'].tolist()
    M['picks'] = [(lv, s - M['levels'][lv]['offset']) for s, lv in zip(sel, level)]
    n = len(sel)
    gamma = [_mpf(a, b) for a, b in zip(data['j_gamma_hi'].tolist(),
                                        data['j_gamma_lo'].tolist())]
    prefix = data['j_prefix'].tolist()
    coeffs = data['coeffs']
    M['d'] = {}
    for col in columns:
        c = coeffs[:, col].tolist()
        e = [mp.mpf(x) for x in c[:n]]
        for i, size in enumerate(prefix):
            if c[n + i] != 0.0:
                cn = mp.mpf(c[n + i])
                for t in range(size):
                    e[t] += cn * gamma[t]
        t1 = time.perf_counter()
        M['d'][col] = _solve_upper(data['j_h'], data['j_hlo'], e)
        print(f'column {col}: coefficients on the level functions in '
              f'{time.perf_counter() - t1:.0f}s', flush=True)


def _sections(z, derivs=False):
    """sigma and, with `derivs`, its 4 holomorphic derivatives, for each block."""
    u = [c ** 4 for c in z]
    out = []
    for g in M['groups']:
        pf, dpf = [], []
        for row in g['exps']:
            val = mp.mpc(1)
            for a in range(4):
                if row[a]:
                    val *= u[a] ** row[a]
            pf.append(val)
            if derivs:
                dpf.append([4 * row[a] * val / z[a] for a in range(4)])
        psi = [mp.fsum(f * x for f, x in zip(col, pf)) for col in g['frame_t']]
        dpsi = [[mp.fsum(f * x[i] for f, x in zip(col, dpf)) for i in range(4)]
                for col in g['frame_t']] if derivs else None
        for r, hm in zip(g['res'], g['h']):
            zr = mp.mpc(1)
            for a in range(4):
                if r[a]:
                    zr *= z[a] ** r[a]
            s = [zr * x for x in psi]
            ds = None
            if derivs:
                dzr = [r[a] * zr / z[a] for a in range(4)]
                ds = [[dzr[i] * psi[al] + zr * dpsi[al][i] for i in range(4)]
                      for al in range(len(psi))]
            out.append((s, ds, hm))
    return out


def potential(z):
    """p0 = sigma^* H sigma at one point."""
    p = mp.mpf(0)
    for s, _, hm in _sections(z):
        for a in range(len(s)):
            p += (mp.conj(s[a]) * mp.fsum(hm[a][c] * s[c] for c in range(len(s)))).real
    return p


def phi0(z):
    """ddbar log p0 at one point, from the centered form b = dsigma - (p_i/p) sigma."""
    blocks = _sections(z, derivs=True)
    p = mp.mpf(0)
    p_i = [mp.mpc(0)] * 4
    store = []
    for s, ds, hm in blocks:
        d = len(s)
        h_s = [mp.fsum(hm[a][c] * s[c] for c in range(d)) for a in range(d)]
        h_d = [[mp.fsum(hm[a][c] * ds[c][i] for c in range(d)) for i in range(4)]
               for a in range(d)]
        p += mp.fsum(mp.conj(s[a]) * h_s[a] for a in range(d)).real
        for i in range(4):
            p_i[i] += mp.fsum(mp.conj(s[a]) * h_d[a][i] for a in range(d))
        store.append((s, ds, h_s, h_d))
    phi = [[mp.mpc(0)] * 4 for _ in range(4)]
    for s, ds, h_s, h_d in store:
        d = len(s)
        b = [[ds[a][i] - (p_i[i] / p) * s[a] for i in range(4)] for a in range(d)]
        hb = [[h_d[a][i] - (p_i[i] / p) * h_s[a] for i in range(4)] for a in range(d)]
        for i in range(4):
            for j in range(4):
                phi[i][j] += mp.fsum(mp.conj(b[a][j]) * hb[a][i] for a in range(d))
    return [[phi[i][j] / p for j in range(4)] for i in range(4)]


def _multipliers(lev, z, r2):
    """The multipliers (y - shift) / scale of a level at one point."""
    out = []
    for d, a_exp, b_exp in lev['entries']:
        total = mp.mpc(0)
        for perm in M['perms']:
            term = mp.mpc(1)
            for c in range(4):
                w = z[perm[c]]
                if a_exp[c]:
                    term *= w ** a_exp[c]
                if b_exp[c]:
                    term *= mp.conj(w) ** b_exp[c]
            total += term
        out.append(total.real / r2 ** d)
    if lev['shift'] is not None:
        out = [(y - a) / b for y, a, b in zip(out, lev['shift'], lev['scale'])]
    return out


def _recurrence(lev, y):
    """q_t = (y q_parent - sum over s < t of h_st q_s) / h_tt for every function."""
    h, h_lo = lev['h'], lev['hlo']
    n = h.shape[0]
    q = [1 / _mpf(h[0, 0], h_lo[0, 0])]
    for t in range(1, n):
        col = [mp.mpf(a) + mp.mpf(b) for a, b in zip(h[:t, t].tolist(), h_lo[:t, t].tolist())]
        grown = y[lev['mults'][t]] * q[lev['parents'][t]]
        q.append((grown - mp.fdot(col, q)) / _mpf(h[t, t], h_lo[t, t]))
    return q


def psi_values(z):
    """psi of every column with coefficients at one point of X, and log p0 as a control."""
    p0 = potential(z)
    r2 = mp.fsum(abs(c) ** 2 for c in z)
    vals = []
    for lev in M['levels']:
        q = _recurrence(lev, _multipliers(lev, z, r2))
        w = r2 ** lev['k1'] * p0 ** (-mp.mpf(lev['k1']) / M['k0'])
        vals.append([w * x for x in q])
    flat = [vals[lv][i] for lv, i in M['picks']]
    out = {col: mp.fdot(d, flat) for col, d in M['d'].items()}
    out['logp'] = mp.log(p0)
    return out


def on_x(center, chart, dt):
    """The point of X with the free coordinates of `center` moved by dt, and z_j = 1."""
    j, m, free = chart
    z = list(center)
    z[j] = mp.mpc(1)
    z[free[0]] += dt[0]
    z[free[1]] += dt[1]
    rest = 1 + z[free[0]] ** 4 + z[free[1]] ** 4
    w = z[m]
    for _ in range(10):
        w -= (w ** 4 + rest) / (4 * w ** 3)
    z[m] = w
    return z


def _offsets(step):
    """The multiples of h of the real directions for the second differences at `step`."""
    out = []
    for u in range(4):
        for sign in (1, -1):
            e = [0, 0, 0, 0]
            e[u] = sign * step
            out.append(tuple(e))
    for u, w in MIXED:
        for su in (1, -1):
            for sw in (1, -1):
                e = [0, 0, 0, 0]
                e[u], e[w] = su * step, sw * step
                out.append(tuple(e))
    return out


def _task(job):
    k, off = job
    center, chart, h = M['centers'][k], M['charts'][k], M['h']
    dt = [mp.mpc(0), mp.mpc(0)]
    for (index, unit), n in zip(DIRS, off):
        dt[index] += n * h * unit
    vals = psi_values(on_x(center, chart, dt))
    return k, off, {col: mp.nstr(v, mp.mp.dps + 5) for col, v in vals.items()}


def _hessian(vals, step, h):
    """The complex Hessian d_i dbar_j psi in t from the values at multiples of h."""
    size = step * h
    zero = vals[(0, 0, 0, 0)]

    def diag(u):
        e = [0, 0, 0, 0]
        e[u] = step
        plus = vals[tuple(e)]
        e[u] = -step
        return (plus + vals[tuple(e)] - 2 * zero) / size ** 2

    def mixed(u, w):
        total = mp.mpf(0)
        for su in (1, -1):
            for sw in (1, -1):
                e = [0, 0, 0, 0]
                e[u], e[w] = su * step, sw * step
                total += su * sw * vals[tuple(e)]
        return total / (4 * size ** 2)
    xx0, yy0, xx1, yy1 = (diag(u) for u in range(4))
    x0x1, y0y1, x0y1, y0x1 = (mixed(u, w) for u, w in MIXED)
    h01 = (x0x1 + y0y1 + 1j * (x0y1 - y0x1)) / 4
    return [[(xx0 + yy0) / 4, h01], [mp.conj(h01), (xx1 + yy1) / 4]]


def _volume(z, chart, phi, g_add):
    """det(R phi R^dag + g_add) |dP/dz_m|^2, and R phi R^dag, at a point of X."""
    j, m, free = chart
    grad = [4 * c ** 3 for c in z]
    rows = []
    for t in free:
        row = [mp.mpc(0)] * 4
        row[t] = mp.mpc(1)
        row[m] = -grad[t] / grad[m]
        rows.append(row)
    g0 = [[mp.fsum(rows[i][a] * phi[a][b] * mp.conj(rows[k][b])
                   for a in range(4) for b in range(4)) for k in range(2)] for i in range(2)]
    g = [[g0[i][k] + g_add[i][k] for k in range(2)] for i in range(2)]
    det = (g[0][0] * g[1][1] - g[0][1] * g[1][0]).real
    return det * abs(grad[m]) ** 2, g0


def reference(args):
    data = np.load(args.data)
    mp.mp.dps = args.dps
    coeffs = data['coeffs']
    columns = [c for c in range(coeffs.shape[1]) if np.any(coeffs[:, c] != 0.0)]
    t0 = time.perf_counter()
    prepare(data, columns)
    z_hi, z_lo, patch = data['z_hi'], data['z_lo'], data['patch']
    n_pts = z_hi.shape[0]
    M['h'] = mp.mpf(2) ** -args.log2_step
    M['charts'], M['centers'] = [], []
    for k in range(n_pts):
        m, f0, f1 = (int(x) for x in patch[k])
        j = ({0, 1, 2, 3} - {m, f0, f1}).pop()
        chart = (j, m, (f0, f1))
        center = [_mpc(a, b) for a, b in zip(z_hi[k], z_lo[k])]
        M['charts'].append(chart)
        M['centers'].append(on_x(center, chart, (mp.mpc(0), mp.mpc(0))))
    jobs = [(k, off) for k in range(n_pts)
            for off in [(0, 0, 0, 0)] + _offsets(1) + _offsets(2)]
    print(f'{len(jobs)} values of psi for {n_pts} points with {args.procs} processes, '
          f'{args.dps} digits, h = 2^-{args.log2_step}', flush=True)
    t1 = time.perf_counter()
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        results = pool.map(_task, jobs, chunksize=1)
    print(f'values in {time.perf_counter() - t1:.0f}s', flush=True)
    table = {}
    for k, off, vals in results:
        table.setdefault(k, {})[off] = {col: mp.mpf(v) for col, v in vals.items()}

    for k in range(n_pts):
        center, chart = M['centers'][k], M['charts'][k]
        phi = phi0(center)
        _, g0 = _volume(center, chart, phi, [[0, 0], [0, 0]])
        scale = max(abs(g0[i][j]) for i in range(2) for j in range(2))
        logp = {off: v['logp'] for off, v in table[k].items()}
        l1, l2 = _hessian(logp, 1, M['h']), _hessian(logp, 2, M['h'])
        control = max(abs((4 * l1[i][j] - l2[i][j]) / 3 - g0[i][j])
                      for i in range(2) for j in range(2))
        print(f'point {k}, min/max abs(z) {float(data["ratio"][k]):.3f}, chart (j, m, free) '
              f'{chart}; control: the differences of log p0 against R phi0 R^dag '
              f'{float(control / scale):.1e}', flush=True)
        for col in range(coeffs.shape[1]):
            if col in columns:
                vals = {off: v[col] for off, v in table[k].items()}
                g1 = _hessian(vals, 1, M['h'])
                g2 = _hessian(vals, 2, M['h'])
                g_add = [[(4 * g1[i][j] - g2[i][j]) / 3 for j in range(2)] for i in range(2)]
                trunc = max(abs(g1[i][j] - g2[i][j]) for i in range(2) for j in range(2))
                size = max(abs(g1[i][j]) for i in range(2) for j in range(2))
            else:
                g_add, trunc, size = [[0, 0], [0, 0]], mp.mpf(0), mp.mpf(0)
            v_ref, _ = _volume(center, chart, phi, g_add)
            v1 = _mpf(data['v1_hi'][k, col], data['v1_lo'][k, col])
            v2 = _mpf(data['v2_hi'][k, col], data['v2_lo'][k, col])
            print(f'    column {col}: max abs(dg) {float(size / scale):.3e} of max abs(g0), '
                  f'h against 2h {float(trunc / scale):.1e}; v_dd / v_ref - 1: '
                  f'{float(v1 / v_ref - 1):+.3e} in the first chart, '
                  f'{float(v2 / v_ref - 1):+.3e} in the second', flush=True)
    print(f'all in {time.perf_counter() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('export')
    ex.add_argument('--k0', type=int, required=True)
    ex.add_argument('--theta', type=str, required=True)
    ex.add_argument('--k1', type=int, nargs='+', required=True)
    ex.add_argument('--load', type=str, required=True)
    ex.add_argument('--load-joint', type=str, required=True)
    ex.add_argument('--coeffs', type=str, required=True)
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
    ref.add_argument('--procs', type=int, default=32)
    args = ap.parse_args()
    export(args) if args.stage == 'export' else reference(args)


if __name__ == '__main__':
    main()
