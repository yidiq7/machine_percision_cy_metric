"""The Hermitian forms N_L of the levels of a one-stage level fit, and their size.

A level function G = W_L q, with W_L = (abs(z)^2 p0^(-1/k0))^L, is N / p0^(L/k0), where
N = abs(z)^(2L) q is a real form of bidegree (L, L), invariant under the symmetry group.
So the sum of level L over its effective weights is N_L / p0^(L/k0), and N_L is an
invariant Hermitian form of the sections of degree L. A Taylor model of the potential
can take N_L as such a form, and not through the V+A recurrence, whose ball bounds
grow by approximately 10 for each degree.

The script fits N_L in the invariant forms of degree L by least squares, in ball
arithmetic (python-flint), on random points of X, and checks the fit on other points.
Each form is a sum over S_4 of one block of each orbit, so it is exactly invariant,
although the FP64 frames and matrices of the layout are invariant only to 1e-15.
It reports how much each form cancels: the absolute-value form over the value. A
Taylor model needs that factor as extra digits.

The input is the export of `level_oracle.py`. Usage, on CPU cores:
    JAX_PLATFORMS=cpu uv run --extra gpu --with python-flint python -u \\
        scripts/level_matrices.py --data runs_tm/r20_data.npz --col 3 --procs 32 \\
        --out runs_tm/r20_forms.npz
"""
import argparse
import itertools
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import flint
from flint import acb, acb_mat, arb, arb_mat

ROOT = Path(__file__).resolve().parents[1]
PERMS = list(itertools.permutations(range(4)))

# The model and the layouts, set by `setup` before the pool forks.
M = {}


def _mat(rows, cols, values):
    return arb_mat(rows, cols, [float(x) for x in np.asarray(values).ravel()])


def _cmat(rows, cols, values):
    return acb_mat(rows, cols, [acb(float(x)) for x in np.asarray(values).ravel()])


def _arb(hi, lo=0.0):
    return arb(float(hi)) + arb(float(lo))


def _text(x):
    """The midpoint of a ball as a decimal string, so that it passes between processes."""
    return x.mid().str(90, radius=False)


def _levels(data):
    """The V+A recurrence of each level by degree blocks, as `level_oracle` holds it."""
    levels, offset = [], 0
    for i, k1 in enumerate(data['k1'].tolist()):
        h, hl = data[f'l{i}_h'], data[f'l{i}_hlo']
        n = h.shape[0]
        mults, parents, degs = data[f'l{i}_mults'], data[f'l{i}_parents'], data[f'l{i}_deg']
        deg = np.zeros(n, int)
        for t in range(1, n):
            deg[t] = deg[parents[t]] + degs[mults[t]]
        blocks, a = [], 1
        while a < n:
            b = a
            while b < n and deg[b] == deg[a]:
                b += 1
            lt = _mat(b - a, a, h[:a, a:b].T) + _mat(b - a, a, hl[:a, a:b].T)
            ut = (_mat(b - a, b - a, np.tril(h[a:b, a:b].T))
                  + _mat(b - a, b - a, np.tril(hl[a:b, a:b].T)))
            blocks.append((a, b, lt, ut.inv()))
            a = b
        shift, scale = data[f'l{i}_shift'], data[f'l{i}_scale']
        levels.append({
            'k1': int(k1), 'n': n, 'blocks': blocks, 'h00': _arb(h[0, 0], hl[0, 0]),
            'mults': mults.tolist(), 'parents': parents.tolist(),
            'entries': list(zip(degs.tolist(), data[f'l{i}_a'].tolist(),
                                data[f'l{i}_b'].tolist())),
            'shift': [arb(float(x)) for x in shift] if shift.size else None,
            'scale': [arb(float(x)) for x in scale] if scale.size else None,
            'offset': offset})
        offset += n
    return levels


def _weights(data, col, levels):
    """The coefficients of psi on the level functions of each level, as `level_oracle`."""
    sel, level = data['j_sel'].tolist(), data['j_level'].tolist()
    gamma = [_arb(a, b) for a, b in zip(data['j_gamma_hi'], data['j_gamma_lo'])]
    jh, jl = data['j_h'], data['j_hlo']
    c = data['coeffs'][:, col].tolist()
    nj = len(sel)
    e = [arb(x) for x in c[:nj]]
    for i, size in enumerate(data['j_prefix'].tolist()):
        if c[nj + i] != 0.0:
            for t in range(size):
                e[t] += c[nj + i] * gamma[t]
    dvec = [arb(0)] * nj
    for t in range(nj - 1, -1, -1):
        acc = arb(0)
        for u in np.flatnonzero((jh[t, t + 1:] != 0) | (jl[t, t + 1:] != 0)):
            acc += _arb(jh[t, t + 1 + u], jl[t, t + 1 + u]) * dvec[t + 1 + u]
        dvec[t] = (e[t] - acc) / _arb(jh[t, t], jl[t, t])
    out = [[] for _ in levels]
    for s, lv, w in zip(sel, level, dvec):
        out[lv].append((s - levels[lv]['offset'], w))
    return out


def _layout(k):
    """For each orbit of degree k, one block: its group, residue and invariant matrices."""
    sys.path.insert(0, str(ROOT))
    from quartic import sections
    lay = sections.layout(k, normalize=False)
    groups, orbits = [], []
    for gi, g in enumerate(lay.groups):
        frame = np.asarray(g.frame)
        groups.append({'exps': np.asarray(g.exps).tolist(), 'm': int(g.m),
                       'frame_t': _cmat(frame.shape[1], frame.shape[0], frame.T)})
        for o, sl in enumerate(g.slices):
            b = int(np.flatnonzero(np.asarray(g.orbit_of) == o)[0])
            rot = np.asarray(g.rot[b])
            mats = []
            for a in range(sl.stop - sl.start):
                m = rot @ np.asarray(g.basis[o, a]) @ rot.T
                mats.append((m + m.T) / 2)
            d = rot.shape[0]
            orbits.append({'group': gi, 'r': np.asarray(g.residues[b]).tolist(), 'd': d,
                           'mats': mats,
                           'cols': _mat(d * d, len(mats),
                                        np.stack([m.ravel() for m in mats], axis=1))})
    n = sum(len(o['mats']) for o in orbits)
    if n != lay.n_theta:
        raise SystemExit(f'degree {k}: {n} matrices against n_theta = {lay.n_theta}')
    return {'n_theta': lay.n_theta, 'groups': groups, 'orbits': orbits}


def setup(args):
    flint.ctx.prec = args.prec
    data = np.load(args.data)
    M['k0'] = int(data['k0'])
    M['perms'] = data['perms'].tolist()
    M['base'] = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        d = frame.shape[1]
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        blocks = [(r, _mat(d, d, hhi[b]) + _mat(d, d, hlo[b]),
                   _mat(d, d, np.abs(hhi[b]) + np.abs(hlo[b])))
                  for b, r in enumerate(data[f'g{i}_res'].tolist())]
        M['base'].append((data[f'g{i}_exps'].tolist(),
                          _cmat(d, frame.shape[0], frame.T), blocks))
    M['levels'] = _levels(data)
    M['w'] = _weights(data, args.col, M['levels'])
    M['layouts'] = [_layout(lev['k1']) for lev in M['levels']]


def _psi(exps, frame_t, powers):
    """The sections of a group without the residue factor, frame^T u^E, as (d, 1)."""
    mono = []
    for row in exps:
        val = acb(1)
        for a in range(4):
            if row[a]:
                val *= powers[a][row[a]]
        mono.append(val)
    return frame_t * acb_mat(len(mono), 1, mono)


def _parts(col):
    """The real and imaginary parts and the absolute values of an acb_mat (d, 1)."""
    d = col.nrows()
    x = arb_mat(d, 1, [col[i, 0].real for i in range(d)])
    y = arb_mat(d, 1, [col[i, 0].imag for i in range(d)])
    a = arb_mat(d, 1, [abs(col[i, 0]) for i in range(d)])
    return x, y, a


def _weight(z2, r):
    w = arb(1)
    for a in range(4):
        if r[a]:
            w *= z2[a] ** r[a]
    return w


def base_potential(z):
    """p0 = sum over blocks of abs(z^r)^2 psi^dag H psi, and the same with abs values."""
    z2 = [abs(c) ** 2 for c in z]
    u = [c ** 4 for c in z]
    top = max(max(max(row) for row in exps) for exps, _, _ in M['base'])
    powers = [[acb(1)] for _ in range(4)]
    for a in range(4):
        for _ in range(top):
            powers[a].append(powers[a][-1] * u[a])
    p, pabs = arb(0), arb(0)
    for exps, frame_t, blocks in M['base']:
        x, y, a = _parts(_psi(exps, frame_t, powers))
        for r, h, habs in blocks:
            w = _weight(z2, r)
            p += w * ((x.transpose() * h * x)[0, 0] + (y.transpose() * h * y)[0, 0])
            pabs += w * (a.transpose() * habs * a)[0, 0]
    return p, pabs


def _multipliers(lev, z, zc, r2):
    out = []
    for d, a_exp, b_exp in lev['entries']:
        total = acb(0)
        for perm in M['perms']:
            term = acb(1)
            for c in range(4):
                if a_exp[c]:
                    term *= z[perm[c]] ** a_exp[c]
                if b_exp[c]:
                    term *= zc[perm[c]] ** b_exp[c]
            total += term
        out.append(total.real / r2 ** d)
    if lev['shift'] is not None:
        out = [(y - a) / b for y, a, b in zip(out, lev['shift'], lev['scale'])]
    return out


def _recurrence(lev, y):
    """q_t = (y q_parent - sum over s < t of h_st q_s) / h_tt, one degree block at a time."""
    q = [arb(0)] * lev['n']
    q[0] = 1 / lev['h00']
    for a, b, lt, uinv in lev['blocks']:
        lq = lt * arb_mat(a, 1, q[:a])
        rhs = [y[lev['mults'][t]] * q[lev['parents'][t]] - lq[t - a, 0] for t in range(a, b)]
        sol = uinv * arb_mat(b - a, 1, rhs)
        for k in range(b - a):
            q[a + k] = sol[k, 0]
    return q


def level_forms(z):
    """N_L = abs(z)^(2L) sum of w q over the level, from the recurrence, for each level."""
    zc = [c.conjugate() for c in z]
    r2 = sum((abs(c) ** 2 for c in z), arb(0))
    out = []
    for lev, weights in zip(M['levels'], M['w']):
        q = _recurrence(lev, _multipliers(lev, z, zc, r2))
        total = sum((w * q[i] for i, w in weights), arb(0))
        out.append(r2 ** lev['k1'] * total)
    return out


def features(z, layout):
    """For each orbit, T = sum over S_4 of abs(z'^r)^2 Re(psi psi^dag), with z' = sigma z.

    The invariant forms of the orbit are then <B, T> for its matrices B. The second
    output is the same sum with abs(psi) abs(psi)^T, for the absolute-value form."""
    z2 = [abs(c) ** 2 for c in z]
    u = [c ** 4 for c in z]
    top = max(g['m'] for g in layout['groups'])
    powers = [[acb(1)] for _ in range(4)]
    for a in range(4):
        for _ in range(top):
            powers[a].append(powers[a][-1] * u[a])
    t_sum = [None] * len(layout['orbits'])
    t_abs = [None] * len(layout['orbits'])
    for perm in PERMS:
        pw = [powers[perm[a]] for a in range(4)]
        zp2 = [z2[perm[a]] for a in range(4)]
        cols = [_parts(_psi(g['exps'], g['frame_t'], pw)) for g in layout['groups']]
        for i, orb in enumerate(layout['orbits']):
            x, y, a = cols[orb['group']]
            w = _weight(zp2, orb['r'])
            s = (x * x.transpose() + y * y.transpose()) * w
            sa = (a * a.transpose()) * w
            t_sum[i] = s if t_sum[i] is None else t_sum[i] + s
            t_abs[i] = sa if t_abs[i] is None else t_abs[i] + sa
    feats, absvec = [], []
    for orb, t, ta in zip(layout['orbits'], t_sum, t_abs):
        d = orb['d']
        vec = arb_mat(1, d * d, [t[i, j] for i in range(d) for j in range(d)])
        row = vec * orb['cols']
        feats.extend(row[0, a] for a in range(row.ncols()))
        absvec.append(np.array([float(ta[i, j]) for i in range(d) for j in range(d)]))
    return feats, absvec


def point_from(raw):
    """A point of X at the working precision, from three FP64 coordinates and a branch."""
    re, im, branch = raw
    z = [acb(float(a), float(b)) for a, b in zip(re, im)]
    unit = [acb(1), acb(0, 1), acb(-1), acb(0, -1)][branch]
    z3 = (-(z[0] ** 4 + z[1] ** 4 + z[2] ** 4)).root(4) * unit
    z.append(z3)
    big = max(range(4), key=lambda c: float(abs(z[c]).mid()))
    return [c / z[big] for c in z]


def _task(raw):
    t0 = time.perf_counter()
    z = point_from(raw)
    size = [float(abs(c).mid()) for c in z]
    p, pabs = base_potential(z)
    forms = level_forms(z)
    feats, absvecs = [], []
    for layout in M['layouts']:
        f, a = features(z, layout)
        feats.append([_text(x) for x in f])
        absvecs.append(a)
    rad = max(float(x.rad() / abs(x.mid())) for x in forms)
    return {'rho': min(size) / max(size), 'p0': _text(p), 'p0abs': float(pabs),
            'forms': [_text(x) for x in forms], 'feats': feats, 'abs': absvecs,
            'rad': rad, 'seconds': time.perf_counter() - t0}


def fit(rows, values, prec):
    """theta of the least-squares fit sum_a theta_a F_a = N, at `prec` bits."""
    old = flint.ctx.prec
    flint.ctx.prec = prec
    n_pts, n = len(rows), len(rows[0])
    f = arb_mat(n_pts, n, [arb(s) for row in rows for s in row])
    v = arb_mat(n_pts, 1, [arb(s) for s in values])
    scale = [max(abs(float(f[i, a])) for i in range(n_pts)) or 1.0 for a in range(n)]
    dinv = arb_mat(n, n, [arb(1) / arb(scale[i]) if i == j else arb(0)
                          for i in range(n) for j in range(n)])
    fs = f * dinv
    gram = fs.transpose() * fs
    sol = gram.solve(fs.transpose() * v, algorithm='approx')
    theta = [sol[a, 0] / arb(scale[a]) for a in range(n)]
    flint.ctx.prec = old
    return theta


def apply(rows, theta):
    return [sum((arb(s) * t for s, t in zip(row, theta)), arb(0)) for row in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--col', type=int, required=True, help='the solution column of coeffs')
    ap.add_argument('--n-fit', type=int, default=0, help='0: twice the largest n_theta')
    ap.add_argument('--n-check', type=int, default=150)
    ap.add_argument('--seed', type=int, default=20266001)
    ap.add_argument('--prec', type=int, default=256, help='bits for the point values')
    ap.add_argument('--fit-prec', type=int, default=640, help='bits for the solve')
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    setup(args)
    k1 = [lev['k1'] for lev in M['levels']]
    sizes = [lay['n_theta'] for lay in M['layouts']]
    print(f'levels {k1}, n_theta {sizes}, effective weights per level '
          f'{[len(w) for w in M["w"]]}, setup {time.perf_counter() - t0:.0f}s', flush=True)
    n_fit = args.n_fit or 2 * max(sizes) + 50
    rng = np.random.default_rng(args.seed)
    raws = [(rng.normal(size=3), rng.normal(size=3), int(rng.integers(4)))
            for _ in range(n_fit + args.n_check)]
    t1 = time.perf_counter()
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        res = pool.map(_task, raws, chunksize=1)
    secs = [r['seconds'] for r in res]
    print(f'{len(res)} points in {time.perf_counter() - t1:.0f}s, {np.median(secs):.1f}s '
          f'per point (median); largest relative ball radius of N_L '
          f'{max(r["rad"] for r in res):.1e}', flush=True)
    fit_res, check_res = res[:n_fit], res[n_fit:]

    flint.ctx.prec = args.fit_prec
    k0 = M['k0']
    out = {'k1': np.array(k1), 'n_theta': np.array(sizes),
           'rho_check': np.array([r['rho'] for r in check_res])}
    p0c = [arb(r['p0']) for r in check_res]
    terms, abs_terms = [], []
    print('level  n_theta  fit residual  check error  median and max of abs form over value '
          ' max abs(theta)', flush=True)
    for li, (lev, lay) in enumerate(zip(M['levels'], M['layouts'])):
        rows = [r['feats'][li] for r in fit_res]
        vals = [r['forms'][li] for r in fit_res]
        theta = fit(rows, vals, args.fit_prec)
        nf = [arb(v) for v in vals]
        resid = max(abs(float(a - b)) for a, b in zip(apply(rows, theta), nf))
        resid /= max(abs(float(x)) for x in nf)
        nc = [arb(r['forms'][li]) for r in check_res]
        crow = [r['feats'][li] for r in check_res]
        err = max(abs(float(a - b)) for a, b in zip(apply(crow, theta), nc))
        err /= max(abs(float(x)) for x in nc)
        # The absolute-value form: <abs(C_o), T_abs> summed over the orbits.
        th = np.array([float(t) for t in theta])
        cabs, start = [], 0
        for orb in lay['orbits']:
            n = len(orb['mats'])
            c = sum(t * m for t, m in zip(th[start:start + n], orb['mats']))
            cabs.append(np.abs(c).ravel())
            start += n
        absform = np.array([sum(float(np.dot(ca, ta)) for ca, ta in zip(cabs, r['abs'][li]))
                            for r in check_res])
        size = np.array([abs(float(x)) for x in nc])
        ratio = absform / size
        print(f'{lev["k1"]:5d}  {lay["n_theta"]:7d}  {resid:12.2e}  {err:11.2e}  '
              f'{np.median(ratio):10.2e} {ratio.max():10.2e}  {np.abs(th).max():12.3e}  '
              f'max abs form over max abs(N_L) {absform.max() / size.max():.2e}', flush=True)
        power = arb(-lev['k1']) / k0
        terms.append(np.array([float(x * p ** power) for x, p in zip(nc, p0c)]))
        abs_terms.append(absform * np.array([float(p ** power) for p in p0c]))
        out[f'theta_{lev["k1"]}'] = np.array([_text(t) for t in theta])
        out[f'fit_residual_{lev["k1"]}'] = resid
        out[f'check_error_{lev["k1"]}'] = err
        out[f'abs_over_value_{lev["k1"]}'] = ratio
    terms, abs_terms = np.array(terms), np.array(abs_terms)
    psi = terms.sum(axis=0)
    base_ratio = np.array([r['p0abs'] / float(arb(r['p0'])) for r in check_res])
    print(f'base: abs form over p0, median {np.median(base_ratio):.2e}, '
          f'max {base_ratio.max():.2e}', flush=True)
    print(f'psi = sum of N_L p0^(-L/k0): mean {psi.mean():+.6e}, spread '
          f'{psi.max() - psi.min():.2e}', flush=True)
    s_val = np.abs(terms).sum(axis=0)
    s_abs = abs_terms.sum(axis=0)
    print(f'sum of abs(N_L p0^(-L/k0)): median {np.median(s_val):.2e}, max {s_val.max():.2e}; '
          f'with the abs forms: median {np.median(s_abs):.2e}, max {s_abs.max():.2e}',
          flush=True)
    out.update(terms=terms, abs_terms=abs_terms, base_ratio=base_ratio)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f'saved to {args.out}; total {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
