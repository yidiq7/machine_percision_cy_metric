"""The derivatives of f for a level fit from Taylor jets in dd, with no differences.

At each point the jets of bidegree (2, 2) in the chart coordinates (`quartic.tjets`)
go through the base, the V+A recurrence of each level and the weights, and give
Phi to D D Dbar Dbar. Then f, D f, the Levi form of f and the Ricci form follow
with the same digits as f itself: no step divides the noise of the arithmetic.

psi = sum_t w_t Gt_t over the joint functions equals sum_t what_t G_t over the level
functions, with what from `joint_dd.effective_weights`, so each level needs one
combination of its jets and no joint transform.

The stage `compute` runs on the GPU and saves, for the columns base, first stage
and final fit, f, D f, ddbar f, Ric = -ddbar log(1 + f) and g at every point. The
stage `combine` needs numpy only and prints the norms for G = c Re(g), c =
1/(sqrt(2) pi k0), the metric of volume 1. The stage `check` compares the jets
with the 13-row jets and with f of the residual program at a few points.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/residual_jets.py compute --k0 12 \\
        --theta runs_explore/f2r/theta_k0_12.npy --stage1 runs_explore/s1b \\
        --stage1-k1 24 23 22 21 20 19 18 17 --k1 36 35 34 33 32 31 30 29 \\
        --load runs_explore/f2r --run runs_explore/s2e --exact-mul \\
        --n 32768 --seed 20271001 --out runs_jets_s2e/part_20271001.npz
    python scripts/residual_jets.py combine runs_jets_s2e/part_*.npz
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
COLUMNS = ('base', 'first stage', 'final fit')


def load_model(args):
    """The base, the levels of both stages, and the effective weights of each level."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import MLGeometry as mlg
    mlg.set_precision(64)
    import jax
    import jax.numpy as jnp
    import vpa_dd_levels as vl
    import vpa_dd_stage2 as st
    import vpa_fit as vf
    from quartic import analytic, joint_dd, sections

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
    levels1 = load_levels(args.stage1, args.stage1_k1)
    jm1 = joint_dd.load(Path(args.stage1) / f'joint_map_{s1_tag}.npz')
    sizes1 = [int(lev.basis.h.shape[0]) for lev in levels1]
    prefix1 = np.cumsum([int(np.sum(jm1.level == i)) for i in range(len(levels1))])
    c1 = np.load(Path(args.stage1) / f'vpa_levels_coeffs_{s1_tag}.npy')[:, args.stage1_col]
    w1 = st.stage1_weights(c1, jm1, prefix1)
    tag = f"k0_{args.k0}_s1_{'_'.join(str(k) for k in args.stage1_k1)}_s2_" + \
        '_'.join(str(k) for k in args.k1)
    run = Path(args.run)
    shift = float(json.loads((run / f'vpa_stage2_{tag}.json').read_text())['stage1_shift'])
    levels2 = load_levels(args.load, args.k1)
    jm2 = joint_dd.load(run / f'joint_map_{tag}.npz')
    sizes2 = [int(lev.basis.h.shape[0]) for lev in levels2]
    prefix2 = np.cumsum([int(np.sum(jm2.level == i)) for i in range(len(levels2))])
    c2 = np.load(run / f'vpa_stage2_coeffs_{tag}.npy')[:, args.col]
    w2 = st.stage1_weights(c2, jm2, prefix2)
    what1 = joint_dd.effective_weights(jm1, w1, sizes1)
    what2 = joint_dd.effective_weights(jm2, w2, sizes2)
    s1 = st.Stage1(levels1, jm1, w1, shift)
    return dict(lay0=lay0, parts0=parts0, levels1=levels1, levels2=levels2, what1=what1,
                what2=what2, shift=shift, s1=s1, jm2=jm2, prefix2=prefix2,
                coeffs=np.load(run / f'vpa_stage2_coeffs_{tag}.npy'))


def jet_program(model, k0, chunk):
    """f, D f, ddbar f, Ric and g for the columns base, first stage and final fit."""
    import jax
    import jax.numpy as jnp
    from quartic import arnoldi_dd, dd, tjets as tj, volume, vpa_dd
    import vpa_dd_stage2 as st

    lay0, parts0 = model['lay0'], model['parts0']
    levels1, levels2 = model['levels1'], model['levels2']
    top = 3
    for g in lay0.groups:
        top = max(top, 4 * int(np.max(g.exps)))
    for lev in levels1 + levels2:
        for _, a, b in lev.basis.entries:
            top = max(top, int(np.max(a)), int(np.max(b)))
    mean = tuple(jnp.asarray(x) for x in volume.mean_v_dd(k0))

    def level_sum(levels, hs, whats, table, r2, r20, p0, lp, lr, extra_nil, extra_val):
        total = None
        for lev, (h, hl, uh, ul), (wh, wl) in zip(levels, hs, whats):
            basis = dataclasses.replace(lev.basis, h=h)
            y = tj.centered(tj.multiplier_jets(basis.entries, table, r2), basis)
            q = arnoldi_dd.joint_jets_dd(basis, hl, y, (uh, ul), algebra=tj.T36)
            # Exact dd products and a dd sum: an Ozaki product bounds its error by the largest
            # weight times the largest jet, and the two belong to different functions.
            terms = dd.mul(tj.T36.flat(q), (wh[None, :], wl[None, :]))
            comb = jax.tree_util.tree_map(lambda a: a[:, None], dd.sum_dd(*terms, axis=1))
            big_q = jax.tree_util.tree_map(lambda a: a[0], tj.T36.unflat(comb))
            alpha = vpa_dd.alpha_dd(lev.alpha)
            lnil = dd.csub(tj.analytic._cscale(lr, jnp.asarray(float(lev.k1))),
                           tj._scale(lp, alpha))
            w0 = vpa_dd.weight_dd(p0, r20, lev.k0, lev.k1)
            if extra_nil is not None:
                lnil = dd.csub(lnil, tj._scale(extra_nil, alpha))
                w0 = dd.mul(w0, st.exp_small(dd.neg(dd.mul(alpha, extra_val))))
            term = tj.mul(tj.weight_jet(w0, lnil), big_q)
            total = term if total is None else dd.cadd(total, term)
        return total

    def one(a, hs1, hs2, what1, what2, shift):
        z_hi, z_lo, patch = a
        z_dd = ((jnp.real(z_hi), jnp.real(z_lo)), (jnp.imag(z_hi), jnp.imag(z_lo)))
        ch = tj.chart_jets(z_dd, patch[0], (patch[1], patch[2]))
        table = tj.power_table(ch.hol, top)
        big_p = tj.base_jet(lay0, parts0, ch, table)
        p0 = tj.value(big_p)
        r2 = tj.radius_jet(ch)
        r20 = tj.value(r2)
        lp = tj.log1p_nil(tj._scale(tj.nil(big_p), dd.reciprocal(p0)))
        lr = tj.log1p_nil(tj._scale(tj.nil(r2), dd.reciprocal(r20)))
        psi1 = level_sum(levels1, hs1, what1, table, r2, r20, p0, lp, lr, None, None)
        psi1_val = dd.add(tj.value(psi1), dd.neg(shift))
        psi1_nil = tj.nil(psi1)
        psi2 = level_sum(levels2, hs2, what2, table, r2, r20, p0, lp, lr, psi1_nil, psi1_val)
        phi_a = dd.cadd(lp, psi1_nil)
        phi_b = dd.cadd(phi_a, tj.nil(psi2))
        out = []
        for phi in (lp, phi_a, phi_b):
            f0, df, lev, ric, g0 = tj.metric_parts(phi, ch, mean)
            out.append((dd.to_fp(f0), dd.cto_fp(df), dd.cto_fp(lev), dd.cto_fp(ric),
                        dd.cto_fp(g0)))
        stack = lambda *xs: jnp.stack(xs)
        return jax.tree_util.tree_map(stack, *out)

    import vpa_dd_levels as vl
    hs1, _ = vl.level_arrays(levels1)
    hs2, _ = vl.level_arrays(levels2)
    what1 = tuple((jnp.asarray(h), jnp.asarray(l)) for h, l in model['what1'])
    what2 = tuple((jnp.asarray(h), jnp.asarray(l)) for h, l in model['what2'])
    shift = dd.to_dd(jnp.asarray(model['shift']))
    program = jax.jit(lambda x, hs1, hs2, what1, what2, shift: jax.lax.map(
        lambda a: one(a, hs1, hs2, what1, what2, shift), x, batch_size=chunk))
    return lambda x: program(x, hs1, hs2, what1, what2, shift)


def compute(args):
    t0 = time.perf_counter()

    def say(text):
        print(f'[{time.perf_counter() - t0:7.0f}s] {text}', flush=True)
    model = load_model(args)
    import jax
    import jax.numpy as jnp
    import correction_fit as cf
    say(f"model: k0 = {args.k0}, first stage {args.stage1_k1}, second stage {args.k1}; "
        f"effective weights of {len(model['what1'])} + {len(model['what2'])} levels")
    program = jet_program(model, args.k0, args.chunk)
    batch = cf.load_points(args.n, args.seed)
    if args.other_chart:
        # The same points in the chart of the second largest coordinate, for the noise.
        import seam_levels as sl
        batch = sl.other_chart(batch)
    inputs = (batch['points_hi'], batch['points_lo'], batch['patch'])
    n = inputs[0].shape[0]
    parts = []
    for start in range(0, n, args.block):
        stop = min(start + args.block, n)
        take = jnp.minimum(jnp.arange(start, start + args.block), stop - 1)
        got = program(jax.tree_util.tree_map(lambda a: a[take], inputs))
        parts.append(jax.tree_util.tree_map(lambda a: np.asarray(a[:stop - start]), got))
        say(f'{stop} of {n} points')
    f, df, lev, ric, g = (np.concatenate([p[i] for p in parts]) for i in range(5))
    z = np.asarray(batch['points_hi'])
    ratio = np.abs(z).min(axis=1) / np.abs(z).max(axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f=f, df=df, lev=lev, ric=ric, g=g, mass=np.asarray(batch['mass']),
             ratio=ratio, k0=args.k0, seed=args.seed)
    say(f'saved {out}')


def check(args):
    """The jets against the 13-row jets and the residual program, at a few points."""
    t0 = time.perf_counter()

    def say(text):
        print(f'[{time.perf_counter() - t0:7.0f}s] {text}', flush=True)
    model = load_model(args)
    import jax
    import jax.numpy as jnp
    import correction_fit as cf
    import vpa_dd_levels as vl
    import vpa_dd_stage2 as st
    from quartic import (analytic, arnoldi_dd, correction, dd, dd_metric, tjets as tj,
                         volume, vpa_dd)
    lay0, parts0 = model['lay0'], model['parts0']
    batch = cf.load_points(args.n, args.seed)
    inputs = (batch['points_hi'], batch['points_lo'], batch['patch'])
    lev1, lev2 = model['levels1'][0], model['levels2'][0]
    hs1, _ = vl.level_arrays([lev1])
    hs2, _ = vl.level_arrays([lev2])

    def compare(a, hs1, hs2):
        z_hi, z_lo, patch = a
        z_dd = ((jnp.real(z_hi), jnp.real(z_lo)), (jnp.imag(z_hi), jnp.imag(z_lo)))
        m, free = patch[0], (patch[1], patch[2])
        ch = tj.chart_jets(z_dd, m, free)
        table = tj.power_table(ch.hol, 12)
        rest_dd, _ = dd_metric._restriction_dd(z_dd, m, free)
        out = {}
        # 1. The 13-row jets of q against the low entries of the jets of bidegree (2, 2).
        r2 = tj.radius_jet(ch)
        for name, lev, (h, hl, uh, ul) in (('q, first stage', lev1, hs1[0]),
                                            ('q, second stage', lev2, hs2[0])):
            basis = dataclasses.replace(lev.basis, h=h)
            old = arnoldi_dd.joint_jets_dd(basis, hl, vpa_dd.basis_multiplier_jets(
                basis, z_dd, rest_dd), (uh, ul))
            new = arnoldi_dd.joint_jets_dd(basis, hl, tj.centered(
                tj.multiplier_jets(basis.entries, table, r2), basis), (uh, ul),
                algebra=tj.T36)
            nv = dd.to_fp(tj.value(new))
            nd = dd.cto_fp(analytic._cindex(new, Ellipsis, slice(1, 3), 0))
            nh = dd.cto_fp(analytic._cindex(new, Ellipsis, slice(1, 3), slice(1, 3)))
            ov, od, oh = dd.to_fp(old[0]), dd.cto_fp(old[1]), dd.cto_fp(old[2])
            out[name] = jnp.stack([jnp.max(jnp.abs(nv - ov)) / jnp.max(jnp.abs(ov)),
                                   jnp.max(jnp.abs(nd - od)) / jnp.max(jnp.abs(od)),
                                   jnp.max(jnp.abs(nh - oh)) / jnp.max(jnp.abs(oh))])
        # 2. ddbar log p0 and d log p0 against R phi R^dag and R a of the base state.
        _, hi, p_dd = correction.base_state_dd(lay0, parts0, None, (z_hi, z_lo, patch),
                                               with_p=True)
        big_p = tj.base_jet(lay0, parts0, ch, table)
        lp = tj.log1p_nil(tj._scale(tj.nil(big_p), dd.reciprocal(tj.value(big_p))))
        rest = dd.cto_fp(rest_dd)
        g_old = rest @ dd.cto_fp(hi[0]) @ jnp.conj(rest).T
        d_old = rest @ dd.cto_fp(hi[1])
        g_new = dd.cto_fp(analytic._cindex(lp, slice(1, 3), slice(1, 3)))
        d_new = dd.cto_fp(analytic._cindex(lp, slice(1, 3), 0))
        out['base'] = jnp.stack([
            jnp.abs(dd.to_fp(tj.value(big_p)) / dd.to_fp(p_dd) - 1.0),
            jnp.max(jnp.abs(d_new - d_old)) / jnp.max(jnp.abs(d_old)),
            jnp.max(jnp.abs(g_new - g_old)) / jnp.max(jnp.abs(g_old))])
        return out
    run = jax.jit(jax.vmap(compare, in_axes=(0, None, None)))
    take = jnp.arange(args.n)
    got = run(jax.tree_util.tree_map(lambda a: a[take], inputs), hs1, hs2)
    for name, val in got.items():
        v = np.asarray(val)
        say(f'{name}: largest relative difference of value {v[:, 0].max():.2e}, of D '
            f'{v[:, 1].max():.2e}, of D Dbar {v[:, 2].max():.2e}')
    # 3. f of the final fit from the jets against the residual program.
    program = jet_program(model, args.k0, args.chunk)
    f_jets = np.asarray(program(inputs)[0])                       # (n, 3)
    joint = (model['jm2'], None, tuple(int(x) for x in model['prefix2']))
    res = st.residual_program(lay0, parts0, model['s1'], model['levels2'], args.chunk, joint)
    import vpa_fit as vf
    v = res(vf.point_inputs(batch), jnp.asarray(model['coeffs']))
    f_old = np.asarray(analytic.residual_from_volumes(v, None, mean=volume.mean_v_dd(args.k0)))
    for col, name in ((0, 'first stage'), (1, 'final fit')):
        diff = np.abs(f_jets[:, col + 1] - f_old[:, col])
        say(f'f, {name}: largest difference to the residual program {diff.max():.2e}, '
            f'largest abs(f) {np.abs(f_old[:, col]).max():.2e}')


def norms(f, df, lev, ric, g, w, c):
    """The norms of one column for the metric G = c Re(g) of volume 1."""
    ginv = np.linalg.inv(g)
    grad = np.sqrt(np.maximum(4 * np.real(np.einsum('ni,nij,nj->n', np.conj(df), ginv, df)),
                              0.0) / c)
    lap = 4 * np.real(np.einsum('nij,nji->n', ginv, lev)) / c
    eig = 2 * np.linalg.eigvals(np.einsum('nij,njk->nik', ginv, ric)).real / c
    ric_sup = np.abs(eig).max(axis=1)
    ric_norm = np.sqrt(2 * np.sum(eig ** 2, axis=1))
    return {
        'sigma': float(np.sum(w * np.abs(f))),
        'l2_f': float(math.sqrt(np.sum(w * f ** 2))),
        'max_f': float(np.abs(f).max()),
        'l2_grad': float(math.sqrt(np.sum(w * grad ** 2))),
        'max_grad': float(grad.max()),
        'int_grad_sq': float(np.sum(w * grad ** 2)),
        'int_f_lap': float(-np.sum(w * f * lap)),
        'l2_lap': float(math.sqrt(np.sum(w * lap ** 2))),
        'max_lap': float(np.abs(lap).max()),
        'l2_ric': float(math.sqrt(np.sum(w * ric_norm ** 2))),
        'max_ric': float(ric_sup.max()),
        'q9999_ric': float(np.quantile(ric_sup, 0.9999)),
    }, grad


def combine(paths):
    import json as _json
    parts = [np.load(p) for p in paths]
    k0 = int(parts[0]['k0'])
    c = 1.0 / (math.sqrt(2.0) * math.pi * k0)
    arr = {key: np.concatenate([p[key] for p in parts]) for key in
           ('f', 'df', 'lev', 'ric', 'g', 'mass', 'ratio')}
    w = arr['mass'] / arr['mass'].sum()
    print(f"{len(w)} points from {len(parts)} samples; G = c Re(g), c = {c:.6e}, volume 1")
    result = {}
    for col, name in enumerate(COLUMNS):
        row, grad = norms(arr['f'][:, col], arr['df'][:, col], arr['lev'][:, col],
                          arr['ric'][:, col], arr['g'][:, col], w, c)
        regions = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sel = (arr['ratio'] >= lo) & (arr['ratio'] < hi)
            ws = w[sel] / w[sel].sum()
            regions.append({'bin': [lo, hi], 'weight': float(w[sel].sum()),
                            'l2_f': float(math.sqrt(np.sum(ws * arr['f'][sel, col] ** 2))),
                            'l2_grad': float(math.sqrt(np.sum(ws * grad[sel] ** 2))),
                            'max_grad': float(grad[sel].max())})
        row['regions'] = regions
        result[name] = row
        print(f'column {col}, {name}:')
        for key, val in row.items():
            if key != 'regions':
                print(f'    {key:14s} {val:.4e}')
        for reg in regions:
            print(f'    min/max abs(z) in {reg["bin"][0]:.2f} to {reg["bin"][1]:.2f}: weight '
                  f'{reg["weight"]:.4f}, L2 f {reg["l2_f"]:.3e}, L2 grad f '
                  f'{reg["l2_grad"]:.3e}, max grad f {reg["max_grad"]:.3e}')
    out = Path(paths[0]).parent / 'jets_summary.json'
    out.write_text(_json.dumps(result, indent=1))
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['compute', 'combine', 'check'])
    ap.add_argument('parts', nargs='*')
    ap.add_argument('--k0', type=int, default=12)
    ap.add_argument('--theta', type=str)
    ap.add_argument('--k1', type=int, nargs='+')
    ap.add_argument('--load', type=str, help='the directory of the second-stage level bases')
    ap.add_argument('--stage1', type=str, help='the first-stage run of vpa_dd_levels.py')
    ap.add_argument('--stage1-k1', type=int, nargs='+')
    ap.add_argument('--stage1-col', type=int, default=-1)
    ap.add_argument('--run', type=str, help='the output directory of vpa_dd_stage2.py')
    ap.add_argument('--col', type=int, default=1,
                    help='the column of the second-stage coefficients for the final fit')
    ap.add_argument('--exact-mul', action='store_true')
    ap.add_argument('--n', type=int, default=32_768)
    ap.add_argument('--seed', type=int, default=20271001)
    ap.add_argument('--other-chart', action='store_true',
                    help='write each point in the chart of its second largest coordinate')
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--block', type=int, default=4096)
    ap.add_argument('--out', type=str)
    args = ap.parse_args()
    if args.stage == 'compute':
        compute(args)
    elif args.stage == 'check':
        check(args)
    else:
        combine(args.parts)


if __name__ == '__main__':
    main()
