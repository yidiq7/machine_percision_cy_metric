"""The norms and derivatives of f for a level fit, from f in dd at moved points of X.

Each sample point moves along X in its own chart, t -> t + s v. Here s is a power of
2 and v is one of 8 complex directions. 4 Newton steps in dd put the point back on X,
so it has the chart coordinates t + s v exactly. Central differences of f
give grad f, and second differences along v and i v give the Levi form
ddbar f(v, vbar). Richardson's rule on the steps s and 2s removes the s^2 term.

The stage `compute` runs on the GPU and saves f at every moved point, and g0, the
metric of the HN base in the chart coordinates. The levels change g by at most 2e-12
of itself, so g0 serves for every column. The stage `combine` needs numpy only. It
takes the saved parts and prints the norms for the Riemannian metric
G = c Re(g0), with c = 1/(sqrt(2) pi k0), so that G has volume 1.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/residual_derivatives.py compute --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 44 40 36 \\
        --load runs_levels_load --load-joint MAP --coeffs COEFFS --exact-mul \\
        --n 32768 --seed 20271001 --out runs_derivatives/part_20271001.npz
    python scripts/residual_derivatives.py combine runs_derivatives/part_*.npz

For a two-stage fit of `vpa_dd_stage2.py`, give --stage1, --stage1-k1 and --run in
place of --load-joint and --coeffs; --load is then the directory of the second-stage
level bases.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The 8 complex directions v. The second of each pair is i v, for the Levi form.
DIRECTIONS = np.array([[1, 0], [1j, 0], [0, 1], [0, 1j],
                       [1, 1], [1j, 1j], [1, 1j], [1j, -1]], dtype=complex)
H1, H2 = 2.0 ** -9, 2.0 ** -7          # the steps for grad f, and for the Levi form
AXIS_STEPS = (H1, 2 * H1, H2, 2 * H2)
DIAGONAL_STEPS = (H2, 2 * H2)
NEWTON_STEPS = 4
EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def plan():
    """The list of (direction, step) of the moved points; entry 0 is the point itself."""
    out = [(0, 0.0)]
    for d in range(8):
        for s in (AXIS_STEPS if d < 4 else DIAGONAL_STEPS):
            out += [(d, s), (d, -s)]
    return out


def index_of(entries):
    return {(d, s): i for i, (d, s) in enumerate(entries)}


def compute(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import MLGeometry as mlg
    mlg.set_precision(64)
    import jax
    import jax.numpy as jnp
    import correction_fit as cf
    import vpa_dd_levels as vl
    import vpa_fit as vf
    from quartic import analytic, dd, dd_metric, joint_dd, sections, volume

    t0 = time.perf_counter()

    def say(text):
        print(f'[{time.perf_counter() - t0:7.0f}s] {text}', flush=True)

    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    analytic.EXACT_MUL = args.exact_mul
    sections.PREPARE_DD = True
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    mats0 = sections.prepare(lay0, theta0)
    counts = vf.hn_counts(max(args.k1 + (args.stage1_k1 or [])))

    def load_levels(directory, degrees):
        class LoadArgs:
            load = directory
        return [vl.build_level(args.k0, k1, lay0, mats0, None, None, counts, LoadArgs,
                               Path('.'))[0] for k1 in degrees]
    levels = load_levels(args.load, args.k1)
    s1 = None
    if args.stage1 is not None:
        # A two-stage fit of vpa_dd_stage2.py: the first stage gives psi_1, and the
        # weights of the second stage take exp(-(k1/k0)(psi_1 - C)) with the saved C.
        import vpa_dd_stage2 as st
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
        jm = joint_dd.load(run / f'joint_map_{tag}.npz')
        coeffs = jnp.asarray(np.load(run / f'vpa_stage2_coeffs_{tag}.npy'))
    else:
        jm = joint_dd.load(args.load_joint)
        coeffs = jnp.asarray(np.load(args.coeffs))
    prefix = np.cumsum([int(np.sum(jm.level == i)) for i in range(len(levels))])
    joint = (jm, None, tuple(int(x) for x in prefix))
    say(f'model: k0 = {args.k0}, levels {args.k1}'
        f"{'' if s1 is None else f' on the first stage {args.stage1_k1}'}, "
        f'{coeffs.shape[1]} columns of c')

    batch = cf.load_points(args.n, args.seed)
    entries = plan()
    dirs = jnp.asarray(np.array([DIRECTIONS[d] for d, _ in entries]))
    steps = jnp.asarray(np.array([s for _, s in entries]))

    def move(z_hi, z_lo, patch):
        """The dd points t + s v of one sample point, for every entry of the plan."""
        z_dd = ((jnp.real(z_hi), jnp.real(z_lo)), (jnp.imag(z_hi), jnp.imag(z_lo)))
        m, free = patch[0], (patch[1], patch[2])
        rest, _ = dd_metric._restriction_dd(z_dd, m, free)

        def one(v, s):
            # s v has parts 0 or +-s, a power of 2, so each product is exact in dd.
            total = z_dd
            zero = jnp.zeros_like(s)
            for i in range(2):
                c = ((jnp.real(v[i]) * s, zero), (jnp.imag(v[i]) * s, zero))
                total = dd.cadd(total, dd.cmul(analytic._cindex(rest, i), c))
            for _ in range(NEWTON_STEPS):
                total = dd_metric.newton_step(total, m)
            z2 = dd.cmul(total, total)
            p_val = dd_metric._creduce(dd.cmul(z2, z2), 1)
            return (total[0][0] + 1j * total[1][0], total[0][1] + 1j * total[1][1],
                    jnp.abs(dd.cto_fp(p_val)))
        return jax.vmap(one)(dirs, steps)

    moved_hi, moved_lo, p_res = jax.jit(jax.vmap(move))(
        batch['points_hi'], batch['points_lo'], batch['patch'])
    n, e = moved_hi.shape[:2]
    say(f'{n} points, {e} evaluations each; abs(P) of the moved points <= '
        f'{float(jnp.max(p_res)):.2e}')

    def one(a, cc, hs, ja, arrays1):
        if s1 is None:
            _, hi, (z_dd, m, free), p = vl.point_hessians(lay0, parts0, levels, hs, a,
                                                          vl._with_arrays(joint, ja))
            base = None
        else:
            _, hi, (z_dd, m, free), p, psi = st.point_state(
                lay0, parts0, s1, levels, a, arrays1, hs, vl._with_arrays(joint, ja))
            base = psi[2]
        rest = dd.cto_fp(dd_metric._restriction_dd(z_dd, m, free)[0])
        # The metric of the norms: the HN base, and the first stage when there is one.
        g0 = rest @ dd.cto_fp(hi[0]) @ jnp.conj(rest).T
        if base is not None:
            g0 = g0 + dd.cto_fp(base)

        def column(c):
            delta = tuple(dd.dot_dd_fp(comp, c[:, None, None], axis=0) for comp in p)
            if base is not None:
                delta = dd.cadd(base, delta)
            return analytic._volume_from_phi_dd(hi[0], z_dd, m, free, keep=True,
                                                delta=delta)
        return jax.vmap(column)(cc.T), g0
    program = jax.jit(lambda x, cc, hs, ja, arrays1: jax.lax.map(
        lambda a: one(a, cc, hs, ja, arrays1), x, batch_size=args.chunk))
    hs, ja = vl.level_arrays(levels, joint)
    arrays1 = None if s1 is None else st.stage1_arrays(s1)

    flat_hi = moved_hi.reshape(n * e, 4)
    flat_lo = moved_lo.reshape(n * e, 4)
    flat_patch = jnp.repeat(batch['patch'], e, axis=0)
    unused = jnp.zeros((n * e, 2, 4), dtype=flat_hi.dtype)
    inputs = (flat_hi, unused, (flat_hi, flat_lo, flat_patch))
    mean = volume.mean_v_dd(args.k0)
    fs, gs = [], []
    total = n * e
    for start in range(0, total, args.block):
        stop = min(start + args.block, total)
        take = jnp.minimum(jnp.arange(start, start + args.block), stop - 1)
        v, g0 = program(jax.tree_util.tree_map(lambda a: a[take], inputs), coeffs, hs, ja,
                        arrays1)
        f = analytic.residual_from_volumes(v, None, mean=mean)
        fs.append(np.asarray(f[:stop - start]))
        gs.append(np.asarray(g0[:stop - start]))
        say(f'{stop} of {total} evaluations')
    f = np.concatenate(fs).reshape(n, e, -1)
    g0 = np.concatenate(gs).reshape(n, e, 2, 2)[:, 0]
    z = np.asarray(batch['points_hi'])
    ratio = np.abs(z).min(axis=1) / np.abs(z).max(axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f=f, g0=g0, mass=np.asarray(batch['mass']), ratio=ratio,
             k0=args.k0, seed=args.seed, entries=np.array(entries))
    say(f'saved {out}')


def derivatives(f, entries):
    """d f, the Levi form H = ddbar f, and two step checks, from f at the moved points."""
    at = index_of([(int(d), float(s)) for d, s in entries])
    f0 = f[:, at[(0, 0.0)]]

    def first(d, h):
        return (f[:, at[(d, h)]] - f[:, at[(d, -h)]]) / (2 * h)

    def second(d, h):
        return (f[:, at[(d, h)]] + f[:, at[(d, -h)]] - 2 * f0) / h ** 2

    def rich(a, b):
        return (4 * a - b) / 3
    dr = [rich(first(d, H1), first(d, 2 * H1)) for d in range(4)]
    dr_check = [rich(first(d, H2), first(d, 2 * H2)) for d in range(4)]
    s = [rich(second(d, H2), second(d, 2 * H2)) for d in range(8)]
    s_raw = [second(d, H2) for d in range(8)]
    df = np.stack([(dr[0] - 1j * dr[1]) / 2, (dr[2] - 1j * dr[3]) / 2], axis=1)
    df_check = np.stack([(dr_check[0] - 1j * dr_check[1]) / 2,
                         (dr_check[2] - 1j * dr_check[3]) / 2], axis=1)

    def levi(ss):
        lev = [(ss[2 * k] + ss[2 * k + 1]) / 4 for k in range(4)]
        h00, h11 = lev[0], lev[1]
        h01 = (lev[2] - h00 - h11) / 2 + 1j * (lev[3] - h00 - h11) / 2
        return np.stack([np.stack([h00 + 0j, h01], -1),
                         np.stack([np.conj(h01), h11 + 0j], -1)], -2)
    return f0, df, df_check, levi(s), levi(s_raw)


def combine(paths):
    parts = [np.load(p) for p in paths]
    k0 = int(parts[0]['k0'])
    c = 1.0 / (math.sqrt(2.0) * math.pi * k0)
    f = np.concatenate([p['f'] for p in parts])
    g0 = np.concatenate([p['g0'] for p in parts])
    mass = np.concatenate([p['mass'] for p in parts])
    ratio = np.concatenate([p['ratio'] for p in parts])
    entries = parts[0]['entries']
    w = mass / mass.sum()
    ginv = np.linalg.inv(g0)
    print(f'{f.shape[0]} points from {len(parts)} samples, {f.shape[1]} evaluations each; '
          f'G = c Re(g0), c = {c:.6e}, volume 1')
    result = {}
    for col in range(f.shape[2]):
        f0, df, df2, lev, lev_raw = derivatives(f[:, :, col], entries)
        # abs(grad f)^2 = 4 (df)^dag g^-1 df for Re(g); G = c Re(g) divides it by c.
        grad = np.sqrt(np.maximum(4 * np.real(np.einsum('ni,nij,nj->n', np.conj(df), ginv,
                                                        df)), 0.0) / c)
        grad_t = np.sqrt(np.maximum(4 * np.real(np.einsum('nij,ni,nj->n', ginv, df,
                                                          np.conj(df))), 0.0) / c)
        lap = 4 * np.real(np.einsum('nij,nji->n', ginv, lev)) / c
        # Ric = 2 Re(R) against Re(g) with R = -ddbar log(1 + f); the eigenvalues of
        # g^-1 R, times 2/c, are those of Ric against G, each twice.
        eig = 2 * np.linalg.eigvals(np.einsum('nij,njk->nik', ginv, -lev)).real / c
        ric_sup = np.abs(eig).max(axis=1)
        ric_norm = np.sqrt(2 * np.sum(eig ** 2, axis=1))
        d1 = np.linalg.norm(df - df2, axis=1) / np.maximum(np.linalg.norm(df, axis=1), 1e-300)
        d2 = (np.linalg.norm((lev - lev_raw).reshape(len(w), -1), axis=1)
              / np.maximum(np.linalg.norm(lev.reshape(len(w), -1), axis=1), 1e-300))
        row = {
            'sigma': float(np.sum(w * np.abs(f0))),
            'l2_f': float(math.sqrt(np.sum(w * f0 ** 2))),
            'max_f': float(np.abs(f0).max()),
            'l2_grad': float(math.sqrt(np.sum(w * grad ** 2))),
            'max_grad': float(grad.max()),
            'l2_grad_transposed': float(math.sqrt(np.sum(w * grad_t ** 2))),
            'int_f_lap': float(-np.sum(w * f0 * lap)),
            'max_lap': float(np.abs(lap).max()),
            'l2_lap': float(math.sqrt(np.sum(w * lap ** 2))),
            'max_ric': float(ric_sup.max()),
            'l2_ric': float(math.sqrt(np.sum(w * ric_norm ** 2))),
            'step_check_grad_median': float(np.median(d1)),
            'step_check_grad_p99': float(np.quantile(d1, 0.99)),
            'step_check_levi_median': float(np.median(d2)),
            'step_check_levi_p99': float(np.quantile(d2, 0.99)),
            'max_grad_over_max_f': float(grad.max() / np.abs(f0).max()),
        }
        regions = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sel = (ratio >= lo) & (ratio < hi)
            ws = w[sel] / w[sel].sum()
            regions.append({'bin': [lo, hi], 'weight': float(w[sel].sum()),
                             'l2_f': float(math.sqrt(np.sum(ws * f0[sel] ** 2))),
                             'l2_grad': float(math.sqrt(np.sum(ws * grad[sel] ** 2))),
                             'max_grad': float(grad[sel].max())})
        row['regions'] = regions
        result[col] = row
        print(f'column {col}:')
        for key, val in row.items():
            if key != 'regions':
                print(f'    {key:26s} {val:.4e}')
        for reg in regions:
            print(f'    min/max abs(z) in {reg["bin"][0]:.2f} to {reg["bin"][1]:.2f}: weight '
                  f'{reg["weight"]:.4f}, L2 f {reg["l2_f"]:.3e}, L2 grad f '
                  f'{reg["l2_grad"]:.3e}, max grad f {reg["max_grad"]:.3e}')
    out = Path(paths[0]).parent / 'derivatives_summary.json'
    out.write_text(json.dumps(result, indent=1))
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['compute', 'combine'])
    ap.add_argument('parts', nargs='*')
    ap.add_argument('--k0', type=int, default=28)
    ap.add_argument('--theta', type=str)
    ap.add_argument('--k1', type=int, nargs='+')
    ap.add_argument('--load', type=str)
    ap.add_argument('--load-joint', type=str)
    ap.add_argument('--coeffs', type=str)
    ap.add_argument('--stage1', type=str, default=None,
                    help='for a two-stage fit: the first-stage run of vpa_dd_levels.py')
    ap.add_argument('--stage1-k1', type=int, nargs='+', default=None)
    ap.add_argument('--stage1-col', type=int, default=-1)
    ap.add_argument('--run', type=str, default=None,
                    help='for a two-stage fit: the output directory of vpa_dd_stage2.py')
    ap.add_argument('--exact-mul', action='store_true')
    ap.add_argument('--n', type=int, default=32_768)
    ap.add_argument('--seed', type=int, default=20271001)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--block', type=int, default=65_536)
    ap.add_argument('--out', type=str)
    args = ap.parse_args()
    if args.stage == 'compute':
        compute(args)
    else:
        combine(args.parts)


if __name__ == '__main__':
    main()
