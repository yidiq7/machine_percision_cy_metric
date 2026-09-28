"""Does the linear model of LM hold where the fit at k = 28 stalls?

LM predicts the residual after a step `delta` as `r + J delta`. `optimistix`
accepts the step when the true decrease of the loss is at least 1 percent of
the decrease that this model predicts. The model drops the second derivative of
`r`. In the loss that term is `(1/2) delta^T S delta` with
`S = sum_i r_i Hess(r_i)`, and it vanishes where `r = 0`. The test problem of
`lm_solve_test.py` has an exact zero minimum, so it cannot detect the term.

At each theta the test scans the damping `lambda` of the LM step
`delta = -(J^T J + lambda I)^-1 J^T r`, and it compares the true decrease of
the loss with the predicted decrease. Two sets of coordinates give two families
of steps for the same h:

  * plain: the coordinates of the fits,
  * whitened: the L2(X)-whitened coordinates. There `cond(J)*eps` is below 1,
    so the step is accurate up to the Gauss-Newton limit.

For each step the test also measures the part of the residual that the model
drops, `d(t) = r(theta + t delta) - r - t J delta`, at t = 1 and t = 1/2. The
ratio `|d(1)|/|d(1/2)|` is near 4 for a second-order term, near 2 for an error
of `J`, and near 1 for round-off. The shortfall of the true decrease splits
exactly into `(r + J delta)^T d`, the part of `S`, and `|d|^2/2`, the curvature
of the residual along the step.

The controls are points part way through a fit that still descends: legs of
`--leg` LM steps from theta_FS, on the same fit set and in the same arithmetic.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/lm_model_test.py --k 28 --n-fit 59312 \\
        --chunk 2653 --section-dd --theta runs/theta_k28_seed20260813.npy
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd_metric, fit, sections, whiten   # noqa: E402

EPS = float(np.finfo(np.float64).eps)
ACCEPT = 0.01          # low_cutoff of optimistix.ClassicalTrustRegion
RELIABLE = 0.1         # largest eps*s_max/sqrt(lambda) for an accurate step
T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def to_whitened(lay_p, lay_w, theta_p):
    """theta in the whitened coordinates for the same h as `theta_p`.

    H_rep = W H_white W, so H_white = W^-1 H_rep W^-1. `fubini_study_theta`
    applies the same map to the Fubini-Study point.
    """
    raw = np.asarray(theta_p) * np.asarray(lay_p.scale)
    out = np.zeros(lay_p.n_theta)
    struct = lay_p.structure
    for oi, (orb, sl) in enumerate(zip(struct.orbits, struct.param_slices())):
        h_rep = np.einsum('a,aij->ij', raw[sl], orb.basis)
        inv = np.asarray(lay_w.whiten_inv[oi])
        out[sl] = np.einsum('aij,ij->a', orb.basis, inv @ h_rep @ inv)
    return jnp.asarray(out / np.asarray(lay_w.scale))


def h_difference(lay_a, th_a, lay_b, th_b):
    """Largest relative difference of the block matrices of two layouts."""
    worst = 0.0
    for a, b in zip(sections.prepare(lay_a, th_a), sections.prepare(lay_b, th_b)):
        worst = max(worst, float(jnp.abs(a - b).max() / jnp.abs(a).max()))
    return worst


def scan(res, jac_fn, theta, gauge=None):
    """One family of damped steps at theta. Returns the rows and a summary.

    J = Q R and R = U S V^T. The SVD is backward stable, so each step is the
    exact damped step of a J within approximately eps*norm(J) of the true one.
    """
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(res(jnp.asarray(theta)))
    loss = 0.5 * float(r @ r)
    jac = jac_fn(jnp.asarray(theta))
    q, upper = jnp.linalg.qr(jac, mode='reduced')
    u, s, vt = jnp.linalg.svd(upper)
    coef = np.asarray(u.T @ (q.T @ jnp.asarray(r)))
    s, vt = np.asarray(s), np.asarray(vt)
    del q, upper, u
    s_max = float(s[0])

    # h -> c h leaves v unchanged, so J has one null direction along theta.
    # Keep its singular vector out of every step and out of every share. For a
    # correction on a base, the caller gives that direction.
    gauge = theta if gauge is None else np.asarray(gauge, dtype=np.float64)
    gauge = gauge / np.linalg.norm(gauge)
    overlap = np.abs(vt @ gauge)
    ig = int(np.argmax(overlap))
    keep = np.ones(s.size, dtype=bool)
    keep[ig] = False
    s_small = max(float(s[keep].min()), EPS * s_max)

    # The decrease that the Gauss-Newton limit predicts, by decade of s/s_max.
    share = np.where(keep, 0.5 * coef ** 2 / loss, 0.0)
    decade = np.floor(np.log10(np.maximum(s / s_max, 1e-300))).astype(int)
    by_decade = {int(dec): float(share[decade == dec].sum())
                 for dec in sorted(set(decade[keep].tolist()), reverse=True)}

    rows = []
    lam = 1e2 * s_max ** 2
    while lam >= 1e-2 * s_small ** 2:
        filt = np.where(keep, s / (s * s + lam), 0.0)
        delta = -(vt.T @ (filt * coef))
        delta -= (gauge @ delta) * gauge
        jd = np.asarray(jac @ jnp.asarray(delta))
        model = r + jd
        pred = loss - 0.5 * float(model @ model)
        r1 = np.asarray(res(jnp.asarray(theta + delta)))
        rh = np.asarray(res(jnp.asarray(theta + 0.5 * delta)))
        true = loss - 0.5 * float(r1 @ r1)
        d1 = r1 - model
        dh = rh - (r + 0.5 * jd)
        gap = pred - true
        cross = float(model @ d1)
        square = 0.5 * float(d1 @ d1)
        rows.append({
            'lambda_over_smax2': lam / s_max ** 2,
            'solve_err': EPS * s_max / math.sqrt(lam),
            'step_rel': float(np.linalg.norm(delta) / np.linalg.norm(theta)),
            'pred': pred / loss,
            'true': true / loss,
            'rho': true / pred if pred > 0 else float('nan'),
            'order': float(np.linalg.norm(d1) / max(np.linalg.norm(dh), 1e-300)),
            'd_over_jd': float(np.linalg.norm(d1) / max(np.linalg.norm(jd), 1e-300)),
            'cross_share': cross / gap if gap != 0 else float('nan'),
            'square_share': square / gap if gap != 0 else float('nan'),
        })
        lam /= 100.0
    del jac

    ok = [w for w in rows if w['solve_err'] < RELIABLE and np.isfinite(w['true'])
          and w['pred'] > 0]
    top = max(ok, key=lambda w: w['pred']) if ok else None
    taken = [w for w in ok if w['rho'] >= ACCEPT]
    summary = {
        'loss': loss,
        'grad_norm': float(np.linalg.norm(s * coef)),
        's_max': s_max,
        'cond': s_max / s_small,
        'gauge_overlap': float(overlap[ig]),
        'gn_share': float(share.sum()),
        'gn_share_by_decade': by_decade,
        'max_pred': top['pred'] if top else float('nan'),
        'rho_at_max_pred': top['rho'] if top else float('nan'),
        'order_at_max_pred': top['order'] if top else float('nan'),
        'cross_at_max_pred': top['cross_share'] if top else float('nan'),
        'square_at_max_pred': top['square_share'] if top else float('nan'),
        'best_accepted': max((w['true'] for w in taken), default=0.0),
    }
    return rows, summary


def show(tag, rows, summary):
    print(f"\n  {tag}: cond(J) = {summary['cond']:.3e}, norm(J^T r) = "
          f"{summary['grad_norm']:.3e}, gauge overlap = "
          f"{summary['gauge_overlap']:.6f}")
    print(f"  the Gauss-Newton limit predicts a decrease of "
          f"{summary['gn_share']:.4e} of the loss. By decade of s/s_max:")
    parts = [f"1e{dec}: {val:.2e}" for dec, val in summary['gn_share_by_decade'].items()
             if val > 1e-8]
    for i in range(0, len(parts), 5):
        print('    ' + '   '.join(parts[i:i + 5]))
    print(f"  {'lam/smax^2':>10} {'solve':>6} {'|dth|/|th|':>10} {'pred/L':>10} "
          f"{'true/L':>10} {'rho':>9} {'|d1|/|dh|':>9} {'|d1|/|Jd|':>9} "
          f"{'cross':>6} {'square':>6}")
    for w in rows:
        mark = ' ' if w['solve_err'] < RELIABLE else '*'
        print(f"  {w['lambda_over_smax2']:>10.0e} {w['solve_err']:>6.0e}{mark}"
              f"{w['step_rel']:>10.2e} {w['pred']:>10.3e} {w['true']:>10.3e} "
              f"{w['rho']:>9.3f} {w['order']:>9.2f} {w['d_over_jd']:>9.2e} "
              f"{w['cross_share']:>6.2f} {w['square_share']:>6.2f}")
    print("  * marks a step whose solve error bound is above "
          f"{RELIABLE}; the summary leaves it out.")


def measure(name, theta_p, lay_p, lay_w, fns, weights):
    """Both families at one point, and one summary line for each."""
    res_p, res_p2, jac_p, res_w, jac_w = fns
    r = np.asarray(res_p(theta_p))
    r2 = np.asarray(res_p2(theta_p))
    loss, loss2 = 0.5 * float(r @ r), 0.5 * float(r2 @ r2)
    sigma = float(np.sum(np.sqrt(weights) * np.abs(r)))
    noise = abs(loss2 - loss) / loss
    say(f"{name}: fit-set sigma = {sigma:.4e}, loss = {loss:.4e}, relative "
        f"noise of the loss between two blocks = {noise:.2e}")
    out = {'name': name, 'sigma_fit': sigma, 'loss_noise': noise}

    rows, summ = scan(res_p, jac_p, theta_p)
    show('plain', rows, summ)
    out['plain'] = {'rows': rows, 'summary': summ}

    if lay_w is not None:
        theta_w = to_whitened(lay_p, lay_w, theta_p)
        diff_h = h_difference(lay_p, theta_p, lay_w, theta_w)
        rw = np.asarray(res_w(theta_w))
        diff_r = float(np.linalg.norm(rw - r) / np.linalg.norm(r))
        say(f"{name}: the whitened theta gives the same H to {diff_h:.2e} and "
            f"the same residual to {diff_r:.2e}")
        rows, summ = scan(res_w, jac_w, theta_w)
        show('whitened', rows, summ)
        out['whitened'] = {'rows': rows, 'summary': summ,
                           'h_difference': diff_h, 'r_difference': diff_r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None,
                    help='the saved theta of the fit. Without it only the '
                         'control points are measured.')
    ap.add_argument('--n-fit', type=int, required=True,
                    help='the fit points of the run that saved --theta')
    ap.add_argument('--n-train', type=int, default=200_000,
                    help='the --n-train of that run. The fit set is the first '
                         '--n-fit points of that sample.')
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, required=True,
                    help='the lax.map block of the fit')
    ap.add_argument('--chunk2', type=int, default=None,
                    help='a second block, for the noise of the loss')
    ap.add_argument('--legs', type=int, default=2,
                    help='control points: legs of LM from theta_FS')
    ap.add_argument('--leg', type=int, default=40, help='LM steps in each leg')
    ap.add_argument('--gram-dd', choices=['off', 'inner', 'on'], default='on')
    ap.add_argument('--section-dd', action='store_true')
    ap.add_argument('--prepare-dd', action='store_true',
                    help='form H in dd and round it once, as in run_fit.py')
    ap.add_argument('--h-low', action='store_true',
                    help='carry the low part of H into the Gram stage')
    ap.add_argument('--phi-dd', action='store_true',
                    help='stage 5 in dd, as in run_fit.py')
    ap.add_argument('--end-dd', action='store_true',
                    help='the whole path in dd from dd points, as in run_fit.py')
    ap.add_argument('--w-centered', action='store_true',
                    help='form W from the centered vectors, as in run_fit.py')
    ap.add_argument('--j-rows', action='store_true',
                    help='form J from one row for each point, as in run_fit.py')
    ap.add_argument('--remat-w', action='store_true',
                    help='form W again in the tangent pass, as in run_fit.py, so '
                         'that the control legs fit in memory at k = 32')
    ap.add_argument('--h-base', type=str, default=None,
                    help='the base theta of a fit with --h-base in run_fit.py. '
                         '--theta is then the correction, and only that point '
                         'is measured')
    ap.add_argument('--correction-whitened', action='store_true',
                    help='the correction is in the whitened basis')
    ap.add_argument('--zero-correction', action='store_true',
                    help='measure at a correction of exactly 0, so at the base '
                         'itself. That point does not depend on W, which moves '
                         'between processes.')
    ap.add_argument('--no-whiten', action='store_true',
                    help='leave out the whitened family')
    ap.add_argument('--whiten-file', type=str, default=None,
                    help='W and W^-1 saved by a whitened fit, in place of a new Gram')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    dev = jax.devices()[0]
    say(f"device: {dev.platform} {dev.device_kind}")
    if dev.platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[args.gram_dd]
    analytic.SECTION_DD = args.section_dd
    sections.PREPARE_DD = args.prepare_dd
    analytic.H_LOW = args.h_low
    analytic.PHI_DD = args.phi_dd
    analytic.END_DD = args.end_dd
    analytic.REMAT_W = args.remat_w
    analytic.W_CENTERED = args.w_centered
    analytic.J_ROWS = args.j_rows
    if args.section_dd and args.gram_dd != 'on':
        raise SystemExit('--section-dd needs --gram-dd on')
    if args.end_dd and not (args.h_low and args.prepare_dd and args.gram_dd == 'on'):
        raise SystemExit('--end-dd needs --h-low, --prepare-dd and --gram-dd on')

    lay_p = sections.layout(args.k)
    lay_c = lay_p
    base = None
    if args.h_base is not None:
        if args.theta is None and not args.zero_correction:
            raise SystemExit('--h-base needs --theta, the correction, or '
                             '--zero-correction')
        side = Path(args.h_base).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != lay_p.fingerprint():
            raise SystemExit('the basis of --h-base is absent or not the plain basis')
        base = np.asarray(np.load(args.h_base))
        sections.H_BASE = jax.jit(lambda t: sections.base_matrices(lay_p, t))(
            jnp.asarray(base))
        if args.correction_whitened and args.whiten_file is not None:
            lay_c = whiten.load_layout(args.k, args.whiten_file)
        elif args.correction_whitened:
            lay_c = whiten.whitened_layout(args.k, n_pairs=args.gram_pairs,
                                           seed=args.gram_seed)
    theta_saved = None
    if args.zero_correction:
        theta_saved = jnp.zeros(lay_c.n_theta)
    elif args.theta is not None:
        theta_saved = jnp.asarray(np.load(args.theta))
        if theta_saved.shape != (lay_c.n_theta,):
            raise SystemExit(f'theta has {theta_saved.shape}, k = {args.k} needs '
                             f'({lay_c.n_theta},)')
        side = Path(args.theta).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != lay_c.fingerprint():
            raise SystemExit('the basis of --theta is absent or not the basis '
                             'of this build')

    if args.n_train < args.n_fit:
        raise SystemExit('--n-train must be at least --n-fit')
    sample = data.make(math.ceil(args.n_train / 4), args.seed)
    batch = {key: val[:args.n_fit] for key, val in sample.items()}
    if args.end_dd:
        batch.update(dd_metric.point_extras(batch['points']))
    mass = np.asarray(batch['mass'])
    weights = mass / mass.sum()
    chunk2 = args.chunk2 or max(64, args.chunk - 97)

    if base is not None:
        # One point: the correction on its base. The null direction of J is the
        # scale of the whole H, so the gauge is the base plus the correction.
        res_c = jax.jit(fit.residual_fn(lay_c, batch, args.chunk))
        res_c2 = jax.jit(fit.residual_fn(lay_c, batch, chunk2))
        jac_c = jax.jit(jax.jacfwd(fit.residual_fn(lay_c, batch, args.chunk)))
        start = (to_whitened(lay_p, lay_c, base) if args.correction_whitened
                 else jnp.asarray(base))
        gauge = np.asarray(start) + np.asarray(theta_saved)
        r = np.asarray(res_c(theta_saved))
        r2 = np.asarray(res_c2(theta_saved))
        loss, loss2 = 0.5 * float(r @ r), 0.5 * float(r2 @ r2)
        sigma = float(np.sum(np.sqrt(weights) * np.abs(r)))
        say(f"the correction {args.theta or 0} on {args.h_base}: fit-set sigma = "
            f"{sigma:.4e}, loss = {loss:.4e}, relative noise of the loss between "
            f"two blocks = {abs(loss2 - loss) / loss:.2e}")
        family = 'whitened' if args.correction_whitened else 'plain'
        rows, summ = scan(res_c, jac_c, theta_saved, gauge=gauge)
        show(f'{family} correction', rows, summ)
        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f'model_correction_k{args.k}_seed{args.seed}.json'
        path.write_text(json.dumps({'args': vars(args), 'sigma_fit': sigma,
                                    'rows': rows, 'summary': summ}, indent=1))
        say(f'saved to {path}')
        return
    say(f"k = {args.k}, n_theta = {lay_p.n_theta}, {args.n_fit} of {args.n_train} "
        f"points, seed {args.seed}, blocks {args.chunk} and {chunk2}, Gram "
        f"stage {args.gram_dd}, section dd {args.section_dd}, end to end dd "
        f"{args.end_dd}")

    lay_w = None
    if not args.no_whiten:
        lay_w = whiten.whitened_layout(args.k, n_pairs=args.gram_pairs,
                                       seed=args.gram_seed)
        say(f"whitened layout from {lay_w.gram_id}")

    res_p = jax.jit(fit.residual_fn(lay_p, batch, args.chunk))
    res_p2 = jax.jit(fit.residual_fn(lay_p, batch, chunk2))
    jac_p = jax.jit(jax.jacfwd(fit.residual_fn(lay_p, batch, args.chunk)))
    res_w = jac_w = None
    if lay_w is not None:
        res_w = jax.jit(fit.residual_fn(lay_w, batch, args.chunk))
        jac_w = jax.jit(jax.jacfwd(fit.residual_fn(lay_w, batch, args.chunk)))
    fns = (res_p, res_p2, jac_p, res_w, jac_w)

    points = []
    theta = sections.fubini_study_theta(lay_p)
    for leg in range(args.legs):
        theta, stats = fit.fit(lay_p, batch, theta0=theta, max_steps=args.leg,
                               chunk=args.chunk)
        name = f'{(leg + 1) * args.leg} steps from theta_FS'
        say(f"{name}: leg of {stats['steps']} steps, {stats['seconds']:.0f}s, "
            f"converged {stats['converged']}")
        points.append((name, theta))
    if theta_saved is not None:
        points.append((f'the saved theta {args.theta}', theta_saved))

    results = []
    for name, th in points:
        print(flush=True)
        results.append(measure(name, th, lay_p, lay_w, fns, weights))

    print('\nSummary. "max pred" is the largest decrease that an accurate step')
    print('predicts, as a share of the loss. "rho" and "order" are at that step.')
    print('"best taken" is the largest true decrease among the accurate steps')
    print(f'that optimistix accepts, which needs rho of at least {ACCEPT}.\n')
    print(f"  {'point':>34} {'family':>8} {'sigma':>10} {'GN share':>9} "
          f"{'max pred':>9} {'rho':>8} {'order':>6} {'cross':>6} {'square':>6} "
          f"{'best taken':>10} {'noise':>8}")
    for res in results:
        for fam in ('plain', 'whitened'):
            if fam not in res:
                continue
            sm = res[fam]['summary']
            print(f"  {res['name'][-34:]:>34} {fam:>8} {res['sigma_fit']:>10.3e} "
                  f"{sm['gn_share']:>9.2e} {sm['max_pred']:>9.2e} "
                  f"{sm['rho_at_max_pred']:>8.3f} {sm['order_at_max_pred']:>6.2f} "
                  f"{sm['cross_at_max_pred']:>6.2f} {sm['square_at_max_pred']:>6.2f} "
                  f"{sm['best_accepted']:>10.2e} {res['loss_noise']:>8.1e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'model_k{args.k}_seed{args.seed}.json'
    path.write_text(json.dumps({'args': vars(args), 'results': results}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
