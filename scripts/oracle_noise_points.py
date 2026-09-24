"""Where the rest of the noise of the residual lives: which points, which stage.

With H in dd, the noise of the computed residual at the stalled theta of k = 28
is 12 to 14 percent of norm(r), and 1 percent of the points carries 76 percent
of it. The Gram stage in dd changes it by 1.6 times and stages 1 to 3 in dd by
nothing, so most of it comes after the Gram sums, or from the rounding of H.

For the points that carry most of the noise, and for as many random points,
the script measures the noise of v from second differences along one random
step, in three forms that differ in the arithmetic only:

  * fit:    `analytic.volume_at`, the path of the fit,
  * phi dd: phi from `dd_metric.phi_dd`, which holds stages 1 to 5 in dd,
            rounded once, and then R phi R^dag, det and the division by
            OmegaOmegabar in FP64 as in the fit,
  * all dd: the same chain with every step in dd.

All three take the same H, formed in dd and rounded once, and the stored FP64
R and OmegaOmegabar. R and OmegaOmegabar do not depend on theta, so their own
errors are fixed and cannot make noise. So the noise of `all dd` is what the
rounding of H alone causes, `phi dd` minus `all dd` is the chain after phi, and
`fit` minus `phi dd` is stages 1 to 5 of the fit.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/oracle_noise_points.py --k 28 \\
        --n-fit 59312 --chunk 2653 --theta runs/theta_k28_seed20260813.npy
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

from quartic import analytic, data, dd, dd_metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def v_phi_dd(lay, mats, z, rr, om):
    """phi in dd and rounded once, then the rest of the chain in FP64."""
    phi, _ = dd_metric.phi_dd(lay, mats, dd.cto_dd(z))
    g = rr @ dd.cto_fp(phi) @ jnp.conj(rr).T
    return jnp.real(jnp.linalg.det(g)) / om


def v_all_dd(lay, mats, z, rr, om):
    """phi, R phi R^dag, det and the division by OmegaOmegabar, all in dd."""
    phi, _ = dd_metric.phi_dd(lay, mats, dd.cto_dd(z))
    rest = dd.cto_dd(rr)
    rest_conj = (rest[0], dd.neg(rest[1]))
    left = analytic._cindex(rest, slice(None), None, slice(None), None)
    mid = analytic._cindex(phi, None, None, slice(None), slice(None))
    right = analytic._cindex(rest_conj, None, slice(None), None, slice(None))
    g = dd_metric._creduce_last_two(dd.cmul(dd.cmul(left, mid), right))
    det = dd.csub(dd.cmul(dd_metric._at(g, 0, 0), dd_metric._at(g, 1, 1)),
                  dd.cmul(dd_metric._at(g, 0, 1), dd_metric._at(g, 1, 0)))
    return dd.to_fp(dd.mul(det[0], dd.reciprocal(dd.to_dd(om))))


def properties(lay, mats, pts, rr, mass, f, sigma):
    """Medians of the properties that can make a point noisy."""
    z = np.asarray(pts)
    ratio = np.abs(z).min(axis=1) / np.abs(z).max(axis=1)

    @jax.jit
    def cond_g(zz, rest):
        phi, _ = analytic.phi_analytic(lay, mats, zz)
        g = rest @ phi @ jnp.conj(rest).T
        ev = jnp.linalg.eigvalsh(0.5 * (g + jnp.conj(g).T))
        return ev[-1] / ev[0]

    conds = np.array([float(cond_g(pts[i], rr[i])) for i in range(z.shape[0])])
    return {
        'mass over the median of all points': float(np.median(mass)),
        'abs(f) over sigma': float(np.median(np.abs(f)) / sigma),
        'min abs(z_a) over max abs(z_a)': float(np.median(ratio)),
        'cond(g|X)': float(np.median(conds)),
        'largest entry of R': float(np.median(np.abs(np.asarray(rr)).max(axis=(1, 2)))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--n-fit', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--points', type=int, default=300,
                    help='points in each of the two sets')
    ap.add_argument('--step', type=float, default=1e-12)
    ap.add_argument('--block', type=int, default=32,
                    help='points in each block of the dd reference')
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD = True
    analytic.SECTION_DD = False
    sections.PREPARE_DD = True

    lay = sections.layout(args.k)
    theta = np.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis')
    sample = data.make(math.ceil(args.n_train / 4), args.seed)
    batch = {key: val[:args.n_fit] for key, val in sample.items()}
    pts, rr, om = batch['points'], batch['restriction'], batch['Omega_Omegabar']
    mass = np.asarray(batch['mass'])
    w = mass / mass.sum()

    u = theta * np.random.default_rng(11).normal(size=theta.size)
    make = jax.jit(lambda t: sections.prepare(lay, t))
    mats = [make(jnp.asarray(theta + c * args.step * u)) for c in (0.0, 1.0, -1.0)]
    say(f'k = {args.k}, {args.n_fit} points, seed {args.seed}, at {args.theta}, '
        f'step {args.step:.0e}')

    def mapped(fn, block):
        # The points go in as arguments, so that XLA holds no large constants.
        return jax.jit(lambda m, p, q, o: jax.lax.map(
            lambda a: fn(lay, m, *a), (p, q, o), batch_size=block))

    fit_all = mapped(analytic.volume_at, args.chunk)
    v0, vp, vm = (np.asarray(fit_all(m, pts, rr, om)) for m in mats)
    mean = float(np.sum(w * v0))
    f = v0 / mean - 1.0
    sigma = float(np.sum(w * np.abs(f)))
    s_fit = (vp + vm - 2.0 * v0) / mean
    share = w * s_fit ** 2
    order = np.argsort(share)[::-1]
    top = order[:args.points]
    rand = np.random.default_rng(5).choice(pts.shape[0], args.points, replace=False)
    say(f'fit-set sigma {sigma:.4e}. The {args.points} noisiest points carry '
        f'{share[top].sum() / share.sum():.3f} of the noise')

    # One compile for each form, shared by the two sets of equal size.
    runs = {'phi dd': mapped(v_phi_dd, args.block),
            'all dd': mapped(v_all_dd, args.block)}

    def over(idx, run):
        sub = (jnp.asarray(pts)[idx], jnp.asarray(rr)[idx], jnp.asarray(om)[idx])
        v = [np.asarray(run(m, *sub)) for m in mats]
        return (v[1] + v[2] - 2.0 * v[0]) / mean

    rows = {}
    for tag, idx in (('noisiest', top), ('random', rand)):
        noise = {'fit': s_fit[idx]}
        for name, run in runs.items():
            noise[name] = over(idx, run)
            say(f'{tag}: {name} done')
        ww = w[idx] / w[idx].sum()
        rows[tag] = {
            'rms': {n: float(np.sqrt(np.sum(ww * s ** 2)) / math.sqrt(6.0))
                    for n, s in noise.items()},
            'median': {n: float(np.median(np.abs(s)) / math.sqrt(6.0))
                       for n, s in noise.items()},
            'properties': properties(lay, mats[0], jnp.asarray(pts)[idx],
                                     jnp.asarray(rr)[idx], mass[idx] / np.median(mass),
                                     f[idx], sigma),
        }

    print('\n  noise of f at one evaluation, by form of the arithmetic:')
    print(f"  {'points':>9} {'measure':>8} {'fit':>11} {'phi dd':>11} {'all dd':>11}")
    for tag, row in rows.items():
        for meas in ('rms', 'median'):
            vals = row[meas]
            print(f"  {tag:>9} {meas:>8} {vals['fit']:>11.3e} {vals['phi dd']:>11.3e} "
                  f"{vals['all dd']:>11.3e}")
    print('\n  medians of the properties of the two sets:')
    for key in rows['noisiest']['properties']:
        print(f"  {key:>36}   noisiest {rows['noisiest']['properties'][key]:.3e}   "
              f"random {rows['random']['properties'][key]:.3e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'noise_points_k{args.k}_seed{args.seed}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
