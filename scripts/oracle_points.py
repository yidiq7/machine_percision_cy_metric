"""How much the FP64 points move f, against the arithmetic of the fit.

The sampler leaves each point off X by `abs(P)/scale` of approximately 3e-16,
and no FP64 point can do better. That error is fixed in theta, so it does not
add noise that LM sees. It biases sigma instead, and the bias adds to sigma in
quadrature, so it matters only where sigma approaches it.

At one theta and for a sample of points, the script computes v three ways:

  * fit:     `analytic.volume_at`, the path of the fit, at the FP64 point, with
             the stored R and OmegaOmegabar,
  * at z:    `dd_metric.volume_dd` at the same FP64 point, with R and
             OmegaOmegabar in dd from the closed forms,
  * on X:    `dd_metric.volume_dd` at the point after one Newton step in dd,
             which puts it on X to approximately 1e-31.

So `on X` minus `at z` is the effect of the point, and `fit` minus `at z` is
the error of the arithmetic of the fit. H is the dd pair of
`sections.prepare_parts` in all three.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/oracle_points.py --k 32 \\
        --theta runs_full/theta_k32_seed20260813.npy --points 2000
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


def patch_indices(points):
    """m, the coordinate that Newton corrects, and the two free coordinates."""
    ms, frees = [], []
    for z in np.asarray(points):
        m = dd_metric.newton_index(z)
        patch = int(np.argmin(np.abs(z - 1.0)))
        frees.append(sorted(set(range(4)) - {patch, m}))
        ms.append(m)
    return np.array(ms), np.array(frees)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--points', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261813,
                    help='the sample; the default is the hold-out set of seed 20260813')
    ap.add_argument('--block', type=int, default=32)
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.PHI_DD = True, True, True
    sections.PREPARE_DD = True

    lay = sections.layout(args.k)
    theta = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis')
    batch = data.make(math.ceil(args.points / 4), args.seed)
    batch = {key: val[:args.points] for key, val in batch.items()}
    pts, rr, om = (jnp.asarray(batch[key]) for key in
                   ('points', 'restriction', 'Omega_Omegabar'))
    mass = np.asarray(batch['mass'])
    w = mass / mass.sum()
    ms, frees = patch_indices(pts)
    mats = jax.jit(lambda t: sections.prepare_parts(lay, t))(theta)
    say(f'k = {args.k}, {pts.shape[0]} points of seed {args.seed}, at {args.theta}')

    def fit_one(a):
        return analytic.volume_at(lay, mats, *a)

    def ref_one(a, refined):
        z, m, f0, f1 = a
        z_dd = dd_metric.refine(z, m, 1)[0] if refined else dd.cto_dd(z)
        return dd.to_fp(dd_metric.volume_dd(lay, mats, z_dd, m, (f0, f1)))

    xs_fit = (pts, rr, om)
    xs_ref = (pts, jnp.asarray(ms), jnp.asarray(frees[:, 0]), jnp.asarray(frees[:, 1]))
    v_fit = np.asarray(jax.jit(lambda x: jax.lax.map(fit_one, x, batch_size=args.block))(xs_fit))
    say('fit path done')
    v_z = np.asarray(jax.jit(lambda x: jax.lax.map(lambda a: ref_one(a, False), x,
                                                   batch_size=args.block))(xs_ref))
    say('dd at the FP64 point done')
    v_x = np.asarray(jax.jit(lambda x: jax.lax.map(lambda a: ref_one(a, True), x,
                                                   batch_size=args.block))(xs_ref))
    say('dd on X done')

    mean = float(np.sum(w * v_x))
    f = v_x / mean - 1.0
    sigma = float(np.sum(w * np.abs(f)))
    point = (v_x - v_z) / mean
    arith = (v_fit - v_z) / mean
    rows = {
        'sigma_on_X': sigma,
        'rms_f_on_X': float(np.sqrt(np.sum(w * f * f))),
        'point_rms': float(np.sqrt(np.sum(w * point * point))),
        'point_median': float(np.median(np.abs(point))),
        'arith_rms': float(np.sqrt(np.sum(w * arith * arith))),
        'arith_median': float(np.median(np.abs(arith))),
        'sigma_fit_path': float(np.sum(w * np.abs(v_fit / float(np.sum(w * v_fit)) - 1.0))),
    }
    print(f"\n  sigma of this sample on X, in dd          {rows['sigma_on_X']:.4e}")
    print(f"  sigma of this sample by the fit path       {rows['sigma_fit_path']:.4e}")
    print(f"  rms of f on X                              {rows['rms_f_on_X']:.4e}")
    print(f"  effect of the FP64 point on f: rms {rows['point_rms']:.3e}, "
          f"median {rows['point_median']:.3e}")
    print(f"  error of the fit arithmetic:   rms {rows['arith_rms']:.3e}, "
          f"median {rows['arith_median']:.3e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'points_k{args.k}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
