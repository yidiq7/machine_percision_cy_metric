"""Where the loss of a potential correction lies, in the singular directions of J.

At a correction C, the Gauss-Newton limit of LM can remove the part of r in the
range of J. This script reports the share of norm(r)^2 in each decade of s over
s_max, and the share outside the range of J, for the column-scaled J of the mixed
form. It does that at C = 0, the base, and at a saved level if one is given.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/correction_share.py --k0 28 \\
        --theta runs_full/theta_k28_seed20260813.npy --k1 32 \\
        --coeffs runs_corr28/corr_k28_level1_k32.npy
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

from quartic import analytic, correction, data, dd_metric, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def load_points(n, seed):
    """n points of the sampler with `seed`, with the dd points of `point_extras`."""
    batch = data.make(math.ceil(n / 4), seed)
    batch = {key: val[:n] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    return batch


def shares(res, c):
    """Singular values of the column-scaled J at c, and where norm(r)^2 lies."""
    r = res(c)
    jac = jax.jit(jax.jacfwd(res))(c)
    norms = jnp.linalg.norm(jac, axis=0)
    q, rr = jnp.linalg.qr(jac / norms)
    u, s, _ = jnp.linalg.svd(rr)
    proj = np.asarray(u.T @ (q.T @ r)) ** 2
    s = np.asarray(s)
    total = float(r @ r)
    rel = s / s[0]
    decades = {}
    for top in range(0, -20, -1):
        mask = (rel <= 10.0 ** top) & (rel > 10.0 ** (top - 1))
        if mask.any():
            decades[f'1e{top - 1}..1e{top}'] = {'count': int(mask.sum()),
                                               'share': float(proj[mask].sum() / total)}
    return {'loss': total, 's_max': float(s[0]), 's_min': float(s[-1]),
            'cond': float(s[0] / s[-1]), 'in_range': float(proj.sum() / total),
            'outside': float(1.0 - proj.sum() / total), 'decades': decades}


def describe(name, out):
    say(f"{name}: loss {out['loss']:.4e}, s_max {out['s_max']:.3e}, cond "
        f"{out['cond']:.3e}; share in the range of J {out['in_range']:.4f}, "
        f"outside {out['outside']:.4f}")
    for key, val in out['decades'].items():
        print(f"    s over s_max in {key}: {val['count']:>5} directions, share "
              f"{val['share']:.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True, help='the plain theta at k0')
    ap.add_argument('--k1', type=int, required=True)
    ap.add_argument('--coeffs', type=str, default=None, help='a saved level at k1')
    ap.add_argument('--n-fit', type=int, default=300_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--fp64-b', action='store_true',
                    help='form b of the tangent in FP64 and not in dd')
    ap.add_argument('--out', type=str, default='runs_corr_share')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.END_DD = True, True, True
    sections.PREPARE_DD = True

    lay0 = sections.layout(args.k0)
    theta0 = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay0.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis at k0')
    parts0 = jax.jit(lambda t: analytic.point_matrices(lay0, t))(theta0)
    lay_c = sections.layout(args.k1, normalize=False)
    alpha = args.k1 / args.k0

    fit_set = load_points(args.n_fit, args.seed)
    xs = (fit_set['points_hi'],
          (fit_set['points_hi'], fit_set['points_lo'], fit_set['patch']))
    state, hi = jax.jit(lambda x: jax.lax.map(
        lambda a: correction.base_state_dd(lay0, parts0, a[0], a[1]), x,
        batch_size=8192))(xs)
    res = correction.residual_fn_mixed(lay_c, fit_set, state, hi, args.chunk,
                                       alpha, dd_b=not args.fp64_b)
    say(f'k0 = {args.k0}, k1 = {args.k1}, n = {lay_c.n_theta}, {args.n_fit} points')

    rows = {'zero': shares(res, jnp.zeros(lay_c.n_theta))}
    describe('at C = 0, the base', rows['zero'])
    if args.coeffs is not None:
        side = Path(args.coeffs).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != lay_c.fingerprint():
            raise SystemExit('the basis of --coeffs is absent or not the layout at k1')
        rows['level'] = shares(res, jnp.asarray(np.load(args.coeffs)))
        describe('at the saved level', rows['level'])

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = '_fp64b' if args.fp64_b else ''
    path = out_dir / f'share_k{args.k0}_k{args.k1}{tag}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
