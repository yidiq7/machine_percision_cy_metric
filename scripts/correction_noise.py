"""The error and the noise of the correction primal at a saved level, FP64 psi against dd psi.

At the saved coefficients c, the script evaluates r with psi in FP64 and with psi
in dd on n points of the fit sample, and it reports the norm of their difference.
The second difference r(c + h u) + r(c - h u) - 2 r(c), with u = c times random
normal numbers, estimates the noise of one evaluation of each form, as in
`oracle_noise.py`: norm(s) / sqrt(6).

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/correction_noise.py --k0 28 \\
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True, help='the plain theta at k0')
    ap.add_argument('--k1', type=int, required=True)
    ap.add_argument('--coeffs', type=str, required=True, help='a saved level at k1')
    ap.add_argument('--n', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--steps', type=float, nargs='+', default=[1e-12, 1e-10])
    ap.add_argument('--out', type=str, default='runs_corr_noise')
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
    side = Path(args.coeffs).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay_c.fingerprint():
        raise SystemExit('the basis of --coeffs is absent or not the layout at k1')
    c = jnp.asarray(np.load(args.coeffs))
    alpha = args.k1 / args.k0

    batch = data.make(math.ceil(args.n / 4), args.seed)
    batch = {key: val[:args.n] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    xs = (batch['points_hi'], (batch['points_hi'], batch['points_lo'], batch['patch']))
    state, hi = jax.jit(lambda x: jax.lax.map(
        lambda a: correction.base_state_dd(lay0, parts0, a[0], a[1]), x,
        batch_size=4096))(xs)

    forms = {name: jax.jit(correction.residual_fn_mixed(
        lay_c, batch, state, hi, args.chunk, alpha, psi_dd=flag))
        for name, flag in (('fp64 psi', False), ('dd psi', True))}
    r = {name: fn(c) for name, fn in forms.items()}
    base = float(jnp.linalg.norm(r['dd psi']))
    gap = r['fp64 psi'] - r['dd psi']
    say(f"k0 = {args.k0}, k1 = {args.k1}, {args.n} points: norm(r) with dd psi {base:.4e}, "
        f"with FP64 psi {float(jnp.linalg.norm(r['fp64 psi'])):.4e}, norm of the "
        f"difference {float(jnp.linalg.norm(gap)):.3e}, "
        f"{float(jnp.linalg.norm(gap)) / base:.3e} of norm(r)")
    root = jnp.sqrt(batch['mass'] / jnp.sum(batch['mass']))
    per_point = np.abs(np.asarray(gap / root))
    size = np.abs(np.asarray(batch['points_hi']))
    ratio = size.min(axis=1) / size.max(axis=1)
    say(f"difference of f at one point: median {np.median(per_point):.3e}, 99th percentile "
        f"{np.percentile(per_point, 99):.3e}, max {per_point.max():.3e}; median at "
        f"min/max abs(z) below 0.1: {np.median(per_point[ratio < 0.1]):.3e}")

    rng = np.random.default_rng(20261002)
    rows = {'norm_r': base, 'gap': float(jnp.linalg.norm(gap)), 'noise': {}}
    for name, fn in forms.items():
        rows['noise'][name] = []
        for h in args.steps:
            for trial in range(2):
                u = c * jnp.asarray(rng.normal(size=c.shape))
                s = fn(c + h * u) + fn(c - h * u) - 2.0 * r[name]
                noise = float(jnp.linalg.norm(s)) / math.sqrt(6.0)
                rows['noise'][name].append({'h': h, 'noise': noise})
                say(f"{name}: h = {h:.0e}, noise {noise:.3e}, {noise / base:.3e} of norm(r)")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'noise_k{args.k0}_k{args.k1}.json'
    path.write_text(json.dumps({'args': vars(args), 'rows': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
