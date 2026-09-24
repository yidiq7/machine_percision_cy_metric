"""How far one ulp of theta moves the residual, against the residual itself.

theta is FP64, so the fit can only reach the points of the FP64 grid of theta.
A random step of one ulp in each component moves r by approximately J delta.
If that change is of the size of norm(r), the grid of theta, and not the
model, can limit the fit. The second difference at the same step checks that
the change is linear, and so that it is not noise.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/oracle_lattice.py --k 32 --n-fit 76259 \\
        --chunk 1024 --end-dd --theta runs_end/theta_k32_seed20260813.npy
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

from quartic import analytic, data, dd_metric, fit, sections   # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--n-fit', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--ulps', type=float, nargs='+', default=[1.0, 16.0])
    ap.add_argument('--end-dd', action='store_true')
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    if jax.devices()[0].platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD, analytic.H_LOW, analytic.PHI_DD = True, True, True
    analytic.END_DD = args.end_dd
    sections.PREPARE_DD = True

    lay = sections.layout(args.k)
    theta = np.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the plain basis')
    sample = data.make(math.ceil(args.n_train / 4), args.seed)
    batch = {key: val[:args.n_fit] for key, val in sample.items()}
    if args.end_dd:
        batch.update(dd_metric.point_extras(batch['points']))
    say(f'k = {args.k}, {args.n_fit} fit points of seed {args.seed}, at {args.theta}, '
        f'end to end dd {args.end_dd}')

    res = jax.jit(fit.residual_fn(lay, batch, args.chunk))
    r0 = np.asarray(res(jnp.asarray(theta)))
    norm_r = float(np.linalg.norm(r0))
    ulp = np.spacing(np.abs(theta))
    say(f'norm(r) = {norm_r:.4e}; ulp of theta: median {np.median(ulp):.2e}, '
        f'max {ulp.max():.2e}; median abs(theta) {np.median(np.abs(theta)):.2e}')

    rng = np.random.default_rng(7)
    rows = []
    for size in args.ulps:
        first, second = [], []
        for _ in range(args.draws):
            step = size * ulp * rng.choice([-1.0, 1.0], size=theta.size)
            plus = np.asarray(res(jnp.asarray(theta + step)))
            minus = np.asarray(res(jnp.asarray(theta - step)))
            first.append(float(np.linalg.norm(plus - minus) / 2.0))
            second.append(float(np.linalg.norm(plus + minus - 2.0 * r0) / math.sqrt(6.0)))
        row = {'ulps': size, 'first': first, 'second': second}
        rows.append(row)
        print(f"  {size:>5.0f} ulp: change of r {np.mean(first):.3e} "
              f"({np.mean(first) / norm_r:.3e} of norm(r)); second difference "
              f"{np.mean(second):.3e}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'lattice_k{args.k}_seed{args.seed}.json'
    path.write_text(json.dumps({'args': vars(args), 'norm_r': norm_r, 'rows': rows},
                               indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
