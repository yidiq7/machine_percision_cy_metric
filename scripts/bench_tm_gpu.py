"""The time of one product of Taylor models of `quartic.tm_gpu` on the GPU, by order and batch.

The product is the main cost of a box: approximately 230 of them for each box. The
script reports the pair count, the time for each product and box, and the peak
memory that JAX reports.

Usage, on one GPU node:
    uv run --extra gpu python -u scripts/bench_tm_gpu.py --orders 16 20 24 26 --batches 8 32
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax                                                  # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                     # noqa: E402
from quartic import tm_gpu as tg                            # noqa: E402


def random_tm(sp, b, seed):
    rng = np.random.default_rng(seed)
    scale = 1.3 ** -sp.deg[None, :]
    parts = [jnp.asarray(rng.normal(size=(b, sp.size)) * scale) for _ in range(2)]
    zero = jnp.zeros((b, sp.size))
    c = ((parts[0], zero), (parts[1], zero))
    return tg.from_coeffs(c, jnp.full((b,), 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', type=int, nargs='+', default=[16, 20, 24])
    ap.add_argument('--batches', type=int, nargs='+', default=[8, 32])
    ap.add_argument('--repeat', type=int, default=5)
    args = ap.parse_args()
    print(jax.devices(), flush=True)
    for n in args.orders:
        t0 = time.perf_counter()
        sp = tg.Space(4, n)
        sp.buckets()
        pairs = sum(len(k) * w for k, _, _, w in sp.buckets())
        print(f'order {n}: {sp.size} coefficients, {pairs:.3e} padded pairs, tables '
              f'{time.perf_counter() - t0:.1f}s', flush=True)
        for b in args.batches:
            x, y = random_tm(sp, b, 1), random_tm(sp, b, 2)
            fn = jax.jit(lambda a, c: tg.mul(sp, a, c))
            t1 = time.perf_counter()
            out = fn(x, y)
            jax.block_until_ready(out.rem)
            compile_s = time.perf_counter() - t1
            t1 = time.perf_counter()
            for _ in range(args.repeat):
                out = fn(x, y)
            jax.block_until_ready(out.rem)
            each = (time.perf_counter() - t1) / args.repeat
            stats = jax.devices()[0].memory_stats() or {}
            peak = stats.get('peak_bytes_in_use', 0) / 1e9
            print(f'    batch {b}: compile {compile_s:.1f}s, {each * 1e3:.1f} ms for each product, '
                  f'{each / b * 1e3:.2f} ms for each product and box, '
                  f'{pairs * b / each:.2e} pairs/s, peak {peak:.1f} GB', flush=True)


if __name__ == '__main__':
    main()
