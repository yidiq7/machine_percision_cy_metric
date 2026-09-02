"""Relative error of the LM gradient g = J^T r, at a saved theta.

`oracle_jacobian.py` measures the error of each entry of `J`, and that is not
what stops the iteration. LM descends on `g = J^T r`. Near a minimum `|g|` goes
to zero, so the relative error of `g` grows without bound even when every entry
of `J` is exact to 4e-15. The fit stalls where `|dg| ~ |g|`, because no descent
direction survives the round-off from that point on.

The test needs no exact reference. The order of the summation inside `lax.map`
is a perturbation of the arithmetic and nothing else, so two block sizes give
two computed values of the same exact gradient. Their difference is the noise
that LM sees.

**A noisy gradient at a converged theta is normal, and it is not by itself a
fault.** The exact gradient is zero at a minimum, so any computed value there is
pure error. Measured at a converged theta on 4,000 points, the relative error of
`g` is 5.8e-2, 2.1e-1 and 1.0 at k = 4, 6 and 8, where every sigma reproduces
exactly. Use this script to compare `norm(g)` against the noise **part way
through** a fit, or to compare one k against another at equal sigma. Do not read
a large relative error at the end point as the cause of anything.

Usage:
    uv run --extra gpu python scripts/oracle_gradient.py --k 28 \\
        --theta runs/theta_k28_seed20260813.npy --gram-dd on
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, fit, sections, whiten    # noqa: E402


def gradient(lay, theta, batch, chunk, analytic_jac=True):
    """g = J^T r, as the gradient of one half of the squared residual.

    This is what LM descends on. `jax.grad` of `0.5*sum(r^2)` gives it without
    the (N, n_theta) matrix, so a large fit set costs no extra memory.
    """
    residual = fit.residual_fn(lay, batch, chunk, analytic_jac)
    loss = lambda t: 0.5 * jnp.sum(residual(t) ** 2)
    return jax.grad(loss)(theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None,
                    help='a saved theta. The default is Fubini-Study, where the '
                         'gradient is large and nothing is visible.')
    ap.add_argument('--n-fit', type=int, default=59_312,
                    help='fit points. Use the value of the run that saved theta.')
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--blocks', type=int, nargs=2, default=(2761, 2983),
                    help='two lax.map block sizes. They change only the order '
                         'of the summation.')
    ap.add_argument('--gram-dd', choices=['off', 'inner', 'on'], default='on')
    ap.add_argument('--whiten', action='store_true')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    args = ap.parse_args()

    analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[args.gram_dd]

    lay = (whiten.whitened_layout(args.k, n_pairs=args.gram_pairs,
                                  seed=args.gram_seed)
           if args.whiten else sections.layout(args.k))

    if args.theta is None:
        theta = sections.fubini_study_theta(lay)
        where = 'theta_FS'
    else:
        theta = jnp.asarray(np.load(args.theta))
        if theta.shape != (lay.n_theta,):
            raise SystemExit(f'theta has {theta.shape}, k={args.k} needs '
                             f'({lay.n_theta},)')
        side = Path(args.theta).with_suffix('.basis')
        if side.exists() and side.read_text().strip() != lay.fingerprint():
            raise SystemExit('theta was saved against another basis. Match '
                             '--whiten to the run that made it.')
        where = args.theta

    batch = data.make(int(np.ceil(args.n_fit / 4)), args.seed)
    batch = {key: val[:args.n_fit] for key, val in batch.items()}

    print(f'k = {args.k}, {args.n_fit} fit points, at {where}, '
          f'Gram stage {args.gram_dd}, '
          f'basis {"whitened" if args.whiten else "plain"}')
    print(f'two lax.map blocks: {args.blocks[0]} and {args.blocks[1]}\n')

    g = [np.asarray(gradient(lay, theta, batch, b)) for b in args.blocks]
    diff = g[1] - g[0]
    scale = max(np.abs(g[0]).max(), np.abs(g[1]).max())

    print(f"{'quantity':>34}   value")
    print(f"{'max abs component of g':>34}   {scale:.4e}")
    print(f"{'norm of g':>34}   {np.linalg.norm(g[0]):.4e}")
    print(f"{'max abs difference of the two g':>34}   "
          f"{np.abs(diff).max():.4e}")
    print(f"{'norm of the difference':>34}   {np.linalg.norm(diff):.4e}")
    print()
    rel_inf = np.abs(diff).max() / scale
    rel_two = np.linalg.norm(diff) / np.linalg.norm(g[0])
    print(f"{'relative error of g, max norm':>34}   {rel_inf:.4e}")
    print(f"{'relative error of g, 2 norm':>34}   {rel_two:.4e}")
    print()
    if rel_two > 0.3:
        print('The gradient is noise, so LM cannot descend from here. At a')
        print('converged theta that is expected and it is not a fault: the exact')
        print('gradient is zero there. It matters only part way through a fit.')
    elif rel_two > 1e-3:
        print('The gradient carries only a few digits, so the last digits of')
        print('theta are not determined.')
    else:
        print('The gradient is well resolved.')


if __name__ == '__main__':
    main()
