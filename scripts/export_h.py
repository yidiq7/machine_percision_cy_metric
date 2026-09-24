"""Save h at full precision, in a form that needs no theta and no basis code.

For each group of blocks with the same m the file holds:

  * `g{i}_exps`: the exponents C of the monomials u^C, where u_a = z_a^4,
  * `g{i}_frame`: the section frame F, so that psi = F^T (u^C) over C,
  * `g{i}_residues`: the residue r of each block,
  * `g{i}_H_hi`, `g{i}_H_lo`: each block matrix H_r as a dd pair.

Then p(z) = sum over the blocks of |z^r|^2 psi^T H_r conj(psi), and the metric
is (1/k) ddbar log p. H already carries the scale of theta, the whitening and
the base of a refined fit, so no other convention enters.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/export_h.py --k 32 \\
        --theta runs_end_remat/theta_k32_seed20260813.npy --out h_k32.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import data, sections, whiten                 # noqa: E402


def potential_from_file(arrays, n_groups, z):
    """p(z) from the saved arrays alone, in numpy FP64."""
    z2 = z * z
    u = z2 * z2
    # Powers by repeated multiplication: a complex ** goes through exp and log.
    top = max(int(arrays[f'g{i}_exps'].max()) for i in range(n_groups))
    table = [np.ones_like(u)]
    for _ in range(top):
        table.append(table[-1] * u)
    table = np.stack(table)                                  # (top + 1, 4)
    total = 0.0
    for i in range(n_groups):
        exps = arrays[f'g{i}_exps']
        mono = np.prod(table[exps, np.arange(4)], axis=1)
        psi = arrays[f'g{i}_frame'].T @ mono
        abs2 = np.abs(z) ** 2
        powers = np.stack([np.ones(4), abs2, abs2 * abs2, abs2 * abs2 * abs2])
        weights = np.prod(powers[arrays[f'g{i}_residues'], np.arange(4)], axis=1)
        h = arrays[f'g{i}_H_hi'] + arrays[f'g{i}_H_lo']
        total += float(np.sum(weights * np.real(np.einsum('i,bij,j->b', psi, h,
                                                          np.conj(psi)))))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, required=True)
    ap.add_argument('--h-base', type=str, default=None)
    ap.add_argument('--whiten', action='store_true')
    ap.add_argument('--gram-pairs', type=int, default=whiten.GRAM_PAIRS)
    ap.add_argument('--gram-seed', type=int, default=whiten.GRAM_SEED)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    sections.PREPARE_DD = True
    plain = sections.layout(args.k)
    lay = (whiten.whitened_layout(args.k, n_pairs=args.gram_pairs, seed=args.gram_seed)
           if args.whiten else plain)
    theta = jnp.asarray(np.load(args.theta))
    side = Path(args.theta).with_suffix('.basis')
    if not side.exists() or side.read_text().strip() != lay.fingerprint():
        raise SystemExit('the basis of --theta is absent or not the basis of this build')
    if args.h_base is not None:
        side = Path(args.h_base).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != plain.fingerprint():
            raise SystemExit('the basis of --h-base is absent or not the plain basis')
        base = jnp.asarray(np.load(args.h_base))
        sections.H_BASE = jax.jit(lambda t: sections.base_matrices(plain, t))(base)

    parts = jax.jit(lambda t: sections.prepare_parts(lay, t))(theta)
    arrays = {}
    for i, (g, (hi, lo)) in enumerate(zip(lay.groups, parts)):
        arrays[f'g{i}_m'] = np.array(g.m)
        arrays[f'g{i}_exps'] = np.asarray(g.exps)
        arrays[f'g{i}_frame'] = np.asarray(g.frame)
        arrays[f'g{i}_residues'] = np.asarray(g.residues)
        arrays[f'g{i}_H_hi'] = np.asarray(hi)
        arrays[f'g{i}_H_lo'] = np.asarray(lo)
    meta = {'k': args.k, 'theta': args.theta, 'h_base': args.h_base,
            'whiten': bool(args.whiten), 'fingerprint': lay.fingerprint(),
            'n_groups': len(lay.groups),
            'convention': 'p(z) = sum over blocks of |z^r|^2 psi^T H_r conj(psi), '
                          'psi = frame^T (u^C), u = z^4, H_r = H_hi + H_lo'}
    arrays['meta'] = np.array(json.dumps(meta))

    # The file alone must give the p of the code path.
    points = np.asarray(data.make(2, 5)['points'])
    mats = tuple(hi for hi, _ in parts)
    worst = max(abs(potential_from_file(arrays, len(lay.groups), z)
                    / float(sections.potential_at(lay, mats, jnp.asarray(z))) - 1.0)
                for z in points)
    np.savez_compressed(args.out, **arrays)
    print(f"saved {args.out}: {len(lay.groups)} groups, "
          f"{sum(a.shape[0] for key, a in arrays.items() if key.endswith('_H_hi'))} "
          f"blocks; p from the file matches the code path to {worst:.1e}")


if __name__ == '__main__':
    main()
