"""Accuracy of `sections.prepare`, the step theta -> H, and its effect on r.

Every precision oracle of this project takes the block matrices H_r as exact
input, and the end-to-end dd path in `dd_metric.py` does too. `prepare` forms
them in FP64: a sum over the invariant basis, and then H_r = rot H_rep rot^T for
each block that is not an orbit representative. H spans many orders of
magnitude and rot is dense, so a small entry of H_r can get an absolute error
of order eps * max abs(H).

This script forms H in double-double and rounds each entry once. At a saved
theta it compares the residual from the two forms of H, on the fit set of the
run. It also measures how rough the computed residual is as a function of
theta: the change of r when each component of theta moves by one unit in the
last place. A smooth residual changes by approximately 1e-18 there, so any
larger change is round-off, and LM cannot descend below it.

Usage, on the GPU node:
    uv run --extra gpu python -u scripts/oracle_prepare.py --k 28 --n-fit 59312 \\
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

from quartic import analytic, data, dd, fit, sections       # noqa: E402

T0 = time.perf_counter()


def say(text):
    print(f'[{time.perf_counter() - T0:7.0f}s] {text}', flush=True)


def prepare_dd(lay, theta):
    """The block matrices of `sections.prepare`, formed in dd and rounded once.

    theta * scale is exact as a dd product. The sum over the basis and both
    products with rot run in dd, so the only error left in each entry of H_r is
    the final rounding to FP64.
    """
    if lay.whiten_inv:
        raise ValueError('prepare_dd covers the plain basis only')
    th = dd.mul(dd.to_dd(jnp.asarray(theta)), dd.to_dd(jnp.asarray(lay.scale)))
    out = []
    for g in lay.groups:
        basis = jnp.asarray(g.basis)
        his, los = [], []
        for i, sl in enumerate(g.slices):
            n = sl.stop - sl.start
            terms = jnp.moveaxis(basis[i, :n], 0, -1)             # (d, d, n)
            hi, lo = dd.dot_dd_fp((th[0][sl], th[1][sl]), terms, axis=-1)
            his.append(hi)
            los.append(lo)
        rep = (jnp.stack(his)[g.orbit_of], jnp.stack(los)[g.orbit_of])
        rot = jnp.asarray(g.rot)                                   # (b, d, d)
        # T[b, k, j] = sum_l H[b, k, l] rot[b, j, l], which is H rot^T.
        t = dd.dot_dd_fp((rep[0][:, :, None, :], rep[1][:, :, None, :]),
                         rot[:, None, :, :], axis=-1)
        # H_r[b, i, j] = sum_k rot[b, i, k] T[b, k, j], which is rot T.
        t_t = (jnp.swapaxes(t[0], 1, 2)[:, None, :, :],
               jnp.swapaxes(t[1], 1, 2)[:, None, :, :])
        out.append(dd.to_fp(dd.dot_dd_fp(t_t, rot[:, :, None, :], axis=-1)))
    return tuple(out)


def residual_from_mats(lay, mats, batch, chunk):
    """sqrt(w) f from given block matrices, by the per-point path of the fit.

    This is `analytic.volumes_analytic` with `mats` as an input in place of
    `sections.prepare(lay, theta)`.
    """
    def one(args):
        z, restriction, omega = args
        phi, _ = analytic.phi_analytic(lay, mats, z)
        g = restriction @ phi @ jnp.conj(restriction).T
        return jnp.real(jnp.linalg.det(g)) / omega

    xs = (batch['points'], batch['restriction'], batch['Omega_Omegabar'])
    v = jax.lax.map(one, xs, batch_size=chunk)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return jnp.sqrt(weights) * (v / jnp.sum(weights * v) - 1.0)


def h_error(mats_a, mats_b):
    """Two measures of the difference of two forms of H, over every group.

    `max_rel` scales by the largest entry of the group. `entry_rel` is the
    largest relative error of one entry, over the entries above 1e-12 of that
    largest entry.
    """
    max_rel = entry_rel = 0.0
    for a, b in zip(mats_a, mats_b):
        a, b = np.asarray(a), np.asarray(b)
        top = np.abs(b).max()
        diff = np.abs(a - b)
        max_rel = max(max_rel, float(diff.max() / top))
        big = np.abs(b) > 1e-12 * top
        entry_rel = max(entry_rel, float((diff[big] / np.abs(b[big])).max()))
    return max_rel, entry_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--theta', type=str, default=None,
                    help='a saved theta. The default is theta_FS.')
    ap.add_argument('--n-fit', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--steps', type=int, default=3,
                    help='random one-ulp steps of theta for the roughness')
    ap.add_argument('--gram-dd', choices=['off', 'inner', 'on'], default='on')
    ap.add_argument('--section-dd', action='store_true')
    ap.add_argument('--out', type=str, default='runs_modeltest')
    args = ap.parse_args()

    dev = jax.devices()[0]
    if dev.platform != 'gpu':
        raise SystemExit('no GPU. Run with `uv run --extra gpu`.')
    analytic.GRAM_DD = {'off': False, 'inner': 'inner', 'on': True}[args.gram_dd]
    analytic.SECTION_DD = args.section_dd

    lay = sections.layout(args.k)
    if args.theta is None:
        theta = sections.fubini_study_theta(lay)
        where = 'theta_FS'
    else:
        theta = jnp.asarray(np.load(args.theta))
        side = Path(args.theta).with_suffix('.basis')
        if not side.exists() or side.read_text().strip() != lay.fingerprint():
            raise SystemExit('the basis of --theta is absent or not the plain '
                             'basis of this build')
        where = args.theta
    sample = data.make(math.ceil(args.n_train / 4), args.seed)
    batch = {key: val[:args.n_fit] for key, val in sample.items()}
    say(f'k = {args.k}, {args.n_fit} of {args.n_train} points, seed {args.seed}, '
        f'at {where}, Gram stage {args.gram_dd}, section dd {args.section_dd}')

    res_theta = jax.jit(fit.residual_fn(lay, batch, args.chunk))
    res_mats = jax.jit(lambda mats: residual_from_mats(lay, mats, batch, args.chunk))
    make_fp = jax.jit(lambda t: sections.prepare(lay, t))
    make_dd = jax.jit(lambda t: prepare_dd(lay, t))

    mats_fp, mats_dd = make_fp(theta), make_dd(theta)
    max_rel, entry_rel = h_error(mats_fp, mats_dd)
    say(f'H from prepare against H from dd: {max_rel:.3e} of the largest entry, '
        f'{entry_rel:.3e} for the worst entry')

    r_theta = np.asarray(res_theta(theta))
    r_fp = np.asarray(res_mats(mats_fp))
    r_dd = np.asarray(res_mats(mats_dd))
    sq = np.sqrt(np.asarray(batch['mass']) / np.sum(np.asarray(batch['mass'])))
    sigma = lambda r: float(np.sum(sq * np.abs(r)))
    norm = lambda r: float(np.linalg.norm(r))
    say(f'the path from mats repeats the fit residual to '
        f'{norm(r_fp - r_theta) / norm(r_theta):.2e}')
    rows = {'h_max_rel': max_rel, 'h_entry_rel': entry_rel,
            'sigma_fp': sigma(r_fp), 'sigma_dd': sigma(r_dd),
            'norm_r_fp': norm(r_fp), 'norm_r_dd': norm(r_dd),
            'norm_diff': norm(r_fp - r_dd)}
    print(f"\n  fit-set sigma with H from prepare  {rows['sigma_fp']:.4e}")
    print(f"  fit-set sigma with H from dd       {rows['sigma_dd']:.4e}")
    print(f"  norm of r, prepare and dd          {rows['norm_r_fp']:.4e}   "
          f"{rows['norm_r_dd']:.4e}")
    print(f"  norm of the difference of r        {rows['norm_diff']:.4e}, "
          f"{rows['norm_diff'] / rows['norm_r_dd']:.3e} of norm(r)")

    rng = np.random.default_rng(7)
    rough_fp, rough_dd = [], []
    base = np.asarray(theta)
    for _ in range(args.steps):
        sign = rng.choice([-np.inf, np.inf], size=base.size)
        moved = jnp.asarray(np.nextafter(base, sign))
        rough_fp.append(norm(np.asarray(res_mats(make_fp(moved))) - r_fp))
        rough_dd.append(norm(np.asarray(res_mats(make_dd(moved))) - r_dd))
    rows['rough_fp'] = rough_fp
    rows['rough_dd'] = rough_dd
    print(f"\n  change of r for a one-ulp step of every component of theta:")
    print(f"    H from prepare: " + '  '.join(f'{x:.3e}' for x in rough_fp))
    print(f"    H from dd:      " + '  '.join(f'{x:.3e}' for x in rough_dd))
    print(f"  as a share of norm(r): prepare {np.mean(rough_fp) / rows['norm_r_fp']:.2e}, "
          f"dd {np.mean(rough_dd) / rows['norm_r_dd']:.2e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'prepare_k{args.k}_seed{args.seed}.json'
    path.write_text(json.dumps({'args': vars(args), 'result': rows}, indent=1))
    say(f'saved to {path}')


if __name__ == '__main__':
    main()
