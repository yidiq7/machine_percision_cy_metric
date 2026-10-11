"""The check that the local forms of each reference center hold the certified potential.

scripts/cert_run.py takes the forms of p0 and of the level sum from the files
center_<r>.npz of scripts/tm_local_forms.py. This script ties those files to the two files
that define the certified potential, the base (--data) and the forms N_L (--forms):

1. At each test point t on X, it evaluates p0 and each N_L from --data and --forms in arb,
   in homogeneous coordinates z = (t_0, t_1, z_2, 1):
     p0(z)  = sum over blocks b of abs(z^r_b)^2 U^T C_b conj(U),  U = ((z^4)^E)_E,
     N_L(z) = sum over sigma in S_4 and orbits o of abs(w^r_o)^2 psi^T C_o conj(psi),
              w = z o sigma,  psi = F^T ((w^4)^E)_E.
   This path does not use the chart forms, the shift or the split in Delta of
   quartic.cert.forms.
2. The stored forms give at t the value sum over j of Delta^j m^T R_j conj(m), with
   m = ((t - c)^alpha)_alpha and Delta = abs(z_2)^2 - s2, for the base (p0) and for each
   M~_m (sum over L of C(L, m) x0^L N_L), with the c, x0 and s2 of the file.
3. The two values must agree within sum over j of abs(Delta)^j abs(m)^T rad_j abs(m), the
   bound that the radii of the file give.

The first test point of a reference center is the corner farthest from c of its cover
cell farthest from c, so that the terms of high degree count; the other points are random
points of random cells of that center. The script also gives the SHA-256 of --data and
--forms, and with --manifest the SHA-256 and the result of each file.

Usage, on CPU cores (about 20 s for each point and file on one core):
    uv run --extra gpu --with cupy-cuda12x --with python-flint python -u \\
        scripts/cert_local_values.py --cover runs_tm/cover_s0.03_c6_v2.npz \\
        --local /n/netscratch/.../tm_local_f6t_td --data runs_tm/f6t_data.npz \\
        --forms runs_tm/f6t_forms.npz --points 2 --procs 16 --manifest <file>
"""
import argparse
import hashlib
import itertools
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np
from flint import acb, acb_mat, arb, arb_mat, ctx

S4 = list(itertools.permutations(range(4)))
G = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def monomials2(d):
    """The exponents (a, b) of t_0^a t_1^b of degree <= d, in the order of the stored forms."""
    return [(a, s - a) for s in range(d + 1) for a in range(s, -1, -1)]


def upper(x):
    """An upper bound of abs(x) for an arb or acb ball, as a float."""
    a = abs(x)
    return (abs(float(a.mid())) + float(a.rad())) * (1 + 1e-12) + 1e-300


def setup(data_path, forms_path):
    """The base and the forms N_L as arb matrices, from the files that define them."""
    data, forms = np.load(data_path), np.load(forms_path)
    base = []
    for i in range(int(data['n_groups'])):
        frame = data[f'g{i}_frame']
        # The invariant base is in the monomial basis of each block.
        if not np.array_equal(frame, np.eye(frame.shape[0])):
            raise SystemExit(f'base group {i}: the frame is not the identity')
        exps = [tuple(int(x) for x in e) for e in data[f'g{i}_exps']]
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        n = len(exps)
        blocks = []
        for b, r in enumerate(data[f'g{i}_res'].tolist()):
            c = arb_mat(n, n, [arb(float(x)) + arb(float(y))
                               for x, y in zip(hhi[b].ravel(), hlo[b].ravel())])
            blocks.append((tuple(int(x) for x in r), c))
        base.append((exps, blocks))
    levels = []
    for L in [int(x) for x in forms['k1']]:
        groups = {}
        for gi in range(int(forms[f'L{L}_n_groups'])):
            exps = [tuple(int(x) for x in e) for e in forms[f'L{L}_g{gi}_exps']]
            fr = forms[f'L{L}_g{gi}_frame']
            ft = arb_mat(fr.shape[1], fr.shape[0], [arb(float(x)) for x in fr.T.ravel()])
            groups[gi] = (exps, ft)
        orbits = []
        for oi in range(int(forms[f'L{L}_n_orbits'])):
            # The entries of C_o are 90-digit decimal strings; arb holds each in a ball.
            cs = forms[f'L{L}_o{oi}_C']
            orbits.append((int(forms[f'L{L}_o{oi}_group']),
                           tuple(int(x) for x in forms[f'L{L}_o{oi}_r']),
                           arb_mat(cs.shape[0], cs.shape[1], [arb(str(x)) for x in cs.ravel()])))
        levels.append((L, groups, orbits))
    G.update(k0=int(data['k0']), base=base, levels=levels, top=max(L for L, _, _ in levels))


def upow(w, e):
    """prod_a (w_a^4)^e_a."""
    v = acb(1)
    for a in range(4):
        if e[a]:
            v *= w[a] ** (4 * e[a])
    return v


def weight(w, r):
    """prod_a abs(w_a)^(2 r_a)."""
    v = arb(1)
    for a in range(4):
        if r[a]:
            v *= abs(w[a]) ** (2 * r[a])
    return v


def quad_real(c, u):
    """u^T C conj(u) for a real arb_mat C and an acb column u."""
    uc = acb_mat(u.nrows(), 1, [u[i, 0].conjugate() for i in range(u.nrows())])
    return (u.transpose() * (acb_mat(c) * uc))[0, 0]


def global_values(z):
    """p0(z) and N_L(z) for each level L, from --data and --forms."""
    p0 = acb(0)
    for exps, blocks in G['base']:
        u = acb_mat(len(exps), 1, [upow(z, e) for e in exps])
        for r, c in blocks:
            p0 += weight(z, r) * quad_real(c, u)
    nl = {}
    for L, groups, orbits in G['levels']:
        total = acb(0)
        for s in S4:
            w = [z[s[a]] for a in range(4)]
            psi = {gi: acb_mat(ft) * acb_mat(len(exps), 1, [upow(w, e) for e in exps])
                   for gi, (exps, ft) in groups.items()}
            for gi, r, c in orbits:
                total += weight(w, r) * quad_real(c, psi[gi])
        nl[L] = total
    return p0, nl


def stored_values(z, name, pts):
    """sum over j of Delta^j m^T R_j conj(m) of the form `name` at each point, and the bound
    of its error that the radii of the file give. pts holds, for each point, the monomials
    m as a list, abs(m), Delta and abs(Delta)."""
    vals, tols = [acb(0) for _ in pts], [0.0 for _ in pts]
    size = z[f'{name}_j0'].shape[1]
    cols = [(acb_mat(size, 1, ms[:size]), acb_mat(size, 1, [v.conjugate() for v in ms[:size]]))
            for ms, _, _, _ in pts]
    for j in range(4):
        a = z[f'{name}_j{j}']
        rad = z[f'{name}_j{j}_rad']
        if a.shape != (6, size, size) or rad.shape != (size, size):
            raise ValueError(f'{name}_j{j}: shape {a.shape}, radii {rad.shape}')
        # Each td entry is the sum of its three limbs; arb holds that sum in a ball.
        re, im = (acb_mat(sum((arb_mat(size, size, a[p + k].ravel().tolist()) for k in (1, 2)),
                              arb_mat(size, size, a[p].ravel().tolist()))) for p in (0, 3))
        for k, ((m, mc), (_, mabs, dl, dabs)) in enumerate(zip(cols, pts)):
            q = (m.transpose() * (re * mc))[0, 0] + acb(0, 1) * (m.transpose() * (im * mc))[0, 0]
            vals[k] += dl ** j * q
            tols[k] += dabs ** j * float(mabs[:size] @ rad @ mabs[:size])
    return vals, [t * 1.01 for t in tols]


def test_points(r, c_mid, rng):
    """The far corner of the cell of reference r farthest from c, and random cell points."""
    cells = np.flatnonzero(G['ref'] == r)
    side = G['side']
    lo = G['idx'][cells] * side
    off = lo + side / 2 - c_mid
    far = int(np.argmax(np.abs(off).max(axis=1)))
    pts = [lo[far] + side * (off[far] > 0)]
    for _ in range(G['points'] - 1):
        pts.append(lo[rng.integers(len(cells))] + side * rng.uniform(size=4))
    return pts


def one(r):
    ctx.prec = G['prec']
    t0 = time.perf_counter()
    path = Path(G['local']) / f'center_{r}.npz'
    digest = sha256(path) if G['manifest'] else ''
    z = np.load(path)
    if int(z['k0']) != G['k0'] or int(z['top']) != G['top']:
        return r, math.inf, math.inf, digest, 0.0, 'k0 or top differs from the data'
    c = z['c']
    cc = [acb(arb(c[i][0]) + arb(c[i][1]), arb(c[i][2]) + arb(c[i][3])) for i in range(2)]
    c_mid = np.array([c[0][0] + c[0][1], c[0][2] + c[0][3], c[1][0] + c[1][1], c[1][2] + c[1][3]])
    x0 = arb(float(z['x0'][0])) + arb(float(z['x0'][1]))
    s2 = arb(float(z['s2'][0])) + arb(float(z['s2'][1]))
    top = G['top']
    # The monomials go by degree, so the first ones are those of the base, of degree <= k0.
    mons = monomials2(top)
    pts, exact = [], []
    for x in test_points(r, c_mid, np.random.default_rng(G['seed'] + r)):
        t = [acb(float(x[0]), float(x[1])), acb(float(x[2]), float(x[3]))]
        y2 = (-(1 + t[0] ** 4 + t[1] ** 4)).root(4)
        p0, nl = global_values([t[0], t[1], y2, acb(1)])
        mt = []
        for m in range(top + 1):
            g = acb(0)
            for L, val in nl.items():
                if m <= L:
                    g += math.comb(L, m) * x0 ** L * val
            mt.append(g)
        exact.append([p0] + mt)
        ms = [(t[0] - cc[0]) ** a * (t[1] - cc[1]) ** b for a, b in mons]
        delta = abs(y2) ** 2 - s2
        pts.append((ms, np.array([upper(v) for v in ms]), delta, upper(delta)))
    worst, worst_rel = 0.0, 0.0
    names = ['base'] + [f'mt{m}' for m in range(top + 1)]
    for k, name in enumerate(names):
        vals, tols = stored_values(z, name, pts)
        for i, (v, tol) in enumerate(zip(vals, tols)):
            gap = upper(v - exact[i][k])
            worst = max(worst, gap / tol)
            worst_rel = max(worst_rel, gap / max(abs(float(exact[i][k].real.mid())), 1e-300))
    return r, worst, worst_rel, digest, time.perf_counter() - t0, ''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cover', type=str, required=True)
    ap.add_argument('--local', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--forms', type=str, required=True)
    ap.add_argument('--points', type=int, default=2, help='test points for each reference center')
    ap.add_argument('--refs', type=int, nargs='*', default=None,
                    help='only these reference centers (default: every one of the cover)')
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--prec', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--manifest', type=str, default='',
                    help='write the SHA-256 and the result of each file to this text file')
    args = ap.parse_args()
    t0 = time.perf_counter()
    cov = np.load(args.cover)
    refs = args.refs if args.refs else list(range(len(cov['ref_centers'])))
    G.update(local=args.local, prec=args.prec, manifest=bool(args.manifest), seed=args.seed,
             points=max(1, args.points), side=float(cov['side']), ref=np.asarray(cov['ref']),
             idx=np.asarray(cov['idx']))
    ctx.prec = args.prec
    setup(args.data, args.forms)
    sums = {name: sha256(p) for name, p in (('data', args.data), ('forms', args.forms))}
    print(f'data {args.data}: SHA-256 {sums["data"]}', flush=True)
    print(f'forms {args.forms}: SHA-256 {sums["forms"]}', flush=True)
    print(f'k0 {G["k0"]}, levels {[L for L, _, _ in G["levels"]]}; {len(refs)} reference '
          f'centers, {G["points"]} points each; setup {time.perf_counter() - t0:.0f}s', flush=True)
    res = []
    with multiprocessing.get_context('fork').Pool(args.procs) as pool:
        for out in pool.imap_unordered(one, refs):
            res.append(out)
            if not out[1] <= 1 or len(res) % 40 == 0:
                print(f'reference {out[0]}: largest gap over its bound {out[1]:.3e}, over '
                      f'abs(value) {out[2]:.3e} {out[5]}; {out[4]:.0f}s; done {len(res)}',
                      flush=True)
    res.sort()
    bad = [x[0] for x in res if not x[1] <= 1]
    if args.manifest:
        with open(args.manifest, 'w') as fh:
            fh.write(f'# data {args.data} {sums["data"]}\n# forms {args.forms} {sums["forms"]}\n')
            fh.write('# reference, SHA-256 of center_<r>.npz, largest gap over its bound\n')
            for r, w, _, digest, _, _ in res:
                fh.write(f'{r} {digest} {w:.3e}\n')
    print(f'{len(res)} reference centers: largest gap over its bound '
          f'{max(x[1] for x in res):.3e}, largest gap over abs(value) '
          f'{max(x[2] for x in res):.3e}; {len(bad)} above the bound {bad[:10]}; '
          f'{time.perf_counter() - t0:.0f}s', flush=True)
    print('the local forms hold the certified potential' if not bad and len(res) == len(refs)
          else 'THE LOCAL FORM CHECK FAILS', flush=True)


if __name__ == '__main__':
    main()
