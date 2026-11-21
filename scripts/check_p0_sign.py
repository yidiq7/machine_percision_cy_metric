"""The sign of the base p0 of the certified potential, with no use of the box run.

p0(z) = sum over blocks b of abs(z^r_b)^2 U^T C_b conj(U), U = ((z^4)^E)_E, with the real
symmetric C_b = hhi + hlo of --data (the formula of scripts/cert_local_values.py).

1. Each C_b is positive definite: an LDL^T factorization in exact rationals, no pivot
   exchange, gives only pivots > 0. Then each term of p0 is >= 0.
2. The block with r = 0 has the weight 1 and holds the four powers u_a^(k0/4), so its U is
   not 0 for z != 0. With 1, p0 > 0 on C^4 minus 0.
3. p0 at the point [1 : e^(i pi/4) : 0 : 0] of X, in Arb at 300 bits, is a ball above 0.
   This point and the connectedness of X give the sign of p0 on X even if 1 fails.

For the potential f6t, test 1 fails: only 2 of the 64 blocks are positive definite. Test 3
gives the sign of p0 on X.

Usage, on one CPU core, from the repo root:
    uv run --extra gpu --locked python -u scripts/check_p0_sign.py \\
        --data runs_tm/f6t_data.npz --forms runs_tm/f6t_forms.npz
"""
import argparse
import hashlib
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import acb, acb_mat, arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('clv', ROOT / 'scripts' / 'cert_local_values.py')
clv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clv)


def ldl_pivots(a):
    """The pivots of the LDL^T factorization of a symmetric matrix of Fractions."""
    a = [row[:] for row in a]
    n = len(a)
    piv = []
    for k in range(n):
        d = a[k][k]
        piv.append(d)
        if d <= 0:
            break
        for i in range(k + 1, n):
            f = a[i][k] / d
            if f:
                for j in range(k + 1, n):
                    a[i][j] -= f * a[k][j]
    return piv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--forms', required=True)
    args = ap.parse_args()
    for p in (args.data, args.forms):
        print(f'sha256 {p} {hashlib.sha256(Path(p).read_bytes()).hexdigest()}')

    data = np.load(args.data)
    k0 = int(data['k0'])
    n_blocks, all_pd, r0_ok = 0, True, False
    worst = None
    for i in range(int(data['n_groups'])):
        exps = [tuple(int(x) for x in e) for e in data[f'g{i}_exps']]
        hhi, hlo = data[f'g{i}_hhi'], data[f'g{i}_hlo']
        n = len(exps)
        for b, r in enumerate(data[f'g{i}_res'].tolist()):
            c = [[Fraction(float(hhi[b][p][q])) + Fraction(float(hlo[b][p][q]))
                  for q in range(n)] for p in range(n)]
            sym = all(c[p][q] == c[q][p] for p in range(n) for q in range(n))
            piv = ldl_pivots(c)
            pd = sym and len(piv) == n and all(d > 0 for d in piv)
            ratio = float(min(piv) / max(c[p][p] for p in range(n)))
            n_blocks += 1
            all_pd &= pd
            if worst is None or ratio < worst[0]:
                worst = (ratio, i, tuple(r))
            if not pd:
                print(f'group {i} block {b} r={tuple(r)} n={n}: symmetric={sym}, '
                      f'pivots > 0: {len(piv) == n and all(d > 0 for d in piv)}, '
                      f'min pivot over max diagonal {ratio:.3e}')
            if all(x == 0 for x in r):
                powers = [tuple(k0 // 4 if a == j else 0 for a in range(4)) for j in range(4)]
                r0_ok = pd and all(e in exps for e in powers)
                print(f'block r=0: group {i}, n={n}, positive definite={pd}, '
                      f'holds the four powers u_a^{k0 // 4}: {all(e in exps for e in powers)}')
    print(f'{n_blocks} blocks, all positive definite: {all_pd}; '
          f'smallest min pivot over max diagonal {worst[0]:.3e} (group {worst[1]}, r={worst[2]})')
    print(f'TEST 1+2 (p0 > 0 on C^4 minus 0): {"PASS" if all_pd and r0_ok else "FAIL"}')

    clv.setup(args.data, args.forms)
    ctx.prec = 300
    h = arb(1) / arb(2).sqrt()
    z = [acb(1), acb(h, h), acb(0), acb(0)]
    p = sum((zi ** 4 for zi in z), acb(0))
    p0 = acb(0)
    for exps, blocks in clv.G['base']:
        u = acb_mat(len(exps), 1, [clv.upow(z, e) for e in exps])
        for r, c in blocks:
            p0 += clv.weight(z, r) * clv.quad_real(c, u)
    print(f'P(z) = {p}')
    print(f'p0(z) = {p0}')
    print(f'TEST 3 (p0 > 0 at [1 : e^(i pi/4) : 0 : 0]): {"PASS" if p0.real > 0 else "FAIL"}')


if __name__ == '__main__':
    main()
