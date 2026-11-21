"""Exact checks of the two files of the potential, for the S_4 invariance of p0 and the
symmetry of every stored matrix.

For the base file (the export of level_oracle.py --invariant-base):
* each frame is the identity, and each exponent list and each residue list is closed under S_4;
* the residues of one group are distinct;
* for every block b and every permutation s, the block of the residue s(r_b) holds the numbers
  of block b at the permuted places, hhi and hlo bit for bit (stabilizers included);
* hhi and hlo of every block are exactly symmetric, and finite.
For the forms file (level_matrices.py): each C_o is square, its texts (i, j) and (j, i) are the
same string, each text is a decimal number, and each exponent list is closed under S_4.
The script prints the SHA-256 of the bytes that it reads, so the output names the files.

The permutations act on 4-tuples by moving slot i to slot s[i]; the property holds for all of
S_4, so the direction of the action does not matter. Exit status 1 on any failure.

Usage, on one CPU core:
    uv run python -u scripts/check_file_symmetry.py runs_tm/f6t_data.npz runs_tm/f6t_forms.npz
"""
import hashlib
import io
import itertools
import re
import sys

import numpy as np

PERMS = list(itertools.permutations(range(4)))
DECIMAL = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def load(path):
    """The npz file and the SHA-256 of the bytes that the checks read."""
    raw = open(path, 'rb').read()
    print(f'sha256 {path} {hashlib.sha256(raw).hexdigest()}', flush=True)
    return np.load(io.BytesIO(raw))


def act(s, vec):
    out = [0] * 4
    for i, v in enumerate(vec):
        out[s[i]] = v
    return tuple(out)


def check_base(path):
    z = load(path)
    bad = []
    pairs = 0
    for i in range(int(z['n_groups'])):
        frame = z[f'g{i}_frame']
        exps = [tuple(int(x) for x in e) for e in z[f'g{i}_exps']]
        res = [tuple(int(x) for x in r) for r in z[f'g{i}_res']]
        hhi, hlo = z[f'g{i}_hhi'], z[f'g{i}_hlo']
        n = len(exps)
        if frame.shape != (n, n) or not np.array_equal(frame, np.eye(n)):
            bad.append(f'group {i}: the frame is not the identity')
        index = {e: k for k, e in enumerate(exps)}
        if len(index) != n:
            bad.append(f'group {i}: repeated exponents')
        if len(set(res)) != len(res):
            bad.append(f'group {i}: repeated residues')
        if hhi.shape != (len(res), n, n) or hlo.shape != hhi.shape:
            bad.append(f'group {i}: shapes {hhi.shape} {hlo.shape} for {len(res)} blocks of {n}')
            continue
        if not (np.all(np.isfinite(hhi)) and np.all(np.isfinite(hlo))):
            bad.append(f'group {i}: a value that is not finite')
        for b in range(len(res)):
            if not (np.array_equal(hhi[b], hhi[b].T) and np.array_equal(hlo[b], hlo[b].T)):
                bad.append(f'group {i} block {b}: not exactly symmetric')
        for s in PERMS:
            img = [act(s, e) for e in exps]
            if any(e not in index for e in img):
                bad.append(f'group {i}: exponents not closed under {s}')
                continue
            pi = np.array([index[e] for e in img])
            for b, r in enumerate(res):
                target = act(s, r)
                if target not in res:
                    bad.append(f'group {i} block {b}: residue {target} = {s}({r}) missing')
                    continue
                bt = res.index(target)
                if not (np.array_equal(hhi[bt][np.ix_(pi, pi)], hhi[b])
                        and np.array_equal(hlo[bt][np.ix_(pi, pi)], hlo[b])):
                    bad.append(f'group {i} block {b}: block {bt} is not its image under {s}')
                pairs += 1
    print(f'base {path}: {int(z["n_groups"])} groups, {pairs} (block, permutation) pairs '
          f'tested, {len(bad)} failures', flush=True)
    return bad


def check_forms(path):
    z = load(path)
    bad = []
    count = 0
    for k1 in [int(x) for x in z['k1']]:
        for gi in range(int(z[f'L{k1}_n_groups'])):
            exps = {tuple(int(x) for x in e) for e in z[f'L{k1}_g{gi}_exps']}
            if any(act(s, e) not in exps for s in PERMS for e in exps):
                bad.append(f'level {k1} group {gi}: exponents not closed under S_4')
            if not np.all(np.isfinite(z[f'L{k1}_g{gi}_frame'])):
                bad.append(f'level {k1} group {gi}: a frame value that is not finite')
        for oi in range(int(z[f'L{k1}_n_orbits'])):
            c = z[f'L{k1}_o{oi}_C']
            if c.ndim != 2 or c.shape[0] != c.shape[1]:
                bad.append(f'level {k1} orbit {oi}: C of shape {c.shape}')
                continue
            for a in range(c.shape[0]):
                for b in range(c.shape[1]):
                    if not DECIMAL.match(str(c[a, b])):
                        bad.append(f'level {k1} orbit {oi}: C[{a},{b}] is not a decimal text')
                    if str(c[a, b]) != str(c[b, a]):
                        bad.append(f'level {k1} orbit {oi}: C[{a},{b}] differs from C[{b},{a}]')
            count += 1
    print(f'forms {path}: {count} orbit matrices tested, {len(bad)} failures', flush=True)
    return bad


def main():
    bad = check_base(sys.argv[1]) + check_forms(sys.argv[2])
    for line in bad[:20]:
        print('  ', line)
    print('ALL FILE CHECKS PASS' if not bad else 'THE FILE CHECKS FAIL', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
