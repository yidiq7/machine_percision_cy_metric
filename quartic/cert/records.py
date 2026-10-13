"""The record files of scripts/cert_run.py: the columns, and one run from one or more parts.

A record file holds `records` (one row for each final box, with the columns COLUMNS),
`target`, `columns`, `commit` (the git commit of the code of the run) and, for each file
of local forms that the run read, its reference center in `local_refs` and its SHA-256 in
`local_sha`. A run on several GPUs writes one file for each part, and `load` joins them.
"""
import hashlib

import numpy as np

COLUMNS = ('ref', 'depth', 'lo0', 'lo1', 'lo2', 'lo3', 'side', 'delta', 'bound', 'center',
           'polynomial', 'remainder', 'hmax', 'lam_big', 'order', 'a', 'grad_tau', 'grad_g',
           'lam_low')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def load(paths):
    """The records, the target, the commit and {ref: SHA-256} of the parts of one run.

    The parts must have the columns COLUMNS, one target and one commit, and no reference
    center may appear in two parts. A file of a run before the hashes has no commit and no
    hashes; it gives commit None and an empty dict.
    """
    recs, targets, commits, shas, seen = [], set(), set(), {}, set()
    for path in paths:
        z = np.load(path)
        if tuple(str(c) for c in z['columns']) != COLUMNS:
            raise ValueError(f'{path}: the columns differ from quartic.cert.records.COLUMNS')
        targets.add(float(z['target']))
        commits.add(str(z['commit']) if 'commit' in z.files else None)
        part = {}
        if 'local_sha' in z.files:
            part = {int(r): str(s) for r, s in zip(z['local_refs'], z['local_sha'])}
        refs = set(np.unique(z['records'][:, 0]).astype(int).tolist()) | set(part)
        if refs & seen:
            raise ValueError(f'{path}: a reference center of this part is also in another part')
        seen |= refs
        shas.update(part)
        recs.append(np.asarray(z['records']))
    if len(targets) != 1 or len(commits) != 1:
        raise ValueError(f'the parts have targets {targets} and commits {commits}')
    return np.concatenate(recs), targets.pop(), commits.pop(), shas
